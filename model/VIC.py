import torch
import torch.nn as nn
from functools import partial
from model.points_from_den import *
from misc.layer import Gaussianlayer
from model.ViT.models_crossvit import CrossAttnModule, FeatureMerger
from model.VGG.VGG16_FPN import VGG16FeatureExtractor
from model.ResNet.ResNet50_FPN import ResNet50FeatureExtractor
from model.decoder import DensityDecoder, SharedRegionDecoder, FlowDecoder


class VideoIndividualCounter(nn.Module):
    """
    Video Individual Counting model for drone footage.
    Uses cross-attention between frames to identify shared regions and track individuals.

    Args:
        cfg: Main configuration object
        cfg_data: Dataset-specific configuration
    """

    def __init__(self, cfg, cfg_data):
        super(VideoIndividualCounter, self).__init__()

        self.config = cfg
        self.data_config = cfg_data

        # Initialize backbone feature extractor
        self.feature_extractor = self._build_backbone(cfg.encoder)

        # Cross-attention modules for temporal feature matching
        normalization_layer = partial(nn.LayerNorm, eps=1e-6)

        self.temporal_cross_attention = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        CrossAttnModule(
                            cfg.cross_attn_embed_dim,
                            cfg.cross_attn_num_heads,
                            cfg.mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            norm_layer=normalization_layer,
                        )
                        for _ in range(cfg.cross_attn_depth)
                    ]
                )
                for _ in range(3)  # 3 pyramid levels
            ]
        )

        self.cross_attention_norm = normalization_layer(cfg.cross_attn_embed_dim)

        # Feature fusion module
        self.feature_merger = FeatureMerger(self.config.FEATURE_DIM)

        # Decoders for different density map types
        self.global_density_decoder = DensityDecoder()
        self.shared_region_decoder = SharedRegionDecoder()
        self.flow_density_decoder = FlowDecoder()

        # Loss and Gaussian layer
        self.loss_function = torch.nn.MSELoss()
        self.gaussian_layer = Gaussianlayer()

    def _build_backbone(self, encoder_type):
        """
        Build feature extraction backbone based on configuration.

        Args:
            encoder_type: Type of encoder ('VGG16_FPN', 'ResNet_50_FPN', etc.)

        Returns:
            Feature extractor module
        """
        if encoder_type == "VGG16_FPN":
            return VGG16FeatureExtractor()
        elif encoder_type == "ResNet_50_FPN":
            return ResNet50FeatureExtractor()
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

    def forward(self, image_batch, targets):
        """
        Forward pass for training.

        Args:
            image_batch: Batch of image pairs [B*2, 3, H, W]
            targets: List of target dictionaries with points and masks

        Returns:
            Tuple of (predicted_global_density, gt_global_density,
                     predicted_shared_density, gt_shared_density,
                     predicted_flow_density, gt_flow_density, loss_dict)
        """
        # Extract multi-scale features
        pyramid_features = self.feature_extractor(image_batch)
        batch_size, channels, feat_height, feat_width = pyramid_features[-1].shape
        self.feature_scale_factor = feat_height / image_batch.shape[2]

        # Decode global density from highest level features
        predicted_global_density = self.global_density_decoder(pyramid_features[-1])

        # Initialize loss dictionary and ground truth maps
        loss_components = {}
        gt_flow_dot_map = torch.zeros_like(predicted_global_density)
        gt_shared_dot_map = torch.zeros_like(predicted_global_density)

        # Process frame pairs
        num_pairs = image_batch.size(0) // 2
        assert image_batch.size(0) % 2 == 0, "Batch size must be even (frame pairs)"

        # Apply cross-attention across pyramid levels
        shared_features = None
        for level_idx in range(len(self.temporal_cross_attention)):
            pair_features = []

            # Fuse with previous level if available
            if shared_features is not None:
                fused_features = self.feature_merger(
                    shared_features, pyramid_features[level_idx]
                )

            # Process each frame pair
            for pair_idx in range(num_pairs):
                # Get query and key-value features for first frame
                if shared_features is not None:
                    query_frame1 = (
                        fused_features[pair_idx * 2]
                        .unsqueeze(0)
                        .flatten(2)
                        .permute(0, 2, 1)
                        .contiguous()
                    )
                else:
                    query_frame1 = (
                        pyramid_features[level_idx][pair_idx * 2]
                        .unsqueeze(0)
                        .flatten(2)
                        .permute(0, 2, 1)
                        .contiguous()
                    )

                key_value_frame1 = (
                    pyramid_features[level_idx][pair_idx * 2 + 1]
                    .unsqueeze(0)
                    .flatten(2)
                    .permute(0, 2, 1)
                    .contiguous()
                )

                # Apply cross-attention layers
                for attn_layer in self.temporal_cross_attention[level_idx]:
                    query_frame1 = attn_layer(query_frame1, key_value_frame1)

                query_frame1 = self.cross_attention_norm(query_frame1)

                # Get query and key-value features for second frame
                if shared_features is not None:
                    query_frame2 = (
                        fused_features[pair_idx * 2 + 1]
                        .unsqueeze(0)
                        .flatten(2)
                        .permute(0, 2, 1)
                        .contiguous()
                    )
                else:
                    query_frame2 = (
                        pyramid_features[level_idx][pair_idx * 2 + 1]
                        .unsqueeze(0)
                        .flatten(2)
                        .permute(0, 2, 1)
                        .contiguous()
                    )

                key_value_frame2 = (
                    pyramid_features[level_idx][pair_idx * 2]
                    .unsqueeze(0)
                    .flatten(2)
                    .permute(0, 2, 1)
                    .contiguous()
                )

                # Apply cross-attention layers
                for attn_layer in self.temporal_cross_attention[level_idx]:
                    query_frame2 = attn_layer(query_frame2, key_value_frame2)

                query_frame2 = self.cross_attention_norm(query_frame2)

                # Collect results
                pair_features.append(query_frame1)
                pair_features.append(query_frame2)

            # Concatenate and reshape
            shared_features = torch.cat(pair_features, dim=0)
            shared_features = (
                shared_features.permute(0, 2, 1)
                .reshape(batch_size, channels, feat_height, feat_width)
                .contiguous()
            )

        # Build ground truth dot maps for shared regions and flow
        for pair_idx in range(num_pairs):
            frame1_points = targets[pair_idx * 2]["points"]
            frame2_points = targets[pair_idx * 2 + 1]["points"]

            frame1_shared_mask = targets[pair_idx * 2]["share_mask0"]
            frame1_outflow_mask = targets[pair_idx * 2]["outflow_mask"]
            frame2_shared_mask = targets[pair_idx * 2 + 1]["share_mask1"]
            frame2_inflow_mask = targets[pair_idx * 2 + 1]["inflow_mask"]

            # Extract coordinates for shared regions
            frame1_shared_coords = frame1_points[frame1_shared_mask].long()
            frame2_shared_coords = frame2_points[frame2_shared_mask].long()

            gt_shared_dot_map[
                pair_idx * 2, 0, frame1_shared_coords[:, 1], frame1_shared_coords[:, 0]
            ] = 1
            gt_shared_dot_map[
                pair_idx * 2 + 1,
                0,
                frame2_shared_coords[:, 1],
                frame2_shared_coords[:, 0],
            ] = 1

            # Extract coordinates for inflow/outflow
            outflow_coords = frame1_points[frame1_outflow_mask].long()
            inflow_coords = frame2_points[frame2_inflow_mask].long()

            gt_flow_dot_map[
                pair_idx * 2, 0, outflow_coords[:, 1], outflow_coords[:, 0]
            ] = 1
            gt_flow_dot_map[
                pair_idx * 2 + 1, 0, inflow_coords[:, 1], inflow_coords[:, 0]
            ] = 1

        # Decode shared region density
        predicted_shared_density = self.shared_region_decoder(shared_features)

        # Compute flow density from residual
        residual_density = predicted_global_density - predicted_shared_density
        predicted_flow_density = self.flow_density_decoder(residual_density)

        # Compute losses
        # Global density loss
        gt_global_dot_map = torch.zeros_like(predicted_global_density)
        for idx, target_dict in enumerate(targets):
            point_coords = target_dict["points"].long()
            gt_global_dot_map[idx, 0, point_coords[:, 1], point_coords[:, 0]] = 1

        gt_global_density = self.gaussian_layer(gt_global_dot_map)
        assert predicted_global_density.size() == gt_global_density.size()

        global_loss = self.loss_function(
            predicted_global_density, gt_global_density * self.data_config.DEN_FACTOR
        )
        predicted_global_density = (
            predicted_global_density.detach() / self.data_config.DEN_FACTOR
        )
        loss_components["global"] = global_loss

        # Shared region density loss
        gt_shared_density = self.gaussian_layer(gt_shared_dot_map)
        assert predicted_shared_density.size() == gt_shared_density.size()

        shared_loss = self.loss_function(
            predicted_shared_density, gt_shared_density * self.data_config.DEN_FACTOR
        )
        predicted_shared_density = (
            predicted_shared_density.detach() / self.data_config.DEN_FACTOR
        )
        loss_components["share"] = shared_loss * 10  # Weighted loss

        # Flow density loss
        gt_flow_density = self.gaussian_layer(gt_flow_dot_map)
        assert predicted_flow_density.size() == gt_flow_density.size()

        flow_loss = self.loss_function(
            predicted_flow_density, gt_flow_density * self.data_config.DEN_FACTOR
        )
        predicted_flow_density = (
            predicted_flow_density.detach() / self.data_config.DEN_FACTOR
        )
        loss_components["in_out"] = flow_loss

        return (
            predicted_global_density,
            gt_global_density,
            predicted_shared_density,
            gt_shared_density,
            predicted_flow_density,
            gt_flow_density,
            loss_components,
        )

    def test_forward(self, image_batch):
        """
        Forward pass for inference (no ground truth required).

        Args:
            image_batch: Batch of image pairs [B*2, 3, H, W]

        Returns:
            Tuple of (predicted_global_density, predicted_shared_density, predicted_flow_density)
        """
        # Extract multi-scale features
        pyramid_features = self.feature_extractor(image_batch)
        batch_size, channels, feat_height, feat_width = pyramid_features[-1].shape

        # Decode global density
        predicted_global_density = self.global_density_decoder(pyramid_features[-1])

        # Process frame pairs
        num_pairs = image_batch.size(0) // 2
        assert image_batch.size(0) % 2 == 0

        # Apply cross-attention across pyramid levels
        shared_features = None
        for level_idx in range(len(self.temporal_cross_attention)):
            pair_features = []

            # Fuse with previous level if available
            if shared_features is not None:
                fused_features = self.feature_merger(
                    shared_features, pyramid_features[level_idx]
                )

            # Process each frame pair
            for pair_idx in range(num_pairs):
                # First frame cross-attention
                if shared_features is not None:
                    query_frame1 = (
                        fused_features[pair_idx * 2]
                        .unsqueeze(0)
                        .flatten(2)
                        .permute(0, 2, 1)
                        .contiguous()
                    )
                else:
                    query_frame1 = (
                        pyramid_features[level_idx][pair_idx * 2]
                        .unsqueeze(0)
                        .flatten(2)
                        .permute(0, 2, 1)
                        .contiguous()
                    )

                key_value_frame1 = (
                    pyramid_features[level_idx][pair_idx * 2 + 1]
                    .unsqueeze(0)
                    .flatten(2)
                    .permute(0, 2, 1)
                    .contiguous()
                )

                for attn_layer in self.temporal_cross_attention[level_idx]:
                    query_frame1 = attn_layer(query_frame1, key_value_frame1)

                query_frame1 = self.cross_attention_norm(query_frame1)

                # Second frame cross-attention
                if shared_features is not None:
                    query_frame2 = (
                        fused_features[pair_idx * 2 + 1]
                        .unsqueeze(0)
                        .flatten(2)
                        .permute(0, 2, 1)
                        .contiguous()
                    )
                else:
                    query_frame2 = (
                        pyramid_features[level_idx][pair_idx * 2 + 1]
                        .unsqueeze(0)
                        .flatten(2)
                        .permute(0, 2, 1)
                        .contiguous()
                    )

                key_value_frame2 = (
                    pyramid_features[level_idx][pair_idx * 2]
                    .unsqueeze(0)
                    .flatten(2)
                    .permute(0, 2, 1)
                    .contiguous()
                )

                for attn_layer in self.temporal_cross_attention[level_idx]:
                    query_frame2 = attn_layer(query_frame2, key_value_frame2)

                query_frame2 = self.cross_attention_norm(query_frame2)

                pair_features.append(query_frame1)
                pair_features.append(query_frame2)

            shared_features = torch.cat(pair_features, dim=0)
            shared_features = (
                shared_features.permute(0, 2, 1)
                .reshape(batch_size, channels, feat_height, feat_width)
                .contiguous()
            )

        # Decode densities
        predicted_shared_density = self.shared_region_decoder(shared_features)
        residual_density = predicted_global_density - predicted_shared_density
        predicted_flow_density = self.flow_density_decoder(residual_density)

        # Normalize predictions
        predicted_global_density = (
            predicted_global_density.detach() / self.data_config.DEN_FACTOR
        )
        predicted_shared_density = (
            predicted_shared_density.detach() / self.data_config.DEN_FACTOR
        )
        predicted_flow_density = (
            predicted_flow_density.detach() / self.data_config.DEN_FACTOR
        )

        return (
            predicted_global_density,
            predicted_shared_density,
            predicted_flow_density,
        )
