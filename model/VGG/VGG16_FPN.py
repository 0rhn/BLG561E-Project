import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from misc.utils import *
from misc.layer import *
from model.necks import FeaturePyramidNetwork


class ResidualBlock(nn.Module):
    """
    Residual block with dilated convolutions and normalization.

    Args:
        in_dim: Input channel dimension
        out_dim: Output channel dimension
        dilation: Dilation rate for convolutions
        norm: Normalization type ('bn' for BatchNorm)
    """

    def __init__(self, in_dim, out_dim, dilation=0, norm="bn"):
        super(ResidualBlock, self).__init__()

        if dilation == 0:
            self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Conv2d(
                in_dim, out_dim, kernel_size=3, padding=dilation, dilation=dilation
            )

        if norm == "bn":
            self.norm1 = nn.BatchNorm2d(out_dim)
        else:
            self.norm1 = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

        if dilation == 0:
            self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        else:
            self.conv2 = nn.Conv2d(
                out_dim, out_dim, kernel_size=3, padding=dilation, dilation=dilation
            )

        if norm == "bn":
            self.norm2 = nn.BatchNorm2d(out_dim)
        else:
            self.norm2 = nn.Identity()

        # Projection shortcut if dimensions don't match
        if in_dim != out_dim:
            self.shortcut = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """Forward pass with residual connection."""
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.relu(out)

        return out


class VGG16FeatureExtractor(nn.Module):
    """
    VGG16 backbone with Feature Pyramid Network for multi-scale features.

    Args:
        pretrained: Whether to load ImageNet pretrained weights
    """

    def __init__(self, pretrained=True):
        super(VGG16FeatureExtractor, self).__init__()

        # Load pretrained VGG16 with batch normalization
        vgg_model = models.vgg16_bn(
            weights=models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        )
        vgg_features = list(vgg_model.features.children())

        # Extract hierarchical features at different scales
        self.stage1 = nn.Sequential(*vgg_features[0:23])  # Up to conv4_3
        self.stage2 = nn.Sequential(*vgg_features[23:33])  # Up to conv5_1
        self.stage3 = nn.Sequential(*vgg_features[33:43])  # Up to conv5_3

        # Feature Pyramid Network for multi-scale fusion
        input_channels = [256, 512, 512]
        self.feature_pyramid = FeaturePyramidNetwork(
            input_channels, out_channels=256, num_outs=len(input_channels)
        )

        # Feature refinement head
        self.refinement_head = nn.Sequential(
            nn.Dropout2d(0.2),
            ResidualBlock(in_dim=768, out_dim=384, dilation=0, norm="bn"),
            ResidualBlock(in_dim=384, out_dim=256, dilation=0, norm="bn"),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, input_image):
        """
        Extract multi-scale features from input image.

        Args:
            input_image: Input tensor [B, 3, H, W]

        Returns:
            List of feature maps at different scales
        """
        # Extract hierarchical features
        feature_list = []
        feat1 = self.stage1(input_image)
        feature_list.append(feat1)

        feat2 = self.stage2(feat1)
        feature_list.append(feat2)

        feat3 = self.stage3(feat2)
        feature_list.append(feat3)

        # Apply FPN for multi-scale fusion
        fpn_features = self.feature_pyramid(feature_list)

        # Align all FPN features to same spatial resolution
        aligned_features = []
        aligned_features.append(
            F.interpolate(
                fpn_features[0], scale_factor=0.25, mode="bilinear", align_corners=True
            )
        )
        aligned_features.append(
            F.interpolate(
                fpn_features[1], scale_factor=0.5, mode="bilinear", align_corners=True
            )
        )
        aligned_features.append(fpn_features[2])

        # Concatenate multi-scale features
        multi_scale_features = torch.cat(aligned_features, dim=1)

        # Refine concatenated features
        refined_features = self.refinement_head(multi_scale_features)

        # Return all feature levels plus refined features
        output_features = list(fpn_features)
        output_features.append(refined_features)

        return output_features
