import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramidNetwork(nn.Module):
    """
    Implementation of Feature Pyramid Networks (FPN).
    Reference: https://arxiv.org/abs/1612.03144

    Args:
        in_channels: List of input channel counts for each scale
        out_channels: Output channel count (uniform across scales)
        num_outs: Number of output feature maps
        start_level: Index of first input scale to use
        end_level: Index of last input scale to use (-1 for all)
        bn: Whether to use batch normalization
    """

    def __init__(
        self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, bn=True
    ):
        super(FeaturePyramidNetwork, self).__init__()

        assert isinstance(in_channels, list), "in_channels must be a list"

        self.input_channels = in_channels
        self.output_channels = out_channels
        self.num_inputs = len(in_channels)
        self.num_outputs = num_outs
        self.start_idx = start_level
        self.end_idx = end_level
        self.use_bn = bn

        # Determine backbone end level
        if end_level == -1:
            self.backbone_end_idx = self.num_inputs
            assert num_outs >= self.num_inputs - start_level
        else:
            self.backbone_end_idx = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level

        # Build lateral and output convolution layers
        self.lateral_layers = nn.ModuleList()
        self.output_layers = nn.ModuleList()

        for idx in range(self.start_idx, self.backbone_end_idx):
            lateral = ConvBlock(
                in_channels[idx],
                out_channels,
                kernel_size=1,
                bn=bn,
                bias=not bn,
                same_padding=True,
            )

            output = ConvBlock(
                out_channels,
                out_channels,
                kernel_size=3,
                bn=bn,
                bias=not bn,
                same_padding=True,
            )

            self.lateral_layers.append(lateral)
            self.output_layers.append(output)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize convolution weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, feature_maps):
        """
        Forward pass through FPN.

        Args:
            feature_maps: List of input feature maps from backbone

        Returns:
            Tuple of output feature maps
        """
        assert len(feature_maps) == len(self.input_channels)

        # Apply lateral convolutions
        lateral_features = [
            lateral_layer(feature_maps[i + self.start_idx])
            for i, lateral_layer in enumerate(self.lateral_layers)
        ]

        # Build top-down pathway with skip connections
        num_levels = len(lateral_features)
        for i in range(num_levels - 1, 0, -1):
            target_shape = lateral_features[i - 1].shape[2:]
            upsampled = F.interpolate(
                lateral_features[i], size=target_shape, mode="nearest"
            )
            lateral_features[i - 1] = lateral_features[i - 1] + upsampled

        # Apply output convolutions
        outputs = [
            self.output_layers[i](lateral_features[i]) for i in range(num_levels)
        ]

        return tuple(outputs)


class ConvBlock(nn.Module):
    """
    Convolutional block with optional batch normalization and activation.

    Args:
        in_channels: Input channel count
        out_channels: Output channel count
        kernel_size: Convolution kernel size
        stride: Convolution stride
        NL: Nonlinearity type ('relu', 'prelu', or None)
        same_padding: Whether to use same padding
        bn: Whether to use batch normalization
        bias: Whether to use bias in convolution
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        NL="relu",
        same_padding=False,
        bn=True,
        bias=True,
    ):
        super(ConvBlock, self).__init__()

        padding = int((kernel_size - 1) // 2) if same_padding else 0

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias
        )

        self.batch_norm = nn.BatchNorm2d(out_channels) if bn else None

        # Configure activation function
        if NL == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif NL == "prelu":
            self.activation = nn.PReLU()
        else:
            self.activation = None

    def forward(self, x):
        """Forward pass through convolution block."""
        x = self.conv(x)

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        if self.activation is not None:
            x = self.activation(x)

        return x
