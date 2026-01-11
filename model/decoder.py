import torch.nn as nn
import torch.nn.functional as F
from misc.utils import *
from misc.layer import *
from model.necks import FeaturePyramidNetwork


class DensityDecoder(nn.Module):
    """
    Global density map decoder with progressive upsampling.
    Decodes features to full-resolution density maps through 4 stages.
    """

    def __init__(self, input_channels=256, num_groups=8):
        super(DensityDecoder, self).__init__()

        self.stage1 = self._build_decode_stage(
            input_channels, input_channels, num_groups
        )
        self.stage2 = self._build_decode_stage(
            input_channels, input_channels, num_groups
        )
        self.stage3 = self._build_decode_stage(
            input_channels, input_channels, num_groups
        )
        self.stage4 = self._build_final_stage(input_channels, num_groups)

    def _build_decode_stage(self, in_ch, out_ch, groups):
        """Build a decoding stage with GroupNorm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def _build_final_stage(self, in_ch, groups):
        """Build final stage with 1x1 conv to produce single-channel output."""
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groups, in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 1, kernel_size=1, stride=1),
        )

    def forward(self, features):
        """
        Progressive upsampling through 4 stages (16x total upsampling).

        Args:
            features: Input feature map

        Returns:
            Single-channel density map
        """
        x = F.interpolate(
            self.stage1(features), scale_factor=2, mode="bilinear", align_corners=False
        )
        x = F.interpolate(
            self.stage2(x), scale_factor=2, mode="bilinear", align_corners=False
        )
        x = F.interpolate(
            self.stage3(x), scale_factor=2, mode="bilinear", align_corners=False
        )
        x = F.interpolate(
            self.stage4(x), scale_factor=2, mode="bilinear", align_corners=False
        )
        return x


class SharedRegionDecoder(nn.Module):
    """
    Decoder for shared region density maps between consecutive frames.
    Uses same architecture as DensityDecoder but for temporal consistency.
    """

    def __init__(self, input_channels=256, num_groups=8):
        super(SharedRegionDecoder, self).__init__()

        self.stage1 = self._build_decode_stage(
            input_channels, input_channels, num_groups
        )
        self.stage2 = self._build_decode_stage(
            input_channels, input_channels, num_groups
        )
        self.stage3 = self._build_decode_stage(
            input_channels, input_channels, num_groups
        )
        self.stage4 = self._build_final_stage(input_channels, num_groups)

    def _build_decode_stage(self, in_ch, out_ch, groups):
        """Build a decoding stage with GroupNorm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def _build_final_stage(self, in_ch, groups):
        """Build final stage with 1x1 conv to produce single-channel output."""
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groups, in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 1, kernel_size=1, stride=1),
        )

    def forward(self, features):
        """
        Progressive upsampling through 4 stages.

        Args:
            features: Input feature map from cross-attention

        Returns:
            Single-channel shared region density map
        """
        x = F.interpolate(
            self.stage1(features), scale_factor=2, mode="bilinear", align_corners=False
        )
        x = F.interpolate(
            self.stage2(x), scale_factor=2, mode="bilinear", align_corners=False
        )
        x = F.interpolate(
            self.stage3(x), scale_factor=2, mode="bilinear", align_corners=False
        )
        x = F.interpolate(
            self.stage4(x), scale_factor=2, mode="bilinear", align_corners=False
        )
        return x


class FlowDecoder(nn.Module):
    """
    Decoder for inflow/outflow density maps.
    Processes residual between global and shared densities.
    """

    def __init__(self, input_channels=1, hidden_channels=256, num_groups=8):
        super(FlowDecoder, self).__init__()

        self.flow_head = nn.Sequential(
            nn.Conv2d(
                input_channels, hidden_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.GroupNorm(num_groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.GroupNorm(num_groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.GroupNorm(num_groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1),
        )

    def forward(self, residual_density):
        """
        Process residual density to extract inflow/outflow.

        Args:
            residual_density: Difference between global and shared densities

        Returns:
            Inflow/outflow density map
        """
        return self.flow_head(residual_density)
