# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class GaussianFilter(nn.Module):
    """
    Gaussian convolution filter for generating density maps from dot maps.

    Args:
        in_channels: Number of input channels
        sigma_list: List of sigma values for Gaussian kernels
        kernel_size: Size of the Gaussian kernel
        stride: Convolution stride
        padding: Padding size
        freeze_weights: Whether to freeze kernel weights
    """

    def __init__(
        self,
        in_channels,
        sigma_list,
        kernel_size=64,
        stride=1,
        padding=0,
        freeze_weights=True,
    ):
        super(GaussianFilter, self).__init__()

        out_channels = len(sigma_list) * in_channels
        center = kernel_size // 2

        # Create Gaussian function generator
        def create_gaussian_func(position):
            return lambda sigma_val: math.exp(
                -((position - center) ** 2) / float(2 * sigma_val**2)
            )

        # Generate Gaussian functions for each position
        gaussian_functions = [create_gaussian_func(pos) for pos in range(kernel_size)]

        # Build Gaussian kernels for each sigma
        kernel_list = []
        for sigma_value in sigma_list:
            # Generate 1D Gaussian
            gaussian_1d = torch.Tensor(
                [func(sigma_value) for func in gaussian_functions]
            )
            gaussian_1d /= gaussian_1d.sum()

            # Convert to 2D Gaussian kernel
            gaussian_1d_col = gaussian_1d.unsqueeze(1)
            gaussian_2d = (
                gaussian_1d_col.mm(gaussian_1d_col.t())
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
            )

            # Expand for all input channels
            expanded_kernel = Variable(
                gaussian_2d.expand(
                    in_channels, 1, kernel_size, kernel_size
                ).contiguous()
            )
            kernel_list.append(expanded_kernel)

        # Stack and reshape kernels
        stacked_kernels = torch.stack(kernel_list)
        stacked_kernels = stacked_kernels.permute(1, 0, 2, 3, 4)
        weight_tensor = stacked_kernels.reshape(
            out_channels, in_channels, kernel_size, kernel_size
        )

        # Create convolution layer
        self.gaussian_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.gaussian_conv.weight = torch.nn.Parameter(weight_tensor)

        if freeze_weights:
            self._freeze_parameters()

    def forward(self, dot_maps):
        """Apply Gaussian filter to dot maps."""
        density_maps = self.gaussian_conv(dot_maps)
        return density_maps

    def _freeze_parameters(self):
        """Freeze all parameters to prevent training."""
        for param in self.parameters():
            param.requires_grad = False


class SumPooling2D(nn.Module):
    """
    2D sum pooling layer (average pooling scaled by kernel area).

    Args:
        kernel_size: Size of pooling kernel
    """

    def __init__(self, kernel_size):
        super(SumPooling2D, self).__init__()

        self.avg_pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)

        # Calculate kernel area
        if isinstance(kernel_size, (tuple, list)):
            self.kernel_area = kernel_size[0] * kernel_size[1]
        else:
            self.kernel_area = kernel_size * kernel_size

    def forward(self, dot_map):
        """Apply sum pooling to input."""
        return self.avg_pool(dot_map) * self.kernel_area
