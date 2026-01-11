# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np


class DiamondKernelInflation(nn.Module):
    """
    Diamond-shaped kernel inflation for expanding point annotations.

    Args:
        kernel_size: Size of the diamond kernel
        stride: Convolution stride
        padding: Padding size (auto-calculated if None)
    """

    def __init__(self, kernel_size=15, stride=1, padding=None):
        super(DiamondKernelInflation, self).__init__()

        # Create diamond-shaped kernel
        kernel_weights = np.zeros((kernel_size, kernel_size))
        center_pos = (kernel_size - 1) / 2

        for row in range(kernel_size):
            for col in range(kernel_size):
                # Diamond shape: Manhattan distance from center
                manhattan_dist = abs(row - center_pos) + abs(col - center_pos)
                if manhattan_dist <= center_pos:
                    kernel_weights[row, col] = 1

        # Auto-calculate padding if not provided
        if padding is None:
            padding = kernel_size // 2

        # Create convolution layer
        self.inflation_conv = nn.Conv2d(
            1, 1, kernel_size, stride=stride, padding=padding, bias=False
        )

        # Set kernel weights
        weight_tensor = torch.from_numpy(
            kernel_weights.reshape(1, 1, kernel_size, kernel_size).astype(np.float32)
        )
        self.inflation_conv.weight = torch.nn.Parameter(weight_tensor)

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input_map):
        """Apply diamond kernel inflation."""
        # Add batch and channel dimensions
        inflated = input_map.unsqueeze(0).unsqueeze(0)
        inflated = self.inflation_conv(inflated)
        return inflated.squeeze()


class RegionExpansion(torch.nn.Module):
    """
    Region expansion using average pooling.

    Expands regions in the input map using a 15x15 average pooling kernel.
    """

    def __init__(self):
        super(RegionExpansion, self).__init__()

        self.expansion_pool = torch.nn.AvgPool2d(15, stride=1, padding=7)

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input_map):
        """Apply region expansion."""
        # Add batch dimension
        expanded = input_map.unsqueeze(0)
        expanded = self.expansion_pool(expanded)
        return expanded.squeeze()
