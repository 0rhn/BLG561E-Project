import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIMatchingExtractor:
    """
    Extract regions of interest and matching information between frame pairs.

    Args:
        train_size: Tuple of (height, width) for training images
        radius: Radius for ROI extraction around each point
    """

    def __init__(self, train_size, radius=8):
        self.height = train_size[0]
        self.width = train_size[1]
        self.roi_radius = radius

    def __call__(self, frame_a_target, frame_b_target, noise=None, shape=None):
        """
        Extract ROIs and matching information for a pair of frames.

        Args:
            frame_a_target: Target dict for first frame
            frame_b_target: Target dict for second frame
            noise: Noise type to add ('ab', 'a', 'b', or None)
            shape: Optional shape override (height, width)

        Returns:
            match_info: Dictionary with matching indices
            rois: Concatenated ROIs for both frames
        """
        points_a = frame_a_target["points"]
        points_b = frame_b_target["points"]

        # Update dimensions if shape provided
        if shape is not None:
            self.height = shape[0]
            self.width = shape[1]

        # Add noise if specified
        if noise == "ab":
            noise_a = torch.randn(points_a.size()).to(points_a) * 2
            noise_b = torch.randn(points_b.size()).to(points_b) * 2
            points_a = points_a + noise_a
            points_b = points_b + noise_b
        elif noise == "a":
            points_a = points_a + torch.randn(points_a.size()).to(points_a)
        elif noise == "b":
            points_b = points_b + torch.randn(points_b.size()).to(points_b)

        # Build ROIs for frame A
        roi_frame_a = torch.zeros(points_a.size(0), 5).to(points_a)
        roi_frame_a[:, 0] = 0  # Batch index
        roi_frame_a[:, 1] = torch.clamp(points_a[:, 0] - self.roi_radius, min=0)
        roi_frame_a[:, 2] = torch.clamp(points_a[:, 1] - self.roi_radius, min=0)
        roi_frame_a[:, 3] = torch.clamp(
            points_a[:, 0] + self.roi_radius, max=self.width - 1
        )
        roi_frame_a[:, 4] = torch.clamp(
            points_a[:, 1] + self.roi_radius, max=self.height - 1
        )

        # Build ROIs for frame B
        roi_frame_b = torch.zeros(points_b.size(0), 5).to(points_b)
        roi_frame_b[:, 0] = 1  # Batch index
        roi_frame_b[:, 1] = torch.clamp(points_b[:, 0] - self.roi_radius, min=0)
        roi_frame_b[:, 2] = torch.clamp(points_b[:, 1] - self.roi_radius, min=0)
        roi_frame_b[:, 3] = torch.clamp(
            points_b[:, 0] + self.roi_radius, max=self.width - 1
        )
        roi_frame_b[:, 4] = torch.clamp(
            points_b[:, 1] + self.roi_radius, max=self.height - 1
        )

        # Concatenate ROIs
        combined_rois = torch.cat([roi_frame_a, roi_frame_b], dim=0)

        # Extract person IDs and find matches
        ids_a = frame_a_target["person_id"]
        ids_b = frame_b_target["person_id"]

        # Compute pairwise ID differences
        id_diff = ids_a.unsqueeze(1).expand(-1, len(ids_b)) - ids_b.unsqueeze(0).expand(
            len(ids_a), -1
        )
        id_diff = id_diff.abs()

        # Find matched and unmatched indices
        matched_idx_a, matched_idx_b = torch.where(id_diff == 0)
        matched_pairs = torch.stack([matched_idx_a, matched_idx_b], dim=1)

        unmatched_a = torch.where(id_diff.min(1)[0] > 0)[0]
        unmatched_b = torch.where(id_diff.min(0)[0] > 0)[0]

        match_info = {"a2b": matched_pairs, "un_a": unmatched_a, "un_b": unmatched_b}

        return match_info, combined_rois


def extract_local_maxima(density_map, gaussian_peak, radius=8.0):
    """
    Extract local maximum points from density map.

    Args:
        density_map: Predicted density map tensor
        gaussian_peak: Maximum value of Gaussian kernel
        radius: Radius for ROI extraction

    Returns:
        Dictionary containing:
            - num: Number of detected points
            - points: Coordinates of detected points
            - rois: ROIs around detected points
    """
    density_map = density_map.detach()
    batch_size, channels, height, width = density_map.size()

    # Apply smoothing with 3x3 average filter
    smoothing_kernel = torch.ones(3, 3) / 9.0
    smoothing_kernel = smoothing_kernel.unsqueeze(0).unsqueeze(0).cuda()
    kernel_weight = nn.Parameter(data=smoothing_kernel, requires_grad=False)
    smoothed_map = F.conv2d(density_map, kernel_weight, stride=1, padding=1)

    # Apply non-maximum suppression
    pooled = F.max_pool2d(smoothed_map, (5, 5), stride=2, padding=2)
    upsampled = F.interpolate(pooled, scale_factor=2)
    local_max_mask = (upsampled == smoothed_map).float()
    filtered_map = local_max_mask * smoothed_map

    # Threshold based on Gaussian peak
    threshold = 0.25 * gaussian_peak
    filtered_map[filtered_map < threshold] = 0
    filtered_map[filtered_map > 0] = 1

    # Count detected points
    point_count = int(torch.sum(filtered_map).item())

    # Extract point coordinates (b, c, h, w -> b, c, w, h)
    point_coords = torch.nonzero(filtered_map)[:, [0, 1, 3, 2]].float()

    # Build ROIs around points
    rois = torch.zeros((point_coords.size(0), 5)).float().to(density_map)
    rois[:, 0] = point_coords[:, 0]  # Batch index
    rois[:, 1] = torch.clamp(point_coords[:, 2] - radius, min=0)
    rois[:, 2] = torch.clamp(point_coords[:, 3] - radius, min=0)
    rois[:, 3] = torch.clamp(point_coords[:, 2] + radius, max=width)
    rois[:, 4] = torch.clamp(point_coords[:, 3] + radius, max=height)

    result = {"num": point_count, "points": point_coords, "rois": rois}

    return result
