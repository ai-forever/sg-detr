"""Module for generating anchors."""

from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from src.utils.span_utils import SpanList


def get_width_cntrs(anchor: np.ndarray) -> Tuple[float, float]:
    """Calculate the width and x center of a given anchor (window).

    Args:
        anchor (np.ndarray): A 1-dimensional array representing the anchor (window).

    Returns:
        Tuple[float, float]: A tuple containing the width and the x center of the anchor.
    """
    width = anchor[1] - anchor[0] + 1
    x_ctr = anchor[0] + 0.5 * (width - 1)
    return width, x_ctr


def make_anchors(width: np.ndarray, x_ctr: float) -> np.ndarray:
    """Generate a set of anchors based on the given widths around a center point.

    Args:
        width (np.ndarray): A vector of widths.
        x_ctr (float): The center x-coordinate around which to generate the anchors.

    Returns:
        np.ndarray: A 2D array representing a set of anchors (windows).
    """
    width = width[:, np.newaxis]
    left_border = x_ctr - 0.5 * (width - 1)
    right_border = x_ctr + 0.5 * (width - 1)
    return np.hstack((left_border, right_border))


def _scale_enum(anchor: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Generate a set of anchors by scaling the base anchor.

    Args:
        anchor (np.ndarray): A 1-dimensional array representing the base anchor.
        scales (np.ndarray): A 1-dimensional array of scaling factors.

    Returns:
        np.ndarray: A 2D array representing the enumerated anchors.
    """
    width, x_center = get_width_cntrs(anchor)
    scaled_width = width * scales
    return make_anchors(scaled_width, x_center)


def generate_anchors(stride: float, sizes: Tuple[int, ...]) -> Tensor:
    """Generate a matrix of anchor boxes in (x1, x2) format.

    Anchors are centered on stride / 2 and have (approximate) sqrt areas of the specified sizes.

    Args:
        stride (float): The stride of the feature map.
        sizes (Tuple[int, ...]): A tuple of anchor sizes.

    Returns:
        Tensor: A tensor representing the generated anchors.
    """
    scales = np.array(sizes, dtype=np.float64) / stride
    anchor = np.array([1, stride], dtype=np.float64) - 0.5
    anchors = _scale_enum(anchor, scales)
    return torch.from_numpy(anchors)


# pylint: disable=abstract-method
class BufferList(nn.Module):
    """A custom module similar to `nn.ParameterList`, but for managing buffers.

    This class stores a list of buffers, which are tensors that do not require gradients.

    Attributes:
        _buffers (OrderedDict): An ordered dictionary containing the registered buffers.
    """

    def __init__(self, buffers: Optional[Iterable[Tensor]] = None) -> None:
        """Initialize the BufferList with an optional iterable of buffers.

        Args:
            buffers (Optional[Iterable[Tensor]]): An optional iterable of buffers to initialize the list with.
        """
        super().__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers: Iterable[Tensor]) -> "BufferList":
        """Add multiple buffers to the BufferList.

        Args:
            buffers (Iterable[Tensor]): An iterable of buffers to be added to the list.

        Returns:
            BufferList: The BufferList instance (self) to allow method chaining.
        """
        offset = len(self)
        for idx, buffer in enumerate(buffers):
            self.register_buffer(str(offset + idx), buffer)
        return self

    def __len__(self) -> int:
        """Return the number of buffers in the list.

        Returns:
            int: The count of buffers.
        """
        return len(self._buffers)

    def __iter__(self):
        """Return an iterator over the buffers.

        Returns:
            Iterator: An iterator over the buffer values.
        """
        return iter(self._buffers.values())


class AnchorGenerator(nn.Module):
    """For a set of image sizes and feature maps, computes a set of anchors."""

    def __init__(
        self,
        anchor_sizes: Tuple[int, ...] = (4, 16, 32, 64),
        anchor_strides: Tuple[float, ...] = (0.5, 1, 2, 4),
    ) -> None:
        """Initialize the AnchorGenerator module.

        Args:
            anchor_sizes (Tuple[int, ...]): A tuple of anchor sizes.
            anchor_strides (Tuple[float, ...]): A tuple of anchor strides.
        """
        super().__init__()
        assert len(anchor_strides) == len(anchor_sizes), "Only support FPN now"
        cell_anchors = [generate_anchors(stride, (size,)) for stride, size in zip(anchor_strides, anchor_sizes)]
        self.strides = anchor_strides
        self.cell_anchors = BufferList(cell_anchors)

    def num_anchors_per_location(self) -> List[int]:
        """Return the number of anchors per location.

        Returns:
            List[int]: A list of integers representing the number of anchors per location.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes: List[int]) -> List[Tensor]:
        """Generate anchors for a set of grid sizes.

        Args:
            grid_sizes (List[int]): A list of grid sizes.

        Returns:
            List[Tensor]: A list of tensors representing the generated anchors.
        """
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            device = base_anchors.device
            shift_x = torch.arange(0, size * stride, step=stride, dtype=torch.float32, device=device)  # noqa: WPS221
            shifts = torch.stack((shift_x, shift_x), dim=1)
            shifts = shifts.view(-1, 1, 2)
            base_anchors = base_anchors.view(1, -1, 2)
            anchors.append((shifts + base_anchors).reshape(-1, 2))
        return anchors

    def forward(self, feature_maps: List[Tensor], real_video_len: Tensor) -> List[List[SpanList]]:
        """Generate anchors for a set of feature maps.

        Args:
            feature_maps (List[Tensor]): A list of feature maps.
            real_video_len (Tensor): A tensor representing the real video length.

        Returns:
            List[List[SpanList]]: A list of lists of SpanList objects representing the generated anchors.
        """
        grid_sizes = [feature_map.shape[1] for feature_map in feature_maps]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors = []
        for size in real_video_len:
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                spanlist = SpanList(anchors_per_feature_map, size, mode="xx")
                anchors_in_image.append(spanlist)
            anchors.append(anchors_in_image)
        return anchors
