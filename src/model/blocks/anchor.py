"""Anchors init."""

from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from src.model.utils.model_utils import inverse_sigmoid


class RandomAnchor(nn.Module):
    """Random anchor Embedding."""

    def __init__(self, num_queries: int, uniform: bool = True) -> None:
        """
        Initialize the AnchorEmbedding module.

        Args:
            num_queries (int): Total number of anchors
            uniform (bool): if True, uniform distribution will be used
        """
        super().__init__()
        self.num_queries = num_queries
        self.center = nn.Embedding(num_queries, 1)
        self.width = nn.Embedding(num_queries, 1)

        # Initialize the weights
        if uniform:
            self.center.weight.data = inverse_sigmoid(torch.Tensor(np.random.uniform(0, 1, size=num_queries)))[:, None]
            self.width.weight.data = inverse_sigmoid(torch.Tensor(np.random.uniform(0, 0.5, size=num_queries)))[:, None]

    def get_reference_points(self) -> Tensor:
        """
        Get reference points as tensor [n_points, 2].

        Returns:
            Tensor: reference points as tensor [n_points, 2]
        """
        return torch.cat([self.center.weight, self.width.weight], dim=-1)

    def forward(self, idx: int) -> Tensor:
        """
        Forward pass for the AnchorEmbedding module.

        Args:
            idx (Tensor): Index tensor for the anchors.

        Returns:
            Tensor: Concatenated embeddings of centers and widths.
        """
        centers = self.center_embedding(idx)  # type: ignore
        widths = self.width_embedding(idx)  # type: ignore
        return torch.cat([centers, widths], dim=-1)


def distribute_points_on_triangle(n_references: int, min_width: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Distributes points uniformly along the perimeter of a triangle with vertices
    at (0, 0), (0.5, 1), (1, 0), adjusting for a minimum width on the bottom side.

    Args:
        n_references (int): Number of points to distribute along the perimeter.
        min_width (float): Minimum width adjustment for the bottom side.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two numpy arrays containing the x and y coordinates
                                        of the distributed points.
    """
    # Coordinates of the triangle vertices
    points = np.array([[0, 0], [0.5, 1], [1, 0]])

    # Lengths of the triangle sides
    side_lengths = np.linalg.norm(np.diff(points[[0, 1, 2, 0], :], axis=0), axis=1)
    perimeter = np.sum(side_lengths)

    # Determining the number of points on each side based on its length
    num_points_per_side = (side_lengths / perimeter * n_references).astype(int)
    num_points_per_side[-1] = n_references - num_points_per_side[:-1].sum()  # Adjusting the last segment

    def interpolate_points(p1: float, p2: float, num_points: int) -> np.ndarray:
        return np.linspace(p1, p2, num_points, endpoint=False)[1:]

    x_points: List[float] = []
    y_points: List[float] = []

    for point_idx in range(3):
        point_1 = points[point_idx]
        point_2 = points[(point_idx + 1) % 3]
        num_points = num_points_per_side[point_idx]

        # Adjusting the starting points for the left and right sides
        if point_idx == 0:
            point_1 = [min_width / 2, min_width]
        if point_idx == 1:
            point_2 = [1 - min_width / 2, min_width]

        interpolated_points = interpolate_points(point_1, point_2, num_points + 1)

        if point_idx == 2:  # Bottom side
            interpolated_points[:, 1] += min_width  # Increasing y for the bottom side

        x_points.extend(interpolated_points[:, 0])
        y_points.extend(interpolated_points[:, 1])

    return np.array(x_points), np.array(y_points)


class TriangleAnchor(nn.Module):
    """
    The embedding class for negative anchors.

    Separates the centers and widths of the anchors, which allows better adjustment during training.
    """

    def __init__(
        self,
        num_queries: int,
        min_width: float = 0.1,
    ) -> None:
        """
        Initialize the NegAnchorEmbedding module.
        """
        super().__init__()
        centers, widths = distribute_points_on_triangle(num_queries, min_width)
        self.num_queries = num_queries
        self.center = nn.Embedding(num_queries, 1)
        self.width = nn.Embedding(num_queries, 1)
        self.center.weight.data = inverse_sigmoid(torch.Tensor(centers))[:, None]
        self.width.weight.data = inverse_sigmoid(torch.Tensor(widths))[:, None]

    def get_reference_points(self) -> Tensor:
        """
        Get reference points as tensor [n_points, 2].

        Returns:
            Tensor: reference points as tensor [n_points, 2]
        """
        return torch.cat([self.center.weight, self.width.weight], dim=-1)

    def forward(self, idx: Tensor) -> Tensor:
        """
        Forward pass for the AnchorEmbedding module.

        Args:
            idx (Tensor): Index tensor for the anchors.

        Returns:
            Tensor: Concatenated embeddings of centers and widths.
        """
        centers = self.center(idx)  # Embedding for centers
        widths = self.width(idx)  # Embedding for widths
        return torch.cat([centers, widths], dim=-1)


class LayerAnchor(nn.Module):
    """
    The embedding class for anchors.

    Separates the centers and widths of the anchors, which allows to better adjust them during training.
    In addition, there is a special initialization for a video width token
    """

    def __init__(self, num_queries: int, ratios: Tuple[float, ...] = (0.5, 0.35), special_anchor: bool = True) -> None:
        """
        Initialize the AnchorEmbedding module.

        Args:
            num_queries (int): Total number of anchors
            ratios (Tuple[float, ...]): Defines anchor levels, each level consists of ratio * num_queries of anchors.
            special_anchor (bool): If True, a special anchor with parameters (0.5, 1) will be added.
        """
        super().__init__()
        self.num_queries = num_queries
        # Create two separate embeddings: one for the centers, the other for the widths
        layers_num_queries = num_queries - 1 if special_anchor else num_queries  # If there is a special anchor,
        # then remove one of the anchor from the distribution
        self.center = nn.Embedding(layers_num_queries, 1)
        self.width = nn.Embedding(layers_num_queries, 1)
        # Determine the number of anchors for each level
        ratios_n = [int(ratio * layers_num_queries) for ratio in ratios]
        # Determine the number of anchors for the last level
        ratios_n.append(layers_num_queries - sum(ratios_n))
        centers: List[float] = []
        widths: List[float] = []
        for ratio_n in ratios_n:
            # distribute the anchors evenly, their width is the same
            centers.extend(np.linspace(0, 1, ratio_n + 2)[1:-1])
            widths.extend([1 / ratio_n] * ratio_n)
        # Add a special anchor
        if special_anchor:
            centers.append(0.5)
            widths.append(1)
        # Initialize the weights
        self.center.weight.data = inverse_sigmoid(torch.Tensor(centers))[:, None]
        self.width.weight.data = inverse_sigmoid(torch.Tensor(widths))[:, None]

    def get_reference_points(self) -> Tensor:
        """
        Get reference points as tensor [n_points, 2].

        Returns:
            Tensor: reference points as tensor [n_points, 2]
        """
        return torch.cat([self.center.weight, self.width.weight], dim=-1)

    def forward(self, idx: int) -> Tensor:
        """
        Forward pass for the AnchorEmbedding module.

        Args:
            idx (Tensor): Index tensor for the anchors.

        Returns:
            Tensor: Concatenated embeddings of centers and widths.
        """
        centers = self.center_embedding(idx)  # type: ignore
        widths = self.width_embedding(idx)  # type: ignore
        return torch.cat([centers, widths], dim=-1)
