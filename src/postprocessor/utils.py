"""General purpose nms."""

from typing import Any, Callable, List

import numpy as np


def general_nms(
    items: List[Any],
    score_function: Callable[[Any], float],
    iou_function: Callable[[Any, Any], float],
    threshold: float,
) -> np.ndarray:
    """
    Apply Non-Maximum Suppression (NMS) to a list of items.

    Args:
        items: List of items to apply NMS.
        score_function: Function to compute the score for each item.
        iou_function: Function to compute IoU (Intersection over Union) between two items.
        threshold: IoU threshold to consider two items as overlapping.

    Returns:
        np.ndarray: indices of items selected by NMS.
    """
    # Compute scores for each item
    scores = np.array([score_function(item) for item in items])

    # Sort items by their scores in descending order
    sorted_indices = np.argsort(-scores)

    # List to keep track of selected indices
    selected_indices = []

    while sorted_indices.size > 0:
        # Always select the item with the highest score
        current_index = sorted_indices[0]
        selected_indices.append(current_index)

        # Compute IoU between the selected item and the rest; keep those with IoU less than threshold
        remaining_indices = sorted_indices[1:]
        remaining_indices = [
            idx
            for idx in remaining_indices
            if iou_function(items[current_index], items[idx]) < threshold  # type: ignore
        ]

        # Update the indices for the next iteration
        sorted_indices = np.array(remaining_indices)

    return np.array(selected_indices)
