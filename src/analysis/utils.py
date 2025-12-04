"""Callbacks utils."""

from typing import Dict, List, Tuple

import numpy as np

MAX_DURATION: int = 150
WINDOW_RANGES: Dict[str, Tuple[float, float]] = {  # noqa: WPS407
    "short": (0, 10 / MAX_DURATION),  # noqa: WPS432
    "middle": (10 / MAX_DURATION, 30 / MAX_DURATION),  # noqa: WPS432
    "long": (30 / MAX_DURATION, 150 / MAX_DURATION),  # noqa: WPS432
}  # Categories by length(normalized) for spans(in seconds)


def categorize_span(span: Tuple[float, float]) -> str:
    """
    Define the category (by length) of the span.

    Args:
        span (Tuple[float, float]): span, beginning and end(normalized from 0 d to 1)

    Raises:
        ValueError: if the end or beginning of span is outside [0, 1]

    Returns:
        str: type of span
    """
    length = span[1] - span[0]
    for category, (start, end) in WINDOW_RANGES.items():
        if start <= length <= end:
            return category
    raise ValueError(f"Unsupported span: {span}")


def calculate_intersection_over_gt_size(  # noqa: WPS118
    pred_span: Tuple[float, float],
    gt_span: Tuple[float, float],
) -> float:
    """
    Calculate the ratio of the intersection of the predicted span with the ground truth span to the size of the gt span.

    Args:
        pred_span: A tuple representing the start and end points of the predicted span.
        gt_span: A tuple representing the start and end points of the ground truth span.

    Returns:
        A float value representing the ratio of the intersection of the predicted span with
        the ground truth span to the size of the ground truth span.
    """
    # Calculate the intersection of pred_span and gt_span
    intersection_start = max(pred_span[0], gt_span[0])
    intersection_end = min(pred_span[1], gt_span[1])
    intersection = max(0, intersection_end - intersection_start)

    # Calculate the size of the ground truth span
    gt_size = gt_span[1] - gt_span[0]

    # Calculate the ratio
    if gt_size > 0:
        return intersection / gt_size
    return 0


def calculate_average_distance_between_spans(covered_gt_spans: List[Tuple[float, float]]) -> float:  # noqa: WPS118
    """
    Calculate the average distance between consecutive spans.

    Args:
        covered_gt_spans: A list of tuples, where each tuple represents a span defined by its start and end points.

    Returns:
        A float representing the average distance between the spans.

    Raises:
        ValueError: If there is only one span or spans overlap.
    """
    # Ensure there are at least two spans to calculate distance between them
    if len(covered_gt_spans) < 2:
        raise ValueError("At least two spans are required to calculate distances.")

    # Sort spans based on their start point
    sorted_spans = sorted(covered_gt_spans, key=lambda x: x[0])  # noqa: WPS111

    # Check for overlapping spans
    for idx in range(len(sorted_spans) - 1):
        if sorted_spans[idx + 1][0] < sorted_spans[idx][1]:
            raise ValueError("Spans overlap, cannot calculate distances.")

    # Calculate distances between the end of one span and the start of the next
    distances = [
        sorted_spans[idx + 1][0] - sorted_spans[idx][1] for idx in range(len(sorted_spans) - 1)  # noqa: WPS221, WPS441
    ]

    # Calculate and return the average distance
    return sum(distances) / len(distances)


def calculate_iou_1d(span_a: Tuple[float, float], span_b: Tuple[float, float]) -> float:  # noqa: WPS114
    """
    Calculate the Intersection over Union (IoU) of two spans.

    Args:
        span_a (Tuple[float, float]): Start and end of the first span.
        span_b (Tuple[float, float]): Start and end of the second span.

    Returns:
        float: The IoU between span_a and span_b.
    """
    start_a, end_a = span_a
    start_b, end_b = span_b

    # Find the intersection of the spans
    intersection_start = max(start_a, start_b)
    intersection_end = min(end_a, end_b)
    intersection = max(intersection_end - intersection_start, 0)

    # Find the union of the spans
    union = max(end_a, end_b) - min(start_a, start_b)

    # Calculate the IoU
    return intersection / union if union > 0 else 0


def find_external_gaps(gt_spans: np.ndarray) -> np.ndarray:
    """
    Identify the gaps outside the provided intervals within the range [0, 1].

    This function calculates and returns the gaps before the first interval if it does not start at 0,
    and after the last interval if it does not end at 1.

    Args:
        gt_spans (np.ndarray): A 2D numpy array where each row represents an interval with
                            the first column being the start and the second column the end of the interval.

    Returns:
        np.ndarray: A 2D numpy array containing the intervals of the external gaps found,
                    if any. Each row represents a gap, where the first column is the start
                    and the second column is the end of the gap.
    """
    if len(gt_spans) == 0:  # noqa: WPS507
        return np.empty((0, 2))
    # Sort spans by the starting point of each segment
    sorted_spans = gt_spans[np.argsort(gt_spans[:, 0])]
    # Initialize a list to store the "external" segments
    external_gaps = []
    # Check and add a gap before the first segment if it doesn't start at 0
    if sorted_spans[0][0] > 0:
        external_gaps.append([0, sorted_spans[0][0]])
    # Check and add a gap after the last segment if it doesn't end at 1
    if sorted_spans[-1][1] < 1:
        external_gaps.append([sorted_spans[-1][1], 1])
    return np.array(external_gaps)


def intersection_length(span1: Tuple[float, float], span2: Tuple[float, float]) -> float:
    """
    Calculate the length of the intersection between two spans.

    Args:
        span1 (Tuple[float, float]): A tuple representing the start and end of the first span.
        span2 (Tuple[float, float]): A tuple representing the start and end of the second span.

    Returns:
        float: The length of the intersection. Returns 0 if there is no intersection.
    """
    # Calculate the start and end of the intersection
    start = max(span1[0], span2[0])
    end = min(span1[1], span2[1])
    # Return the length of the intersection if it exists
    return max(0, end - start)


def compute_intersection_matrix(spans_1: np.ndarray, spans_2: np.ndarray) -> np.ndarray:  # noqa: WPS114
    """
    Compute a matrix representing the intersection lengths between two sets of spans.

    This function calculates the intersection length for every possible pair of spans from two arrays
    and stores the result in a matrix.

    Args:
        spans_1 (np.ndarray): A numpy array where each row represents a span (start and end).
        spans_2 (np.ndarray): A numpy array similar to spans_1 representing another set of spans.

    Returns:
        np.ndarray: A 2D numpy array (matrix) where element (i, j) is the length of the intersection
                    between spans_1[i] and spans_2[j].
    """
    # Number of spans in each list
    n_points_1 = spans_1.shape[0]  # noqa: WPS114
    n_points_2 = spans_2.shape[0]  # noqa: WPS114
    # Create an empty matrix for results
    intersection_matrix = np.zeros((n_points_1, n_points_2))

    # Fill the matrix with intersection lengths
    for idx_i in range(n_points_1):
        for idx_j in range(n_points_2):
            intersection_matrix[idx_i, idx_j] = intersection_length(spans_1[idx_i], spans_2[idx_j])

    return intersection_matrix


def find_gaps(gt_spans: np.ndarray) -> np.ndarray:
    """
    Identify the gaps between consecutive spans provided in a numpy array.

    This function sorts the spans by their starting points and iterates through them to find
    non-overlapping gaps between the end of one span and the start of the next.

    Args:
        gt_spans (np.ndarray): A 2D numpy array where each row represents a span with
                            the first column being the start and the second column the end of the span.

    Returns:
        np.ndarray: A 2D numpy array containing the found gaps. Each row represents a gap,
                    where the first column is the start and the second column is the end of the gap.
    """
    # Sort spans by the starting point of each segment
    sorted_spans = gt_spans[np.argsort(gt_spans[:, 0])]

    # Initialize a list to store the gaps found
    gaps = []

    # Iterate through the sorted spans to find gaps
    for idx in range(1, len(sorted_spans)):  # noqa: WPS518
        end_of_previous = sorted_spans[idx - 1][1]
        start_of_current = sorted_spans[idx][0]
        if end_of_previous < start_of_current:
            gaps.append([end_of_previous, start_of_current])

    return np.array(gaps)
