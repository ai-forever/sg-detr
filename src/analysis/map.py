"""Calculate map."""

from typing import List, Tuple, Union

import numpy as np


def calculate_iou_matrix_1d(  # noqa: WPS114
    span_a: Union[np.ndarray, Tuple[float, float]],
    span_b: Union[np.ndarray, Tuple[float, float]],
) -> np.ndarray:
    """
    Calculate the Intersection over Union (IoU) of two spans or two sets of spans.

    Example:
        >>> span_a = np.array([[0, 2], [1, 4]])
        >>> span_b = np.array([[1, 3], [2, 5]])
        >>> calculate_iou_matrix_1d(span_a, span_b)
        >>> # array([[0.33333333, 0.0],
        >>>          [0.66666667, 0.5]])

    Args:
        span_a (Union[np.ndarray, Tuple[float, float]]): A single span or array of spans, each defined by (start, end).
        span_b (Union[np.ndarray, Tuple[float, float]]): A single span or array of spans, each defined by (start, end).

    Returns:
        np.ndarray: A single float IoU value if span_a and span_b are single spans, or a 2D array of IoU values for
                    each pair of spans from span_a and span_b where spans_a on the y axis and spans_b on the x axis.
    """
    span_a = np.atleast_2d(span_a).astype(float)
    span_b = np.atleast_2d(span_b).astype(float)

    # Calculate intersection
    intersection_start = np.maximum(span_a[:, 0:1], span_b[:, 0].T)  # noqa: WPS221
    intersection_end = np.minimum(span_a[:, 1:2], span_b[:, 1].T)  # noqa: WPS221
    intersection = np.maximum(intersection_end - intersection_start, 0)

    # Calculate union
    union_start = np.minimum(span_a[:, 0:1], span_b[:, 0].T)  # noqa: WPS221
    union_end = np.maximum(span_a[:, 1:2], span_b[:, 1].T)  # noqa: WPS221
    union = union_end - union_start

    # Calculate IoU
    out_zeros = np.zeros_like(intersection, dtype=float)
    return np.divide(intersection, union, out=out_zeros, where=(union != 0))  # pylint: disable=superfluous-parens


def calculate_precision_recall_curve(  # noqa: WPS118
    gt_spans: List[Tuple[float, float]],
    predicted_spans: List[Tuple[float, float]],
    scores: List[float],
    iou_threshold: float,
) -> Tuple[List[float], List[float]]:
    """
    Calculate the precision-recall curve for a given IoU threshold.

    Args:
        gt_spans (List[Tuple[float, float]]): Ground truth spans, each defined by (start, end).
        predicted_spans (List[Tuple[float, float]]): Predicted spans, each defined by (start, end).
        scores (List[float]): Confidence scores for each predicted span.
        iou_threshold (float): IoU threshold to consider a prediction as true positive.

    Returns:
        Tuple[List[float], List[float]]: Lists of precision and recall values at each prediction threshold.
    """
    if len(predicted_spans) == 0:  # noqa: WPS507
        return [0], [0]

    # Sort predicted spans by descending scores
    sorted_indices = np.argsort(scores)[::-1]
    predicted_spans_sorted = [predicted_spans[idx] for idx in sorted_indices]

    # Convert lists to numpy arrays
    gt_spans_np = np.array(gt_spans)
    predicted_spans_np = np.array(predicted_spans_sorted)

    # Calculate IoU for all pairs of spans
    iou_matrix = calculate_iou_matrix_1d(gt_spans_np, predicted_spans_np)
    if iou_matrix.ndim == 1:
        iou_matrix = iou_matrix[:, np.newaxis]

    precisions = []
    recalls = []
    true_positives = 0
    false_positives = 0
    false_negatives = len(gt_spans)

    detected_gt = np.zeros(len(gt_spans), dtype=bool)

    for idx in range(len(predicted_spans_np)):
        if np.any(iou_matrix[:, idx] >= iou_threshold):
            max_iou_idx = np.argmax(iou_matrix[:, idx])
            if not detected_gt[max_iou_idx]:
                true_positives += 1
                false_negatives -= 1
                detected_gt[max_iou_idx] = True
            else:
                false_positives += 1
        else:
            false_positives += 1

        precision = (
            true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0.0
        )
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls


def calculate_average_precision(
    gt_spans: List[Tuple[float, float]],
    predicted_spans: List[Tuple[float, float]],
    scores: List[float],
    iou_threshold: float = 0.5,
    use_11_point_interpolation: bool = False,
) -> float:
    """
    Calculate Average Precision (AP) with an option to use 11-point interpolation for a single IoU threshold.

    Args:
        gt_spans (List[Tuple[float, float]]): Ground truth spans, each defined by (start, end).
        predicted_spans (List[Tuple[float, float]]): Predicted spans, each defined by (start, end).
        scores (List[float]): Confidence scores for each predicted span.
        iou_threshold (float): IoU threshold to consider a prediction as true positive. Defaults to 0.5.
        use_11_point_interpolation (bool): Whether to use 11-point interpolation. Defaults to False.

    Returns:
        float: Average precision (AP) score.
    """
    if len(gt_spans) == 0:  # noqa: WPS507
        return 0.0
    precisions, recalls = calculate_precision_recall_curve(gt_spans, predicted_spans, scores, iou_threshold)

    precisions_np = np.array(precisions)
    recalls_np = np.array(recalls)

    # Ensure the curve starts at (0,1) and ends at (1,0)
    precisions_np = np.concatenate(([1.0], precisions_np, [0.0]))  # type: ignore
    recalls_np = np.concatenate(([0.0], recalls_np, [1.0]))  # type: ignore

    # Interpolate precision to ensure it is non-decreasing
    for idx in range(len(precisions_np) - 2, -1, -1):
        precisions_np[idx] = np.maximum(precisions_np[idx], precisions_np[idx + 1])

    if use_11_point_interpolation:
        # 11-point interpolation
        recall_levels = np.linspace(0, 1, 11)
        precision_at_recall_levels = []

        for recall_level in recall_levels:
            precisions_at_recall = precisions_np[recalls_np >= recall_level]
            if precisions_at_recall.size > 0:
                precision_at_recall_levels.append(np.max(precisions_at_recall))
            else:
                precision_at_recall_levels.append(0.0)

        ap: float = np.mean(precision_at_recall_levels)
    else:
        # Compute the area under the PR curve using the trapezoidal rule
        ap = np.trapz(precisions_np, recalls_np)

    return ap


def calculate_map(
    gt_spans: List[Tuple[float, float]],
    predicted_spans: List[Tuple[float, float]],
    scores: List[float],
    iou_thresholds: np.ndarray = np.linspace(0.5, 0.95, 10),
    use_11_point_interpolation: bool = False,
) -> Tuple[float, List[float]]:
    """
    Calculate Mean Average Precision (MAP) with an option to use 11-point interpolation across multiple IoU thresholds.

    Args:
        gt_spans (List[Tuple[float, float]]): Ground truth spans, each defined by (start, end).
        predicted_spans (List[Tuple[float, float]]): Predicted spans, each defined by (start, end).
        scores (List[float]): Confidence scores for each predicted span.
        iou_thresholds (np.ndarray): IoU thresholds to consider for MAP calculation.
        use_11_point_interpolation (bool): Whether to use 11-point interpolation. Defaults to False.

    Returns:
        Tuple[float, List[float]]: Mean average precision (MAP) score and list of AP scores for each IoU threshold.
    """
    aps = []
    for iou_threshold in iou_thresholds:
        ap = calculate_average_precision(gt_spans, predicted_spans, scores, iou_threshold, use_11_point_interpolation)
        aps.append(ap)

    map_score: float = np.mean(aps)  # type: ignore
    return map_score, aps
