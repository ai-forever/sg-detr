# flake8: noqa
# pylint: skip-file
# type: ignore

# Code were adapted from:
# https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_wbf_1d.py
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np


def prefilter_line_segments(
    boxes: List[np.ndarray], scores: List[np.ndarray], labels: List[np.ndarray], weights: np.ndarray, thr: float
) -> Dict[int, List[np.ndarray]]:
    """
    Prefilter line segments by thresholding scores and normalizing box coordinates.

    Args:
        boxes (List[np.ndarray]): List of arrays containing box coordinates for each model.
        scores (List[np.ndarray]): List of arrays containing scores for each model.
        labels (List[np.ndarray]): List of arrays containing labels for each model.
        weights (np.ndarray): Array of weights for each model.
        thr (float): Score threshold below which boxes are discarded.

    Returns:
        Dict[int, List[np.ndarray]]: A dictionary where keys are labels and values are lists of filtered boxes.
    """
    new_boxes = dict()

    for t in range(len(boxes)):
        if len(boxes[t]) != len(scores[t]):
            raise ValueError(
                f"Length of boxes arrays not equal to length of scores array: {len(boxes[t])} != {len(scores[t])}"
            )

        if len(boxes[t]) != len(labels[t]):
            raise ValueError(
                f"Length of boxes arrays not equal to length of labels array: {len(boxes[t])} != {len(labels[t])}"
            )

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            x1, x2 = map(float, boxes[t][j])

            # Ensure x1 and x2 are within valid bounds
            if x2 < x1:
                warnings.warn("X2 < X1 value in box. Swapping them.")
                x1, x2 = x2, x1
            x1 = np.clip(x1, 0.0, 1.0)
            x2 = np.clip(x2, 0.0, 1.0)

            if (x2 - x1) == 0.0:
                warnings.warn(f"Zero length line segment skipped: {boxes[t][j]}.")
                continue

            # Construct box representation with weighted score
            box_data = [label, score * weights[t], weights[t], t, x1, x2]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(box_data)

    # Sort boxes in each label group by score in descending order
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[np.argsort(current_boxes[:, 1])[::-1]]

    return new_boxes


def get_weighted_box(boxes: np.ndarray, conf_type: str = "avg") -> np.ndarray:
    """
    Create a weighted box from a set of boxes.

    Args:
        boxes (np.ndarray): Array of boxes to fuse.
        conf_type (str): Type of confidence ('avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg').

    Returns:
        np.ndarray: A weighted box in the format [label, score, weight, index, x1, x2].
    """
    box = np.zeros(6, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    for b in boxes:
        box[4:] += b[1] * b[4:]
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]
    box[0] = boxes[0][0]  # label
    box[1] = np.mean(conf_list) if conf_type == "avg" else max(conf_list)
    box[2] = w  # weight
    box[3] = -1  # model index (not used)
    box[4:] /= conf  # normalize coordinates
    return box


def find_matching_line_segment_quickly(
    boxes_list: np.ndarray, new_box: np.ndarray, match_iou: float
) -> Tuple[int, float]:
    """
    Find the best matching box in the list using IoU.

    Args:
        boxes_list (np.ndarray): Array of existing boxes.
        new_box (np.ndarray): The new box to match against.
        match_iou (float): IoU threshold for matching.

    Returns:
        Tuple[int, float]: Index of the best match and its IoU. Returns -1 if no match found.
    """

    def bb_iou_array(boxes, new_box):
        xA = np.maximum(boxes[:, 0], new_box[0])
        xB = np.minimum(boxes[:, 1], new_box[1])
        interSeg = np.maximum(xB - xA, 0)
        lsAArea = boxes[:, 1] - boxes[:, 0]
        lsBArea = new_box[1] - new_box[0]
        iou = interSeg / (lsAArea + lsBArea - interSeg)
        return iou

    if boxes_list.shape[0] == 0:
        return -1, match_iou

    ious = bb_iou_array(boxes_list[:, 4:], new_box[4:])
    ious[boxes_list[:, 0] != new_box[0]] = -1  # Ensure matching labels
    best_idx = np.argmax(ious)
    best_iou = ious[best_idx]

    if best_iou <= match_iou:
        return -1, match_iou

    return best_idx, best_iou


def weighted_boxes_fusion_1d(
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    weights: Union[List[float], None] = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    conf_type: str = "avg",
    allows_overflow: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Weighted Boxes Fusion (WBF) to line segments.

    Args:
        boxes_list (List[np.ndarray]): List of line segment predictions from each model.
        scores_list (List[np.ndarray]): List of confidence scores for each model.
        labels_list (List[np.ndarray]): List of labels for each model.
        weights (Union[List[float], None]): List of weights for each model. Defaults to equal weights.
        iou_thr (float): IoU threshold for determining matches.
        skip_box_thr (float): Minimum score threshold to include a box.
        conf_type (str): Method to calculate fused confidence ('avg', 'max', etc.).
        allows_overflow (bool): Allow confidence scores to exceed 1.0.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of fused boxes, scores, and labels.
    """
    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        warnings.warn(f"Incorrect number of weights: {len(weights)}. Expected: {len(boxes_list)}. Using equal weights.")
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ["avg", "max", "box_and_model_avg", "absent_model_aware_avg"]:
        raise ValueError(
            f'Unknown conf_type: {conf_type}. Must be ["avg", "max", "box_and_model_avg", "absent_model_aware_avg"].'
        )

    filtered_boxes = prefilter_line_segments(boxes_list, scores_list, labels_list, weights, skip_box_thr)

    if len(filtered_boxes) == 0:
        return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label, boxes in filtered_boxes.items():
        new_boxes = []
        weighted_boxes = np.empty((0, 6))

        # Clusterize boxes
        for j in range(len(boxes)):
            index, best_iou = find_matching_line_segment_quickly(weighted_boxes, boxes[j], iou_thr)

            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes = np.vstack((weighted_boxes, boxes[j].copy()))

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            clustered_boxes = np.array(new_boxes[i])
            if conf_type == "box_and_model_avg":
                # weighted average for boxes
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weighted_boxes[i, 2]
                # identify unique model index by model index column
                _, idx = np.unique(clustered_boxes[:, 3], return_index=True)
                # rescale by unique model weights
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * clustered_boxes[idx, 2].sum() / weights.sum()
            elif conf_type == "absent_model_aware_avg":
                # get unique model index in the cluster
                models = np.unique(clustered_boxes[:, 3]).astype(int)
                # create a mask to get unused model weights
                mask = np.ones(len(weights), dtype=bool)
                mask[models] = False
                # absent model aware weighted average
                weighted_boxes[i, 1] = (
                    weighted_boxes[i, 1] * len(clustered_boxes) / (weighted_boxes[i, 2] + weights[mask].sum())
                )
            elif conf_type == "max":
                weighted_boxes[i, 1] = weighted_boxes[i, 1] / weights.max()
            elif not allows_overflow:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * min(len(weights), len(clustered_boxes)) / weights.sum()
            else:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weights.sum()
        overall_boxes.append(weighted_boxes)

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 4:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels
