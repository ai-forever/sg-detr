"""Filter queries with ground truth window length in the specified length range."""

import copy
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

DataContainerType = List[Dict[str, Any]]


def check_len(window: List[int], min_l: int, max_l: int) -> int:
    """Get the length of a window.

    Args:
        window (List[int]): [start, end]
        min_l (int): minimum length
        max_l (int): maximum length

    Returns:
        int: length of the window
    """
    return min_l < (window[1] - window[0]) <= max_l


def get_data_by_range(
    submissions: DataContainerType,
    targets: DataContainerType,
    len_range: Tuple[int, int],
) -> Tuple[DataContainerType, DataContainerType]:  # noqa: WPS221
    """Keep queries with ground truth window length in the specified length range.

    Args:
        submissions (DataContainerType): submissions to be filtered
        targets (DataContainerType): Targets to be filtered
        len_range (Tuple[int, int]): Target window length range

    Returns:
        Tuple[DataContainerType, DataContainerType]: filtered submissions and ground truth
    """
    targets_in_range = []
    target_qids_in_range = set()
    for target in targets:
        rel_windows_in_range = [window for window in target["relevant_windows"] if check_len(window, *len_range)]
        if rel_windows_in_range:
            target = copy.deepcopy(target)
            target["relevant_windows"] = rel_windows_in_range
            targets_in_range.append(target)
            target_qids_in_range.add(target["qid"])

    # keep only submissions for ground_truth_in_range
    submission_in_range = []
    for submission in submissions:
        if submission["qid"] in target_qids_in_range:
            submission_in_range.append(copy.deepcopy(submission))

    return submission_in_range, targets_in_range


def prepare_data(
    submissions: DataContainerType,
    targets: DataContainerType,
    max_pred_windows: Optional[int],
    max_gt_windows: Optional[int],
) -> Tuple[dict, dict]:
    """Prepare data for evaluation.

    Args:
        submissions (DataContainerType): Model predictions
        targets (DataContainerType): Ground truth data
        max_pred_windows (Optional[int]): Maximum number of predicted windows to keep
        max_gt_windows (Optional[int]): Maximum number of ground truth windows to keep

    Returns:
        Tuple[dict, dict]: Processed predicted and ground truth data
    """
    pred_qid2data = defaultdict(list)
    for submission in submissions:
        pred_windows = (
            submission["pred_relevant_windows"][:max_pred_windows]
            if max_pred_windows is not None
            else submission["pred_relevant_windows"]
        )
        qid = submission["qid"]
        for window in pred_windows:
            pred_qid2data[qid].append(
                {
                    "video-id": submission["qid"],
                    "t-start": float(window[0]),
                    "t-end": float(window[1]),
                    "score": float(window[2]),
                },
            )

    gt_qid2data = defaultdict(list)
    for target in targets:
        gt_windows = (
            target["relevant_windows"][:max_gt_windows] if max_gt_windows is not None else target["relevant_windows"]
        )
        qid = target["qid"]
        for window in gt_windows:
            data_line = {"video-id": target["qid"], "t-start": window[0], "t-end": window[1]}
            gt_qid2data[qid].append(data_line)

    return pred_qid2data, gt_qid2data
