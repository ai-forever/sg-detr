import numpy as np


def compute_temporal_iou_batch_paired(pred_windows: np.ndarray, gt_windows: np.ndarray) -> float:
    """Compute IoU along temporal axis for each pair of windows in pred_windows and gt_windows.

    Args:
        pred_windows: np.ndarray, (N, 2), [st (float), ed (float)] * N
        gt_windows: np.ndarray, (N, 2), [st (float), ed (float)] * N

    Returns:
        float: iou np.ndarray, (N, )
    """
    possible_inter = np.minimum(pred_windows[:, 1], gt_windows[:, 1]) - np.maximum(pred_windows[:, 0], gt_windows[:, 0])
    intersection = np.maximum(0, possible_inter)
    union = np.maximum(pred_windows[:, 1], gt_windows[:, 1]) - np.minimum(pred_windows[:, 0], gt_windows[:, 0])
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)


def compute_temporal_iou_batch_cross(spans1: np.ndarray, spans2: np.ndarray):
    """
    Compute temporal intersection over union between spans1 and spans2.

    Args:
        spans1 (np.ndarray): each row defines a span [st, ed] (N, 2)
        spans2 (np.ndarray): each row defines a span [st, ed] (M, 2)

    Returns:
        iou: (N, M) np.ndarray
        union: (N, M) np.ndarray
    """
    areas1 = spans1[:, 1] - spans1[:, 0]  # (N, )
    areas2 = spans2[:, 1] - spans2[:, 0]  # (M, )

    left = np.maximum(spans1[:, None, 0], spans2[None, :, 0])  # (N, M)
    right = np.minimum(spans1[:, None, 1], spans2[None, :, 1])  # (N, M)

    inter = np.clip(right - left, 0, None)  # (N, M)
    union = areas1[:, None] + areas2[None, :] - inter  # (N, M)

    iou = inter / union
    return iou, union


def interpolated_precision_recall(precision: np.ndarray, recall: np.ndarray) -> float:
    """Interpolated AP - VOCdevkit from VOC 2011.

    Args:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.

    Returns:
        float: Average precision score.
    """
    mprecision = np.hstack([[0], precision, [0]])
    mrecall = np.hstack([[0], recall, [1]])
    for i in range(len(mprecision) - 1)[::-1]:
        mprecision[i] = max(mprecision[i], mprecision[i + 1])
    idx = np.where(mrecall[1::] != mrecall[0:-1])[0] + 1
    return np.sum((mrecall[idx] - mrecall[idx - 1]) * mprecision[idx])
