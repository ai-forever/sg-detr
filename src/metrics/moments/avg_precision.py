import multiprocessing as mp
from functools import partial
from typing import Any, List

import numpy as np

from src.metrics.moments.misc import prepare_data
from src.metrics.moments.utils import (
    compute_temporal_iou_batch_cross,
    interpolated_precision_recall,
)


def compute_average_precision_detection(
    ground_truth: List[dict],
    prediction: List[dict],
    tiou_thresholds: List[int],
) -> np.ndarray:
    """Compute average precision (detection task) between ground truth and predictions data frames.

    If multiple predictions occurs for the same predicted segment, only the one with highest score is matches as true
    positive. This code is greatly inspired by Pascal VOC devkit.

    Args:
        ground_truth (List[dict]): List containing the ground truth instances dict['video-id', 't-start', 't-end']
        prediction (List[dict]): List containing the prediction instances dict['video-id', 't-start', 't-end', 'score']
        tiou_thresholds (List[int]): indicates the temporal intersection over union threshold, which is optional.

    Returns:
        np.ndarray: ap, Average precision score for each IoU threshold.
    """
    num_thresholds = len(tiou_thresholds)
    num_gts = len(ground_truth)
    num_preds = len(prediction)
    ap = np.zeros(num_thresholds)
    if len(prediction) == 0:
        return ap

    num_positive = float(num_gts)
    lock_gt = np.ones((num_thresholds, num_gts)) * -1
    # Sort predictions by decreasing score order.
    prediction.sort(key=lambda x: -x["score"])
    # Initialize true positive and false positive vectors.
    tp = np.zeros((num_thresholds, num_preds))
    fp = np.zeros((num_thresholds, num_preds))

    # Adaptation to query faster
    ground_truth_by_videoid: dict = {}
    for i, item in enumerate(ground_truth):
        item["index"] = i
        ground_truth_by_videoid.setdefault(item["video-id"], []).append(item)

    # Assigning true positive to truly grount truth instances.
    for idx, pred in enumerate(prediction):
        if pred["video-id"] in ground_truth_by_videoid:
            gts = ground_truth_by_videoid[pred["video-id"]]
        else:
            fp[:, idx] = 1
            continue

        _pred = np.array([[pred["t-start"], pred["t-end"]]])
        _gt = np.array([[gt["t-start"], gt["t-end"]] for gt in gts])
        tiou_arr = compute_temporal_iou_batch_cross(_pred, _gt)[0]

        tiou_arr = tiou_arr.reshape(-1)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for t_idx, tiou_threshold in enumerate(tiou_thresholds):
            for j_idx in tiou_sorted_idx:
                if tiou_arr[j_idx] < tiou_threshold:
                    fp[t_idx, idx] = 1
                    break
                if lock_gt[t_idx, gts[j_idx]["index"]] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[t_idx, idx] = 1
                lock_gt[t_idx, gts[j_idx]["index"]] = idx
                break

            if fp[t_idx, idx] == 0 and tp[t_idx, idx] == 0:
                fp[t_idx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / num_positive

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for t_idx, _ in enumerate(tiou_thresholds):
        ap[t_idx] = interpolated_precision_recall(precision_cumsum[t_idx, :], recall_cumsum[t_idx, :])
    return ap


def compute_average_precision_detection_wrapper(input_triple, tiou_thresholds):
    qid, ground_truth, prediction = input_triple
    scores = compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=tiou_thresholds)
    return qid, scores


def compute_mr_ap(
    submission,
    ground_truth,
    iou_thds,
    max_gt_windows=None,
    max_pred_windows=10,
    num_workers=0,
    chunksize=50,
):
    iou_thds: List[float] = [float(f"{thd:.2f}") for thd in iou_thds]  # type: ignore
    pred_qid2data, gt_qid2data = prepare_data(submission, ground_truth, max_pred_windows, max_gt_windows)

    qid2ap_list = {}
    data_triples = [[qid, gt_qid2data[qid], pred_qid2data[qid]] for qid in pred_qid2data]

    compute_ap_from_triple = partial(compute_average_precision_detection_wrapper, tiou_thresholds=iou_thds)

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for qid, scores in pool.imap_unordered(compute_ap_from_triple, data_triples, chunksize=chunksize):
                qid2ap_list[qid] = scores
    else:
        for data_triple in data_triples:
            qid, scores = compute_ap_from_triple(data_triple)
            qid2ap_list[qid] = scores  # noqa: WPS441

    ap_array = np.array(list(qid2ap_list.values()))  # (#queries, #thd)
    ap_thds = ap_array.mean(0)  # mAP at different IoU thresholds.
    iou_thd2ap = dict(zip([str(e) for e in iou_thds], ap_thds))
    iou_thd2ap["average"] = np.mean(ap_thds)
    # formatting
    return {k: float(f"{100 * v:.2f}") for k, v in iou_thd2ap.items()}
