"""Torchmetrics for moment retrieval evaluation."""

# import multiprocessing as mp  # noqa: E800
from functools import partial
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from src.metrics.moments.avg_precision import (
    compute_average_precision_detection_wrapper,
)
from src.metrics.moments.misc import DataContainerType, get_data_by_range, prepare_data
from src.metrics.moments.utils import (
    compute_temporal_iou_batch_cross,
    compute_temporal_iou_batch_paired,
)

LengthRanges: List[Tuple[int, int]] = [
    (0, 150),
    (0, 10),
    (10, 30),
    (30, 150),
]

RangeNames: Tuple[str, str, str, str] = ("short", "middle", "long", "full")
RangeTypes = Literal["short", "middle", "long", "full"]


class MRAveragePrecision(Metric):
    """MRAveragePrecision implementation based on torchmetrics.Metric."""

    window_ranges: Dict[str, Tuple[int, int]] = {
        "short": (0, 10),
        "middle": (10, 30),
        "long": (30, 150),
        "full": (0, 150),
    }

    def __init__(
        self,
        window_range: RangeTypes,
        max_pred_windows: int = 10,
        max_gt_windows: int = 10,
        num_workers: int = 2,
        **kwargs: Any,
    ) -> None:
        """Initialize the MRAveragePrecision metric.

        Args:
            window_range (RangeTypes): The name of the range to use.
            max_pred_windows (int): The maximum number of predicted windows to use.
            max_gt_windows (int): The maximum number of ground truth windows to use.
            num_workers (int): The number of workers to use for multiprocessing.
            kwargs (Any): Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.add_state("avg_precision", default=[], dist_reduce_fx="cat")
        self.window_range = window_range
        self.iou_trhds = torch.linspace(0.5, 0.95, 10)  # noqa: WPS432
        self.max_pred_windows = max_pred_windows
        self.max_gt_windows = max_gt_windows
        self.num_workers = num_workers

    # pylint: disable=arguments-differ
    def update(self, submissions: DataContainerType, targets: DataContainerType) -> None:  # noqa: WPS210
        """Update the state with new metric values.

        Args:
            submissions (DataContainerType): The model's predictions.
            targets (DataContainerType): The ground truth data.
        """
        window_length = self.window_ranges[self.window_range]
        ranged_submissions, ranged_targets = get_data_by_range(submissions, targets, window_length)
        pred_qid2data, gt_qid2data = prepare_data(
            ranged_submissions,
            ranged_targets,
            self.max_pred_windows,
            self.max_gt_windows,
        )

        qid2ap_list = {}
        data_triples = [[qid, gt_qid2data[qid], pred_qid2data[qid]] for qid in pred_qid2data]

        compute_ap_from_triple = partial(compute_average_precision_detection_wrapper, tiou_thresholds=self.iou_trhds)

        for data_triple in data_triples:
            qid, scores = compute_ap_from_triple(data_triple)
            qid2ap_list[qid] = scores  # noqa: WPS441

        # with mp.Pool(self.num_workers) as pool:
        #     for qid, scores in pool.imap_unordered(compute_ap_from_triple, data_triples):
        #         qid2ap_list[qid] = scores  # noqa: E800

        ap_array = np.array(list(qid2ap_list.values()))  # (#queries, #thd)
        self.avg_precision.append(torch.tensor(ap_array, device=self.device))

    def compute(self) -> Dict[str, Tensor]:
        """Compute the average precision per IoU threshold.

        Returns:
            Dict[str, Tensor]: The Average Presion Score somputed per each IoU threshold.
        """
        ap_per_query_per_iou = dim_zero_cat(self.avg_precision)
        if ap_per_query_per_iou.nelement() == 0:
            ap_per_query_per_iou = torch.zeros(1, len(self.iou_trhds)).to(self.device)
        ap_per_iou = ap_per_query_per_iou.mean(0) * 100  # mAP at different IoU thresholds.
        str_ious = [str(float(f"{iou:.2f}")) for iou in self.iou_trhds]
        iou_thd2ap = dict(zip(str_ious, ap_per_iou))
        iou_thd2ap["Avg"] = torch.mean(ap_per_iou)
        return iou_thd2ap


class MRRecallAt1(Metric):
    """MRRecallAt1 implementation based on torchmetrics.Metric."""

    window_ranges: Dict[str, Tuple[int, int]] = {
        "short": (0, 10),
        "middle": (10, 30),
        "long": (30, 150),
        "full": (0, 150),
    }

    def __init__(
        self,
        window_range: RangeTypes,
        **kwargs: Any,
    ) -> None:
        """Initialize the MRRecallAt1 metric.

        Args:
            window_range (RangeTypes): The name of the range to use.
            kwargs (Any): Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.add_state("computed_ious", default=[], dist_reduce_fx="cat")
        self.window_range = window_range
        self.iou_trhds = torch.linspace(0.3, 0.95, 14)  # noqa: WPS432

    @staticmethod
    def prepare_data(submissions: DataContainerType, targets: DataContainerType) -> Tuple[Any, Any]:  # noqa: WPS602
        """Prepare the data for the metric computation.

        Args:
            submissions (DataContainerType): The model's predictions.
            targets (DataContainerType): The ground truth data.

        Returns:
            Tuple[Any, Any]: The prepared data.
        """
        pred_qid2window = {
            submission["qid"]: submission["pred_relevant_windows"][0][:2] for submission in submissions
        }  # noqa: WPS221

        gt_qid2window = {}
        for target in targets:
            cur_gt_windows = target["relevant_windows"]
            cur_qid = target["qid"]
            cur_max_iou_idx = 0
            if cur_gt_windows:
                # select the GT window that has the highest IoU
                cur_ious = compute_temporal_iou_batch_cross(
                    np.array([pred_qid2window[cur_qid]]),
                    np.array(cur_gt_windows),
                )[0]
                cur_max_iou_idx = np.argmax(cur_ious)  # type: ignore
            gt_qid2window[cur_qid] = cur_gt_windows[cur_max_iou_idx]

        return pred_qid2window, gt_qid2window

    # pylint: disable=arguments-differ
    def update(self, submissions: DataContainerType, targets: DataContainerType) -> None:  # noqa: WPS210
        """Update the state with new metric values.

        Args:
            submissions (DataContainerType): The model's predictions.
            targets (DataContainerType): The ground truth data.
        """
        window_length = self.window_ranges[self.window_range]
        ranged_submissions, ranged_targets = get_data_by_range(submissions, targets, window_length)
        if ranged_targets:
            pred_qid2window, gt_qid2window = self.prepare_data(ranged_submissions, ranged_targets)

            # get window arrays
            qids = list(pred_qid2window.keys())
            pred_windows = np.array([pred_qid2window[qid] for qid in qids]).astype(float)
            gt_windows = np.array([gt_qid2window[qid] for qid in qids]).astype(float)

            # compute temporal IoU
            pred_gt_iou = compute_temporal_iou_batch_paired(pred_windows, gt_windows)
            self.computed_ious.append(torch.tensor(pred_gt_iou, device=self.device))

    def compute(self) -> Dict[str, Tensor]:
        """Compute the average precision per IoU threshold.

        Returns:
            Dict[str, Tensor]: The Average Presion Score somputed per each IoU threshold.
        """
        if not self.computed_ious:
            zero_tensor = torch.tensor(0, dtype=torch.float32)
            iou_thd2recall_at_one: Dict[str, Tensor] = {
                str(float(f"{thd:.2f}")): zero_tensor for thd in self.iou_trhds  # noqa: WPS221
            }
            iou_thd2recall_at_one["mIoU"] = zero_tensor
            return iou_thd2recall_at_one

        computed_ious = dim_zero_cat(self.computed_ious)
        computed_ious = computed_ious.reshape(-1)

        iou_thd2recall_at_one = {}
        for thd in self.iou_trhds:
            ious_higher_thresh = (computed_ious >= thd).type(torch.float32)
            recall_at1 = torch.mean(ious_higher_thresh) * 100
            iou_thd2recall_at_one[str(float(f"{thd:.2f}"))] = recall_at1
        iou_thd2recall_at_one["mIoU"] = torch.mean(computed_ious)
        return iou_thd2recall_at_one
