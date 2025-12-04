"""Torchmetrics for highlights evaluation."""

from typing import Any, Dict, List, Literal

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from src.metrics.highlights.avg_precision import compute_hl_ap
from src.metrics.highlights.hit1 import compute_hl_hit1, mk_saliency_scores

# Note: Binary will be usefull for datasets like YouTube, where binary annotation
ThresholdsTypes = Literal["Binary", "Fair", "Good", "VeryGood"]


class HIT1Coef(Metric):
    """HIT at 1 coefficient implemented based on torchmetrics.Metric."""

    saliency_thresholds: Dict[str, int] = {"Binary": 1, "Fair": 2, "Good": 3, "VeryGood": 4}

    def __init__(self, threshold: ThresholdsTypes, **kwargs: Any) -> None:
        """Initialize the HIT1Coef metric.

        Args:
            threshold (ThresholdsTypes): The threshold type to use.
            kwargs (Any): Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.add_state("hit1_values", default=[], dist_reduce_fx="cat")
        self.threshold = threshold

    # pylint: disable=arguments-differ
    def update(self, submissions: List[Dict[str, Any]], targets: List[Dict[str, Any]]) -> None:  # noqa: WPS221
        """Update the state with new metric values.

        Args:
            submissions (List[Dict[str, Any]]): The model's predictions.
            targets (List[Dict[str, Any]]): The ground truth data.
        """
        # prepare inputs
        qid_preds = {submission["qid"]: submission["pred_saliency_scores"] for submission in submissions}
        saliency_gts = {target["qid"]: mk_saliency_scores(target) for target in targets}

        # apply threshold
        threshold = self.saliency_thresholds[self.threshold]
        qid_saliency_thresholded = {qid: (saliency >= threshold) for qid, saliency in saliency_gts.items()}

        # compute HIT1 for each query
        hit1_values = compute_hl_hit1(qid_preds, qid_saliency_thresholded, self.device)

        # update state
        self.hit1_values.append(hit1_values)

    def compute(self) -> Tensor:
        """Compute the HIT1 coefficient.

        Returns:
            Tensor: The HIT1 coefficient.
        """
        hits = dim_zero_cat(self.hit1_values)
        return torch.mean(hits) * 100


class AveragePrecision(Metric):
    """Average Precision implemented based on torchmetrics.Metric."""

    saliency_thresholds: Dict[str, int] = {"Binary": 1, "Fair": 2, "Good": 3, "VeryGood": 4}

    def __init__(self, threshold: ThresholdsTypes, **kwargs: Any) -> None:
        """Initialize the AveragePrecision metric.

        Args:
            threshold (ThresholdsTypes): The threshold type to use.
            kwargs (Any): Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.add_state("avg_precision", default=[], dist_reduce_fx="cat")
        self.threshold = threshold

    # pylint: disable=arguments-differ
    def update(self, submissions: List[Dict[str, Any]], targets: List[Dict[str, Any]]) -> None:  # noqa: WPS221
        """Update the state with new metric values.

        Args:
            submissions (List[Dict[str, Any]]): The model's predictions.
            targets (List[Dict[str, Any]]): The ground truth data.
        """
        # prepare inputs
        qid_preds = {submission["qid"]: submission["pred_saliency_scores"].cpu() for submission in submissions}
        saliency_gts = {target["qid"]: mk_saliency_scores(target) for target in targets}

        # apply threshold
        threshold = self.saliency_thresholds[self.threshold]
        qid_saliency_thresholded = {qid: (saliency >= threshold) for qid, saliency in saliency_gts.items()}

        # compute AvgPrecision for each query
        avg_precision = compute_hl_ap(qid_preds, qid_saliency_thresholded, self.device)

        # update state
        self.avg_precision.append(avg_precision)

    def compute(self) -> Tensor:
        """Compute the AvgPrecision coefficient.

        Returns:
            Tensor: The AvgPrecision coefficient.
        """
        hits = dim_zero_cat(self.avg_precision)
        return torch.mean(hits)


class MAPTop5ForAnnotators(Metric):
    """mAP top-5 for 20 annotators implemented based on torchmetrics.Metric."""

    def __init__(self, n_annotators: int = 20, **kwargs: Any) -> None:
        """Initialize the MAPTop5ForAnnotators metric.

        Args:
            n_annotators (int): number of annotation for each clip
            kwargs (Any): Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.add_state("map_top5", default=[], dist_reduce_fx="cat")
        self.n_annotators = n_annotators

    # pylint: disable=arguments-differ
    def update(self, submissions: List[Dict[str, Any]], targets: List[Dict[str, Any]]) -> None:  # noqa: WPS221
        """Update the state with new metric values.

        Args:
            submissions (List[Dict[str, Any]]): The model's predictions.
            targets (List[Dict[str, Any]]): The ground truth data.
        """
        # Prepare predictions and ground truth annotations
        qid_preds = {submission["qid"]: submission["pred_saliency_scores"].cpu() for submission in submissions}
        qid_targets = {target["qid"]: torch.tensor(target["saliency_scores"]) for target in targets}

        # Iterate over all annotators
        for annotator_idx in range(self.n_annotators):
            video_ap = []

            for qid, preds in qid_preds.items():
                # Sort predictions in descending order
                sorted_inds = torch.argsort(preds, descending=True)

                # Extract the corresponding ground truth annotations for the current annotator
                label = qid_targets[qid][:, annotator_idx]

                # Binarize the labels based on the median
                # ATTENTION: If all zeros are present, it may cause an error!
                threshold = label.median()
                label = torch.where(label > threshold, 1.0, 0.0)

                # Sort labels according to the sorted predictions and take top-5
                label_sorted = label[sorted_inds][:5]

                # Calculate Average Precision for this query
                num_gt = label_sorted.sum().item()
                if num_gt == 0:
                    video_ap.append(torch.tensor(0.0))  # Ensure this is a tensor
                    continue

                hits = ap = rec = 0
                prc = 1

                for j, gt in enumerate(label_sorted):
                    hits += gt

                    _rec = hits / num_gt
                    _prc = hits / (j + 1)

                    ap += (_rec - rec) * (prc + _prc) / 2  # type: ignore
                    rec, prc = _rec, _prc  # type: ignore

                video_ap.append(torch.tensor(ap))  # Ensure this is a tensor

            # Average the APs over all videos for this annotator
            mean_ap = torch.tensor(video_ap).mean() if video_ap else torch.tensor(0.0)
            self.map_top5.append(mean_ap)

    def compute(self) -> Tensor:
        """Compute the mAP top-5 coefficient.

        Returns:
            Tensor: The mAP top-5 coefficient.
        """
        collected_aps = dim_zero_cat(self.map_top5)
        return torch.mean(collected_aps)
