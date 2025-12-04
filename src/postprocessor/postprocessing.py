"""Module to post-process the model outputs."""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as func
from torchvision.ops import nms

from src.postprocessor.weighted_boxes_fusion import (  # type: ignore[attr-defined]
    weighted_boxes_fusion_1d,
)
from src.utils.span_utils import SpanList, decode_spans, span_cxw_to_xx

ROUND_DIGITS: int = 10000


class Preparator:
    """Prepare model outputs."""

    def __init__(self, pre_nms_thresh: float = 0.05, nms_thresh: float = 0.5, fpn_post_nms_top_n: int = 10) -> None:
        """Initialize outputs preparator.

        Args:
            pre_nms_thresh (float): Threshold to filter out low probability predictions.
            nms_thresh (float): Threshold for non-maximum suppression.
            fpn_post_nms_top_n (int): Maximum number of detections to keep after NMS.
        """
        self.pre_nms_thresh = pre_nms_thresh
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n

    def select_over_all_levels(
        self,
        spanlists: List[Tensor],
        metas: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:  # noqa: WPS221
        """Select the best detections among all levels.

        Args:
            spanlists (List[Tensor]): List of tensors containing the predicted spans.
            metas (List[Dict[str, Any]]): Sample's meta.

        Returns:
            List[Dict[str, Any]]: The post-processed predictions, after applying box decoding and NMS.
        """
        results = []
        for preds, meta in zip(spanlists, metas):
            xmin, xmax, scores = preds.split(1, 1)  # type: ignore
            ymin = torch.zeros_like(xmin)
            ymax = torch.ones_like(xmax)
            boxes = torch.cat([xmin, ymin, xmax, ymax], dim=1)
            keep = nms(boxes=boxes, scores=scores.squeeze(1), iou_threshold=self.nms_thresh)
            after_nms_preds = preds[keep].cpu()
            number_of_detections = len(after_nms_preds)

            # Limit to max_per_seq detections
            if (self.fpn_post_nms_top_n > 0) and (number_of_detections > self.fpn_post_nms_top_n):  # noqa: WPS333
                scores = after_nms_preds[:, 2]
                thresh, _ = torch.kthvalue(scores, number_of_detections - self.fpn_post_nms_top_n + 1)
                keep = scores >= thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                after_nms_preds = after_nms_preds[keep]
            cur_query_pred = {
                "qid": meta["qid"],
                "query": meta["query"],
                "vid": meta["vid"],
                "pred_relevant_windows": after_nms_preds,
            }
            results.append(cur_query_pred)
        return results

    def prepare_single_feature_map(
        self,
        anchors: List[SpanList],
        box_cls: Tensor,
        box_regression: Tensor,
        centerness: Tensor,
        durations: List[int],
    ) -> List[Tensor]:
        """Prepare a single feature map for validation.

        This function processes the classification scores, box regression outputs, and centerness scores to generate a
        list of detected objects for each feature map in the batch. The detections are adjusted based on the
        provided anchors and durations.

        Args:
            anchors (List[SpanList]): A list of SpanList objects representing the anchors for each seq in the batch.
            box_cls (Tensor): A tensor containing the classification scores for each anchor.
            box_regression (Tensor): A tensor containing the box regression outputs for each anchor.
            centerness (Tensor): A tensor containing the centerness scores for each anchor.
            durations (List[int]): A list of integers representing the duration (in frames) for each seq in the batch.

        Returns:
            List[Tensor]: A list of tensors where each tensor contains the detected objects for an seq in the batch.
        """
        batch_size, _, _ = box_cls.shape

        # apply sigmoid to classification and centerness scores
        box_cls = box_cls.squeeze(-1).sigmoid()
        centerness = centerness.squeeze(-1).sigmoid()

        # multiply the classification scores with centerness scores
        box_cls = torch.sqrt(box_cls * centerness)

        candidate_inds = box_cls > self.pre_nms_thresh

        results = []
        for idx in range(batch_size):
            per_candidate_inds = candidate_inds[idx]
            per_box_cls = box_cls[idx][per_candidate_inds]
            per_box_regression = box_regression[idx][per_candidate_inds]
            per_anchors = anchors[idx].spans[per_candidate_inds]
            duration = durations[idx]
            detections = decode_spans(per_box_regression, per_anchors) * 2  # multiply by 2 to get seconds
            detections.clip_(min=0, max=duration)
            obj_prediction = torch.cat([detections, per_box_cls[:, None]], dim=1)
            results.append(obj_prediction)
        return results

    def prepare_aux_outputs(
        self,
        all_anchors: List[List[SpanList]],
        all_logits: List[Tensor],
        all_cntr: List[Tensor],
        all_offsets: List[Tensor],
        metas: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Prepare focs predictions.

        Args:
            all_anchors (List[List[SpanList]]): computed anchors for each scale.
            all_logits (List[Tensor]): predicted classification logits for each scale.
            all_cntr (List[Tensor]): predicted centerness logits for each scale.
            all_offsets (List[Tensor]): predicted offsets for each scale
            metas (List[Dict[str, Any]]): Meta information.

        Returns:
            List[Tensor]: the post-processed predictions, after applying box decoding and NMS
        """
        sampled_spans: List[List[Tensor]] = []
        durations = [meta["duration"] for meta in metas]
        all_anchors = list(zip(*all_anchors))  # type: ignore
        for anchors, logits, offsets, cntr in zip(all_anchors, all_logits, all_offsets, all_cntr):
            sampled_spans.append(self.prepare_single_feature_map(anchors, logits, offsets, cntr, durations))
        revered_sampled_spans = list(zip(*sampled_spans))
        spanlists: List[Tensor] = [torch.cat(spans) for spans in revered_sampled_spans]
        return self.select_over_all_levels(spanlists, metas)

    # pylint: disable=too-many-locals
    def prepare_detr_outputs(  # noqa: WPS210
        self,
        query_meta: List[Dict[str, Any]],
        model_inputs: Dict[str, Any],
        model_outputs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Prepare the MR results.

        Args:
            query_meta (Dict[str, Any]): Meta data of the queries.
            model_inputs (Dict[str, Any]): Model inputs.
            model_outputs (Dict[str, Any]): Model outputs.

        Returns:
            Dict[str, Any]: Postprocessed MR results.
        """
        mr_res = []

        # predicted scores
        prob = func.sigmoid(model_outputs["pred_logits"])  # (batch_size, #queries, 1)
        scores = prob[..., 0].cpu()  # (batch_size, #queries)
        pred_quality_scores = func.sigmoid(model_outputs["pred_quality_scores"])
        pred_quality_scores = pred_quality_scores[..., 0].cpu()

        # predicted spans
        valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
        pred_spans = model_outputs["pred_spans"].cpu()  # (bsz, #queries, 2)

        # predicted saliency scores
        if model_outputs["saliency_scores"] is not None:
            predicted_saliency_scores = model_outputs["saliency_scores"].half()  # (bsz, L)
        else:
            predicted_saliency_scores = model_outputs["local_saliency_scores"].half()  # (bsz, L)
        saliency_scores = []
        for idx, valid_length in enumerate(valid_vid_lengths):
            saliency_scores.append(predicted_saliency_scores[idx, : int(valid_length)])

        # compose predictions
        for idx, (meta, spans, score, pred_quality_score) in enumerate(  # noqa: WPS352
            zip(query_meta, pred_spans, scores, pred_quality_scores),
        ):  # noqa: WPS440
            # converts spans to secs
            spans = span_cxw_to_xx(spans) * meta["duration"]
            spans = torch.clamp(spans, 0, meta["duration"])

            # concat spans and scores, sort predictions by score and round to 4 digits
            indexes = torch.argsort(score, dim=0, descending=True)
            cur_ranked_preds = torch.cat([spans, score[:, None], pred_quality_score[:, None]], dim=1)  # noqa: WPS221
            cur_ranked_preds = cur_ranked_preds[indexes]

            # add meta data to results
            cur_query_pred = {
                "qid": meta["qid"],
                "query": meta["query"],
                "vid": meta["vid"],
                "pred_relevant_windows": cur_ranked_preds,
                "pred_saliency_scores": saliency_scores[idx],
            }

            mr_res.append(cur_query_pred)

        return mr_res

    @torch.no_grad()
    def __call__(  # noqa: WPS210, WPS234
        self,
        query_meta: List[Dict[str, Any]],
        model_inputs: Dict[str, Any],
        model_outputs: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:  # noqa: WPS221
        """Prepare the MR results.

        Args:
            query_meta (Dict[str, Any]): Meta data of the queries.
            model_inputs (Dict[str, Any]): Model inputs.
            model_outputs (Dict[str, Any]): Model outputs.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Prepared MR results.
        """
        detr_predictions = self.prepare_detr_outputs(query_meta, model_inputs, model_outputs)
        aux_predictions = self.prepare_aux_outputs(
            all_anchors=model_outputs["locations_aux"],
            all_logits=model_outputs["pred_logits_aux"],
            all_cntr=model_outputs["pred_cntrness_aux"],
            all_offsets=model_outputs["pred_spans_aux"],
            metas=query_meta,
        )
        return aux_predictions, detr_predictions


class OutputCombiner:
    """
    Combiner that combines model predictions into a new combined prediction
    """
    def __init__(
        self,
        pre_nms_thresh: float,
        nms_thresh: float,
        fpn_post_nms_top_n: int,
        ranking_attribute: str,
        iou_thr: float,
        skip_box_thr: float,
        weights: List[float],
        max_length: int = 152
    ) -> None:
        """_summary_

        Args:
            pre_nms_thresh (float): probability threshold for ATSS model(before combain)
            nms_thresh (float): NMS threshold ATSS model(before combain)
            fpn_post_nms_top_n (int): num spans from ATSS model(before combain)
            ranking_attribute (str): ranking attribute for DETR model as probability
            iou_thr (float): IOU threshold for weighted boxes fusion
            skip_box_thr (float): probability threshold for weighted boxes fusion
            weights (List[float]): model weights for weighted boxes fusion
            max_length (int, optional): max span length. Defaults to 152.
        """
        self._preparator = Preparator(
            pre_nms_thresh=pre_nms_thresh,
            nms_thresh=nms_thresh,
            fpn_post_nms_top_n=fpn_post_nms_top_n,
        )
        assert ranking_attribute in {"probs", "iou", "combo"}
        self.ranking_attribute = ranking_attribute
        self.iou_thr = iou_thr
        self.skip_box_thr = skip_box_thr
        self.weights = weights
        self.max_length = max_length

    @torch.no_grad()
    def __call__(  # noqa: WPS210, WPS234
        self,
        query_meta: List[Dict[str, Any]],
        model_inputs: Dict[str, Any],
        model_outputs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:  # noqa: WPS221
        # preprocess models outputs
        aux_predictions, detr_predictions = self._preparator(query_meta, model_inputs, model_outputs)
        # combine model predictions
        return self.boxes_fusion_1d(detr_predictions, aux_predictions)

    def boxes_fusion_1d(
        self,
        detr_outputs: List[Dict[str, Any]],
        aux_outputs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Fusion models spans

        Args:
            detr_outputs (List[Dict[str, Any]]): DETR outputs
            aux_outputs (List[Dict[str, Any]]): ATSS outputs

        Returns:
            List[Dict[str, Any]]: Combained outputs
        """
        new_outputs = []
        for detr_output, aux_output in zip(detr_outputs, aux_outputs):
            new_output = deepcopy(detr_output)

            detr_pred_relevant_windows = detr_output["pred_relevant_windows"].detach().cpu().numpy()
            spans_1 = detr_pred_relevant_windows[:, :2] / self.max_length
            scores = detr_pred_relevant_windows[:, 2]
            iou_scores = detr_pred_relevant_windows[:, 3]
            if self.ranking_attribute == "probs":
                probs_1 = scores
            elif self.ranking_attribute == "iou":
                probs_1 = iou_scores
            else:
                probs_1 = np.sqrt(scores * iou_scores)

            aux_pred_relevant_windows = aux_output["pred_relevant_windows"]
            spans_2 = aux_pred_relevant_windows[:, :2] / self.max_length
            probs_2 = aux_pred_relevant_windows[:, 2]

            spans_list = [spans_1, spans_2]
            scores_list = [probs_1, probs_2]
            labels_list = [[1] * len(spans_1), [1] * len(spans_2)]
            boxes, scores, _ = weighted_boxes_fusion_1d(
                spans_list,
                scores_list,
                labels_list,
                weights=self.weights,
                iou_thr=self.iou_thr,
                skip_box_thr=self.skip_box_thr,
            )
            boxes = boxes * self.max_length
            new_relevant_windows = np.concatenate([boxes, scores[:, None]], axis=1)
            new_relevant_windows = torch.tensor(new_relevant_windows)
            new_output["pred_relevant_windows"] = new_relevant_windows
            new_outputs.append(new_output)
        return new_outputs


class PostProcessorDETR:
    """Class to post-process the model outputs for DETR."""

    def __init__(
        self,
        clip_length: int = 2,
        min_ts_val: int = 0,
        max_ts_val: int = 150,
        min_w_l: int = 2,
        max_w_l: int = 150,
        move_window_method: str = "center",
        nms_threshold: Optional[float] = None,
        nms_attribute: str = "iou_score",
        ranking_attribute: str = "probs",
    ):
        """Initialize the post-processor.

        Args:
            clip_length (int): Clip length in seconds.
            min_ts_val (int): Minimum timestamp value.
            max_ts_val (int): Maximum timestamp value.
            min_w_l (int): Minimum window length.
            max_w_l (int): Maximum window length.
            move_window_method (str): One of left (left unchanged), center (center unchanged), right (right unchanged).
            nms_threshold (Optional[float]): threshold to use for NMS, if None, NMS will be not applied
            nms_attribute (str): which attribute will be used for NMS (`probs`, `iou_score` or `combo`)
            ranking_attribute (str): which attribute will be used for ranking spans, `probs`, `iou_score` or `combo`
        """
        assert nms_attribute in {"iou_score", "probs", "combo"}
        assert ranking_attribute in {"iou_score", "probs", "combo"}
        self.clip_length = clip_length
        self.min_ts_val = min_ts_val
        self.max_ts_val = max_ts_val
        self.min_w_l = min_w_l
        self.max_w_l = max_w_l
        self.move_window_method = move_window_method
        self.nms_threshold = nms_threshold
        self.nms_attribute = nms_attribute
        self.ranking_attribute = ranking_attribute

    def clip_min_max_timestamps(self, windows: Tensor) -> Tensor:
        """Ensure timestamps for all windows is within [min_val, max_val], clip is out of boundaries.

        Args:
            windows (Tensor): Tensor of shape (#windows, 2) containing the start and end timestamps.

        Returns:
            Tensor: Tensor of shape (#windows, 2) containing the clipped start and end timestamps.
        """
        return torch.clamp(windows, min=self.min_ts_val, max=self.max_ts_val)

    def round_to_multiple_clip_lengths(self, windows: Tensor) -> Tensor:
        """Round the start and end timestamps to multiples of clip_length.

        Args:
            windows (Tensor): Tensor of shape (#windows, 2) containing the start and end timestamps.

        Returns:
            Tensor: Tensor of shape (#windows, 2) containing the rounded start and end timestamps.
        """
        return torch.round(windows / self.clip_length) * self.clip_length

    def clip_window_lengths(self, windows: Tensor) -> Tensor:
        """Ensure window lengths are within [min_w_l, max_w_l], clip if out of boundaries.

        Args:
            windows (Tensor): Tensor of shape (#windows, 2) containing the start and end timestamps.

        Returns:
            Tensor: Tensor of shape (#windows, 2) containing the clipped start and end timestamps.
        """
        window_lengths = windows[:, 1] - windows[:, 0]
        small_rows = window_lengths < self.min_w_l
        if torch.sum(small_rows) > 0:
            windows = self.move_windows(windows, small_rows, self.min_w_l)
        large_rows = window_lengths > self.max_w_l
        if torch.sum(large_rows) > 0:
            windows = self.move_windows(windows, large_rows, self.max_w_l)
        return windows

    def move_windows(self, windows: Tensor, row_selector: Tensor, new_length: int) -> Tensor:
        """Move the windows to the left, center or right.

        Args:
            windows (Tensor): Tensor of shape (#windows, 2) containing the start and end timestamps.
            row_selector (Tensor): Tensor of shape (#windows) containing the rows to move.
            new_length (int): New window length.

        Raises:
            ValueError: If move_window_method is not one of left, center or right.

        Returns:
            Tensor: Tensor of shape (#windows, 2) containing the moved start and end timestamps.
        """
        if self.move_window_method == "left":
            windows[row_selector, 1] = windows[row_selector, 0] + new_length
        elif self.move_window_method == "right":
            windows[row_selector, 0] = windows[row_selector, 1] - new_length
        elif self.move_window_method == "center":
            center = (windows[row_selector, 1] + windows[row_selector, 0]) / 2
            windows[row_selector, 0] = center - new_length / 2
            windows[row_selector, 1] = center + new_length / 2
        else:
            raise ValueError(f"Unknown move_window_method: {self.move_window_method}")
        return windows

    def nms(self, segments: torch.Tensor) -> torch.Tensor:
        """
        Apply Non-Maximum Suppression (NMS) on a set of segments.

        Args:
            segments (torch.Tensor):  A tensor of shape (N, 4), were each segment is [start, end, prob, pred_iou]

        Returns:
            np.ndarray: A tensor containing the indices of the segments that are kept after applying NMS.
        """
        if self.nms_attribute == "probs":
            attribute_idx = 2
        elif self.nms_attribute == "iou_score":
            attribute_idx = 3
        else:
            attribute_idx = 4

        segments_new = segments[:, [0, 1, attribute_idx]]
        xmin, xmax, scores = segments_new.split(1, 1)  # type: ignore
        ymin = torch.zeros_like(xmin)
        ymax = torch.ones_like(xmax)
        boxes = torch.cat([xmin, ymin, xmax, ymax], dim=1)
        keep = nms(boxes=boxes, scores=scores.squeeze(1), iou_threshold=self.nms_threshold)
        return segments[keep]

    def __call__(
        self,
        model_outputs: List[Dict[str, Any]],
        aux_head: bool = False,
    ) -> List[Dict[str, Any]]:  # noqa: WPS221
        """Post-process the model outputs.

        Args:
            model_outputs (List[Dict[str, Any]]): Model outputs.
            aux_head (bool): Postprocessing for aux head or not.

        Returns:
            List[Dict[str, Any]]: Post-processed model outputs.
        """
        processed_lines = []
        for output in model_outputs:
            windows_and_scores = output["pred_relevant_windows"]
            windows = windows_and_scores[:, :2]
            windows = self.clip_min_max_timestamps(windows)
            windows = self.round_to_multiple_clip_lengths(windows)
            windows = self.clip_window_lengths(windows)
            if aux_head:
                scores = torch.round(windows_and_scores[:, 2:3] * ROUND_DIGITS) / ROUND_DIGITS  # noqa: WPS221
                output["pred_relevant_windows"] = torch.cat([windows, scores], dim=1)
                processed_lines.append(output)
                continue

            scores = torch.round(windows_and_scores[:, 2:3] * ROUND_DIGITS) / ROUND_DIGITS  # noqa: WPS221
            pred_iou_scores = torch.round(windows_and_scores[:, 3:4] * ROUND_DIGITS) / ROUND_DIGITS  # noqa: WPS221
            combo = torch.sqrt(scores * pred_iou_scores)
            pred_relevant_windows = torch.cat([windows, scores, pred_iou_scores, combo], dim=1)
            if self.nms_threshold is not None:
                pred_relevant_windows = self.nms(pred_relevant_windows)

            if self.ranking_attribute == "probs":
                attribute_idx = 2
            elif self.ranking_attribute == "iou_score":
                attribute_idx = 3
            else:
                attribute_idx = 4
            output["pred_relevant_windows"] = pred_relevant_windows[:, [0, 1, attribute_idx]]
            processed_lines.append(output)
        return processed_lines
