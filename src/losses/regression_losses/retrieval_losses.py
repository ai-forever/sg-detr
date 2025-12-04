"""Module for computing losses."""

from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss
from torch.nn import functional as func

from src.losses.focal import FocalLossBinary
from src.losses.utils import get_src_permutation_idx
from src.utils.span_utils import (
    generalized_temporal_iou,
    span_cxw_to_xx,
    temporal_intersection_criteria,
    temporal_iou,
)


class MainRegressionLosses(nn.Module):  # noqa: WPS230,WPS214
    """Compute the MR losses for DETR."""

    foreground = 0
    background = 1

    def __init__(  # noqa: WPS211
        self,
        backgorund_weight: float,
        weights_loss_params: Dict[str, float],
        use_focal: bool = True,
        gamma: float = 2,
        alpha: float = 0.25,
        encoder_coef: float = 1,
        multiple_criterion: Optional[str] = None,
        use_span_weights: bool = False,
        use_label_weights: bool = False,
        quality_scoring_mode: str = "positive",
        center_offset_margin: float = 0.1,
        width_offset_margin: float = 0.15,
    ) -> None:
        """Init MomentRetrievalLosses class.

        Args:
            backgorund_weight (float): relative classification weight applied to the no-object category
            weights_loss_params (Dict[str, float]): other loss related params
            use_focal: whether to use focal loss or not.
            gamma (float): gamma for focal loss. Defaults to 2.0.
            alpha (float): alpha for focal loss. Defaults to 0.25.
            encoder_coef (float): encoder loss scale
            multiple_criterion (str): type of multiple criterion loss function. One of {"iou", "intersection"}.
            use_span_weights (bool): whether to use weighted spans by size
            use_label_weights (bool): whether to use weighted labels by size
            quality_scoring_mode (str): If `all`, calculated for all spans, if `positive`, then only the matched spans.
            center_offset_margin (float): offsets regularization parameter.
            width_offset_margin (float): offsets regularization parameter.
        """
        super().__init__()
        self.use_focal = use_focal
        self.enc_coef = encoder_coef
        self.backgorund_weight = backgorund_weight

        # label loss
        self.label_loss = (
            FocalLossBinary(gamma=gamma, alpha=alpha, reduction="none")
            if use_focal
            else BCEWithLogitsLoss(reduction="none")
        )

        # add more weight for short spans
        self.use_span_weights = use_span_weights
        self.use_label_weights = use_label_weights
        self.weights_loss_params = weights_loss_params

        # init multiple criterion
        self.init_multiple_criterion(multiple_criterion)

        # The model also predicts the IOU between the predicted segment and the true segment.
        assert quality_scoring_mode in {"all", "positive"}
        self.quality_scoring_mode = quality_scoring_mode

        # offsets regularization params
        self.center_offset_margin = center_offset_margin
        self.width_offset_margin = width_offset_margin

    def init_multiple_criterion(self, multiple_criterion: Optional[str] = None):
        """Init multiple criteriation loss.

        Args:
            multiple_criterion (str): type of multiple criterion loss function. One of {"iou", "intersection"}.
        """
        # The predicted span can additionally intersect not only with the matched true span, but with other true span
        # If "intersection" the model will be penalized as Intersection/length(gt_span),
        # if "iou" the model will be penalized as IOU(pred_span, gt_span)
        assert multiple_criterion is None or multiple_criterion in {"iou", "intersection"}
        if multiple_criterion == "iou":
            self.multiple_criterion: Optional[partial[tuple[Tensor, Tensor]]] = partial(temporal_iou)  # noqa: WPS234
        elif multiple_criterion == "intersection":
            self.multiple_criterion = partial(temporal_intersection_criteria)
        else:
            self.multiple_criterion = None

    def _compute_span_weights(self, tgt_spans: Tensor) -> Tensor:
        assert self.weights_loss_params is not None
        gamma = self.weights_loss_params.get("gamma", 5)
        max_weight = self.weights_loss_params.get("max_weight", 2)
        widths = tgt_spans[:, 1]
        weights = -((1 - widths) ** gamma) * torch.log(widths) + 1  # Similar to the focal loss function, +1 is the
        # minimum loss for intervals that has width = video length
        return torch.clip(weights, 1, max_weight)

    def _compute_label_weights(self, targets_spans: Tensor, indices, weights_shape) -> Tensor:
        idx = get_src_permutation_idx(indices)
        tgt_spans = torch.cat(
            [target["spans"][idx] for target, (_, idx) in zip(targets_spans, indices)],  # type: ignore
            dim=0,
        )
        weights = self._compute_span_weights(tgt_spans)
        loss_weights = torch.full(
            weights_shape,
            1,
            dtype=torch.float,
            device=weights.device,
        )
        loss_weights[idx] = weights
        return loss_weights

    def loss_spans(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        indices: List[Tuple[Tensor, Tensor]],
        **_: Any,
    ) -> Dict[str, Tensor]:
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Args:
            outputs (Dict[str, Any]): See the output specification of the model for the format
            targets (Dict[str, Any]): Targets dicts
            indices (List[Tuple[Tensor, Tensor]]): Output from the matcher.
            _ (Any): unused arguments

        Note:
            Targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
            The target spans are expected in format (center_x, w), normalized by the image size.

        Returns:
            Dict[str, Tensor]: A dict containing the L1 regression and gIoU losses.
        """
        assert "pred_spans" in outputs
        targets = targets["span_labels"]
        idx = get_src_permutation_idx(indices)
        src_spans = outputs["pred_spans"][idx]  # (#spans, max_video_length * 2)
        tgt_spans = torch.cat(
            [target["spans"][idx] for target, (_, idx) in zip(targets, indices)],  # type: ignore
            dim=0,
        )  # (#spans, 2)
        loss_span = func.smooth_l1_loss(src_spans, tgt_spans, reduction="none")
        if self.use_span_weights:
            weights = self._compute_span_weights(tgt_spans)
            loss_span = loss_span * weights[:, None]
        loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))

        losses: Dict[str, Tensor] = {}
        losses["loss_span"] = loss_span.mean()
        losses["loss_giou"] = loss_giou.mean()
        return losses

    def loss_labels(  # noqa: WPS221
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        indices,
        **_: Any,
    ) -> Dict[str, Tensor]:
        """Background and foreground classification loss.

        Args:
            outputs (Dict[str, Tensor]): A dict containing the outputs of the model.
            targets (Dict[str, Any]): Targets dicts
            indices (List[List[int]]): Output from the matcher.
            _ (Any): unused arguments

        Returns:
            Dict[str, Tensor]: A dict containing the classification loss.
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"][..., 0]  # (batch_size, #queries, 1)

        idx = get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)  # noqa:WPS221
        target_classes[idx] = 1

        loss_ce = self.label_loss(src_logits, target_classes.float())
        if not self.use_focal:
            loss_ce = torch.where(target_classes.bool(), loss_ce, loss_ce * self.backgorund_weight)

        if self.use_label_weights:
            weights = self._compute_label_weights(
                targets_spans=targets["span_labels"],
                indices=indices,
                weights_shape=src_logits.shape[:2],
            )
            loss_ce = loss_ce * weights
        return {"loss_label": loss_ce.mean()}

    def loss_quality_scoring(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        indices: List[Tuple[Tensor, Tensor]],
        **_: Any,
    ) -> Dict[str, Tensor]:
        """
        Calculate quality iou score as sum(pred_iou - true_iou).

        Args:
            outputs (Dict[str, Any]): dict of tensors, see the output specification of the model for the format
            targets (Dict[str, Any]): list of dicts, such that len(targets) == batch_size.
            indices (List[Tuple[Tensor, Tensor]]): Output from the matcher.
            _ (Any): unused arguments

        Returns:
            Dict[str, Tensor]: dict with quality iou score
        """
        targets = targets["span_labels"]
        idx = get_src_permutation_idx(indices)
        predicted_spans = outputs["pred_spans"][idx]  # (#spans, max_video_length * 2)
        tgt_spans = torch.cat(
            [target["spans"][idx] for target, (_, idx) in zip(targets, indices)],  # type: ignore
            dim=0,
        )  # (#spans, 2)
        # calculate the iou between the true and predicted spans
        iou_scores = torch.diag(temporal_iou(span_cxw_to_xx(predicted_spans), span_cxw_to_xx(tgt_spans))[0])
        iou_scores = iou_scores.detach()
        if self.quality_scoring_mode == "all":
            # Create a matrix where IOU are for the matched spans and 0 are at all other positions
            gt_iou_scores = torch.full(
                outputs["pred_quality_scores"].shape[:2],
                0,
                dtype=torch.float32,
                device=predicted_spans.device,
            )  # (batch_size, #queries)
            gt_iou_scores[idx] = iou_scores
            predicted_iou_score = outputs["pred_quality_scores"][..., 0]
            # Create a matrix where 1 are for the matched spans and `backgorund_weight` are at all other positions
            weights = torch.full(
                gt_iou_scores.shape[:2],
                self.backgorund_weight,
                dtype=torch.float32,
                device=predicted_spans.device,
            )  # (batch_size, #queries)
            weights[idx] = 1
        else:
            gt_iou_scores = iou_scores
            # for the matched spans, extract the predictions of the iou score
            predicted_iou_score = outputs["pred_quality_scores"][idx][:, 0]
            weights = torch.ones_like(iou_scores)
        # calculate the difference between true and predicted iou scores
        loss_quality = func.binary_cross_entropy_with_logits(
            predicted_iou_score,
            gt_iou_scores,
            weight=weights,
            reduction="mean",
        )
        return {"loss_quality": loss_quality}

    def loss_multiple_spans(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        indices: List[Tuple[Tensor, Tensor]],
        **_: Any,
    ) -> Dict[str, Tensor]:
        """
        Use to prevent a situation when one predicted segment contains several true ones.

        loss = IOU(pred_span, unmatch_gt_span), for iou type
        loss = intersection(pred_span, unmatch_gt_span) / length(unmatch_gt_span), for intersection type

        Args:
            outputs (Dict[str, Any]): dict of tensors, see the output specification of the model for the format
            targets (Dict[str, Any]): list of dicts, such that len(targets) == batch_size.
            indices (List[Tuple[Tensor, Tensor]]): Output from the matcher.
            _ (Any): unused arguments

        Returns:
            Dict[str, Tensor]: dict with quality iou score
        """
        assert self.multiple_criterion is not None
        gt_spans: List[Dict[str, Tensor]] = targets["span_labels"]
        pred_spans = outputs["pred_spans"]
        batch_size = pred_spans.shape[0]
        # Only spans where there are many of them are taken into account
        overlap_n = torch.tensor(0, device=pred_spans.device, dtype=torch.float)
        total_overlap = torch.tensor(0, device=pred_spans.device, dtype=torch.float)
        for in_batch_idx in range(batch_size):
            pred_idx, gt_idx = indices[in_batch_idx]
            n_spans = pred_idx.shape[0]  # type: ignore
            if n_spans > 1:
                cost_matrics = self.multiple_criterion(
                    span_cxw_to_xx(pred_spans[in_batch_idx][pred_idx]),
                    span_cxw_to_xx(gt_spans[in_batch_idx]["spans"][gt_idx]),
                )[0]
                # Exclude the intersection with the matched spans
                overlap = cost_matrics.sum() - torch.diag(cost_matrics).sum()
                total_overlap += overlap
                overlap_n += n_spans
        losses: Dict[str, Tensor] = {}
        losses["loss_multiple_spans"] = total_overlap / overlap_n
        return losses

    def losss_offset_regularization(
        self,
        outputs: Dict[str, Any],
        **_: Any,
    ) -> Dict[str, Tensor]:
        """
        Compute L1 regularization loss for model offsets.

        Args:
            outputs (Dict[str, Any]): dict of tensors, see the output specification of the model for the format
            _ (Any): any kwargs

        Returns:
            Dict[str, Tensor]: dict with L1 offset loss
        """
        losses: Dict[str, Tensor] = {}
        center_offset = torch.abs(outputs["offset"][:, :, 0])  # (batch_size, num_queries)
        width_offset = torch.abs(outputs["offset"][:, :, 1])  # (batch_size, num_queries)
        center_offset = (center_offset - self.center_offset_margin).clamp(0, 1)
        width_offset = (width_offset - self.width_offset_margin).clamp(0, 1)
        center_l1_reg = center_offset.mean()
        width_l1_ref = width_offset.mean()

        losses["center_l1_ref"] = center_l1_reg
        losses["width_l1_ref"] = width_l1_ref
        return losses

    def loss_labels_vfl(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        indices: List[List[int]],
        **_: Any,
    ):
        """Iou aware classification loss for the Query Selector Output.

        Args:
            outputs (Dict[str, Tensor]): A dict containing the outputs of the model.
            targets (Dict[str, Any]): Targets dicts
            indices (List[List[int]]): Output from the matcher.
            _ (Any): unused arguments

        Returns:
            Dict[str, Tensor]: A dict containing the classification loss.
        """
        idx = get_src_permutation_idx(indices)
        src_logits = outputs["pred_logits"][..., 0]
        predicted_spans = outputs["pred_spans"][idx]

        # compute ious
        tgt_spans = torch.cat(
            [target["spans"][idx] for target, (_, idx) in zip(targets["span_labels"], indices)],  # type: ignore
            dim=0,
        )  # (#spans, 2)
        ious, _ = temporal_iou(span_cxw_to_xx(predicted_spans), span_cxw_to_xx(tgt_spans))  # type: ignore
        ious = torch.diag(ious).detach()

        # get targets
        target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)  # noqa:WPS221
        target_classes[idx] = 1

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o * target_classes

        pred_score = func.sigmoid(src_logits).detach()
        weight = pred_score.pow(2) * (1 - target_classes) + target_score

        loss_ce = func.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        return {"loss_vfl": loss_ce.mean()}

    def forward(  # noqa: WPS234
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        indices: List[Tuple[Tensor, Tensor]],
        enc_indices: Optional[List[Tuple[Tensor, Tensor]]] = None,
    ) -> Dict[str, Any]:  # noqa: WPS221
        """
        Compute the MR losses for the model during training.

        Args:
            outputs (Dict[str, Any]): dict of tensors, see the output specification of the model for the format
            targets (Dict[str, Any]): Targets to use.
            indices (List[Tuple[Tensor, Tensor]]): Output from the matcher for main head.
            enc_indices (Optional[List[Tuple[Tensor, Tensor]]]): Output from the matcher for encoder_head.

        Returns:
            Dict[str, Any]: dict of tensors, with the loss values.
        """
        losses: Dict[str, Any] = {}
        if enc_indices is not None:
            enc_outputs = outputs["encoder_outputs"]
            # compute enc losses
            span_losses = self.loss_spans(enc_outputs, targets, enc_indices)
            cls_losses = self.loss_labels(enc_outputs, targets, enc_indices)
            quality_scoring = self.loss_quality_scoring(enc_outputs, targets, enc_indices)
            cls_iou_aware_loss = self.loss_labels_vfl(enc_outputs, targets, enc_indices)  # type: ignore
            # add enc losses
            losses.update({f"enc_{key}": value * self.enc_coef for key, value in span_losses.items()})
            losses.update({f"enc_{key}": value * self.enc_coef for key, value in cls_losses.items()})
            losses.update({f"enc_{key}": value * self.enc_coef for key, value in quality_scoring.items()})
            losses.update({f"enc_{key}": value * self.enc_coef for key, value in cls_iou_aware_loss.items()})
        # compute decoder losses
        losses.update(self.loss_spans(outputs, targets, indices))
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.losss_offset_regularization(outputs))
        if self.multiple_criterion is not None:
            losses.update(self.loss_multiple_spans(outputs, targets, indices))
        losses.update(self.loss_quality_scoring(outputs=outputs, targets=targets, indices=indices))
        return losses
