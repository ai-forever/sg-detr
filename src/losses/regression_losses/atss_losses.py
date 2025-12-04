"""Module to define ATSSRetrievalLoss class."""

from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as func
from torchvision.ops import sigmoid_focal_loss

from src.utils.codetr_utils import prepare_targets
from src.utils.span_utils import cat_boxlist, decode_spans, span_cxw_to_xx

INF: int = 100000000
EPS: float = 1e-7


class ATSSRetrievalLoss:
    """ATSSRetrievalLoss class."""

    def __init__(self, top_k_positive_anchors: int = 9):
        """Initialize ATSSRetrievalLoss.

        Args:
            top_k_positive_anchors (int): number of top-k candidates to select
        """
        self.top_k_positive_anchors = top_k_positive_anchors

    def loss_labels(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the classification loss for the aux head.

        Args:
            logits (Tensor): Predicted logits.
            targets (Tensor): Targets.

        Returns:
            Tensor: Computed classification loss.
        """
        return sigmoid_focal_loss(logits, targets.float(), reduction="sum")

    def loss_centerness(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the centerness loss for the aux head.

        Args:
            logits (Tensor): Predicted centerness logits.
            targets (Tensor): Target centerness score.

        Returns:
            Tensor: Computed classification loss.
        """
        return nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="sum")

    def compute_span_losses(
        self,
        pred: Tensor,
        target: Tensor,
        anchor: Tensor,
        weight: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the generalized IoU loss and l1 loss.

        Args:
            pred (Tensor): Predicted spans.
            target (Tensor): Target spans.
            anchor (Tensor): Anchors.
            weight (Tensor): centerness targets as weights.

        Returns:
            Tuple[Tensor, Tensor]: Computed generalized IoU loss and L1 loss.
        """
        pred_spans = decode_spans(pred.view(-1, 2), anchor.view(-1, 2)).float()  # noqa: WPS221
        pred_x1 = pred_spans[:, 0]
        pred_x2 = pred_spans[:, 1]
        pred_x2 = torch.max(pred_x1, pred_x2)
        pred_area = pred_x2 - pred_x1

        gt_spans = decode_spans(target.view(-1, 2), anchor.view(-1, 2)).float()  # noqa: WPS221
        target_x1 = gt_spans[:, 0]
        target_x2 = gt_spans[:, 1]
        target_area = target_x2 - target_x1

        x1_intersect = torch.max(pred_x1, target_x1)
        x2_intersect = torch.min(pred_x2, target_x2)
        area_intersect = torch.zeros(pred_x1.size()).to(pred)
        mask = x2_intersect > x1_intersect
        area_intersect[mask] = x2_intersect[mask] - x1_intersect[mask]

        x1_enclosing = torch.min(pred_x1, target_x1)
        x2_enclosing = torch.max(pred_x2, target_x2)
        area_enclosing = x2_enclosing - x1_enclosing + EPS

        area_union = pred_area + target_area - area_intersect + EPS
        ious = area_intersect / area_union
        gious = ious - (area_enclosing - area_union) / area_enclosing

        giou_loss = 1 - gious
        l1_loss = func.smooth_l1_loss(pred_spans, gt_spans, reduction="none").sum(1)
        if weight is not None and weight.sum() > 0:
            giou_loss = (giou_loss * weight).sum()
            l1_loss = (l1_loss * weight).sum()
            return giou_loss, l1_loss
        return giou_loss.sum(), l1_loss.sum()

    def compute_centerness_targets(self, reg_targets: Tensor, anchors: Tensor) -> Tensor:
        """Compute the centerness targets.

        Args:
            reg_targets (Tensor): Regression targets.
            anchors (Tensor): Anchors.

        Returns:
            Tensor: computed centerness targets.
        """
        gts = decode_spans(reg_targets, anchors)

        anchors_cx = (anchors[:, 0] + anchors[:, 1]) / 2
        left = anchors_cx - gts[:, 0]
        right = gts[:, 1] - anchors_cx
        left_right = torch.stack([left, right], dim=1)

        min_offset = left_right.min(dim=-1)[0]
        max_offset = left_right.max(dim=-1)[0]
        centerness = min_offset / max_offset

        assert not torch.isnan(centerness).any()
        return centerness

    # def __call__(self, box_cls, box_regression, centerness, targets, anchors):
    def __call__(  # noqa: WPS210
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
    ) -> Dict[str, Tensor]:
        """
        Compute the aux head MR losses during training.

        Args:
            outputs (Dict[str, Any]): dict of tensors, see the output specification of the model for the format
            targets (Dict[str, Any]): Targets to use.
            meta (List[Dict[str, Any]]): Meta information.

        Returns:
            Dict[str, Any]: dict of tensors, with the loss values.
        """
        targets_xx = [
            (span_cxw_to_xx(target["spans"]) * sample_meta["duration"]).type(torch.int) / 2
            for target, sample_meta in zip(targets["span_labels"], meta)
        ]
        anchors = outputs["locations_aux"]
        box_cls = outputs["pred_logits_aux"]
        centerness = outputs["pred_cntrness_aux"]
        box_regression = outputs["pred_spans_aux"]

        # prepare targets
        labels, reg_targets = prepare_targets(targets_xx, anchors, self.top_k_positive_anchors)

        labels_flatten = torch.cat(labels, dim=0)
        reg_targets_flatten = torch.cat(reg_targets, dim=0)
        anchors_flatten = torch.cat([cat_boxlist(anchors_per_image).spans for anchors_per_image in anchors], dim=0)

        # prepare outputs
        box_cls_flatten = torch.cat(box_cls, dim=1).reshape(-1)
        box_regression_flatten = torch.cat(box_regression, dim=1).reshape(-1, 2)
        centerness_flatten = torch.cat(centerness, dim=1).reshape(-1)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        losses: Dict[str, Any] = {}
        # classification loss: Focal loss
        label_loss = self.loss_labels(box_cls_flatten, labels_flatten)
        losses["aux_loss_label"] = label_loss / max(1, pos_inds.numel())

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        anchors_flatten = anchors_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten, anchors_flatten)
            sum_centerness_targets = centerness_targets.sum().item()

            giou_loss, l1_loss = self.compute_span_losses(
                box_regression_flatten,
                reg_targets_flatten,
                anchors_flatten,
                weight=centerness_targets,
            )

            cntrness_loss = self.loss_centerness(centerness_flatten, centerness_targets)

            losses["aux_loss_giou"] = giou_loss / sum_centerness_targets
            losses["aux_loss_span"] = l1_loss / sum_centerness_targets
            losses["aux_loss_ctrness"] = cntrness_loss / pos_inds.numel()
        else:
            losses["aux_loss_giou"] = box_regression_flatten.sum()
            losses["aux_loss_ctrness"] = centerness_flatten.sum()

        return losses
