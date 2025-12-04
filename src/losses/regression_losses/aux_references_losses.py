"""Reference losses."""

from typing import Any, Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss
from torch.nn import functional as func

from src.losses.focal import FocalLossBinary
from src.utils.span_utils import generalized_temporal_iou, span_cxw_to_xx


class AuxRefLosses(nn.Module):
    """Compute the Aux Reference losses for DETR."""

    def __init__(self, use_focal: bool = True, gamma: float = 2, colab_ref_reducer: float = 0.5) -> None:
        """Initialize AuxRefLosses.

        Args:
            use_focal: whether to use focal loss or not.
            gamma (float): Gamma factor for focal loss calculation. Defaults to 2.0.
            colab_ref_reducer (float): weight reducer for colab ref loss
        """
        super().__init__()
        self.use_focal = use_focal
        self.label_loss = (
            FocalLossBinary(reduction="none", alpha=None, gamma=gamma)
            if use_focal
            else BCEWithLogitsLoss(reduction="none")
        )
        self.colab_ref_reducer = colab_ref_reducer

    def loss_spans(self, src_spans: torch.Tensor, tgt_spans: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the losses related to the noised spans (the L1 regression loss and the GIoU loss).

        Args:
            src_spans (torch.Tensor): predicted spans
            tgt_spans (torch.Tensor): target spans

        Returns:
            Dict[str, torch.Tensor]: A dict containing the L1 regression and gIoU losses.
        """
        src_spans = span_cxw_to_xx(src_spans)
        tgt_spans = span_cxw_to_xx(tgt_spans)
        loss_span = func.l1_loss(src_spans, tgt_spans, reduction="none")
        loss_giou = 1 - torch.diag(generalized_temporal_iou(src_spans, tgt_spans))

        losses = {}
        losses["co_loss_span"] = loss_span.mean() * self.colab_ref_reducer
        losses["co_loss_giou"] = loss_giou.mean() * self.colab_ref_reducer
        return losses

    def loss_labels(self, src_logits: Tensor, tgt_labels: Tensor) -> Dict[str, Tensor]:
        """Classification loss.

        Args:
            src_logits (Tensor): predicted logits
            tgt_labels (Tensor): target labels

        Returns:
            Dict[str, torch.Tensor]: A dict containing the classification loss and the classification error.q
        """
        src_logits, tgt_labels = src_logits.unsqueeze(0), tgt_labels.unsqueeze(0)
        loss_ce = self.label_loss(src_logits.reshape(-1), tgt_labels.reshape(-1).float())  # noqa:WPS221
        return {"co_loss_label": loss_ce.mean() * self.colab_ref_reducer}

    @staticmethod
    def prepare_for_loss(mask_dict: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:  # noqa: WPS602
        """
        Prepare dn components to calculate loss.

        Args:
            mask_dict: a dict that contains dn information

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, int]: Prepared components.
        """
        output_known_class, output_known_coord = mask_dict["output_known_lbs_bboxes"]
        known_labels, known_bboxs = mask_dict["known_lbs_bboxes"]

        # prepare logits
        output_known_class = output_known_class.permute(1, 2, 0, 3)
        output_known_class = output_known_class.flatten(0, 1).permute(1, 0, 2)

        # prepare reg preds
        output_known_coord = output_known_coord.permute(1, 2, 0, 3)
        output_known_coord = output_known_coord.flatten(0, 1)[known_labels.bool()]
        output_known_coord = output_known_coord.permute(1, 0, 2)
        return known_labels, known_bboxs, output_known_class, output_known_coord

    def forward(self, mask_dict: Dict[str, Any], aux_num: int) -> Dict[str, Tensor]:
        """
        Compute dn loss in criterion.

        Args:
            mask_dict (Dict[str, Any]): a dict for dn information
            aux_num (int): aux loss number

        Returns:
            Dict[str, Tensor]: computed losses.
        """
        losses = {}
        known_labels, known_bboxs, output_known_class, output_known_coord = self.prepare_for_loss(mask_dict)
        losses.update(self.loss_labels(output_known_class[aux_num], known_labels))
        losses.update(self.loss_spans(output_known_coord[aux_num], known_bboxs))
        return losses
