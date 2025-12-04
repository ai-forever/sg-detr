"""Module for computing losses."""

from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss
from torch.nn import functional as func

from src.losses.focal import FocalLossBinary
from src.losses.utils import get_src_permutation_idx
from src.utils.span_utils import generalized_temporal_iou, span_cxw_to_xx

MatchingIdx = List[Tuple[Tensor, Tensor]]


class DenoiseLosses(nn.Module):
    """Compute the Denoise losses for DETR."""

    def __init__(self, use_focal: bool = True, gamma: float = 2, denoise_reducer: float = 0.5) -> None:
        """Initialize DenoiseLosses.

        Args:
            use_focal: whether to use focal loss or not.
            gamma (float): Gamma factor for focal loss calculation. Defaults to 2.0.
            denoise_reducer (float): weight reducer for denoise loss
        """
        super().__init__()
        self.use_focal = use_focal
        self.label_loss = (
            FocalLossBinary(reduction="none", alpha=None, gamma=gamma)
            if use_focal
            else BCEWithLogitsLoss(reduction="none")
        )
        self.denoise_reducer = denoise_reducer

    def loss_spans(
        self,
        output_known_coord: Tensor,
        targets: Dict[str, Any],
        indices: List[Tuple[Tensor, Tensor]],
        **_: Any,
    ) -> Dict[str, Tensor]:
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Args:
            output_known_coord (Tensor): Denoise span coords.
            targets (Dict[str, Any]): Targets dicts
            indices (List[Tuple[Tensor, Tensor]]): Matched indices.
            _ (Any): unused arguments

        Note:
            Targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
            The target spans are expected in format (center_x, w), normalized by the image size.

        Returns:
            Dict[str, Tensor]: A dict containing the L1 regression and gIoU losses.
        """
        targets = targets["span_labels"]
        idx = get_src_permutation_idx(indices)
        src_spans = output_known_coord[idx]
        tgt_spans = torch.cat(
            [target["spans"][idx] for target, (_, idx) in zip(targets, indices)],  # type: ignore
            dim=0,
        )  # (#spans, 2)
        loss_span = func.smooth_l1_loss(src_spans, tgt_spans, reduction="none")
        loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))

        losses: Dict[str, Tensor] = {}
        losses["dn_loss_span"] = loss_span.mean() * self.denoise_reducer
        losses["dn_loss_giou"] = loss_giou.mean() * self.denoise_reducer

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses["log_only_dn_loss_center"] = loss_span[..., :1].mean()
            losses["log_only_dn_loss_width"] = loss_span[..., 1:].mean()
        return losses

    def loss_labels(
        self,
        output_known_class: Tensor,
        indices: List[Tuple[Tensor, Tensor]],
        neg_indices: List[Tuple[Tensor, Tensor]],
        **_: Any,
    ) -> Dict[str, Tensor]:
        """Background and foreground classification loss.

        Args:
            output_known_class (Tensor): Denoise predicted classes.
            indices (List[Tuple[Tensor, Tensor]]): Matched indices.
            neg_indices (List[Tuple[Tensor, Tensor]]): Neg matched indices.
            _ (Any): unused arguments

        Returns:
            Dict[str, Tensor]: A dict containing the classification loss.
        """
        output_known_class = output_known_class.squeeze(-1)
        idx = get_src_permutation_idx(indices)
        neg_idx = get_src_permutation_idx(neg_indices)
        target_classes = torch.full(
            output_known_class.shape[:2],
            0,
            dtype=torch.int64,
            device=output_known_class.device,
        )
        target_classes[idx] = 1

        # apply masking
        real_indices = [torch.cat((pos, neg)) for pos, neg in zip(idx, neg_idx)]  # noqa: WPS221

        loss_ce = self.label_loss(
            output_known_class[real_indices].reshape(-1),
            target_classes[real_indices].reshape(-1).float(),
        )
        return {"dn_loss_label": loss_ce.mean() * self.denoise_reducer}

    @staticmethod
    def find_pos_indices(  # noqa: WPS602
        targets: Dict[str, Any],
        single_pad: int,
        num_dn_groups: int,
        device: torch.device,
    ) -> Tuple[MatchingIdx, MatchingIdx]:
        """Prepare denoise indices.

        Args:
            targets (Dict[str, Any]): Targets to use.
            single_pad (int): max padding length of the denoise group.
            num_dn_groups (int): Number of the denoise groups.
            device (torch.device): torch device.

        Returns:
            Tuple[MatchingIdx, MatchingIdx]: Prepared pos and neg indices.
        """
        dn_pos_idx = []
        dn_neg_idx = []
        for target in targets["span_labels"]:
            if len(target["spans"]) != 0:  # noqa: WPS507, WPS504
                tgt_idx = torch.arange(0, len(target["spans"]), dtype=torch.long, device=device)  # noqa: WPS221
                tgt_idx = tgt_idx.unsqueeze(0).repeat(num_dn_groups, 1)
                # target indices
                tgt_idx_flatten = tgt_idx.flatten()
                # output indices
                output_idx = torch.arange(0, num_dn_groups, dtype=torch.long, device=device) * single_pad
                output_idx = output_idx.unsqueeze(1) + tgt_idx
                output_idx = output_idx.flatten()
            else:
                output_idx = torch.tensor([], dtype=torch.long, device=device)
                tgt_idx = torch.tensor([], dtype=torch.long, device=device)

            dn_pos_idx.append((output_idx, tgt_idx_flatten))
            dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx_flatten))
        return dn_pos_idx, dn_neg_idx

    @staticmethod
    def prepare_for_loss(mask_dict: Dict[str, Any]) -> Tuple[int, int, Tensor, Tensor]:  # noqa: WPS602
        """
        Prepare dn components to calculate loss.

        Args:
            mask_dict: a dict that contains dn information

        Returns:
            Tuple[int, int, Tensor, Tensor, int]: Prepared denoise components.
        """
        output_known_class, output_known_coord = mask_dict["output_known_lbs_bboxes"]
        num_dn_groups, pad_size = mask_dict["num_groups"], mask_dict["pad_size"]
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups
        return single_pad, num_dn_groups, output_known_class, output_known_coord

    def forward(  # noqa: WPS221
        self,
        mask_dict: Dict[str, Any],
        targets: Dict[str, Any],
        aux_num: int,
    ) -> Dict[str, Tensor]:
        """
        Compute dn loss in criterion.

        Args:
            mask_dict (Dict[str, Any]): a dict for dn information.
            targets (Dict[str, Any]): Targets to use.
            aux_num (int): aux loss number

        Returns:
            Dict[str, Tensor]: computed losses.
        """
        losses = {}
        single_pad, num_dn_groups, output_known_class, output_known_coord = self.prepare_for_loss(mask_dict)
        device = output_known_coord.device
        dn_pos_idx, dn_neg_idx = self.find_pos_indices(targets, single_pad, num_dn_groups, device)
        losses.update(self.loss_spans(output_known_coord[aux_num], targets, dn_pos_idx))
        losses.update(self.loss_labels(output_known_class[aux_num], dn_pos_idx, dn_neg_idx))
        return losses
