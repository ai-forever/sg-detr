"""Module for moment2text losses."""

from typing import Any, Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as func

from src.losses.focal import FocalLossMultiClass

EPS: float = 1e-6


class Moment2TextLosses(nn.Module):
    """Compute the MR losses for DETR."""

    def __init__(self) -> None:
        """Init MomentRetrievalLosses class."""
        super().__init__()
        self.align_criterion = FocalLossMultiClass(alpha=0.5, gamma=2)
        self.bce_criterion = nn.BCEWithLogitsLoss(reduction="none")

    def loss_contrastive_moment_sentence(self, outputs: Dict[str, Any], **_: Any) -> Dict[str, Tensor]:  # noqa: WPS118
        """Compute the contrastive loss between the moment and sentence embeddings.

        Args:
            outputs (Dict[str, Any]): See the output specification of the model for the format
            _ (Any): unused arguments

        Returns:
            Dict[str, Tensor]: A dict containing the contrastive loss.
        """
        moment_token = outputs["moment_token"]
        non_moment_token = outputs["non_moment_token"]
        sentence_token = outputs["sent_txt_token"].squeeze(1)

        if outputs["sent_dummy_token"] is None:
            # moment sentence contrastive
            moment_logits = func.normalize(moment_token, dim=1)
            sentence_logits = func.normalize(sentence_token, dim=1)
            similarity_matrix = torch.matmul(moment_logits, sentence_logits.T)  # B B
            labels = torch.eye(similarity_matrix.shape[0]).to(moment_token.device)
            labels = labels.max(dim=1)[1]
            loss_ms_align = self.align_criterion(similarity_matrix, labels)
            return {"loss_ms_align": loss_ms_align}

        sentence_dummy = outputs["sent_dummy_token"].squeeze(1)  # b, 1, d

        moment_logits = func.normalize(moment_token, dim=1)
        nmoment_logits = func.normalize(non_moment_token, dim=1)
        sentence_logits = func.normalize(sentence_token, dim=1)
        dummy_logits = func.normalize(sentence_dummy, dim=1)

        # moment sentence contrastive
        similarity_matrix = torch.matmul(moment_logits, sentence_logits.T)  # B B
        nsimilarity_matrix = torch.matmul(nmoment_logits, sentence_logits.T)  # B B
        similarity_matrix = torch.cat([similarity_matrix, nsimilarity_matrix], dim=1)
        labels = torch.eye(similarity_matrix.shape[0]).to(moment_token.device)
        nlabels = torch.zeros_like(nsimilarity_matrix).to(moment_token.device)
        labels = torch.cat([labels, nlabels], dim=1).max(dim=1)[1]  # noqa: WPS221

        # compute loss
        loss_ms_align = self.align_criterion(similarity_matrix, labels)

        # moment dummy contrastive
        dummy_similarity_matrix = torch.matmul(moment_logits, dummy_logits.T)
        dummy_nsimilarity_matrix = torch.matmul(nmoment_logits, dummy_logits.T)
        dummy_similarity_matrix = torch.cat([dummy_similarity_matrix, dummy_nsimilarity_matrix], dim=1)
        dummy_labels = torch.eye(similarity_matrix.shape[0])
        dummy_labels = dummy_labels.to(moment_token.device).bool()
        dummy_labels = (~dummy_labels).float()  # noqa: WPS221
        dummy_nlabels = torch.ones_like(nsimilarity_matrix).to(moment_token.device)
        dummy_labels = torch.cat([dummy_labels, dummy_nlabels], dim=1).max(dim=1)[1]  # noqa: WPS221

        # compute loss
        dummy_loss_ms_align = self.align_criterion(dummy_similarity_matrix, dummy_labels)
        loss_ms_align = loss_ms_align + dummy_loss_ms_align

        return {"loss_ms_align": loss_ms_align}

    def momcls_loss(  # noqa: WPS221
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        **_: Any,
    ) -> Dict[str, Tensor]:
        """Compute the contrastive loss between the moment and sentence embeddings.

        Args:
            outputs (Dict[str, Any]): See the output specification of the model for the format
            targets (Dict[str, Any]): Targets dicts.
            _ (Any): unused arguments

        Returns:
            Dict[str, Tensor]: A dict containing the contrastive loss.
        """
        moment_token = outputs["moment_token"]
        video_mask = outputs["video_mask"]
        src_vid = outputs["src_vid"]  # [bsz, L_vid, D_vid]

        momtokcls_pred = torch.matmul(moment_token.unsqueeze(1), src_vid.permute(0, 2, 1))  # bsz 1 L_vid
        momtokcls_pred = momtokcls_pred * moment_token.size(1) ** -0.5
        momtokcls_logit = momtokcls_pred.reshape(-1)
        momtokcls_label = torch.clamp(targets["relevant_clips"], 0, 1)
        momtokcls_label = momtokcls_label.reshape(-1)
        momcls_loss = self.bce_criterion(momtokcls_logit, momtokcls_label) * video_mask.reshape(-1)
        return {"loss_momcls": momcls_loss.mean()}

    def forward(self, outputs: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:  # noqa: WPS221
        """
        Compute the MR losses for the model during training.

        Args:
            outputs (Dict[str, Any]): dict of tensors, see the output specification of the model for the format
            targets (Dict[str, Any]): Targets to use.

        Returns:
            Dict[str, Any]: dict of tensors, with the loss values.
        """
        losses: Dict[str, Any] = {}
        losses.update(self.loss_contrastive_moment_sentence(outputs))
        losses.update(self.momcls_loss(outputs, targets))
        return losses
