"""Module for computing losses."""

from typing import Any, Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as func

from src.model.utils.model_utils import inverse_sigmoid


class SaliencyLosses(nn.Module):
    """Compute the Saliency losses for DETR."""

    def __init__(
        self,
        saliency_margin: float,
        contrastive_reducer: float,
        local_saliency_loss_scale: float,
        use_negative_losses: bool,
    ) -> None:
        """Init saliency loss.

        Args:
            saliency_margin (float): saliency margin
            contrastive_reducer (float): weight reducer for constrastive loss
            local_saliency_loss_scale(float): local saliency bce loss scale
        """
        super().__init__()
        self.contrastive_reducer = contrastive_reducer
        self.saliency_margin = saliency_margin
        self.local_saliency_loss_scale = local_saliency_loss_scale
        self.use_negative_losses = use_negative_losses

    def saliency_bce_loss(
        self,
        saliency_scores,
        vid_token_mask: Tensor,
        targets: Dict[str, Any],
        **_: Any,
    ) -> Tensor:
        """Compute saliency bce loss.

        BCE_Loss = sum(BCE(sigmoid(logits),labels) * mask,
        where:
            - labels is saliency_labels > 0, relevant clips
            - logits is model predictions
            - mask is binary padding mask


        Args:
            saliency_scores: predicted saliency scores.
            video_mask: saliency mask
            targets (Dict[str, Any]): Targets dicts.
            _ (Any): unused arguments

        Returns:
            Tensor: the bce saliency loss.
        """
        saliency_binary_label = torch.clamp(targets["saliency_all_labels"], 0, 1)
        vid_token_mask = vid_token_mask.reshape(-1)
        logits = saliency_scores.reshape(-1)
        labels_x = saliency_binary_label.reshape(-1)
        bceloss = func.binary_cross_entropy_with_logits(logits, labels_x, reduction="none")
        bceloss = bceloss[vid_token_mask.bool()]
        return bceloss.mean()

    def saliency_attn_bce_loss(self, outputs: Dict[str, Any], targets: Dict[str, Any], **_: Any) -> Tensor:
        """Compute the t2v attention loss.

        Args:
            outputs (Dict[str, Any]): See the output specification of the model for the format
            targets (Dict[str, Any]): Targets dicts.
            _ (Any): unused arguments

        Returns:
            Tensor: the attn bce saliency loss.
        """
        if outputs["t2vattnvalues"] is None:
            return 0  # type: ignore
        vid_token_mask = outputs["video_mask"]
        saliency_scores = outputs["t2vattnvalues"]
        saliency_binary_label = torch.clamp(targets["saliency_all_labels"], 0, 1)
        vid_token_mask = vid_token_mask.reshape(-1)
        logits = inverse_sigmoid(saliency_scores.reshape(-1))
        labels_x = saliency_binary_label.reshape(-1)
        bceloss = func.binary_cross_entropy_with_logits(logits, labels_x, reduction="none")
        return bceloss[vid_token_mask.bool()].mean()

    def saliency_neg_pair_loss(
        self,
        saliency_scores_neg: Tensor,
        vid_token_mask: Tensor,
        real_neg_mask: Tensor,
        **_: Any,
    ) -> Tensor:
        """Compute saliency negative pairs loss.

        Args:
            saliency_scores_neg (Tensor): Saliency scores
            vid_token_mask (Tensor): Token mask
            real_neg_mask (Tensor): Sample mask
            _ (Any): unused arguments

        Returns:
            Tensor: the neg saliency loss.
        """
        saliency_neg_sigmoid = torch.sigmoid(saliency_scores_neg)
        saliency_neg_sigmoid = saliency_neg_sigmoid.reshape(-1)
        vid_token_mask = vid_token_mask[real_neg_mask].reshape(-1)
        loss_neg_pair = -torch.log(1.0 - saliency_neg_sigmoid)
        return loss_neg_pair[vid_token_mask.bool()].mean()

    def saliency_neg_pair_loss_attn(self, outputs: Dict[str, Any], **_: Any) -> Tensor:
        """Compute saliency negative pairs attn loss.

        Args:
            outputs (Dict[str, Any]): See the output specification of the model for the format
            _ (Any): unused arguments

        Returns:
            Tensor: the neg attn saliency loss.
        """
        if outputs["t2vattnvalues_neg"] is None:
            return 0  # type: ignore

        vid_token_mask = outputs["video_mask"]
        real_neg_mask = outputs["real_neg_mask"]
        saliency_scores_neg = outputs["t2vattnvalues_neg"]  # (N, L)
        loss_neg_pair_attn = -torch.log(1.0 - saliency_scores_neg) * (vid_token_mask[real_neg_mask])
        return loss_neg_pair_attn.sum(dim=1).mean()

    def margin_saliency_loss(self, saliency_scores, targets: Dict[str, Any], **_: Any) -> Tensor:
        """Compute the margin saliency loss.

        Args:
            saliency_scores: predicted saliency scores.
            targets (Dict[str, Any]): Targets dicts.
            _ (Any): unused arguments

        Returns:
            Tensor: the margin saliency loss.
        """
        saliency_scores = torch.sigmoid(saliency_scores)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)],
            dim=1,
        )
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)],
            dim=1,
        )
        margin = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum()
        return margin / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

    def margin_saliency_loss_attn(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        **_: Any,
    ) -> Tensor:
        """Compute the margin saliency attn loss.

        Args:
            outputs (Dict[str, Any]): See the output specification of the model for the format
            targets (Dict[str, Any]): Targets dicts.
            _ (Any): unused arguments

        Returns:
            Tensor: the margin attn saliency loss.
        """
        if outputs["t2vattnvalues"] is None:
            return 0  # type: ignore

        saliency_scores = outputs["t2vattnvalues"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)],
            dim=1,
        )
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)],
            dim=1,
        )
        margin_attn = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum()
        return margin_attn / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

    def saliency_rank_contrastive_loss(
        self,
        pred_saliency_scores: Tensor,
        pred_saliency_scores_neg: Tensor,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        **_: Any,
    ) -> Tensor:
        """Compute the saliency rank contrastive loss.

        Args:
            pred_saliency_scores (Tensor): predicted saliency scores.
            pred_saliency_scores_neg (Tensor): predicted neg saliency scores
            outputs (Dict[str, Any]): See the output specification of the model for the format
            targets (Dict[str, Any]): Targets dicts.
            _ (Any): unused arguments

        Returns:
            Tensor: the rank saliency loss.
        """
        saliency_scores = pred_saliency_scores.clone()  # (N, L)
        saliency_scores_neg = pred_saliency_scores_neg.clone()  # (N, L)

        vid_token_mask = outputs["video_mask"]
        real_neg_mask = outputs["real_neg_mask"]

        # targets
        saliency_contrast_label = targets["saliency_all_labels"]

        # real neg
        realneg_saliency_scores = torch.cat([saliency_scores[real_neg_mask], saliency_scores_neg], dim=1)
        realneg_saliency_contrast_label = torch.cat(
            [saliency_contrast_label[real_neg_mask], torch.zeros_like(saliency_contrast_label)[real_neg_mask]],
            dim=1,
        )
        realneg_vid_token_mask = vid_token_mask[real_neg_mask].repeat([1, 2])
        realneg_saliency_scores = (
            realneg_vid_token_mask * realneg_saliency_scores + (1.0 - realneg_vid_token_mask.float()) * -1e3
        )

        tau = 0.5
        loss_rank_contrastive = 0.0
        # not all 1, ..., 12 may exists ...
        for rand_idx in range(1, 12):
            drop_mask = ~(realneg_saliency_contrast_label > 100)  # no drop
            pos_mask = realneg_saliency_contrast_label >= rand_idx  # positive when equal or higher than rand_idx
            if torch.sum(pos_mask) == 0:  # no positive sample
                continue
            else:
                batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

            # drop higher ranks
            cur_saliency_scores = realneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e3
            # numerical stability
            logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
            # softmax
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

            mean_log_prob_pos = (pos_mask * log_prob * realneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
            loss = -mean_log_prob_pos * batch_drop_mask
            loss_rank_contrastive = loss_rank_contrastive + loss.mean()
        loss_rank_contrastive = loss_rank_contrastive / 12

        false_neg_mask = ~(real_neg_mask)
        if false_neg_mask.sum() != 0:
            if false_neg_mask.sum() == 1:
                falseneg_saliency_scores = saliency_scores[false_neg_mask].unsqueeze(0)
                falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask].unsqueeze(0)
                falseneg_vid_token_mask = vid_token_mask[false_neg_mask].unsqueeze(0)
                falseneg_saliency_scores = (
                    falseneg_vid_token_mask * falseneg_saliency_scores + (1.0 - falseneg_vid_token_mask.float()) * -1e3
                )
            else:
                falseneg_saliency_scores = saliency_scores[false_neg_mask]
                falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask]
                falseneg_vid_token_mask = vid_token_mask[false_neg_mask]
                falseneg_saliency_scores = (
                    falseneg_vid_token_mask * falseneg_saliency_scores + (1.0 - falseneg_vid_token_mask.float()) * -1e3
                )

            tau = 0.5
            falseneg_loss_rank_contrastive = 0.0
            for rand_idx in range(1, 12):
                drop_mask = ~(falseneg_saliency_contrast_label > 100)  # no drop
                pos_mask = falseneg_saliency_contrast_label >= rand_idx  # positive when equal or higher than rand_idx
                if torch.sum(pos_mask) == 0:  # no positive sample
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                # drop higher ranks
                cur_saliency_scores = falseneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e3
                # numerical stability
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                # softmax
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                mean_log_prob_pos = (pos_mask * log_prob * falseneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                loss = -mean_log_prob_pos * batch_drop_mask
                falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive + loss.mean()
            falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive / 12
            loss_rank_contrastive = loss_rank_contrastive + falseneg_loss_rank_contrastive
        return loss_rank_contrastive  # type: ignore

    def saliency_rank_contrastive_loss_attn(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        **_: Any,
    ) -> Tensor:
        """Compute the saliency rank contrastive loss.

        Args:
            outputs (Dict[str, Any]): See the output specification of the model for the format
            targets (Dict[str, Any]): Targets dicts.
            _ (Any): unused arguments

        Returns:
            Tensor: the rank attn saliency loss.
        """
        if outputs["t2vattnvalues"] is None:
            return 0  # type: ignore

        # outputs
        saliency_scores = outputs["t2vattnvalues"].clone()  # (N, L)
        saliency_scores_neg = outputs["t2vattnvalues_neg"].clone()

        # masks
        real_neg_mask = outputs["real_neg_mask"]
        vid_token_mask = outputs["video_mask"]

        # targets
        saliency_contrast_label = targets["saliency_all_labels"]

        # real neg
        realneg_saliency_scores = torch.cat([saliency_scores[real_neg_mask], saliency_scores_neg], dim=1)
        realneg_saliency_contrast_label = torch.cat(
            [saliency_contrast_label[real_neg_mask], torch.zeros_like(saliency_contrast_label)[real_neg_mask]],
            dim=1,
        )
        realneg_vid_token_mask = vid_token_mask[real_neg_mask].repeat([1, 2])
        realneg_saliency_scores = (
            realneg_vid_token_mask * realneg_saliency_scores + (1.0 - realneg_vid_token_mask.float()) * -1e3
        )

        tau = 0.5
        loss_rank_contrastive_attn = 0.0
        for rand_idx in range(1, 12):
            drop_mask = ~(realneg_saliency_contrast_label > 100)  # no drop
            pos_mask = realneg_saliency_contrast_label >= rand_idx  # positive when equal or higher than rand_idx
            if torch.sum(pos_mask) == 0:  # no positive sample
                continue
            else:
                batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

            # drop higher ranks
            cur_saliency_scores = realneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e3
            # numerical stability
            logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
            # softmax
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

            mean_log_prob_pos = (pos_mask * log_prob * realneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
            loss = -mean_log_prob_pos * batch_drop_mask
            loss_rank_contrastive_attn = loss_rank_contrastive_attn + loss.mean()
        loss_rank_contrastive_attn = loss_rank_contrastive_attn / 12

        false_neg_mask = ~(real_neg_mask)
        if false_neg_mask.sum() != 0:
            if false_neg_mask.sum() == 1:
                falseneg_saliency_scores = saliency_scores[false_neg_mask].unsqueeze(0)
                falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask].unsqueeze(0)
                falseneg_vid_token_mask = vid_token_mask[false_neg_mask].unsqueeze(0)
                falseneg_saliency_scores = (
                    falseneg_vid_token_mask * falseneg_saliency_scores + (1.0 - falseneg_vid_token_mask.float()) * -1e3
                )
            else:
                falseneg_saliency_scores = saliency_scores[false_neg_mask]
                falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask]
                falseneg_vid_token_mask = vid_token_mask[false_neg_mask]
                falseneg_saliency_scores = (
                    falseneg_vid_token_mask * falseneg_saliency_scores + (1.0 - falseneg_vid_token_mask.float()) * -1e3
                )

            tau = 0.5
            falseneg_loss_rank_contrastive_attn = 0.0
            for rand_idx in range(1, 12):
                drop_mask = ~(falseneg_saliency_contrast_label > 100)  # no drop
                pos_mask = falseneg_saliency_contrast_label >= rand_idx  # positive when equal or higher than rand_idx
                if torch.sum(pos_mask) == 0:  # no positive sample
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                # drop higher ranks
                cur_saliency_scores = falseneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e3
                # numerical stability
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                # softmax
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                mean_log_prob_pos = (pos_mask * log_prob * falseneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                loss = -mean_log_prob_pos * batch_drop_mask
                falseneg_loss_rank_contrastive_attn = falseneg_loss_rank_contrastive_attn + loss.mean()
            falseneg_loss_rank_contrastive_attn = falseneg_loss_rank_contrastive_attn / 12
            loss_rank_contrastive_attn = loss_rank_contrastive_attn + falseneg_loss_rank_contrastive_attn
        return loss_rank_contrastive_attn  # type: ignore

    def forward(  # noqa: WPS218
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
    ) -> Dict[str, Tensor]:  # noqa: WPS221
        """
        Compute the Saliency losses for the model during training.

        Args:
            outputs (Dict[str, Any]): dict of tensors, see the output specification of the model for the format
            targets (Dict[str, Any]): Targets to use.

        Returns:
            Dict[str, Tensor]: dict of tensors with the loss values.
        """
        assert "saliency_scores" in outputs, "No saliency scores found."
        assert "saliency_pos_labels" in targets, "No saliency positive labels found."
        assert "saliency_neg_labels" in targets, "No saliency negative labels found."
        assert "saliency_all_labels" in targets, "No saliency all labels found."
        assert "video_mask" in outputs, "No video mask found."
        assert "real_neg_mask" in outputs, "No real neg mask found."
        losses = torch.tensor(0, device=outputs["local_saliency_scores"].device).float()

        # bce losses
        saliency_bce_local = self.saliency_bce_loss(outputs["local_saliency_scores"], outputs["video_mask"], targets)
        if outputs["saliency_scores"] is not None:
            saliency_bce_global = self.saliency_bce_loss(outputs["saliency_scores"], outputs["video_mask"], targets)
            losses = losses + saliency_bce_global
        saliency_attn_bce = self.saliency_attn_bce_loss(outputs, targets)
        losses = losses + saliency_bce_local * self.local_saliency_loss_scale
        losses = losses + saliency_attn_bce


        # # neg pairs losses
        if self.use_negative_losses:
            saliency_neg_pair_local = self.saliency_neg_pair_loss(
                outputs["local_saliency_scores_neg"],
                outputs["video_mask"],
                outputs["real_neg_mask"],
            )
            if outputs["saliency_scores_neg"] is not None:
                saliency_neg_pair_global = self.saliency_neg_pair_loss(
                    outputs["saliency_scores_neg"],
                    outputs["video_mask"],
                    outputs["real_neg_mask"],
                )
                losses = losses + saliency_neg_pair_global
            saliency_neg_pair_attn = self.saliency_neg_pair_loss_attn(outputs)
            losses = losses + saliency_neg_pair_local
            losses = losses + saliency_neg_pair_attn

        # margin losses
        margin_saliency_local = self.margin_saliency_loss(outputs["local_saliency_scores"], targets)
        if outputs["saliency_scores"] is not None:
            margin_saliency_global = self.margin_saliency_loss(outputs["saliency_scores"], targets)
            losses = losses + margin_saliency_global
        margin_saliency_attn = self.margin_saliency_loss_attn(outputs, targets)
        losses = losses + margin_saliency_local
        losses = losses + margin_saliency_attn

        saliency_contrastive_local = (
            self.saliency_rank_contrastive_loss(
                outputs["local_saliency_scores"], outputs["local_saliency_scores_neg"], outputs, targets
            )
            * self.contrastive_reducer
        )
        if outputs["saliency_scores"] is not None:
            saliency_contrastive_global = (
                self.saliency_rank_contrastive_loss(
                    outputs["saliency_scores"], outputs["saliency_scores_neg"], outputs, targets
                )
                * self.contrastive_reducer
            )
            losses = losses + saliency_contrastive_global
        saliency_contrastive_attn = (
            self.saliency_rank_contrastive_loss_attn(outputs, targets) * self.contrastive_reducer
        )
        losses = losses + saliency_contrastive_local
        losses = losses + saliency_contrastive_attn

        out = {
            "loss_saliency": losses,
            "log_saliency_bce_local": saliency_bce_local,
            "log_saliency_attn_bce": saliency_attn_bce,
            "log_margin_saliency_local": margin_saliency_local,
            "log_margin_saliency_attn": margin_saliency_attn,
            "log_saliency_contrastive_local": saliency_contrastive_local,
            "log_saliency_contrastive_attn": saliency_contrastive_attn,
        }

        if outputs["saliency_scores"] is not None:
            out["log_margin_saliency_global"] = margin_saliency_global
            out["log_saliency_bce_global"] = saliency_bce_global
            out["log_saliency_contrastive_global"] = saliency_contrastive_global

        if self.use_negative_losses:
            out.update(
                {
                    "log_saliency_neg_pair_local": saliency_neg_pair_local,
                    "log_saliency_neg_pair_attn": saliency_neg_pair_attn,
                }
            )
            if outputs["saliency_scores"] is not None:
                out["log_saliency_neg_pair_global"] = saliency_neg_pair_global
        return out
