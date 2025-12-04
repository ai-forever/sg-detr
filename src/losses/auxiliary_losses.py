"""Module for Auxiliary losses."""

from typing import Any, Dict

import torch
from torch import Tensor, nn

EPS: float = 1e-6


class AuxiliaryLosses(nn.Module):
    """Compute the MR losses for DETR."""

    def loss_orthogonal(self, outputs, **_: Any) -> Dict[str, Tensor]:
        """Compute the loss for the dummy tokens.

        We want the dummy to be orthogonal to each other.

        Args:
            outputs (Dict[str, Any]): see the output specification of the model for the format
            _ (Any): unused arguments

        Returns:
            Dict[str, Tensor]: A dict containing the loss for the dummy tokens.
        """
        assert "dummy_tokens" in outputs, "No dummy tokens found."
        if outputs["dummy_tokens"] is None:
            return {"loss_orthogonal": 0}  # type: ignore

        dummy_tokens = outputs["dummy_tokens"]  # (n_dum, dim)
        dummy_tokens_norm = dummy_tokens / dummy_tokens.norm(dim=2)[:, :, None]
        dummy_tokens_sim = torch.matmul(dummy_tokens_norm, dummy_tokens_norm.permute(0, 2, 1).detach())
        for idx, _ in enumerate(dummy_tokens_sim):
            dummy_tokens_sim[idx].fill_diagonal_(0)
        loss_dummy_ortho = dummy_tokens_sim.abs().mean()
        return {"loss_orthogonal": loss_dummy_ortho}

    def forward(self, outputs: Dict[str, Any]) -> Dict[str, Any]:  # noqa: WPS221
        """
        Compute the MR losses for the model during training.

        Args:
            outputs (Dict[str, Any]): dict of tensors, see the output specification of the model for the format

        Returns:
            Dict[str, Any]: dict of tensors, with the loss values.
        """
        losses: Dict[str, Any] = {}
        losses.update(self.loss_orthogonal(outputs))
        return losses
