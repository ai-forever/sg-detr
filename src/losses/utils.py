"""Losses utils."""

from typing import List, Tuple

import torch
from torch import Tensor

EPS: float = 1e-7


def fix_loss_name(loss_name: str) -> str:  # noqa: WPS212
    """Fix loss name to get appropriate loss weight.

    Args:
        loss_name (str): initial loss name

    Returns:
        str: fixed loss name
    """
    if "dn_" in loss_name:
        return loss_name[3:]

    if "co_" in loss_name:
        return loss_name[3:]

    if "enc_" in loss_name:
        return loss_name[4:]

    return loss_name


def get_src_permutation_idx(indices: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare source indices for permutation based on the indices of the matched pairs.

    Args:
        indices (list): A list of tuples containing the indices of the matched pairs.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the batch indices and the source indices.
    """
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, idx) for idx, (src, _) in enumerate(indices)])  # noqa: WPS221
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx  # two 1D tensors of the same length


def prepare_model_preds(
    lvl_logits: List[Tensor],
    cntrness: List[Tensor],
    regression: List[Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Prepare the model predictions.

    Args:
        lvl_logits (List[Tensor]): List of logits.
        cntrness (List[Tensor]): List of centerness scores.
        regression (List[Tensor]): List of regression targets.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Flattened logits, centerness, and regression targets.
    """
    logits_flatten = []
    regression_flatten = []
    centerness_flatten = []
    for scale_idx, _ in enumerate(lvl_logits):
        logits_flatten.append(lvl_logits[scale_idx].reshape(-1))
        centerness_flatten.append(cntrness[scale_idx].reshape(-1))

        # convert regression offsets to seconds
        regression_scale = regression[scale_idx]
        regression_flatten.append(regression_scale.reshape(-1, 2))

    logits = torch.cat(logits_flatten, dim=0)
    centerness = torch.cat(centerness_flatten, dim=0)
    pred_spans = torch.cat(regression_flatten, dim=0)
    return logits, centerness, pred_spans
