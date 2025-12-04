"""Model utilities."""

from typing import List

import torch
from torch import Tensor, nn

MIN_ANCHOR_LENGTH: float = 0.0125
MAX_ANCHOR_LENGTH: float = 0.9875


def inverse_sigmoid(point: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    Inverse sigmoid function.

    Args:
        point (torch.Tensor): input tensor.
        eps (float): small value to avoid numerical instability.

    Returns:
        torch.Tensor: inverse sigmoid of the input tensor.
    """
    point = point.clamp(min=0, max=1)
    point1 = point.clamp(min=eps)
    point2 = (1 - point).clamp(min=eps)
    return torch.log(point1 / point2)


def init_weights(module: nn.Module) -> None:
    """
    Initialize weights for a module.

    Args:
        module (nn.Module): module to initialize weights for.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0, std=0.02)  # noqa: WPS432
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def get_reference_points(
    spatial_shapes: List[List[int]],
    valid_ratios: Tensor,
    device: torch.device,
) -> Tensor:  # noqa: WPS602
    """Get reference points for deformable attention.

    Args:
        spatial_shapes (List[List[int]]): List of spatial shapes (height, width) for each level.
        valid_ratios (Tensor): Portion of real tokens.
        device (torch.device): Device on which to perform the computation.

    Returns:
        Tensor: Reference points for deformable attention.
    """
    reference_points_list = []
    for lvl, (height, width) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
            torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
        )
        x_scaler = valid_ratios[:, None, lvl, 1] * width
        y_scaler = valid_ratios[:, None, lvl, 1] * height
        ref_x = ref_x.reshape(-1)[None] / x_scaler
        ref_y = ref_y.reshape(-1)[None] / y_scaler
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    return reference_points[:, :, None] * valid_ratios[:, None]


def get_valid_ratio(mask: Tensor) -> Tensor:
    """
    Calculate the valid ratio of a mask tensor along height and width.

    Args:
        mask (Tensor): A tensor of shape (batch_size, height, width) representing the mask.

    Returns:
        Tensor: A tensor of shape (batch_size, 2) where the first column contains the
                valid width ratios and the second column contains the valid height ratios.
    """
    _, height, width = mask.shape
    valid_h = torch.sum(mask[:, :, 0], 1)
    valid_w = torch.sum(mask[:, 0, :], 1)
    valid_ratio_h = valid_h.float() / height
    valid_ratio_w = valid_w.float() / width
    return torch.stack([valid_ratio_w, valid_ratio_h], -1)


def gen_encoder_output_proposals(
    fpn_features: List[Tensor],
    memory_padding_masks: List[Tensor],
    default_widths: List[float],
):
    """
    Generate encoder output proposals.

    Args:
        fpn_features (List[Tensor]): Fpn features (List[batch_size, width, d_model]).
        memory_padding_masks(List[Tensor]): The padding masks tensor with shape (batch_size, width).
        default_widths (List[float]): The default width of the span for proposals.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: A tuple containing:
            - output_memory (Tensor): The output memory tensor with shape (batch_size, sum(hw), d_model).
            - output_proposals (Tensor): The output proposals tensor with shape (batch_size, sum(hw), 4).
            - mask List[int]: mask of each scale.
    """
    spatial_shapes = [seq.size(1) for seq in fpn_features]
    proposals = []
    for lvl, (memory, lvl_mask, lvl_width) in enumerate(zip(fpn_features, memory_padding_masks, spatial_shapes)):
        memory = memory.transpose(0, 1)
        _, batch_size, _ = memory.shape
        scale = torch.sum(lvl_mask, 1).unsqueeze(-1)

        # centers
        grid = torch.linspace(0, lvl_width - 1, lvl_width, dtype=torch.float32, device=memory.device)  # noqa: WPS221
        grid = (grid.unsqueeze(0).expand(batch_size, -1) + 0.5) / scale
        grid = grid.unsqueeze(-1)

        # widths
        width = torch.ones_like(grid) * default_widths[lvl]

        # concat them
        proposal = torch.cat((grid, width), -1)
        proposals.append(proposal)

    # prepare mask
    memory_padding_mask = torch.cat(memory_padding_masks, dim=1).unsqueeze(-1)
    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = (output_proposals > MIN_ANCHOR_LENGTH) & (output_proposals < MAX_ANCHOR_LENGTH)
    output_proposals_valid = output_proposals_valid.all(-1, keepdim=True)
    mask = torch.logical_and(memory_padding_mask, output_proposals_valid)

    # apply mask
    output_proposals = inverse_sigmoid(output_proposals)
    output_proposals = output_proposals.masked_fill(~mask, float("inf"))
    output_memory = torch.cat(fpn_features, 1)
    output_memory = output_memory.masked_fill(~mask, 0)
    return output_memory, output_proposals, mask
