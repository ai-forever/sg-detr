"""Misc functionality."""

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from src.model.utils.schemas import MomentEncoderOutput, SentenceEncoderOutput
from src.utils.basic_utils import element_wise_list_equal


def find_nth(vid: str, underline: str, num: int) -> int:
    """Find the nth occurrence of a substring in a string.

    Args:
        vid (str): video name
        underline (str): substring to find
        num (int): number of occurrences

    Returns:
        int: index of the nth occurrence
    """
    max_len = len(vid)
    start = vid.find(underline)
    while start >= 0 and num > 1:
        start = vid.find(underline, start + len(underline))
        num -= 1
    if start == -1:
        start = max_len
    return start


def get_origin_vid_name(vids: List[str]) -> List[str]:
    """Get the original video name from the video name with timestamp.

    Args:
        vids (List[str]): video name with timestamp

    Returns:
        List[str]: original video name
    """
    count = [vid.count("_") for vid in vids]
    position_to_cut = [find_nth(vid, "_", count[idx] - 1) for idx, vid in enumerate(vids)]  # noqa: WPS221
    return [vid[: position_to_cut[idx]] for idx, vid in enumerate(vids)]


def prepare_real_neg_mask(vids: List[str], device) -> Tensor:
    """Prepare real negative mask.

    Args:
        vids (List[str]): video names
        device (torch.device): device to put the mask on

    Returns:
        Tensor: real negative mask. Shape: [batch_size]
    """
    origin_vids = get_origin_vid_name(vids)
    neg_vid = origin_vids[1:] + origin_vids[:1]
    real_neg_mask = torch.Tensor(element_wise_list_equal(origin_vids, neg_vid)).to(torch.bool)
    return ~real_neg_mask.to(device)


def prepare_negative_tensors(
    src_vid: Tensor,
    src_vid_mask: Tensor,
    dummy_src_txt: Tensor,
    dummy_src_txt_mask: Tensor,
    pos: Tensor,
    real_neg_mask: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Prepare negative inputs.

    Args:
        src_vid (Tensor): video features, (batch_size, L_vid, D_vid)
        src_vid_mask (Tensor): video mask, (batch_size, L_vid)
        dummy_src_txt (Tensor): dummy text features, (batch_size, L_txt, D_txt)
        dummy_src_txt_mask (Tensor): dummy text mask, (batch_size, L_txt)
        pos (Tensor): positional encoding, (batch_size, L_txt, D_txt)
        real_neg_mask (Tensor): real negative mask, (batch_size)

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]:
            - src_neg (Tensor): negative video features, (batch_size, L_vid, D_vid)
            - mask_neg (Tensor): negative video mask, (batch_size, L_vid)
            - pos_neg (Tensor): negative positional encoding, (batch_size, L_txt, D_txt)
            - dummy_src_txt_mask_neg (Tensor): negative dummy text mask, (batch_size, L_txt)
    """
    src_txt_dummy_neg = torch.cat([dummy_src_txt[1:], dummy_src_txt[:1]])
    dummy_src_txt_mask_neg = torch.cat([dummy_src_txt_mask[1:], dummy_src_txt_mask[:1]])
    src_neg = torch.cat([src_vid, src_txt_dummy_neg], dim=1)
    mask_neg = torch.cat([src_vid_mask, dummy_src_txt_mask_neg], dim=1).bool()
    pos_neg = pos.clone()  # since it does not use actual content

    src_neg = src_neg[real_neg_mask]
    mask_neg = mask_neg[real_neg_mask]
    pos_neg = pos_neg[real_neg_mask]
    dummy_src_txt_mask_neg = dummy_src_txt_mask_neg[real_neg_mask]
    return src_neg, mask_neg, pos_neg, dummy_src_txt_mask_neg


def prepare_context_token(src_vid: Tensor, src_vid_mask: Tensor) -> Tensor:
    """Prepare video context token.

    Compute avg embedding for each video in a batch and use it as video context token.

    Args:
        src_vid (Tensor): video features, (batch_size, L_vid, D_vid)
        src_vid_mask (Tensor): video mask, (batch_size, L_vid)

    Returns:
        Tensor: Context token. Shape: [batch_size, 1, D_vid]
    """
    batch_size, _, d_model = src_vid.shape
    context_token = torch.zeros((batch_size, 1, d_model))
    context_token = context_token.to(src_vid.device)
    for idx, _ in enumerate(src_vid):
        mask = src_vid_mask.sum(1)[idx].long()
        mean_src_vid = src_vid[idx][:mask].mean(0)
        context_token[idx] = mean_src_vid.clone().detach()
    return context_token


def moment_txt_similarity(
    sents_schema: SentenceEncoderOutput,
    moments_schema: MomentEncoderOutput,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Compute similarity btwn moments visual emb and dummy-txt emb, non-moments visual tokens and dummy-txt emb.

    Args:
        sents_schema (SentenceEncoderOutput): Sentence encoder output schema
        moments_schema (MomentEncoderOutput): Moment encoder output schema

    Returns:
        Tuple[Tensor, Tensor]: mom2txt_sim, non_mom2txt_sim
    """
    if sents_schema.sent_dummy_memory is not None:
        txt_dummy = torch.cat([sents_schema.sent_dummy_memory, sents_schema.sent_words_memory], dim=0)  # type: ignore

        mom2txt_sim = torch.matmul(
            moments_schema.moment_memory.permute(1, 0, 2),  # type: ignore
            txt_dummy.permute(1, 2, 0),
        )
        non_mom2txt_sim = torch.matmul(
            moments_schema.non_moment_memory.permute(1, 0, 2),  # type: ignore
            txt_dummy.permute(1, 2, 0),
        )
    else:
        assert sents_schema.sent_words_memory is not None
        mom2txt_sim = torch.matmul(
            moments_schema.moment_memory.permute(1, 0, 2),  # type: ignore
            sents_schema.sent_words_memory.permute(1, 2, 0),
        )
        non_mom2txt_sim = None

    return mom2txt_sim, non_mom2txt_sim
