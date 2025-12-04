"""This module contains functions to compute the HIT@1 score for the highlight detection task."""

from typing import Any, Dict

import torch
from torch import Tensor


def mk_saliency_scores(gt_data: Dict[str, Any], clip_length: int = 2) -> torch.Tensor:
    """Create a tensor of saliency scores for the full video based on the ground truth data.

    Args:
        gt_data (Dict[str, Any]): A dictionary containing ground truth data.
        clip_length (int): The length of each clip in seconds.

    Returns:
        torch.Tensor: A tensor containing the saliency scores for the full video.
    """
    num_clips = int(gt_data["duration"] / clip_length)
    saliency_scores_full_video = torch.zeros((num_clips, 3), dtype=torch.long)
    relevant_clip_ids = gt_data["relevant_clip_ids"]  # (#relevant_clip_ids)
    saliency_scores_relevant_clips = torch.tensor(gt_data["saliency_scores"])  # (#relevant_clip_ids, 3)
    saliency_scores_full_video[relevant_clip_ids] = saliency_scores_relevant_clips
    return saliency_scores_full_video  # (#clips_in_video, 3)  the scores are in range [0, 4]


def compute_hl_hit1(
    qid_preds: Dict[int, Tensor],
    qid_saliency: Dict[int, Tensor],
    device: torch.device,
) -> Tensor:  # noqa: WPS221
    """
    Compute the hit scores for query IDs based on their predictions and ground truth saliency scores.

    Args:
        qid_preds (Dict[int, Tensor]): A dictionary with query IDs as keys and predicted saliency as values.
        qid_saliency (Dict[int, Tensor]): A dict with query IDs as keys and thresholded gt saliency scores as values.
        device (torch.device): The device to use.

    Returns:
        Tensor: Hit score for each query.
    """
    # Extract the index of the clip with the maximum scored prediction for each query ID
    qid_max_scored_index = {qid: torch.argmax(pred_data) for qid, pred_data in qid_preds.items()}

    # Prepare an array to hold hit scores for each query ID
    hit_scores = torch.zeros((len(qid_preds), 3)).to(device)

    # Calculate hit scores for each query ID
    for idx, qid in enumerate(qid_preds.keys()):
        predicted_clip_index = qid_max_scored_index[qid]  # Shape: (#clips, 3)
        saliency = qid_saliency[qid]
        if predicted_clip_index < len(saliency):
            hit_scores[idx] = saliency[predicted_clip_index].type(torch.float32)

    # Aggregate scores by taking the max across the 3 separate annotations, then average the scores from all queries
    return torch.max(hit_scores, dim=1).values
