"""Dataset's utility functions."""

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


def initialize_qv_score_array(
    relevant_clip_ids: List[int],
    number_of_clips: int,
    agg_scores: np.ndarray,
) -> np.ndarray:
    """
    Initialize a score array with zeros and assign aggregated scores to corresponding clip IDs.

    Args:
        relevant_clip_ids: A list of integers representing the IDs of relevant clips.
        number_of_clips: The total number of clips as an integer.
        agg_scores: A numpy array containing aggregated scores for the relevant clips.

    Returns:
        np.ndarray: Array representing the scores where indices correspond to clip IDs and values to their scores.
    """
    max_relevant_id = max(relevant_clip_ids)
    max_clip_id = max(max_relevant_id, number_of_clips - 1)
    score_array = np.zeros(max_clip_id + 1)
    score_array[relevant_clip_ids] = agg_scores
    return score_array


# pylint: disable=unnecessary-lambda-assignment
def get_qv_hard_indices(
    relevant_clip_ids: List[int],
    sort_indices: np.ndarray,
    max_n: int,
    number_of_clips: int,
) -> Tuple[List[int], List[int]]:
    """
    Determine indices for hard positive and negative samples based on sorted scores.

    Args:
        relevant_clip_ids (List[int]): List of the IDs of relevant clips.
        sort_indices (np.ndarray): A numpy array of indices that would sort the scores array.
        max_n (int): The maximum number of positive and negative samples to be selected.
        number_of_clips (int): The total number of clips as an integer.

    Returns:
        Tuple[List[int], List[int]]:
        - First list contains indices of hard positive samples.
        - Second list contains indices of hard negative samples.
    """
    get_min_index = lambda idx: min(relevant_clip_ids[idx], number_of_clips - 1)  # noqa
    hard_pos_clip_indices = [get_min_index(idx) for idx in sort_indices[-max_n:]]  # type: ignore
    hard_neg_clip_indices = [get_min_index(idx) for idx in sort_indices[:max_n]]  # type: ignore
    return hard_pos_clip_indices, hard_neg_clip_indices


def sample_qv_easy_negatives(
    easy_neg_pool: List[int],
    relevant_clip_ids: List[int],
    max_n: int,
    hard_pos_indices: List[int],
    hard_neg_indices: List[int],
) -> Tuple[List[int], List[int]]:
    """
    Sample easy negative (and positive if necessary) clips from the available pool or use hard samples as a fallback.

    Args:
        easy_neg_pool (List[int]): A list of clip IDs considered as potential easy negatives.
        relevant_clip_ids (List[int]): A list of integers representing the IDs of relevant clips.
        max_n (int): The maximum number of easy negatives to be sampled.
        hard_pos_indices (List[int]): A list of indices for hard positive samples.
        hard_neg_indices (List[int]): A list of indices for hard negative samples.

    Returns:
        Tuple[List[int], List[int]]:
        - First list contains indices of easy positive samples or hard positives if easy positives are insufficient.
        - Second list contains indices of easy negative samples or hard negatives if easy negatives are insufficient.
    """
    if len(easy_neg_pool) >= max_n:
        easy_pos_clip_indices = random.sample(relevant_clip_ids, k=max_n)
        easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
    else:
        easy_pos_clip_indices = hard_pos_indices
        easy_neg_clip_indices = hard_neg_indices
    return easy_pos_clip_indices, easy_neg_clip_indices


def get_qv_pos_neg_indices(
    relevant_clip_ids: List[int],
    number_of_clips: int,
    hard_pos_indices: List[int],
    hard_neg_indices: List[int],
    add_easy_negative: bool,
    max_n: int,
) -> Tuple[List[int], List[int]]:
    """
    Combine hard and easy sample indices to create final lists of positive and negative clip indices.

    Args:
        relevant_clip_ids (List[int]): A list of integers representing the IDs of relevant clips.
        number_of_clips: int: The total number of clips as an integer.
        hard_pos_indices (List[int]): A list of indices for hard positive samples.
        hard_neg_indices (List[int]): A list of indices for hard negative samples.
        add_easy_negative (bool): A boolean indicating whether to add easy negatives.
        max_n (int): The maximum number of samples (both positive and negative) to be selected.

    Returns:
        A tuple containing two lists:
        - First list contains combined indices of positive samples.
        - Second list contains combined indices of negative samples.
    """
    pos_clip_indices, neg_clip_indices = hard_pos_indices, hard_neg_indices

    if add_easy_negative:
        easy_neg_pool = list(set(range(number_of_clips)) - set(relevant_clip_ids))
        easy_pos_clip_indices, easy_neg_clip_indices = sample_qv_easy_negatives(
            easy_neg_pool,
            relevant_clip_ids,
            max_n,
            hard_pos_indices,
            hard_neg_indices,
        )
        pos_clip_indices += easy_pos_clip_indices
        neg_clip_indices += easy_neg_clip_indices

    return pos_clip_indices, neg_clip_indices


def get_irrelevant_windows(
    relevant_windows: List[Tuple[int, int]],
    duration: int,
) -> List[Tuple[int, int]]:  # noqa: WPS221
    """
    Identify the gaps between consecutive spans provided in a numpy array.

    This function sorts the spans by their starting points and iterates through them to find
    non-overlapping gaps between the end of one span and the start of the next.

    Args:
        relevant_windows (List[Tuple[int, int]]): list of starts ans ends coords of the spans
        duration (int): duration of the clip.

    Returns:
        List[Tuple[int, int]]: A 2D array containing the found gaps. Each row represents a gap,
                    where the first column is the start and the second column is the end of the gap.
    """
    # Sort spans by the starting point of each segment
    sorted_spans = sorted(relevant_windows)

    # Initialize a list to store the gaps found
    gaps = []
    if sorted_spans[0][0] > 0:
        gaps.append((0, sorted_spans[0][0]))
    # Iterate through the sorted spans to find gaps
    for idx in range(1, len(sorted_spans)):  # noqa: WPS518
        end_of_previous = sorted_spans[idx - 1][1]
        start_of_current = sorted_spans[idx][0]
        if end_of_previous < start_of_current:
            gaps.append((end_of_previous, start_of_current))
    if sorted_spans[-1][1] < duration:
        gaps.append((sorted_spans[-1][1], duration))

    return gaps


def add_padding(emb: Optional[Tensor], max_shrink_rate: int) -> Optional[Tensor]:
    """Add padding.

    Args:
        emb (Optional[Tensor]): embedding.
        max_shrink_rate (int): max stride.

    Returns:
        Optional[Tensor]: padded tensor.
    """
    if emb is not None and emb.size(0) % max_shrink_rate != 0:
        padding_size = max_shrink_rate - emb.size(0) % max_shrink_rate
        padding = torch.zeros(padding_size, emb.size(1))
        emb = torch.cat([emb, padding], dim=0)
    return emb
