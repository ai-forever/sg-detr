"""This module creates annotation file for pretrain dataset."""

import math
import os
import random
import re
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from src.utils.utils import load_jsonl, save_jsonl

# Constants for grading saliency scores
MIN_STD: float = 0.005
UP_GRADE_MULTIPLIER = 1.5  # noqa: WPS114
DOWN_GRADE_MULTIPLIER = 1.5  # noqa: WPS114
MIN_GRADE_MULTIPLIER = 3.0  # noqa: WPS114

# video duration
MAX_VIDEO_DURATION: int = 150
MAX_SAMPLES: int = 150000


def col_round(float_num: float) -> int:
    """Round a floating-point number to the nearest integer.

    This function rounds a floating-point number to the nearest integer.
    If the fractional part of the number is less than 0.5, it rounds down
    (using `math.floor`). Otherwise, it rounds up (using `math.ceil`).

    Args:
        float_num (float): The floating-point number to round.

    Returns:
        int: The rounded integer.
    """
    frac = float_num - math.floor(float_num)
    if frac < 0.5:  # noqa: WPS459
        return math.floor(float_num)
    return math.ceil(float_num)


def get_info(file: str) -> Tuple[str, float, float]:
    """Extract information from a filename following a specific pattern.

    This function extracts the name, start interval, and end interval from a filename
    that follows the pattern "<name>_<start_interval>_<end_interval>.pt".

    Args:
        file (str): The filename to extract information from.

    Returns:
        Tuple[str, float, float]: A tuple containing the name, start interval,
        and end interval.

    Raises:
        ValueError: If the filename format is incorrect.
    """
    pattern = r"(.+?)_(\d+\.\d+)_(\d+\.\d+)\.pt"
    match = re.match(pattern, file)
    if match:
        name_itself = match.group(1)
        start_of_interval_float = float(match.group(2))
        end_of_interval_float = float(match.group(3))
    else:
        raise ValueError("Filename format is incorrect")

    return name_itself, start_of_interval_float, end_of_interval_float


def get_embeddings(video_embs_path: str, txt_embs_path: str, sample_info: Dict[str, str]) -> Tuple[Tensor, Tensor]:
    """Fetch text and video embeddings for a given sample and time interval.

    This function retrieves text and video embeddings from specified paths based on the sample information
    and time interval provided.

    Args:
        video_embs_path (str): path to video embs.
        txt_embs_path (str): path to txt embs.
        sample_info (Dict[str, str]): A dictionary containing sample information with keys 'qid' and 'vid'.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the text embedding and video embedding.

    """
    path_to_txt_features = os.path.join(txt_embs_path, f"{sample_info['qid']}.npz")
    path_to_vid_features = os.path.join(
        video_embs_path,
        f"{sample_info['vid']}.pt",  # noqa: WPS221
    )
    txt_emb = torch.from_numpy(np.load(path_to_txt_features)["features"][0])
    vid_emb = torch.load(path_to_vid_features)
    return txt_emb, vid_emb


def prepare_saliency_scores(  # noqa: WPS210
    scores: torch.Tensor,
    mean_score: float,
    max_score: float,
    std_score: float,
) -> List[List[int]]:
    """Prepare saliency scores based on the input scores, mean score, max score, and standard deviation.

    This function assigns saliency grades to the scores based on their range. The grades are as follows:
    - Grade 4: Scores between mean_score + 1.5 * std_score and max_score.
    - Grade 3: Scores between mean_score and mean_score + 1.5 * std_score.
    - Grade 2: Scores between mean_score - 1.5 * std_score and mean_score.
    - Grade 1: Scores between mean_score - 3 * std_score and mean_score - 1.5 * std_score.

    Args:
        scores (torch.Tensor): The tensor of scores to be graded.
        mean_score (float): The mean score.
        max_score (float): The maximum score.
        std_score (float): The standard deviation of the scores.

    Returns:
        List[List[int]]: A list of saliency scores with grades.
    """
    # Grade 4
    up_4 = max_score
    bottom_4 = mean_score + UP_GRADE_MULTIPLIER * std_score
    grade_4_idx = torch.logical_and(scores > bottom_4, scores <= up_4)

    # Grade 3
    up_3 = mean_score + UP_GRADE_MULTIPLIER * std_score
    bottom_3 = mean_score
    grade_3_idx = torch.logical_and(scores > bottom_3, scores <= up_3)

    # Grade 2
    up_2 = mean_score
    bottom_2 = mean_score - DOWN_GRADE_MULTIPLIER * std_score
    grade_2_idx = torch.logical_and(scores > bottom_2, scores <= up_2)

    # Grade 1
    up_1 = mean_score - DOWN_GRADE_MULTIPLIER * std_score
    bottom_1 = mean_score - MIN_GRADE_MULTIPLIER * std_score
    grade_1_idx = torch.logical_and(scores >= bottom_1, scores <= up_1)

    saliency_scores = torch.zeros_like(scores, dtype=torch.int)
    saliency_scores[grade_4_idx] = 4
    saliency_scores[grade_3_idx] = 3
    saliency_scores[grade_2_idx] = 2
    saliency_scores[grade_1_idx] = 1
    saliency_scores_duplicated = saliency_scores[saliency_scores > 0].tolist()
    return [[score, score, score] for score in saliency_scores_duplicated]


def prepare_relevant_clip_ids(
    time_points: List[int],
    scores: torch.Tensor,
    mean_score: float,
    std_score: float,
) -> Tuple[List[int], List[int]]:
    """Prepare relevant clip IDs based on time points and scores.

    This function identifies relevant clips by finding scores above a certain threshold,
    determined by the mean score minus a multiple of the standard deviation.

    Args:
        time_points (List[int]): A list of time points corresponding to the scores.
        scores (torch.Tensor): A 1d tensor of scores.
        mean_score (float): The mean score.
        std_score (float): The standard deviation of the scores.

    Returns:
        Tuple[List[int], List[int]]: A tuple containing two lists:
            - List of indices of relevant scores.
            - List of corresponding time points of the relevant scores.
    """
    # Determine the minimum relevant score
    min_rel_score = mean_score - MIN_GRADE_MULTIPLIER * std_score

    # Select points with scores above the minimum relevant score
    high_rel_points = [idx for idx, score in enumerate(scores) if score >= min_rel_score]
    high_rel_times = [time_points[idx] for idx in high_rel_points]

    return high_rel_points, high_rel_times


def prepare_relevant_windows(high_rel_times: List[int]) -> List[List[int]]:
    """Prepare relevant time windows based on high relevance times.

    This function generates time windows where each window starts from a high relevance
    time point and extends for 2 units. If there is a gap of more than 2 units between
    consecutive high relevance time points, a new window is started.

    Args:
        high_rel_times (List[int]): A list of high relevance time points.

    Returns:
        List[List[int]]: A list of time windows, where each window is represented as a list with start and end points.
    """
    relevant_windows = []
    previous_value: Optional[int] = None
    start_point = 0

    for timing in high_rel_times:
        if previous_value is not None:
            if timing - previous_value > 2:
                relevant_windows.append([start_point, previous_value + 2])
                start_point = timing
        else:
            start_point = timing
        previous_value = timing

    if previous_value is not None:
        relevant_windows.append([start_point, previous_value + 2])

    return relevant_windows


def compute_sims_and_new_intervals(  # noqa: WPS210
    video_embs_path: str,
    txt_embs_path: str,
    sample_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute similarity scores and updates intervals in the sample information.

    This function computes the cosine similarity scores between text and video embeddings,
    determines relevant clip IDs, and updates the relevant windows and saliency scores
    in the sample information dictionary.

    Args:
        video_embs_path (str): path to video embs.
        txt_embs_path (str): path to txt embs.
        sample_info (Dict[str, Any]): A dictionary containing sample information.

    Returns:
        Dict[str, Any]: updated sample info
    """
    # Initial interval
    start, end = sample_info["relevant_windows_old"][0]
    if end - start < 2:
        return sample_info
    start_idx = col_round(start / 2)
    end_idx = col_round(end / 2)

    # Get embeddings
    txt_emb, vid_emb = get_embeddings(video_embs_path, txt_embs_path, sample_info)

    # Compute similarities
    scores = torch.nn.functional.cosine_similarity(vid_emb, txt_emb)  # pylint: disable=not-callable
    time_points: List[int] = np.arange(0, len(scores) * 2, 2).tolist()

    # Initial interval scores
    window_scores = scores[start_idx:end_idx]
    mean_score = float(torch.mean(window_scores))
    std_score = float(torch.std(window_scores)) if len(window_scores) >= 2 else MIN_STD
    std_score = max(std_score, MIN_STD)
    max_score = float(torch.max(scores))

    # Prepare relevant clip IDs
    relevant_clip_ids, high_rel_times = prepare_relevant_clip_ids(time_points, scores, mean_score, std_score)

    sample_info["duration"] = min(vid_emb.shape[0] * 2, MAX_VIDEO_DURATION)
    sample_info["relevant_clip_ids"] = relevant_clip_ids
    sample_info["relevant_windows"] = prepare_relevant_windows(high_rel_times)
    sample_info["saliency_scores"] = prepare_saliency_scores(scores, mean_score, max_score, std_score)
    return sample_info


def convert_chunk_of_df_to_dicts(
    anno: pd.DataFrame,
    vid_infos: List[Tuple[str, float, float]],
) -> List[Dict[str, Any]]:  # noqa: WPS221
    """Convert pandas df to list of dicts contains videos info.

    Args:
        anno (pd.DataFrame): annotation file.
        vid_infos (List[Tuple[int, float, float]]): name, start of interval, end of interval

    Returns:
        List[Dict[str, Any]]: info for each sample in dict format
    """
    samples_info: List[Dict[str, Any]] = []
    for vid_info in vid_infos:
        stem, start, end = vid_info
        anno_sample = anno.loc[anno["YoutubeID"] == stem]
        caption = anno_sample["Caption"].values[0]
        qid = f"{anno_sample.index.values[0]}".rjust(7, "0")  # noqa: WPS237
        dict_sample = {
            "qid": qid,
            "vid": f"{stem}_{start:.2f}_{end:.2f}",
            "duration": None,
            "query": caption,
            "relevant_clip_ids": [],
            "saliency_scores": [],
            "relevant_windows_old": [[start, end]],
        }
        samples_info.append(dict_sample)
    return samples_info


def convert_df_to_dicts(
    anno: pd.DataFrame,
    vid_infos: List[Tuple[str, float, float]],
) -> List[Dict[str, Any]]:
    """Convert pandas df to list of dicts contains videos info.

    Args:
        anno (pd.DataFrame): annotation file.
        vid_infos (List[Tuple[int, float, float]]): name, start of interval, end of interval

    Returns:
        List[Dict[str, Any]]: info for each sample in dict format
    """
    logger.info("Start to convert df to list of dicts...")
    # Determine the number of chunks based on the number of CPU cores
    num_chunks = min(cpu_count(), len(vid_infos))

    # Split vid_infos into chunks
    chunks = [vid_infos[idx::num_chunks] for idx in range(num_chunks)]

    # Create a multiprocessing Pool
    with Pool(processes=num_chunks) as pool:
        # Process each chunk in parallel
        results = pool.starmap(convert_chunk_of_df_to_dicts, [(anno, chunk) for chunk in chunks])

    # Combine results from all chunks
    return [item for sublist in results for item in sublist]


def process_sample(
    sample_info: Dict[str, Any],
    video_embs_path: str,
    txt_embs_path: str,
) -> Optional[Dict[str, Any]]:  # noqa: WPS221
    """
    Process a single sample by computing similarities and new intervals.

    Args:
        sample_info (Dict[str, Any]): The information of the sample to be processed.
        video_embs_path (str): The file path to the video embeddings.
        txt_embs_path (str): The file path to the text embeddings.

    Returns:
        Optional[Dict[str, Any]]: The updated sample information if the sample is valid, otherwise None.
    """
    updated_sample_info = compute_sims_and_new_intervals(video_embs_path, txt_embs_path, sample_info)
    if updated_sample_info["relevant_clip_ids"]:
        return updated_sample_info
    logger.error("Corrupted sample detected.")
    logger.error(f"Video name: {updated_sample_info['vid']}. Duration: {updated_sample_info['duration']}")
    return None


@click.command()
@click.option("--path_to_anno", type=str, default="data/annotations/pretrain_annotation_154k.jsonl")
@click.option("--output_dir", type=str, default="data/annotations")
@click.option("--n_parts", type=int, default=5)
def split_annotation(path_to_anno: str, output_dir: str, n_parts: int) -> None:  # noqa: WPS234
    """Sample parts of the annotation file.

    Args:
        path_to_anno (str): annotation file
        output_dir (str): output dir.
        n_parts (int): number of parts
    """
    # load annotation
    annotation = load_jsonl(path_to_anno)
    annotation = annotation[:MAX_SAMPLES]
    assert 100 % n_parts == 0
    step_percent = 1 / n_parts

    for idx in tqdm(range(n_parts)):
        data_portion = int(len(annotation) * step_percent * (idx + 1))
        portioned_annotation = random.sample(annotation, data_portion)
        output_name = f"pretrain_annotation_{len(portioned_annotation) // 1000}k.jsonl"  # noqa: WPS237
        output_path = os.path.join(output_dir, output_name)
        save_jsonl(portioned_annotation, output_path)


@click.command()
@click.option("--annotation_df_path", type=str, default="data/positive_captions_s3path.csv")
@click.option("--video_embs_path", type=str, default="data/custom_features_v2/video/embeddings")
@click.option("--txt_embs_path", type=str, default="data/custom_features/text_embeddings")
@click.option("--output_dir", type=str, default="data/annotations")
def main(  # noqa: WPS216,WPS210
    annotation_df_path: str,
    video_embs_path: str,
    txt_embs_path: str,
    output_dir: str,
):
    """Prepare annotation file for pretrrain task.

    Args:
        annotation_df_path (str): path to annotation df
        video_embs_path (str): path to video embs.
        txt_embs_path (str): path to txt embs.
        output_dir (str): path to save new annotation.
    """
    anno = pd.read_csv(annotation_df_path, index_col="Unnamed: 0")
    vid_infos = [(get_info(path)) for path in os.listdir(video_embs_path)]
    dict_form_anno = convert_df_to_dicts(anno, vid_infos)

    partial_process_sample = partial(process_sample, video_embs_path=video_embs_path, txt_embs_path=txt_embs_path)
    # Using chunksize to reduce overhead
    chunksize = len(dict_form_anno) // (cpu_count() * 4)
    total = len(dict_form_anno)
    logger.info(f"Chunk size: {chunksize}")
    with Pool(cpu_count()) as pool:
        new_anno = list(
            tqdm(pool.imap(partial_process_sample, dict_form_anno, chunksize=chunksize), total=total),
        )
    new_anno = [sample for sample in new_anno if sample is not None]

    logger.info(f"Processed {len(new_anno)} / {len(dict_form_anno)}")  # noqa: WPS237
    round_len = int(len(new_anno) / 1000)
    output_name = f"pretrain_annotation_{round_len}k.jsonl"
    output_path = os.path.join(output_dir, output_name)
    save_jsonl(new_anno, output_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
