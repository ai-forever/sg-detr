"""This module provides utility functions for data processing."""

import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.media_dataset import FeatureExtractionDataset
from src.utils.schemas import Batch


def get_video_paths(video_dir: str) -> List[Dict[str, Any]]:
    """
    Retrieve the paths to video files based on the provided folder path.

    Args:
        video_dir (str): The path to a feed containing video.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, where each dictionary represents a video file and contains
                        the following keys:
                        - alias: The alias or filename of the video.
                        - video_path: A path to video substream.
    """
    video_paths = []
    for file in sorted(os.listdir(video_dir)):
        if file.endswith(".mp4"):
            alias = file
            video_path = os.path.join(video_dir, file)
            sample_dict = {"alias": alias, "video_path": video_path}
            video_paths.append(sample_dict)
    return video_paths


def get_data_entities(
    video_paths: List[Dict[str, str]],
    interval_duration: int,
    batch_size: int,
    sample_rate: int,
    num_workers: Union[int, None],
) -> Tuple[DataLoader, int, tqdm]:
    """
    Get the data entities for processing a list of video paths.

    Args:
        video_paths (List[Dict[str, str]]): List of video paths.
        interval_duration (int): duration of fragment in secs.
        batch_size (int): Batch size for the DataLoader.
        sample_rate (int): the number of audio samples taken per second
        num_workers (int): Number of worker processes for parallel data loading.

    Returns:
        Tuple[DataLoader, int, tqdm]: Tuple containing DataLoader, Number of worker, and tqdm progress bar.
    """
    dataset = FeatureExtractionDataset(video_paths, interval_duration, desired_sample_rate=sample_rate)
    num_workers = num_workers if num_workers is not None else 0
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    samples_len = len(video_paths)
    pbar = tqdm(total=samples_len)
    return dataloader, num_workers, pbar


def get_start_end_idxs(aliases: List[str]):
    """
    Get the start and end indices for unique aliases.

    This function takes a list of aliases and returns lists of unique aliases,
    their corresponding start indices, and end indices.

    Args:
        aliases (List[str]): A list of aliases.

    Returns:
        Tuple[List[str], np.ndarray, np.ndarray]: A tuple containing:
            - unique_paths (List[str]): List of unique aliases.
            - start_idxs (np.ndarray): Array of start indices for each unique alias.
            - end_idxs (np.ndarray): Array of end indices for each unique alias.

    Raises:
        AssertionError: If two films have the same alias, which could lead to problems.

    Example:
        >>> get_start_end_idxs(["a", "a", "b", "c", "c"])
        (["a", "b", "c"], [0, 2, 3], [2, 3, 5])
    """
    paths = np.array(aliases)
    unique_paths, start_idxs, counts = np.unique(paths, return_index=True, return_counts=True)
    initial_position = np.argsort(start_idxs)
    unique_paths, start_idxs, counts = (
        unique_paths[initial_position],
        start_idxs[initial_position],
        counts[initial_position],
    )

    end_idxs = start_idxs + counts

    # Check if any alias is not grouped together
    for upath in unique_paths:
        positions = [idx for idx, alias in enumerate(aliases) if alias == upath]
        if positions != list(range(positions[0], positions[0] + len(positions))):  # noqa: WPS221
            raise AssertionError(f"Alias '{upath}' is not grouped together! It means two films have the same alias")

    unique_paths = [str(path) for path in unique_paths]
    return unique_paths, start_idxs, end_idxs


def process_batch(batch: Dict[str, Any]) -> Dict[str, Batch]:
    """
    Process a batch of data.

    Args:
        batch (Dict[str, Any]): Input batch dictionary.

    Returns:
        Dict[str, Any]: Processed batch dictionary.
    """
    new_batch = {}
    aliases, start_idxs, end_idxs = get_start_end_idxs(batch["aliases"])
    for idx, path in enumerate(aliases):
        sub_batch = {
            "fps": batch["fps"][start_idxs[idx]].item(),
            "worker_id": batch["worker_id"][start_idxs[idx]].item(),
            "video_state": batch["video_state"][start_idxs[idx] : end_idxs[idx]],
            "video_chunk_len": batch["video_chunk_len"][start_idxs[idx] : end_idxs[idx]],
            "audio_chunk_len": batch["audio_chunk_len"][start_idxs[idx] : end_idxs[idx]],
            "is_last": batch["is_last"][start_idxs[idx] : end_idxs[idx]],
            "video_frames": batch["video"][start_idxs[idx] : end_idxs[idx]],
            "audio_samples": batch["audio"][start_idxs[idx] : end_idxs[idx], : batch["audio_size"][start_idxs[idx]]],
        }
        new_batch[path] = Batch(**sub_batch)
    return new_batch


def get_total_frames(is_last_list: List[bool], video_states: List[int]) -> Union[int, None]:
    """
    Try to extract the total frames.

    Args:
        is_last_list (List[bool]): Is the frames last frame of the sequence or not
        video_states (List[int]): The number of processed frames on each step

    Returns:
        Union[int, None]: The last index of the sequence if the sequence ends here
    """
    if not any(is_last_list):
        return None

    return video_states[-1]
