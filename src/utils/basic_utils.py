"""Basic utils."""

import os
from typing import Any, List, Tuple

import torch


def guess_num_workers() -> int:
    """
    Guesses the number of workers based on the available CPU count and distributed training settings.

    Returns:
        int: The estimated number of workers. If distributed training is enabled, the number of workers is
            determined by dividing the CPU count by the world size. Otherwise, the number of CPUs is returned.
    """
    num_cpus = os.cpu_count()
    if num_cpus is None:
        return 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return num_cpus // torch.distributed.get_world_size()
    return num_cpus


def flat_list_of_lists(list_of_lists: List[List[Any]]) -> List[Any]:
    """Flatten a list of lists [[1,2], [3,4]] to [1,2,3,4].

    Args:
        list_of_lists (List[List[Any]]): List of lists to be flattened

    Returns:
        List[Any]: flattened list
    """
    return [item for sublist in list_of_lists for item in sublist]


def element_wise_list_equal(list_a: List[str], list_b: List[str]) -> List[bool]:
    """Element-wise compare two lists of strings.

    Args:
        list_a (List[str]): list of strings
        list_b (List[str]): list of strings

    Returns:
        List[bool]: list of bools, True if the two elements are equal, False otherwise
    """
    res = []
    for element_a, element_b in zip(list_a, list_b):
        if element_a == element_b:
            res.append(True)
        else:
            res.append(False)
    return res


def convert_to_seconds(hms_time: str) -> float:
    """Convert a time string in the format of HH:MM:SS to seconds.

    Args:
        hms_time (str): time string in the format of HH:MM:SS

    Returns:
        float: time in seconds
    """
    times = [float(time) for time in hms_time.split(":")]
    hours = times[0] * 3600  # noqa: WPS432
    minutes = times[1] * 60
    seconds = times[2]
    return hours + minutes + seconds


def merge_dicts(list_dicts: List[dict]) -> dict:
    """Merge a list of dictionaries into a single dictionary.

    Args:
        list_dicts (List[dict]): list of dictionaries to be merged

    Returns:
        dict: merged dictionary
    """
    merged_dict = list_dicts[0].copy()

    for dict_ in list_dicts[1:]:
        merged_dict.update(dict_)
    return merged_dict


def get_abspaths_by_ext(dir_path: str, ext: Tuple[str] = (".jpg",)):
    """Get absolute paths to files in dir_path with extensions specified by ext.

    Args:
        dir_path (str): The path to the directory containing the files.
        ext (Tuple[str): The file extension(s) to be included. Defaults to ".jpg".

    Returns:
        List[str]: List of absolute paths to the files.

    Note: this function does work recursively.
    """
    files = []

    for root, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith(ext):
                files.append(os.path.join(root, filename))
    return files
