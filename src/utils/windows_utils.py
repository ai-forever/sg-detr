"""This module provides functions for converting between clip indices and time windows in videos."""

from typing import List


def convert_clip_ids_to_windows(clip_ids: List[int]) -> List[List[int]]:
    """Convert a list of clip indices to a list of window indices.

    This function takes a list of clip indices and groups them into windows. Each window is represented by a start
    and end index of clips. A gap between clip indices indicates the start of a new window.

    Args:
        clip_ids (List[int]): A list of clip indices, starting from 0.

    Returns:
        List[List[int]]: A list of windows, where each window is represented as a list containing two integers:
        the start and end indices of clips.

    Examples:
        >>> test_clip_ids = [56, 57, 58, 59, 60, 61, 62] + [64, ] + [67, 68, 69, 70, 71]
        >>> convert_clip_ids_to_windows(test_clip_ids)
        [[56, 62], [64, 64], [67, 71]]
    """
    windows = []
    window = [clip_ids[0], None]
    last_clip_id = clip_ids[0]
    for clip_id in clip_ids:
        if clip_id - last_clip_id > 1:  # find gap
            window[1] = last_clip_id
            windows.append(window)
            window = [clip_id, None]
        last_clip_id = clip_id
    window[1] = last_clip_id
    windows.append(window)
    return windows  # type: ignore


def convert_windows_to_clip_ids(windows: List[List[int]]) -> List[int]:
    """Convert a list of window indices to a list of clip indices.

    This function takes a list of windows, where each window is represented by a start and end index of clips,
    and converts it into a flat list of clip indices.

    Args:
        windows (List[List[int]]): List of windows, where each window is represented as a list containing two integers.

    Returns:
        List[int]: A flat list of clip indices.

    Examples:
        >>> test_windows = [[56, 62], [64, 64], [67, 71]]
        >>> convert_windows_to_clip_ids(test_windows)
        [56, 57, 58, 59, 60, 61, 62] + [64, ] + [67, 68, 69, 70, 71]
    """
    clip_ids = []
    for window in windows:  # noqa: WPS519
        clip_ids += list(range(window[0], window[1] + 1))
    return clip_ids


def convert_clip_window_to_seconds(window, clip_len=2):
    """Convert a window of clip indices to its corresponding time in seconds.

    Args:
        window (List[int]): A list containing two integers, the start and end indices of clips in the window.
        clip_len (int): The duration of each clip in seconds. Default is 2 seconds.

    Returns:
        List[int]: A list containing two integers, the start and end time in seconds of the window.

    Examples:
        >>> convert_clip_window_to_seconds([10, 19])
        [20, 40]
    """
    return [window[0] * clip_len, (window[1] + 1) * clip_len]
