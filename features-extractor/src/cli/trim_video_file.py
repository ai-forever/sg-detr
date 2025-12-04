"""Trim the video file."""

import os
import random
import re
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from threading import Timer
from typing import List, Optional, Tuple

import click
from loguru import logger
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


def process_in_parallel(videos_to_process: List[Tuple[str, str, str, int]], max_workers: int = 10):
    """
    Process the videos in parallel.

    Args:
        videos_to_process (Tuple[str, str, str, int]): A tuple containing the following:
            - the path to the folder of the video
            - the video file name
            - the output folder path
            - target interval length.
        max_workers (int): number of workers.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(trim_video, videos_to_process),
                total=len(videos_to_process),
                desc="Processing videos",
            ),
        )


def check_dir_status(output_dir: str, max_num: int = 1000) -> None:
    """Count number of downloaded and not processed films in the target dir.

    Wait until len(output_dir) <= max_num

    Args:
        output_dir (str): path to output dir
        max_num (int): upper threshold.
    """
    while len(os.listdir(output_dir)) >= max_num:
        logger.info("I've done a good job and deserve a break.")
        time.sleep(10)


def extract_segment(
    duration: float,
    start_sec: float,
    end_sec: float,
    target_duration: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Extract a segment of specified duration from a video interval.

    Args:
        duration (float): Total duration of the video in seconds.
        start_sec (float): Start time of the interval from which to extract the segment.
        end_sec (float): End time of the interval from which to extract the segment.
        target_duration (float): Desired duration of the extracted segment.

    Returns:
        Tuple[Tuple[float, float], Tuple[float, float]]: A tuple containing:
            - A tuple representing the start and end times of the extracted segment.
            - A tuple representing the start and end times of the original interval within the extracted segment.

    Raises:
        ValueError: If target_duration is greater than the total duration of the video.
    """
    if target_duration > duration:
        raise ValueError("Target duration must be less than the total duration of the video")

    interval_length = end_sec - start_sec

    # Ensure the interval is at least as long as target_duration
    if interval_length < target_duration:
        # Calculate the range within which the start of the new segment can be
        min_start = max(0, end_sec - target_duration)
        max_start = min(start_sec, duration - target_duration)

        # Randomly choose a start point within this range
        new_start = round(random.uniform(min_start, max_start), 2)  # noqa: S311
        new_end = new_start + target_duration
    else:
        # If interval_length >= target_duration, adjust the interval to fit the target_duration
        new_start = start_sec
        new_end = start_sec + target_duration

    # Calculate the coordinates of the original interval in the new fragment
    original_interval_start_in_new_segment = start_sec - new_start  # noqa: WPS118
    original_interval_end_in_new_segment = end_sec - new_start  # noqa: WPS118

    return (new_start, new_end), (original_interval_start_in_new_segment, original_interval_end_in_new_segment)


def get_output_info(  # noqa: WPS234
    file: str,
    duration: float,
    target_duration: float,
) -> Tuple[str, Tuple[Optional[float], Optional[float]]]:
    """
    Retrieve information about the output video file and its segment.

    Args:
        file (str): Name of the input video file.
        duration (float): Total duration of the input video file in seconds.
        target_duration (float): Desired duration of the extracted segment.

    Returns:
        Tuple[str, Tuple[Optional[float], Optional[float]]]: A tuple containing:
            - Name of the output video file with the adjusted segment duration.
            - A tuple representing the start and end times of the extracted segment in the input video.

    Raises:
        ValueError: If the filename format is incorrect.
    """
    pattern = r"(.+?)_(\d+\.\d+)_(\d+\.\d+)\.mp4"
    match = re.match(pattern, file)
    if match:
        name_itself = match.group(1)
        start_of_interval_float = float(match.group(2))
        end_of_interval_float = float(match.group(3))
    else:
        raise ValueError("Filename format is incorrect")

    start_of_interval_float = max(0, start_of_interval_float)
    end_of_interval_float = min(end_of_interval_float, duration)

    if duration > target_duration:
        video_start_end, interval_start_end = extract_segment(
            duration,
            start_of_interval_float,
            end_of_interval_float,
            target_duration,
        )
    else:
        video_start_end = (None, None)  # type: ignore
        interval_start_end = (start_of_interval_float, end_of_interval_float)

    name_itself = f"{name_itself}_{interval_start_end[0]:.2f}_{interval_start_end[1]:.2f}.mp4"  # noqa: WPS221
    return name_itself, video_start_end


def exception_output(file: str, exp: Exception):
    """Log exception.

    Args:
        file (str): filename.
        exp (Exception): exception.
    """
    logger.error(f"failed to process {file}.")
    logger.error(f"Error: {exp}")
    with open("failed_trim.txt", "a+", encoding="utf8") as error_logs_file:
        error_logs_file.write(f"{file}\n")


# pylint: disable=too-many-locals
def trim_video(video_details: Tuple[str, str, str, int]) -> None:  # noqa: WPS210,WPS213,C901,WPS212,WPS231
    """
    Process a single video to change its length and keep its audio.

    Args:
        video_details: A tuple containing the following:
            - the path to the folder of the video
            - the video file name
            - the output folder path
            - target interval length.
    """
    folder_path, file, output_folder, target_duration = video_details
    check_dir_status(output_folder)
    video_path = os.path.join(folder_path, file)

    # Load the video file
    try:
        clip = VideoFileClip(video_path)
    except Exception as exp:  # pylint: disable=broad-exception-caught
        exception_output(file, exp)
        os.remove(video_path)
        return

    try:
        duration = clip.duration
    except Exception as exp:  # pylint: disable=broad-exception-caught
        exception_output(file, exp)
        clip.close()
        os.remove(video_path)
        return

    # get output info
    _, ext = os.path.splitext(file)
    file = file.replace(ext, ".mp4")
    file, (start_time, end_time) = get_output_info(file, duration, target_duration)
    output_file = os.path.join(output_folder, file)

    if start_time is None or end_time is None:
        shutil.move(video_path, output_file)
        return

    if end_time - start_time < 2:  # type: ignore
        logger.error("Corrupted sample detected.")
        logger.error(f"Video name: {file}. Start: {start_time}, End: {end_time}.")
        os.remove(video_path)
        return

    # Create temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_path = temp_file.name

    try:  # noqa: WPS229
        logger.debug(f"PID:{os.getpid()} has started the task")
        # Trim the video to the specified start and end times
        trimmed_clip = clip.subclip(start_time, end_time)
        # Save the trimmed video
        trimmed_clip.write_videofile(temp_path, codec="libx264", audio_codec="aac", logger=None)
        # Close the video files
        clip.close()
        trimmed_clip.close()
        shutil.move(temp_path, output_file)
        logger.debug(f"PID:{os.getpid()} has finished the task")
    except Exception as exp:  # pylint: disable=broad-exception-caught
        exception_output(file, exp)
    finally:
        os.remove(video_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)


class VideoHandler(FileSystemEventHandler):
    """A handler class to monitor a folder for new video files and process them at a specified interval."""

    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")

    def __init__(
        self,
        folder_path: str,
        output_folder: str,
        interval_duration: int,
        batch_interval: int = 60,
        num_workers: int = 10,
    ) -> None:
        """
        Initialize the VideoHandler.

        Args:
            folder_path (str): Path to the folder to watch for new video files.
            output_folder (str): Path to the folder to save processed video files.
            interval_duration (int): duration of the target interval.
            batch_interval (int): Interval in seconds for processing new files in batches. Defaults to 60.
            num_workers (int): num workers.
        """
        self.folder_path = folder_path
        self.output_folder = output_folder
        self.interval_duration = interval_duration
        self.num_workers = num_workers
        self.batch_interval = batch_interval
        self.new_files: List[Tuple[str, str, str, int]] = []
        # init timer
        self.timer: Optional[Timer] = None

    def on_start(self) -> None:
        """Add all the files in the input folder to the new files list."""
        # Prepare a list of video details for multiprocessing
        self.new_files = [
            (self.folder_path, file, self.output_folder, self.interval_duration)
            for file in os.listdir(self.folder_path)
            if any(file.endswith(ext) for ext in self.video_extensions)
        ]
        self.process_new_files()

    def on_created(self, event) -> None:
        """
        Call when a file or directory is created.

        Args:
            event: The event representing the file system change.
        """
        logger.info(f"new event detected: {event.src_path}")
        is_any_new_videos = any(event.src_path.endswith(ext) for ext in self.video_extensions)
        if not event.is_directory and is_any_new_videos:
            logger.debug(f"new event recorded: {event.src_path}")
            file = str(os.path.basename(event.src_path))
            self.new_files.append((self.folder_path, file, self.output_folder, self.interval_duration))

    def process_new_files(self) -> None:
        """Process the new files that have been detected."""
        self.stop_timer()
        if self.new_files:
            logger.info(f"Processing {len(self.new_files)} new files...")  # noqa: WPS237
            processing_files = deepcopy(self.new_files)
            process_in_parallel(processing_files, max_workers=self.num_workers)
            self.new_files = self.new_files[len(processing_files) :]
            logger.debug(f"Files left after processing call: {len(self.new_files)}")  # noqa: WPS237
        self.start_timer()

    def start_timer(self) -> None:
        """Start the timer for processing new files."""
        if self.timer:
            self.timer.cancel()
        self.timer = Timer(self.batch_interval, self.process_new_files)
        self.timer.start()

    def stop_timer(self) -> None:
        """Stop the timer for processing new files."""
        if self.timer:
            self.timer.cancel()


@click.command()
@click.option("--folder_path", type=str, default="data/videos_interim")
@click.option("--output_folder", type=str, default="data/videos_trimed")
@click.option("--num_workers", type=int, default=30)  # noqa: WPS432
@click.option("--interval_duration", type=int, default=150)  # noqa: WPS432
@click.option("--batch_interval", type=int, default=60)  # noqa: WPS432
def trim_video_script(  # noqa: WPS213, WPS216
    folder_path: str,
    output_folder: str,
    num_workers: int,
    interval_duration: int,
    batch_interval: int,
):
    """
    Trim of all video files in the specified folder based on interval_duration.

    Args:
        folder_path (str): The path to the folder containing the video files.
        output_folder (str): The path to output folder.
        num_workers (int): The number of workers to use.
        interval_duration (int): target duration of the video.
        batch_interval (int): Interval in seconds for processing new files in batches. Defaults to 60.
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(folder_path, exist_ok=True)

    # Set up a watchdog observer to watch for new files
    event_handler = VideoHandler(
        folder_path,
        output_folder,
        interval_duration=interval_duration,
        batch_interval=batch_interval,
        num_workers=num_workers,
    )
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)  # type: ignore
    observer.start()  # type: ignore
    event_handler.on_start()

    logger.info(f"Watching for new video files in {folder_path}...")

    try:
        while True:  # noqa: WPS457
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()  # type: ignore
    observer.join()
    event_handler.stop_timer()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    trim_video_script()
