"""Correct video fps."""

import os
import random
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


def exception_output(file: str, exp: Exception):
    """Log an error message and write the file name to a log file.

    Args:
        file (str): file name.
        exp (Exception): exception.
    """
    logger.error(f"Failed to process {file}.")
    logger.error(f"Error: {exp}")
    with open("failed_fps.txt", "a+", encoding="utf8") as error_logs_file:
        error_logs_file.write(f"{file}\n")


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


class VideoHandler(FileSystemEventHandler):
    """A handler class to monitor a folder for new video files and process them at a specified interval."""

    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")

    def __init__(
        self,
        folder_path: str,
        output_folder: str,
        target_fps: int,
        batch_interval: int = 60,
        num_workers: int = 10,
    ) -> None:
        """
        Initialize the VideoHandler.

        Args:
            folder_path (str): Path to the folder to watch for new video files.
            output_folder (str): Path to the folder to save processed video files.
            target_fps (int): Target frames per second for processing videos.
            batch_interval (int): Interval in seconds for processing new files in batches. Defaults to 60.
            num_workers (int): num workers.
        """
        self.folder_path = folder_path
        self.output_folder = output_folder
        self.target_fps = target_fps
        self.num_workers = num_workers
        self.batch_interval = batch_interval
        self.new_files: List[Tuple[str, str, int, str]] = []
        # init timer
        self.timer: Optional[Timer] = None

    def on_start(self) -> None:
        """Add all the files in the input folder to the new files list and processed them."""
        # Prepare a list of video details for multiprocessing
        self.new_files = [
            (self.folder_path, file, self.target_fps, self.output_folder)
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
            file = os.path.basename(event.src_path)
            self.new_files.append((self.folder_path, file, self.target_fps, self.output_folder))

    def process_new_files(self) -> None:
        """Process the new files that have been detected."""
        self.stop_timer()
        if self.new_files:
            logger.info(f"Processing {len(self.new_files)} new files...")  # noqa: WPS237
            processing_files = deepcopy(self.new_files)
            process_in_parallel(processing_files, max_workers=self.num_workers)
            self.new_files = self.new_files[len(processing_files) :]
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


def process_video(video_details: Tuple[str, str, int, str]) -> None:  # noqa: WPS210,WPS213
    """
    Process a single video to change its framerate to 30 fps and keep its audio.

    Args:
        video_details: A tuple containing the following:
            - the path to the folder of the video
            - the video file name
            - target frame rate
            - the output folder path.
    """
    folder_path, file, fps, output_folder = video_details
    video_path = os.path.join(folder_path, file)

    _, ext = os.path.splitext(file)
    file = file.replace(ext, ".mp4")

    output_folder = f"{output_folder}/{random.randint(0, 7)}"  # noqa: WPS432,WPS237,S311
    os.makedirs(output_folder, exist_ok=True)
    check_dir_status(output_folder)
    new_file_path = os.path.join(output_folder, file)

    # Create temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_path = temp_file.name

    try:  # noqa: WPS229
        # Load the video file
        clip = VideoFileClip(video_path)
        # Change the framerate to 30 fps and keep the audio
        new_clip = clip.set_fps(fps).set_audio(clip.audio)
        # Save the new video
        new_clip.write_videofile(temp_path, codec="libx264", audio_codec="aac", fps=fps, logger=None)
        # Close the video file
        clip.close()
        shutil.move(temp_path, new_file_path)
    except Exception as exp:  # pylint: disable=broad-exception-caught
        exception_output(video_path, exp)
    finally:
        os.remove(video_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)


def process_in_parallel(videos_to_process: List[Tuple[str, str, int, str]], max_workers: int = 10):
    """
    Process the videos in parallel.

    Args:
        videos_to_process (Tuple[str, str, int, str]): A tuple containing the following:
            - the path to the folder of the video
            - the video file name
            - target frame rate
            - the output folder path.
        max_workers (int): number of workers.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(process_video, videos_to_process),
                total=len(videos_to_process),
                desc="Processing videos",
            ),
        )


@click.command()
@click.option("--folder_path", type=str, default="data/videos_trimed")
@click.option("--output_folder", type=str, default="data/videos_30fps")
@click.option("--target_fps", type=int, default=30)  # noqa: WPS432
@click.option("--num_workers", type=int, default=30)  # noqa: WPS432
@click.option("--batch_interval", type=int, default=60)  # noqa: WPS432
def change_framerate(  # noqa: WPS213, WPS216
    folder_path: str,
    output_folder: str,
    target_fps: int,
    num_workers: int,
    batch_interval: int,
):
    """
    Change the framerate of all video files in the specified folder to 30 fps.

    Args:
        folder_path (str): The path to the folder containing the video files.
        output_folder (str): The path to output folder.
        target_fps (int): Target fps.
        num_workers (int): The number of workers to use.
        batch_interval (int): Interval in seconds for processing new files in batches. Defaults to 60.
    """
    os.makedirs(folder_path, exist_ok=True)

    # Set up a watchdog observer to watch for new files
    event_handler = VideoHandler(
        folder_path,
        output_folder,
        target_fps=target_fps,
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
    change_framerate()
