"""Entrypoint for feature extraction."""

import os
import time
from copy import deepcopy
from pathlib import Path
from threading import Timer
from typing import Any, Dict, List, Optional

import click
import torch
from loguru import logger
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.inference import AudioExtractor, FeaturesSaver, VideoExtractor
from src.inference.tools import get_data_entities, process_batch


class VideoHandler(FileSystemEventHandler):  # noqa: WPS230
    """A handler class to monitor a folder for new video files and process them at a specified interval."""

    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")

    def __init__(  # noqa: WPS211
        self,
        input_folder: str,
        output_folder: str,
        raw_checkpoint: str,
        audio_checkpoint: str,
        video_checkpoint: str,
        sample_rate: int,
        device: torch.device,
        interval_duration: int,
        batch_interval: int = 60,
        num_workers: int = 10,
        batch_size: int = 64,
        load_to_s3: bool = True,
    ) -> None:
        """
        Initialize the VideoHandler.

        Args:
            input_folder (str): Path to the input folder.
            output_folder (str): Path to the folder to save processed video files.
            raw_checkpoint (str): Path to raw audio checkpoint.
            audio_checkpoint (str): Path to the checkpoint file for the audio extractor.
            video_checkpoint (str): Path to the checkpoint file for the video extractor.
            sample_rate (int): the number of audio samples taken per second
            device (torch.device): Device to use for processing.
            interval_duration (int): duration of video fragment in secs
            batch_interval (int): Interval in seconds for processing new files in batches. Defaults to 60.
            num_workers (int): num workers.
            batch_size (int): Batch size for processing.
            load_to_s3 (bool): wether to load to s3.
        """
        # init params
        os.makedirs(output_folder, exist_ok=True)
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.num_workers = num_workers
        self.interval_duration = interval_duration
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.sample_rate = sample_rate
        self.new_files: List[Dict[str, Any]] = []

        # init models
        self.audio_extractor = AudioExtractor(
            raw_checkpoint,
            audio_checkpoint,
            interval_duration,
            sample_rate,
            num_workers,
            device,
        )
        self.video_extractor = VideoExtractor(video_checkpoint, interval_duration, num_workers, device)
        self.features_saver = FeaturesSaver(num_workers, input_folder, output_folder, load_to_s3=load_to_s3)

        # init timer
        self.timer: Optional[Timer] = None

    def check_if_processed(self) -> None:
        """Check if the new files has already been processed."""
        features_dir = os.path.join(self.output_folder, "video", "embeddings")
        if os.path.exists(features_dir):
            processed_files = tuple(Path(file).stem for file in os.listdir(features_dir))
        else:
            processed_files = ()
        new_files_processed = [
            file_info for file_info in self.new_files if Path(file_info["alias"]).stem in processed_files
        ]
        for file in new_files_processed:
            os.remove(file["video_path"])
        processed_aliases = [file_info["alias"] for file_info in new_files_processed]
        logger.info(f"The following files were alredy processed: {processed_aliases}. Removed them.")
        self.new_files = [file_info for file_info in self.new_files if file_info["alias"] not in processed_aliases]

    def on_created(self, event) -> None:
        """
        Call when a file or directory is created.

        Args:
            event: The event representing the file system change.
        """
        logger.info(f"new event detected: {event.src_path}")
        is_any_new_videos = any(event.src_path.endswith(ext) for ext in self.video_extensions)
        if not event.is_directory and is_any_new_videos:
            file_info = {"alias": str(os.path.basename(event.src_path)), "video_path": event.src_path}
            self.new_files.append(file_info)

    def on_start(self) -> None:
        """Add all the files in the input folder to the new files list."""
        for file in os.listdir(self.input_folder):
            file_path = os.path.join(self.input_folder, file)
            self.new_files.append({"alias": file, "video_path": file_path})
        logger.info(f"Found {len(self.new_files)} new files on start.")  # noqa: WPS237
        self.process_new_files()

    def process_new_files(self) -> None:
        """Process the new files that have been detected."""
        self.stop_timer()
        if self.new_files:
            self.check_if_processed()
            logger.info(f"Processing {len(self.new_files)} new files...")  # noqa: WPS237
            processing_files = deepcopy(self.new_files)
            self.extract_features(processing_files)
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

    def extract_features(self, video_paths: List[Dict[str, Any]]):  # noqa: WPS213
        """Extract audio and video features from the video files.

        Args:
            video_paths (List[Dict[str, Any]]): List of dictionaries with video paths and aliases.
        """
        logger.info("Start to extract features.")
        logger.info(f"Current VideoExtractor buffer state: {self.audio_extractor.interval_prep.buffer}")  # noqa: WPS237
        logger.info(f"Current VideoExtractor buffer state: {self.video_extractor.interval_prep.buffer}")  # noqa: WPS237
        logger.info(f"Current FeaturesSaver buffer state: {self.features_saver.buffer_manager.buffers}")  # noqa: WPS237
        dataloader, _, pbar = get_data_entities(
            video_paths,
            self.interval_duration,
            self.batch_size,
            self.sample_rate,
            self.num_workers,
        )
        for batch in dataloader:
            batch = process_batch(batch)
            for sample_name, values in batch.items():
                # get video predictions
                video_results = self.video_extractor(
                    values.video_frames,
                    values.is_last,
                    values.fps,
                    values.video_chunk_len,
                    values.worker_id,
                )

                # get audio predictions
                audio_results = self.audio_extractor(
                    values.audio_samples,
                    values.is_last,
                    values.audio_chunk_len,
                    values.worker_id,
                )

                self.features_saver.update_current_state(
                    worker_id=values.worker_id,
                    sample_name=sample_name,
                    audio_results=audio_results,
                    video_results=video_results,
                    pbar=pbar,
                )
        self.features_saver.pack_up(pbar)
        logger.info("Interation has completed successfully.")
        logger.info(f"Current VideoExtractor buffer state: {self.audio_extractor.interval_prep.buffer}")  # noqa: WPS237
        logger.info(f"Current VideoExtractor buffer state: {self.video_extractor.interval_prep.buffer}")  # noqa: WPS237
        logger.info(f"Current FeaturesSaver buffer state: {self.features_saver.buffer_manager.buffers}")  # noqa: WPS237


# pylint: disable=too-many-arguments, too-many-locals
@click.command()
@click.option("--video_folder", type=str, default="data/videos_30fps/test")
@click.option("--batch_size", type=int, default=32)  # noqa: WPS432
@click.option("--batch_interval", type=int, default=600)  # noqa: WPS432
@click.option("--sample_rate", type=int, default=16000)  # noqa: WPS432
@click.option("--interval_duration", type=int, default=2)
@click.option("--num_workers", type=int, default=7)
@click.option("--video_checkpoint", type=str, default="weights/viclip_vision_v2.pt")
@click.option("--audio_checkpoint", type=str, default="weights/audio_6b.pth")
@click.option("--raw_audio_checkpoint", type=str, default="weights/betas_old.pt")
@click.option("--output_folder", type=str, default="data/custom_features")
@click.option("--load_to_s3", type=bool, default=True)
def main(  # noqa: WPS216,WPS213,WPS211
    video_folder: str,
    batch_size: int,
    batch_interval: int,
    sample_rate: int,
    interval_duration: int,
    num_workers: int,
    video_checkpoint: str,
    audio_checkpoint: str,
    raw_audio_checkpoint: str,
    output_folder: str,
    load_to_s3: bool,
):
    """
    Extract video and audio features.

    Args:
        video_folder (str): Path to the feed.
        batch_size (int): Batch size for processing.
        batch_interval (int): Interval in seconds for processing new files in batches. Defaults to 60.
        sample_rate (int): the number of audio samples taken per second
        interval_duration (int): duration of video fragment in secs
        num_workers (int): Number of worker processes for parallel data loading.
        video_checkpoint (str): Path to the checkpoint file for the video extractor.
        audio_checkpoint (str): Path to the checkpoint file for the audio extractor.
        raw_audio_checkpoint (str): Path to raw audio checkpoint file.
        output_folder (str): Path to folder where to save results.
        load_to_s3 (bool): Wether to load to s3.
    """
    logger.info(f"Started to extract features from {video_folder} folder")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Set up a watchdog observer to watch for new files and process them
    event_handler = VideoHandler(
        input_folder=video_folder,
        output_folder=output_folder,
        raw_checkpoint=raw_audio_checkpoint,
        audio_checkpoint=audio_checkpoint,
        video_checkpoint=video_checkpoint,
        sample_rate=sample_rate,
        device=device,
        interval_duration=interval_duration,
        batch_interval=batch_interval,
        num_workers=num_workers,
        batch_size=batch_size,
        load_to_s3=load_to_s3,
    )
    observer = Observer()
    observer.schedule(event_handler, video_folder, recursive=False)  # type: ignore
    observer.start()  # type: ignore
    event_handler.on_start()

    logger.info(f"Watching for new video files in {video_folder}...")

    try:
        while True:  # noqa: WPS457
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()  # type: ignore
    observer.join()
    event_handler.stop_timer()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
