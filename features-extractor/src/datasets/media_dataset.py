"""This module provides a dataset class for loading video frames from a feed and performing feature extraction."""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import IterableDataset

from src.datasets.streamer import StreamOperator
from src.datasets.transforms import AudioTransforms, VideoTransforms
from src.utils.schemas import MetaModel

CONNECTION_RETRIES: int = 3
VideoContainerType = List[Dict[str, Any]]


# pylint: disable=abstract-method,too-many-instance-attributes,too-many-arguments
class FeatureExtractionDataset(IterableDataset[Dict[str, Any]]):
    """Iterable dataset for loading video frames from feed."""

    def __init__(
        self,
        video_paths: VideoContainerType,
        interval_duration: int = 2,
        min_fps: int = 10,
        target_fps: int = 30,
        shortest_side: int = 224,
        desired_sample_rate: int = 16000,
        desired_channels: int = 1,
        max_iterator_retries: int = 3,
        failed_storage: str = "logs/failed_aliases.txt",
    ):
        """
        Initialize of FeatureExtractionDataset instance.

        Args:
            video_paths (VideoContainerType): List of video paths and their associated information.
            interval_duration (int): Interval duration in seconds. Default is 2.
            min_fps (int): Minimum FPS value. Default is 10.
            target_fps (int): Target FPS value. Default is 30.
            shortest_side (int): Desired length of the shortest side of the video. Default is 224.
            desired_sample_rate (int): Desired sample rate of the audio stream. Default is 16000.
            desired_channels (int): Desired number of channels of the audio stream. Default is 1.
            max_iterator_retries (int): Maximum number of connection retries. Default is 3.
            failed_storage (str): Path to file where to save failed aliases
        """
        self.video_transforms = VideoTransforms()
        self.audio_transforms = AudioTransforms(desired_sample_rate, desired_channels, min_fps)
        window_size = interval_duration * target_fps
        self.streamer_operator = StreamOperator(max_iterator_retries, window_size, min_fps, shortest_side)
        self.interval_duration = interval_duration
        self.desired_sample_rate = desired_sample_rate
        self.desired_channels = desired_channels
        self.video_paths = video_paths
        self._failed_storage = failed_storage
        self._buffer: Optional[Dict[str, object]] = None
        self._log_run_setup(min_fps, shortest_side)

    def _log_run_setup(self, min_fps, shortest_side) -> None:
        """
        Log hyperparams.

        Args:
            min_fps (int): Minimum FPS value. Default is 10.
            shortest_side (int): Desired length of the shortest side of the video. Default is 224.
        """
        logger.info("Dataset class has been initialized with the following params")
        length_of_video_container = len(self.video_paths)
        logger.info(f"{length_of_video_container} samples detected")
        logger.info(f"Min Video FPS: {min_fps}")
        logger.info(f"Shortest Image Side: {shortest_side}")
        logger.info(f"Desired Sample Rate: {self.desired_sample_rate}")

    def _log_failed_alias_to_file(self, alias: str) -> None:
        os.makedirs(os.path.dirname(self._failed_storage), exist_ok=True)
        with open(self._failed_storage, "a+", encoding="utf8") as error_logs_file:
            error_logs_file.write(f"{alias}\n")

    def _split_task_btwn_workers(self) -> Tuple[VideoContainerType, int]:
        """
        Split the task of loading video paths between workers in a multi-process data loading scenario.

        Returns:
            List[str], int: List of video paths assigned to the current worker and worker id.
        """
        worker_info = torch.utils.data.get_worker_info()

        # single-process data loading
        if worker_info is None:
            return self.video_paths, 0

        # If multi-process data loading
        total_videos = len(self.video_paths)
        videos_per_worker = math.ceil(total_videos / worker_info.num_workers)
        start_idx = worker_info.id * videos_per_worker
        end_idx = min(start_idx + videos_per_worker, total_videos)

        return self.video_paths[start_idx:end_idx], worker_info.id

    def _update_buffer(self, current_output: Dict[str, object]) -> Optional[Dict[str, object]]:
        """
        Check if the buffer sample is last sample of the video sequence or not.

        Args:
            current_output (Dict[str, Tensor]): A new sample to yield.

        Returns:
            Optional[Dict[str, Tensor], None]: Previous sample of the sequence with is_last key computed
        """
        # If the buffer is empty, set the current output as the buffer and exit early
        if self._buffer is None:
            self._buffer = current_output
            return None

        # Check if the current sample belongs to the same sequence as the buffer
        is_same_sequence = current_output["aliases"] == self._buffer["aliases"]
        self._buffer["is_last"] = not is_same_sequence

        # Store the previous buffer value to return it
        previous_output = self._buffer
        self._buffer = current_output

        return previous_output

    def _construct_output(
        self,
        worker_id: int,
        alias: str,
        fps: float,
        video: Tensor,
        audio: Tensor,
        desired_audio_size: int,
        video_state: int,
        video_chunk_len: int,
        audio_chunk_len: int,
    ) -> Optional[Dict[str, object]]:
        """
        Construct the output dictionary.

        Args:
            worker_id (int): worker id
            alias (str): The alias of the output.
            fps (float): The frames per second.
            video (Tensor): The frames extracted for FEs with dataset generated padding.
            audio (Tensor): The audio samples prepared for the audio model with dataset generated padding.
            desired_audio_size (int): hz
            video_state (int): The real number of frames processed accumulated total.
            video_chunk_len (int): The real len of video sample processed at the current step.
            audio_chunk_len (int): The real len of audio sample processed at the current step.

        Returns:
            Optional[Dict[str, object]]: The constructed output dictionary.
        """
        output = {
            "worker_id": worker_id,
            "aliases": alias,
            "fps": fps,
            "video": video.permute(1, 0, 2, 3),
            "audio": audio,
            "audio_size": desired_audio_size,
            "video_state": video_state,
            "video_chunk_len": video_chunk_len,
            "audio_chunk_len": audio_chunk_len,
        }

        return self._update_buffer(output)

    def _process_chunk(
        self,
        alias: str,
        video_chunk: Tensor,
        audio_chunk: Tensor,
        meta: MetaModel,
        video_state: int,
        worker_id: int,
    ):
        """
        Process a single chunk of video and audio data.

        This method performs various transformations on the video and audio chunks, ensures they are of
        the same length, and constructs the final output.

        Args:
            alias (str): Alias of the sample being processed.
            video_chunk (Tensor): The chunk of video data to be processed.
            audio_chunk (Tensor): The chunk of audio data to be processed.
            meta (MetaModel): Metadata associated with the chunks.
            video_state (int): Current state of the video processing.
            worker_id (int): ID of the worker processing the chunk.

        Returns:
            output: The processed output after applying various transformations to the audio and video chunks.
        """
        frames = self.video_transforms(video_chunk)
        audio_samples = self.audio_transforms(audio_chunk, meta.current_sample_rate)
        desired_audio_size = math.floor(self.desired_sample_rate * meta.audio_chunk_duration + 0.5)

        video_state += len(frames)
        video_chunk_len = len(frames)
        audio_chunk_len = len(audio_samples)

        # Make sure that the video and audio chunks are of the same length on each yield itteration
        frames = self.video_transforms.pad_video(frames, target_length=self.streamer_operator.window_size)
        audio = self.audio_transforms.pad_audio(audio_samples, number_of_frames=meta.video_chunk_length)

        return video_state, self._construct_output(
            worker_id=worker_id,
            alias=alias,
            fps=meta.fps,
            video=frames,
            audio=audio,
            desired_audio_size=desired_audio_size,
            video_state=video_state,
            video_chunk_len=video_chunk_len,
            audio_chunk_len=audio_chunk_len,
        )

    def process_sample(self, sample: Dict[str, Any], worker_id: int):
        """
        Process a single sample of data.

        This method takes a sample and a worker ID, performs the necessary processing including
        breaking the sample into chunks and processing each chunk, and yields the processed chunks.

        Args:
            sample (dict): The sample of data to be processed. It should contain alias and video_substream.
            worker_id (int): The ID of the worker processing the sample.

        Yields:
            output: The processed chunks of the sample.

        """
        video_state: int = 0
        alias = sample["alias"]
        video_path = sample["video_path"]
        logger.info(f"Starting to process: {alias} video")

        general_stream_iter, general_stream, meta = self.streamer_operator.global_connect(video_path)

        if (general_stream_iter is None) or (general_stream is None) or (meta is None):
            logger.critical(f"Empty stream: {alias}")
            self._log_failed_alias_to_file(alias)
            return

        while True:
            video_chunk, audio_chunk, general_stream_iter = self.streamer_operator.get_chunks(
                alias,
                general_stream_iter,
            )

            if (video_chunk is None) or (audio_chunk is None):
                break

            video_state, output = self._process_chunk(
                alias=alias,
                video_chunk=video_chunk,
                audio_chunk=audio_chunk,
                meta=meta,
                video_state=video_state,
                worker_id=worker_id,
            )

            if output:
                yield output

    def __iter__(self):
        """
        Yield video clips from the video files in the dataset.

        Yields:
            Dict[str, Any]: A dict containing the video path, video tensor, start time, and end time of a video clip.
        """
        samples, worker_id = self._split_task_btwn_workers()
        if samples:
            logger.info(f"The worker {worker_id} has begun the process of completing its task")

        for sample in samples:
            yield from self.process_sample(sample, worker_id)

        if self._buffer is not None:
            self._buffer["is_last"] = True
            yield self._buffer
