"""Video preparation module."""

import math
from typing import List, Tuple

import torch
from torch import Tensor

from src.inference.base.base_preparator import BasePreparator


class VideoBasePreparator(BasePreparator):
    """Base batch processing class."""

    def combine_with_buffer(self, batch: Tensor, worker_id: int) -> Tensor:
        """Combine the batch with an existing buffer for a specific worker.

        Args:
            batch (Tensor): Input batch tensor.
            worker_id (int): Worker ID.

        Returns:
            Tensor: Combined batch.
        """
        if self.buffer[worker_id] is not None:
            batch = torch.cat((self.buffer[worker_id], batch), dim=0)  # type: ignore
        return batch

    def calculate_frames_per_interval(self, fps: float) -> Tuple[int, float]:
        """Calculate frames per interval using the fps.

        Args:
            fps (float): frames per second

        Returns:
            int: frames per interval
        """
        real_interval_size = fps * self.interval_duration
        rounded_interval_size = math.floor(real_interval_size + 0.5)
        rounding_error_per_sample = rounded_interval_size - real_interval_size
        return rounded_interval_size, rounding_error_per_sample

    @staticmethod
    def batch_reshape(batch: Tensor) -> Tuple[Tensor, Tuple[int, int, int]]:
        """
        Combine batch and time axes.

        Args:
            batch (Tensor): Input batch tensor.

        Returns:
            Tuple[Tensor, Tuple[int, int, int]]: Reshaped batch, and its dims
        """
        batch = batch.permute(0, 2, 1, 3, 4)
        B, T, C, H, W = batch.shape  # noqa: WPS236
        batch = batch.reshape(B * T, C, H, W)
        return batch, (C, H, W)

    @staticmethod
    def temporal_sampling(tensor: Tensor, frames_per_interval: int = 16, dim: int = 1) -> Tensor:
        """
        Perform temporal sampling on the video interval to tensor applicable for model input.

        Args:
            tensor (Tensor): Video tensor.
            frames_per_interval (int): Number of input frames per each interval
            dim (int): The dimension to be sampled by

        Returns:
            Tensor: Sampled video tensor with clips.
        """
        index = torch.linspace(0, tensor.size(dim), frames_per_interval)
        index = torch.clamp(index, 0, tensor.size(dim) - 1).long()
        return torch.index_select(tensor, dim, index)


class IntervalBasedBatchPreparation(VideoBasePreparator):
    """Reconstructs batches based on 2-second intervals.

    This class is responsible for preparing video data batches by dividing them into 2-second intervals.
    """

    def __init__(self, num_workers: int, interval_duration: int) -> None:
        """Initialize IntervalBasedBatchPreparation instance.

        Args:
            num_workers (int): Number of worker processes for parallel data loading.
            interval_duration (int): Interval duration in secs.
        """
        super().__init__(num_workers=num_workers, interval_duration=interval_duration)
        self.rounding_error_buffer: List[float] = [0 for _ in range(num_workers)]

    def _handle_rounding_error(
        self,
        batch: torch.Tensor,
        worker_id: int,
        batch_size: int,
        rounding_error_per_sample: float,
        is_last: List[bool],
    ) -> None:
        """Handle rounding error and buffer operations.

        This method updates the rounding error buffer and fixes rounding errors by adjusting the buffer.

        Args:
            batch (Tensor): Input batch tensor.
            worker_id (int): Worker ID.
            batch_size (int): Batch size.
            rounding_error_per_sample (float): Rounding error per sample.
            is_last (List[bool]): Indicates whether each sample is the last one.
        """
        # update rounding error buffer
        if any(is_last):
            self.rounding_error_buffer[worker_id] = 0
        else:
            self.rounding_error_buffer[worker_id] += batch_size * rounding_error_per_sample

        # fixing rounding error
        if self.rounding_error_buffer[worker_id] >= 1:
            if self.buffer[worker_id] is not None:
                self.buffer[worker_id] = torch.cat((self.buffer[worker_id], batch[-1:]), 0)  # type: ignore
            else:
                self.buffer[worker_id] = batch[-1:]

    # pylint: disable=too-many-locals
    def __call__(
        self,
        batch: Tensor,
        is_last: List[bool],
        fps: float,
        video_chunk_lens: List[int],
        worker_id: int,
    ) -> Tensor:
        """Reconstruct video frames into 2-second intervals and apply temporal sampling.

        Args:
            batch (Tensor): Input batch tensor.
            is_last (List[bool]): Indicates whether each sample is the last one.
            fps (float): Frames per second.
            video_chunk_lens (List[int]): Video chunk lengths without padding.
            worker_id (int): Worker ID.

        Returns:
            Tensor: Reconstructed batch.
        """
        batch, (C, H, W) = self.batch_reshape(batch)
        batch = self.combine_with_buffer(batch, worker_id)
        frames_per_interval, rounding_error_per_sample = self.calculate_frames_per_interval(fps)
        batch, batch_size, _ = self.apply_batch_processing(
            batch,
            video_chunk_lens,
            worker_id,
            frames_per_interval,
            is_last,
        )

        self._handle_rounding_error(batch, worker_id, batch_size, rounding_error_per_sample, is_last)

        # reshape to desirable shape
        batch = batch.reshape(batch_size, frames_per_interval, C, H, W)

        # apply temporal sampling
        batch = self.temporal_sampling(batch)

        return batch.permute(0, 2, 1, 3, 4)
