"""Audio Preparation."""

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from src.inference.base.base_preparator import BasePreparator

MIN_AUDIO_LENGTH: int = 10000


class IntervalBasedBatchPreparation(BasePreparator):
    """Class to reconstruct batch based on interval_duration sec intervals."""

    def __init__(self, sample_rate: int, interval_duration: int, num_workers: int):
        """
        Initialize IntervalBasedBatchPreparation.

        Args:
            sample_rate (int): The number of audio samples taken per second
            interval_duration (int): Interval duration.
            num_workers (int): Number of worker processes for parallel data loading.
        """
        num_workers = 1 if num_workers == 0 else num_workers
        super().__init__(num_workers, interval_duration)
        self.sample_rate = sample_rate

    def __call__(
        self,
        batch: Tensor,
        is_last: List[bool],
        audio_chunk_lens: List[int],
        worker_id: int,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Reconstruct batch based on intervals.

        Args:
            batch (Tensor): Input batch tensor.
            is_last (List[bool]): Is last sample or not
            audio_chunk_lens (List[int]): audio chunk len without padding
            worker_id (int): worker id

        Returns:
            Tuple[Tensor, Optional[Tensor]]: reconstructed batch
        """
        batch = batch.reshape(-1)
        if self.buffer[worker_id] is not None:
            batch = torch.cat((self.buffer[worker_id], batch), dim=0)  # type: ignore
        sample_len = self.sample_rate * self.interval_duration
        batch, batch_size, mask = self.apply_batch_processing(batch, audio_chunk_lens, worker_id, sample_len, is_last)
        mask = mask.reshape(batch_size, -1) if mask is not None else None
        return batch.reshape(batch_size, -1), mask
