"""Base Preparator class common for video and audio extractors."""

from abc import ABC
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor


class BasePreparator(ABC):  # noqa: B024
    """Base batch processing class."""

    def __init__(self, num_workers: int, interval_duration: int):
        """
        Initialize BasePreparator.

        Args:
            num_workers (int): Number of worker processes for parallel data loading.
            interval_duration (int): Duration of the interval in secs.
        """
        self.interval_duration = interval_duration
        self.buffer: Dict[int, Union[Tensor, None]] = {worker: None for worker in range(num_workers)}

    def remove_dummy_dataset_generated_padding(  # noqa: WPS118
        self,
        batch: Tensor,
        chunk_lens: List[int],
        worker_id: int,
    ) -> Tensor:
        """
        Remove dummy dataset generated padding and adds true padding if necessary.

        Args:
            batch (Tensor): Input batch tensor.
            chunk_lens (List[int]): real chunk len without dataset generated padding
            worker_id (int): worker id

        Returns:
            Tensor: reconstructed batch
        """
        # calculate real number of samples extracted from the video
        video_chunk_len = sum(chunk_lens)

        # We have already added buffer to the batch, so we need to adjust video_chunk_len by its number of samples
        if self.buffer[worker_id] is not None:
            video_chunk_len += len(self.buffer[worker_id])  # type: ignore

        return batch[:video_chunk_len]

    def add_batch_padding(self, batch: Tensor, samples_per_interval: int) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Add padding in order to make batch divisible by samples_per_interval.

        Args:
            batch (Tensor): Input batch tensor.
            samples_per_interval (int): Number of input samples per each interval

        Returns:
            Tuple[Tensor, Optional[Tensor]]: Reconstructed batch and its mask.
        """
        mask: Optional[Tensor] = None
        if len(batch) % samples_per_interval != 0:
            padding_len = samples_per_interval - len(batch) % samples_per_interval
            last_frame_padding = torch.cat(padding_len * [batch[-1:]])
            batch = torch.cat((batch, last_frame_padding), 0)
            mask = torch.zeros(batch.size(0), device=batch.device, dtype=torch.bool)
            mask[-padding_len:] = 1  # noqa: WPS362
        return batch, mask

    def buffering(  # noqa: WPS221
        self,
        batch: Tensor,
        sample_len: int,
        worker_id: int,
        is_last: List[bool],
    ) -> Tuple[Tensor, int]:
        """Save part of entities to buffer because they do not fit the batch sample len yet.

        Args:
            batch (Tensor): Input batch tensor.
            sample_len (int): Desirable number of entities inside interval
            worker_id (int): worker id
            is_last (List[bool]): Is last sample or not

        Returns:
            Tensor: Tensor with number of samples enough to construct the batch, batch size
        """
        batch_size = len(batch) // sample_len
        break_point = batch_size * sample_len

        self.buffer[worker_id] = None if any(is_last) else batch[break_point:]

        return batch[:break_point], batch_size

    def apply_batch_processing(
        self,
        batch: Tensor,
        chunk_lens: List[int],
        worker_id: int,
        samples_per_interval: int,
        is_last: List[bool],
    ) -> Tuple[Tensor, int, Optional[Tensor]]:
        """Apply batch processing operations.

        This method handles padding, buffering, and other batch processing operations.

        Args:
            batch (Tensor): Input batch tensor.
            chunk_lens (List[int]): chunk length of the samples without padding.
            worker_id (int): Worker ID.
            samples_per_interval (int): Frames per interval.
            is_last (List[bool]): Indicates whether each sample is the last one.

        Returns:
            Tuple[Tensor, int, Optional[Tensor]]: Processed batch, batch size and mask.
        """
        mask: Optional[Tensor] = None
        if any(is_last):
            batch = self.remove_dummy_dataset_generated_padding(batch, chunk_lens, worker_id)
            batch, mask = self.add_batch_padding(batch, samples_per_interval)
        batch, batch_size = self.buffering(batch, samples_per_interval, worker_id, is_last)
        # cut out part of frames to make batch size divisible by samples_per_interval and save them in buffer
        return batch, batch_size, mask
