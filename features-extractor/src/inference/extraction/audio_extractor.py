"""This module provides Audio Features extraction class."""

from typing import List, Union

import torch
from torch import Tensor, nn

from src.inference.preparation.audio_preparation import (
    IntervalBasedBatchPreparation,
)
from src.models.beats.beats import BEATs, BEATsConfig


# pylint: disable=too-many-arguments
class AudioExtractor:
    """Class for extracting audio embeddings using the PANNs model."""

    def __init__(
        self,
        raw_checkpoint: str,
        checkpoint: str,
        interval_duration: int,
        sample_rate: int,
        num_workers: int,
        device: Union[str, torch.device],
    ) -> None:
        """
        Initialize the AudioExtractor.

        Args:
            raw_checkpoint (str): Path to intial checkpoint file.
            checkpoint (str): Path to the checkpoint file.
            interval_duration (int): Interval duration.
            sample_rate (int): the number of audio samples taken per second
            num_workers (int): Number of worker processes for parallel data loading.
            device (Union[str, torch.device]): Torch device to use for computation.
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.raw_checkpoint = raw_checkpoint
        self.checkpoint = checkpoint
        self.sample_rate = sample_rate
        self.interval_prep = IntervalBasedBatchPreparation(sample_rate, interval_duration, num_workers)
        self.model = self._init_model()

    def _init_model(self) -> nn.Module:
        """
        Initialize the PANNs model from a checkpoint file.

        Returns:
            nn.Module: Initialized PANNs model.
        """
        checkpoint = torch.load(self.checkpoint)
        raw_checkpoint = torch.load(self.raw_checkpoint)
        cfg = BEATsConfig(raw_checkpoint["cfg"])
        audio_model = BEATs(cfg)
        audio_model.load_state_dict(checkpoint)
        audio_model.to(self.device)
        return audio_model.eval()

    def __call__(
        self,
        batch: Tensor,
        is_last: List[bool],
        audio_chunk_len: List[int],
        worker_id: int,
    ) -> Tensor:
        """
        Extract audio embeddings from the input batch.

        Args:
            batch (Tensor): Input batch tensor.
            is_last (List[bool]): Is last sample or not
            audio_chunk_len (List[int]): audio chunk len without padding
            worker_id (int): worker id

        Returns:
            Tensor: Extracted audio embeddings.
        """
        with torch.no_grad():
            batch, mask = self.interval_prep(batch, is_last, audio_chunk_len, worker_id)
            embeddings = self.model(batch.to(self.device), mask)
        return embeddings.cpu()
