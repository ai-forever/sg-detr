"""This module provides Video Features extraction class."""

from typing import List, Union

import torch
from torch import Tensor, nn

from src.inference.preparation.video_preparation import (
    IntervalBasedBatchPreparation,
)


# pylint: disable=too-few-public-methods
class VideoInference:
    """A class for performing inferences using the InternVidV2 model.

    Attributes:
        model (nn.Module): Initialized InternVidV2 model for inference.
        renorm_transform (torch.tensor): Pre-computed renormalization transformation tensor.
    """

    imagenet_mean = torch.tensor([0.45, 0.45, 0.45]).view(1, -1, 1, 1, 1)  # noqa: WPS221
    imagenet_std = torch.tensor([0.225, 0.225, 0.225]).view(1, -1, 1, 1, 1)  # noqa: WPS221
    viclip_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1, 1)  # noqa: WPS221
    viclip_std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1, 1)  # noqa: WPS221
    fnum: int = 4  # number of frames to be used by the model

    def __init__(self, checkpoint: str, device: torch.device) -> None:
        """Initialize the ClipInference class.

        Args:
            checkpoint (str): Path to the checkpoint file.
            device (torch.device): The device on which the model will be loaded.
        """
        self.device = device
        self.model = self._init_model(checkpoint, device)

    def _init_model(self, checkpoint, device: torch.device) -> nn.Module:
        """
        Initialize the InternVidV2 model from a checkpoint file.

        Args:
            checkpoint (str): Path to the checkpoint file.
            device (torch.device): The device on which the model will be loaded.

        Returns:
            nn.Module: Initialized InternVidV2 model.
        """
        model = torch.jit.load(checkpoint, map_location=device)  # type: ignore
        model.to(device)
        return model.eval()

    def _renormalization(self, batch: Tensor) -> Tensor:
        """
        Perform renormalization, converting ImageNet norm to WebImageText norm.

        Args:
            batch (Tensor): Input batch tensor.

        Returns:
            Tensor: Renormalized batch tensor.
        """
        denorm_batch = batch.mul(self.imagenet_std).add(self.imagenet_mean)
        return denorm_batch.sub(self.viclip_mean).div(self.viclip_std)

    def _extract_relevant_frames(self, batch: Tensor) -> Tensor:
        """Extract relevant frames from the batch.

        Args:
            batch (Tensor): Input batch tensor.

        Returns:
            Tensor: Batch consists of relevant frames.
        """
        indexes = list(range(batch.size(2)))
        assert len(indexes) >= self.fnum
        step = len(indexes) // self.fnum
        indexes = indexes[::step]
        indexes = indexes[: self.fnum]
        selected = torch.tensor(indexes)
        return torch.index_select(batch, 2, selected.to(batch.device))

    def __call__(self, batch: Tensor) -> Tensor:
        """Inference using the ViCLIP model.

        Args:
            batch (Tensor): Input batch tensor.

        Returns:
            Tensor: ViCLIP embeddings.
        """
        batch = self._extract_relevant_frames(batch)
        batch = self._renormalization(batch)

        with torch.no_grad():
            embeddings = self.model(batch.to(self.device)).float()
        return embeddings.cpu()


# pylint: disable=too-many-instance-attributes, too-many-arguments
class VideoExtractor:
    """Class for extracting video embeddings using the Hiera model."""

    def __init__(
        self,
        checkpoint: str,
        interval_duration: int,
        num_workers: int,
        device: Union[str, torch.device],
    ) -> None:
        """
        Initialize the VideoExtractor.

        Args:
            checkpoint (str): Path to the checkpoint file.
            interval_duration (int): Interval duration.
            num_workers (int): Number of worker processes for parallel data loading.
            device (Union[str, torch.device]): Torch device to use for computation.
        """
        if isinstance(device, str):
            device = torch.device(device)
        num_workers = max(num_workers, 1)

        self.interval_prep = IntervalBasedBatchPreparation(num_workers, interval_duration)
        self.video_model = VideoInference(checkpoint, device)

    def __call__(
        self,
        batch: Tensor,
        is_last: List[bool],
        fps: float,
        video_chunk_len: List[int],
        worker_id: int,
    ) -> Tensor:
        """
        Extract video embeddings from the input batch.

        Args:
            batch (Tensor): Input batch tensor.
            is_last (List[bool]): Is last sample or not
            fps (float): frames per second
            video_chunk_len (List[int]): video chunk len without padding
            worker_id (int): worker id

        Returns:
            Tensor: Extracted video embeddings.
        """
        batch = self.interval_prep(batch, is_last, fps, video_chunk_len, worker_id)
        return self.video_model(batch)  # forward pass of the video model
