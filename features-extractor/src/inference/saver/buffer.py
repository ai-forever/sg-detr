"""Buffer related operations."""

import copy
from typing import Dict, Generator, Tuple

import torch
from torch import Tensor

from src.utils.schemas import BufferTypes

BaseDictType = Dict[str, Dict[int, Dict[str, Tensor]]]


class BufferManager:  # noqa: WPS214
    """
    Manage and handle buffer related operations, especially regarding different types of features and their storage.

    Attributes:
        num_workers (int): Number of workers.
        buffers (dict): Nested dictionary holding the buffers.
    """

    def __init__(self, num_workers: int):
        """
        Initialize the BufferManager object.

        Args:
            num_workers (int): The number of worker processes to be used.
        """
        self.num_workers = num_workers
        self._initialize_buffers()
        self._create_features_buffers()

    def _initialize_buffers(self) -> None:
        """Initialize the buffers for each buffer type and worker as empty dictionaries."""
        self.buffers: BaseDictType = {
            buffer_type.value: {worker: {} for worker in range(self.num_workers)} for buffer_type in BufferTypes
        }

    def _create_features_buffers(self) -> None:
        """Create feature buffers for audio and video, and optionally for interval extraction tasks."""
        video_buffer_dict: Dict[str, Tensor] = {}
        audio_buffer_dict: Dict[str, Tensor] = {}

        # audio buffer
        for worker in range(self.num_workers):
            self.buffers[BufferTypes.AUDIO.value][worker] = copy.deepcopy(audio_buffer_dict)

        # video buffer
        for worker in range(self.num_workers):
            self.buffers[BufferTypes.VIDEO.value][worker] = copy.deepcopy(video_buffer_dict)

    def is_sample_buffered(self, worker_id: int, sample_name: str) -> bool:
        """
        Return True if the sample is buffered, False otherwise.

        Args:
            sample_name (str): The name of the sample.
            worker_id (int): The ID of the worker.

        Returns:
            bool: True if the sample is buffered, False otherwise.
        """
        return sample_name in self.buffers[BufferTypes.VIDEO.value][worker_id]

    def buffer_length(self, worker_id: int) -> int:
        """
        Return the length of the buffer.

        Args:
            worker_id (int): The ID of the worker.

        Returns:
            int: The length of the buffer.
        """
        return len(self.buffers[BufferTypes.VIDEO.value][worker_id])

    def get_buffered_sample_names(self, worker_id: int) -> Tuple[str]:
        """
        Return the names of the samples in the buffer.

        Args:
            worker_id (int): The ID of the worker.

        Returns:
            Tuple[str]: The names of the samples in the buffer.
        """
        return tuple(self.buffers[BufferTypes.VIDEO.value][worker_id].keys())  # type: ignore

    def _update_buffer(
        self,
        worker_id: int,
        sample_name: str,
        embedding: Tensor,
        buffer_type: str,
    ) -> None:
        """
        Update the specified buffer with the given sample name and embedding tensor.

        Args:
            worker_id (int): The ID of the worker.
            sample_name (str): The name of the sample.
            embedding (Tensor): The tensor of embedding values.
            buffer_type (str): The type of buffer to be updated.
        """
        buffer = self.buffers[buffer_type][worker_id]

        if sample_name not in buffer:
            buffer[sample_name] = torch.Tensor()

        buffer[sample_name] = torch.cat((buffer[sample_name], embedding))  # type: ignore

    # pylint: disable=too-many-arguments
    def update_buffers(
        self,
        worker_id: int,
        sample_name: str,
        audio_features: Tensor,
        video_features: Tensor,
    ):
        """
        Update various buffers (audio, and video) with relevant features.

        Args:
            worker_id (int): The ID of the worker.
            sample_name (str): The name of the sample.
            audio_features (Tensor): Dictionary of audio features.
            video_features (Tensor): video features.
        """
        # Update buffers
        self._update_buffer(worker_id, sample_name, audio_features, BufferTypes.AUDIO.value)
        self._update_buffer(worker_id, sample_name, video_features, BufferTypes.VIDEO.value)

    def iterate_over_buffers(self, worker_id: int) -> Generator[Tuple[str, str, Tensor], None, None]:
        """
        Iteratesover the buffers and returns the buffer for the given worker.

        Args:
            worker_id (int): The ID of the worker.

        Yields:
            Tuple[str, BufferTypes, Tensor]: Sample name, buffer type, embedding
        """
        for buffer_type, buffer in self.buffers.items():
            for sample_name, embedding in buffer[worker_id].items():  # noqa: WPS526
                yield sample_name, buffer_type, embedding

    def clear_buffer(self) -> None:
        """Clear the entire buffer."""
        self._initialize_buffers()
        self._create_features_buffers()

    def clear_worker_buffer(self, worker_id: int) -> None:
        """
        Clear the buffer for a specific worker.

        Args:
            worker_id (int): The ID of the worker whose buffer needs to be cleared.
        """
        video_buffer_dict: Dict[str, Tensor] = {}
        audio_buffer_dict: Dict[str, Tensor] = {}

        self.buffers[BufferTypes.AUDIO.value][worker_id] = copy.deepcopy(audio_buffer_dict)
        self.buffers[BufferTypes.VIDEO.value][worker_id] = copy.deepcopy(video_buffer_dict)
