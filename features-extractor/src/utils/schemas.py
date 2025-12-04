"""Data schemas."""

from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor


class BufferTypes(Enum):
    """
    Enumeration of different types of buffers.

    Attributes:
        AUDIO (str): Represents an audio buffer.
        VIDEO (str): Represents a video buffer.
    """

    AUDIO = "audio_buffer"
    VIDEO = "video_buffer"


class Batch(BaseModel):
    """
    Batch schema.

    Attributes:
        fps (float): The frames per second.
        worker_id (int): Worker id.
        video_state (int): The number of processed frames on each step accumulated.
        video_chunk_len (int): len of video sample without padding.
        audio_chunk_len (int): len of audio sample without padding.
        is_last (List[bool]): is last batch or not
        video_frames (Tensor): The video chunk.
        audio_samples (Tensor): The audio chunk.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    fps: float = Field(..., description="The frames per second.")
    worker_id: int = Field(..., description="Worker id")
    video_state: List[int] = Field(..., description="The number of processed frames on each step accumulated")
    video_chunk_len: List[int] = Field(..., description="Length of video sample without padding.")
    audio_chunk_len: List[int] = Field(..., description="Length of audio sample without padding.")
    is_last: List[bool] = Field(..., description="Is last batch or not")
    video_frames: Tensor = Field(..., description="The video chunk.")
    audio_samples: Tensor = Field(..., description="The audio chunk.")


class MetaModel(BaseModel):
    """General Meta Model."""

    video_idx: int
    audio_idx: int
    fps: float
    height: int
    width: int
    current_sample_rate: int
    audio_chunk_duration: float
    audio_chunk_length: int
    video_chunk_length: int
