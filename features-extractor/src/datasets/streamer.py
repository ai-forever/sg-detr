"""Module to manage media streams."""

import math
import time
from typing import Iterator, Optional, Tuple

from loguru import logger
from torch import Tensor
from torchaudio.io import StreamReader
from torchaudio.io._stream_reader import SourceStream

from src.utils.schemas import MetaModel

TIMEOUT: float = 30000
BACKOFF: float = 1000
IteratorType = Iterator[Tuple[Optional[Tensor], ...]]


class StreamConnectionManager:
    """
    A manager for establishing and managing connections to media streams.

    This class provides functionality to connect to video streams using a specified path.
    It will make multiple attempts to establish a connection if the initial attempts fail.
    Connection results and errors are logged using the provided logger.

    Attributes:
        _max_iterator_retries (int): The maximum number of times the connection to a stream will be retried.
        min_fps (int): Minimum allowed fps value.
    """

    def __init__(self, max_iterator_retries: int, min_fps: int) -> None:
        """
        Initialize StreamConnectionManager.

        Args:
            max_iterator_retries (int): Maximum retries for connecting to stream.
            min_fps (int): Minimum permissible fps value.
        """
        self._max_iterator_retries = max_iterator_retries
        self.min_fps = min_fps

    # pylint: disable=broad-except
    def get_stream_reader(self, path: str) -> Optional[StreamReader]:
        """
        Connect to a stream located at the specified path.

        Args:
            path (str): Path to the stream.

        Returns:
            Optional[StreamReader]: A StreamReader object if the connection is successful, otherwise None.
        """
        retries = 1

        while retries <= self._max_iterator_retries:
            try:
                return StreamReader(path, buffer_size=10**9)
            except Exception:
                logger.error(
                    f"Stream {path} is not accessible (attempt {retries} of {self._max_iterator_retries})",
                )
                retries += 1
                time.sleep(3)

        return None

    def establish_connection(
        self,
        video_path: str,
    ) -> Tuple[Optional[StreamReader], Optional[SourceStream], Optional[SourceStream]]:
        """
        Connect to the video stream and retrieve metadata.

        Args:
            video_path (str): Path to the video stream.

        Returns:
            Tuple[Optional[StreamReader], Optional[SourceStream], Optional[SourceStream]]: A tuple containing the
            general stream object and video stream metadata, if connection is successful, otherwise (None, None, None).
        """
        general_stream = self.get_stream_reader(video_path)

        if general_stream is None:
            logger.error(f"Failed to connect to stream {video_path} after {self._max_iterator_retries} attempts")
            return None, None, None

        if general_stream.default_video_stream is None:
            logger.error(f"Stream {video_path} has no video stream")
            return None, None, None

        video_meta = general_stream.get_src_stream_info(general_stream.default_video_stream)
        if video_meta.frame_rate < self.min_fps:
            logger.error(f"Stream {video_path} has fps lower than min permissible value")
            return None, None, None

        audio_stream_index = general_stream.default_audio_stream
        audio_meta = general_stream.get_src_stream_info(audio_stream_index)

        return general_stream, video_meta, audio_meta


class StreamPlugger:
    """Responsible for processing and enhancing media streams."""

    def __init__(self) -> None:
        """Initialize a new instance of StreamPlugger."""

    def plug_audio_stream(self, general_stream: StreamReader, meta: MetaModel) -> Optional[StreamReader]:
        """
        Add a basic audio stream with the specified parameters to the general stream.

        Args:
            general_stream (StreamReader): The general stream instance.
            meta (MetaModel): General stream meta info

        Returns:
            Optional[StreamReader]: General stream with connected audio substream
        """
        try:
            general_stream.add_basic_audio_stream(
                meta.audio_chunk_length,
                stream_index=meta.audio_idx,
                decoder="aac",
                buffer_chunk_size=-1,
            )
        except RuntimeError as error:
            logger.critical("Failed to add basic audio stream!")
            logger.critical(f"Problem description: {error}")
            return None

        return general_stream

    def plug_video_stream(self, general_stream: StreamReader, meta: MetaModel) -> Optional[StreamReader]:
        """
        Add a basic video stream with the specified parameters to the general stream.

        Args:
            general_stream (StreamReader): The general stream containing the video stream.
            meta (MetaModel): General stream meta info

        Returns:
            Optional[StreamReader]: General stream with connected video substream
        """
        try:
            general_stream.add_basic_video_stream(
                meta.video_chunk_length,
                stream_index=meta.video_idx,
                decoder="h264",
                format="rgb24",
                width=meta.width,
                height=meta.height,
                buffer_chunk_size=-1,
            )
        except RuntimeError as error:
            logger.critical("Failed to add basic video stream!")
            logger.critical(f"Problem description: {error}")
            return None

        return general_stream

    def __call__(
        self,
        general_stream: StreamReader,
        meta: MetaModel,
    ) -> Optional[StreamReader]:
        """
        Establish the connection and configuration for video and audio streams.

        Args:
            general_stream (StreamReader): The general stream object.
            meta (MetaModel): General stream meta info

        Returns:
            Optional[StreamReader]: General stream with connected video and audio substream
        """
        # Plug video stream
        general_stream = self.plug_video_stream(general_stream, meta)
        if general_stream is None:
            return None

        return self.plug_audio_stream(general_stream, meta)


class StreamOperator:
    """
    Provides utilities to operate on and manage media streams.

    The class offers functionalities like managing stream connections, adjusting
    stream resolutions while maintaining aspect ratio, and preparing meta
    information for streams.

    Attributes:
        window_size (int): Size of the TransNetV2 window.
        shortest_side (int): Length of the shortest side for resolution adjustment.
        connection_manager (StreamConnectionManager): Handles stream connections and related operations.
        stream_plugger (StreamPlugger): Enhances media streams.
    """

    def __init__(self, max_iterator_retries: int, window_size: int, min_fps: int, shortest_side: int):
        """
        Initialize an instance of the StreamOperator class.

        Args:
            max_iterator_retries (int): Maximum number of retries for stream iteration.
            window_size (int): Size of the TransNetV2 window.
            min_fps (int): Minimum acceptable frames per second for a stream.
            shortest_side (int): Length of the shortest side for resolution adjustment.
        """
        self.window_size = window_size
        self.shortest_side = shortest_side
        self.connection_manager = StreamConnectionManager(max_iterator_retries, min_fps)
        self.stream_plugger = StreamPlugger()

    def _get_new_hw(self, height: int, width: int) -> Tuple[int, int]:
        """
        Calculate new dimensions (height and width) based on the shortest side, preserving aspect ratio.

        Args:
            height (int): Original height.
            width (int): Original width.

        Returns:
            Tuple[int, int]: New dimensions - height and width.
        """
        if width < height:
            # Scale based on width
            new_width = self.shortest_side
            new_height = int(self.shortest_side * (float(height) / width))
        else:
            # Scale based on height
            new_height = self.shortest_side
            new_width = int(self.shortest_side * (float(width) / height))

        return new_height, new_width

    def prepare_general_meta(
        self,
        general_stream: StreamReader,
        video_meta: SourceStream,
        audio_meta: SourceStream,
    ) -> MetaModel:
        """
        Prepare metadata model from provided stream and its video and audio meta info.

        Args:
            general_stream (StreamReader): The general stream containing the video stream.
            video_meta (SourceStream): Metadata information of the video stream.
            audio_meta (SourceStream): Metadata information of the audio stream.

        Returns:
            MetaModel: An instance containing all the necessary metadata.
        """
        fps: float = video_meta.frame_rate
        height, width = self._get_new_hw(video_meta.height, video_meta.width)

        current_sample_rate: int = audio_meta.sample_rate
        audio_chunk_duration: float = self.window_size / fps
        audio_chunk_length: int = math.floor(current_sample_rate * audio_chunk_duration + 0.5)

        return MetaModel(
            video_idx=general_stream.default_video_stream,
            audio_idx=general_stream.default_audio_stream,
            fps=fps,
            height=height,
            width=width,
            current_sample_rate=current_sample_rate,
            audio_chunk_duration=audio_chunk_duration,
            audio_chunk_length=audio_chunk_length,
            video_chunk_length=self.window_size,
        )

    @staticmethod
    def get_iterator(general_stream: StreamReader) -> IteratorType:  # noqa: WPS602
        """Generate iterators from the stream.

        Args:
            general_stream (StreamReader): The general stream containing the video stream and audio stream (optional).

        Returns:
            IteratorType: generated iterator
        """
        return general_stream.stream(timeout=TIMEOUT, backoff=BACKOFF)

    # pylint: disable=too-many-locals
    def global_connect(
        self,
        video_path: str,
    ) -> Tuple[Optional[IteratorType], Optional[StreamReader], Optional[MetaModel]]:
        """
        Establish a connection to the stream, prepare metadata, and provide access to the enhanced stream.

        Args:
            video_path (str): Path to the video stream.

        Returns:
            Tuple[Optional[IteratorType], Optional[StreamReader], Optional[MetaModel]]:
            Stream iterator, enhanced stream reader, and metadata. If any step fails,
            all elements in the tuple will be set to None.
        """
        general_stream, video_meta, audio_meta = self.connection_manager.establish_connection(video_path)
        if (general_stream is None) or (video_meta is None) or (audio_meta is None):
            return None, None, None

        meta = self.prepare_general_meta(general_stream, video_meta, audio_meta)
        general_stream = self.stream_plugger(general_stream, meta)

        if general_stream is None:
            return None, None, None

        general_stream_iter = self.get_iterator(general_stream)

        return general_stream_iter, general_stream, meta

    def get_chunks(
        self,
        alias: str,
        general_stream_iter: IteratorType,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], IteratorType]:
        """
        Get next portion of data from the stream.

        Args:
            alias (str): stream alias
            general_stream_iter (IteratorType): video stream iterator

        Returns:
            Tuple[Optional[Tensor], Optional[Tensor], IteratorType]: Video and Audio portion of data
        """
        try:
            general_iterator_output = next(general_stream_iter)
        except StopIteration:
            # Handle the end of the iterator
            return None, None, general_stream_iter
        except RuntimeError as error:
            logger.error(f"Iterator: {alias} is not accessible.")
            logger.error(f"Problem description: {error}")
            return None, None, general_stream_iter

        video_chunk = general_iterator_output[0]
        audio_chunk = general_iterator_output[1]

        if (video_chunk is None) or (audio_chunk is None):
            return None, None, general_stream_iter

        return video_chunk, audio_chunk, general_stream_iter
