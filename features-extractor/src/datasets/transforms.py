"""Dataset transforms module.

Classes:
    - VideoTransforms: A class for applying video transformations to each video sample.
    - AudioTransforms: A class for applying audio transformations to each audio sample.
"""

from typing import Tuple

import torch
from torch import Tensor
from torchaudio.sox_effects import apply_effects_tensor

BASE_SCALE_FACTOR: int = 255
TRANSNET_INPUT_RESOLUTION: Tuple[int, int] = (27, 48)


class VideoTransforms:
    """Video transformation class for preprocessing video frames."""

    def __init__(
        self,
        crop_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        """
        Initialize VideoTransforms instance.

        Args:
            crop_size (Tuple[int, int]): Desired crop size [crop_height, crop_width] for FE. Default is [224, 224].
            mean (Tuple[float, float, float]): Mean values for normalization. Default is [0.485, 0.456, 0.406].
            std (Tuple[float, float, float]): STD values for normalization. Default is [0.229, 0.224, 0.225].
        """
        self.crop_size = crop_size
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    def pad_video(self, video: Tensor, target_length: int) -> Tensor:
        """
        Pad a video to the desired clip length.

        Args:
            video (Tensor): Video tensor to be padded.
            target_length (int): The desired length of the padded video.

        Returns:
            Tensor: Padded video tensor.
        """
        frames_to_pad = target_length - len(video)
        last_fragment_frame = video[-1:, :, :, :]
        padding_frames = last_fragment_frame.expand(frames_to_pad, -1, -1, -1)
        return torch.cat((video, padding_frames), 0)

    def center_crop(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply center crop to a batch of images.

        Args:
            images (Tensor): Input image tensor with shape (batch_size, C, H, W).

        Returns:
            Tensor: Center cropped image tensor with shape (batch_size, C, crop_height, crop_width).
        """
        _, _, height, width = images.shape
        crop_height, crop_width = self.crop_size

        # Calculate the starting position for cropping
        start_h = max(0, (height - crop_height) // 2)
        start_w = max(0, (width - crop_width) // 2)

        # calculate the endin position for cropping
        end_h = min(height, start_h + crop_height)
        end_w = min(width, start_w + crop_width)

        return images[:, :, start_h:end_h, start_w:end_w]

    def normalize(self, images: Tensor) -> Tensor:
        """
        Normalize the pixel values of the images.

        Args:
            images (Tensor): Input image tensor with shape (batch_size, C, H, W).

        Returns:
            Tensor: Normalized image tensor with shape (batch_size, C, H, W).
        """
        # Create mean and std tensors from the given lists
        mean_tensor = self.mean.to(images.device)
        std_tensor = self.std.to(images.device)
        return images.sub_(mean_tensor).div_(std_tensor)

    def __call__(self, images: Tensor) -> Tensor:
        """
        Apply video transformations to a batch of images.

        Args:
            images (Tensor): Input image tensor with shape (batch_size, C, H, W).

        Returns:
            Tensor: Transformed images for FE
        """
        images = images.float().div(BASE_SCALE_FACTOR)
        images = self.center_crop(images=images)
        return self.normalize(images=images)


class AudioTransforms:
    """
    AudioTransforms class for padding and processing audio data.

    Attributes:
        desired_sample_rate (int): The desired sample rate for audio processing.
        desired_channels (int): The desired number of audio channels.
        min_fps (float): The minimum frames per second for padding audio.
    """

    def __init__(self, desired_sample_rate: int, desired_channels: int, min_fps: float):
        """
        Initialize an AudioTransforms instance.

        Args:
            desired_sample_rate (int): The desired sample rate for audio processing.
            desired_channels (int): The desired number of audio channels.
            min_fps (float): The minimum frames per second for padding audio.
        """
        self.desired_sample_rate = desired_sample_rate
        self.desired_channels = desired_channels
        self.min_fps = min_fps

    def pad_audio(self, audio: Tensor, number_of_frames: int) -> Tensor:
        """
        Pad audio tensor to match the desired length.

        Args:
            audio (Tensor): Audio tensor.
            number_of_frames (int): The desired length of the video.

        Returns:
            Tensor: Padded audio tensor.
        """
        desired_length = int(self.desired_sample_rate * number_of_frames / self.min_fps)
        current_length = len(audio)

        if current_length >= desired_length:
            return audio  # No need to pad if audio is longer or equal to desired length

        number_of_pads = desired_length - current_length
        padding = torch.zeros(number_of_pads, dtype=audio.dtype)
        return torch.cat((audio, padding), 0)

    def __call__(self, audio_tensor: Tensor, current_sample_rate: int) -> Tensor:
        """
        Process an audio chunk by applying effects.

        Args:
            audio_tensor (Tensor): Audio tensor.
            current_sample_rate (int): Current sample rate of the audio tensor.

        Returns:
            Tensor: Processed audio tensor.
        """
        effects = [
            ["rate", str(self.desired_sample_rate)],
            ["channels", str(self.desired_channels)],
        ]

        # Apply the effects chain to the audio tensor
        processed_audio_tensor, _ = apply_effects_tensor(
            audio_tensor,
            int(current_sample_rate),
            effects=effects,
            channels_first=False,
        )
        return processed_audio_tensor.squeeze()
