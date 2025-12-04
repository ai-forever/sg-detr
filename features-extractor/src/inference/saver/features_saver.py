"""FeaturesSaver class which handles the saving of features, managing Kafka messages, updating the buffers."""

import os
from pathlib import Path
from typing import Optional

import boto3
import torch
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from src.inference.saver.buffer import BufferManager

load_dotenv()


def create_s3_client() -> boto3.client:
    """Create and return an S3 client using environment variables for credentials.

    Returns:
        boto3.client: A boto3 S3 client instance.

    Raises:
        BotoCoreError: If there is an error creating the boto3 session.
        ClientError: If there is an error with the boto3 client.
    """
    try:
        session = boto3.session.Session()
        return session.client(
            service_name="s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            endpoint_url=os.getenv("ENDPOINT"),
        )
    except (BotoCoreError, ClientError) as exp:
        logger.error(f"Failed to create S3 client: {exp}")
        raise


# pylint: disable=too-many-instance-attributes,too-many-arguments
class FeaturesSaver:
    """
    Manage the saving of features and updating the state of the buffers.

    Attributes:
        num_workers (int): The number of worker processes for parallel data loading.
        buffer_manager (BufferManager): Manager for buffer-related tasks.
    """

    def __init__(
        self,
        num_workers: int,
        video_folder: str,
        output_folder: str,
        bucket: str,
        s3_video_folder: str = "moment_retrieval_datasets/pretrain_30fps/",
        load_to_s3: bool = True,
    ) -> None:
        """
        Initialize the FeaturesSaver object.

        Args:
            num_workers (int): Number of worker processes for parallel data loading.
            video_folder (str): Path to video folder.
            output_folder (str): Path to save features localy.
            bucket (str): s3 bucker name.
            s3_video_folder (str): Path to s3 video folder.
            load_to_s3 (bool): whether to load to s3
        """
        self.load_to_s3 = load_to_s3
        self.local_video_folder = video_folder
        self.s3_video_folder = s3_video_folder
        self.bucket = bucket
        self.output_folder = output_folder
        self.num_workers = max(num_workers, 1)
        self.buffer_manager = BufferManager(self.num_workers)
        self.s3_client = create_s3_client() if load_to_s3 else None

    @staticmethod
    def save_locally(model_preds: Tensor, local_path: str):  # noqa: WPS602
        """
        Save the model predictions locally.

        Args:
            model_preds (Tensor): The model predictions to be saved.
            local_path (str): The local path where the predictions will be saved.
        """
        dir_name = os.path.dirname(local_path)
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model_preds, local_path)

    # pylint: disable=broad-exception-caught
    def save_video_to_s3(self, video_name: str):  # noqa: WPS231
        """Move video to s3.

        Args:
            video_name (str): Local video name.
        """
        if not self.load_to_s3:
            logger.info("Skip s3 loading.")
            return
        local_video_path = os.path.join(self.local_video_folder, video_name)
        if os.path.exists(local_video_path):
            target_path = os.path.join(self.s3_video_folder, video_name)
            logger.info(f"Trying to save {local_video_path} to {target_path}")
            count = 0
            done = False
            while not done and count < 3:
                try:  # noqa: WPS229
                    self.s3_client.upload_file(local_video_path, self.bucket, target_path)
                    done = True
                    os.remove(local_video_path)
                    logger.info(f"File {local_video_path} was uploaded to s3 and removed locally.")
                except Exception as exp:
                    count += 1
                    logger.warning(f"Failed to upload {local_video_path} to s3. Try: {count} / 3")
                    logger.warning(f"Exception: {exp}")
            if not done:
                logger.error(f"Failed to upload {local_video_path} to s3.")

    def save_features(self, worker_id: int):
        """
        Iterate over the buffers and saves the features.

        Args:
            worker_id (int): The ID of the worker.
        """
        for sample_name, buffer_type, embedding in self.buffer_manager.iterate_over_buffers(worker_id):
            stem = Path(sample_name).stem
            sample_name = f"{stem}.pt"
            video_name = f"{stem}.mp4"
            buffer_dir = buffer_type.split("_")[0]
            local_path = os.path.join(self.output_folder, buffer_dir, sample_name)
            self.save_locally(embedding, local_path)
            self.save_video_to_s3(video_name)

    def update_current_state(
        self,
        worker_id: int,
        sample_name: str,
        audio_results: Tensor,
        video_results: Tensor,
        pbar: Optional[tqdm],
    ):
        """
        Update the current states of the buffer, produce message to Kafka.

        Args:
            worker_id (int): The ID of the worker.
            sample_name (str): The name of the sample.
            audio_results (Tensor): The audio results.
            video_results (Tensor): The video results.
            pbar (Optional[tqdm]): Optional progress bar to update.
        """
        is_buffered = self.buffer_manager.is_sample_buffered(worker_id, sample_name)
        buffer_length = self.buffer_manager.buffer_length(worker_id)

        if not is_buffered and buffer_length != 0:
            self.save_features(worker_id)
            self.buffer_manager.clear_worker_buffer(worker_id)
            if pbar is not None:
                pbar.update()

        self.buffer_manager.update_buffers(worker_id, sample_name, audio_results, video_results)

    def pack_up(self, pbar: Optional[tqdm]):
        """
        Save any remaining buffered features to the storage and clears the buffer.

        Args:
            pbar (Optional[tqdm]): Optional progress bar to update.
        """
        for worker_id in range(self.num_workers):
            sample_names = self.buffer_manager.get_buffered_sample_names(worker_id)
            if len(sample_names) == 0:  # noqa: WPS507
                continue
            self.save_features(worker_id)
            self.buffer_manager.clear_worker_buffer(worker_id)
            if pbar is not None:
                pbar.update()
