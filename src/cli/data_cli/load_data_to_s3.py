"""Upload files to s3."""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import boto3
import click
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm


# pylint: disable=broad-exception-caught
def upload_to_s3(task_info: Tuple[str, str, str]) -> None:
    """Upload to s3 and remove.

    Args:
        task_info (Tuple[str, str, str]): task paths.
    """
    count = 0
    done = False
    while not done and count < 3:
        try:
            local_video_path, bucket, target_path = task_info
            s3_client.upload_file(local_video_path, bucket, target_path)
            done = True
        except Exception as exp:
            count += 1
            logger.warning(f"Failed to upload {local_video_path} to s3. Try: {count} / 3")
            logger.warning(f"Exception: {exp}")
    if not done:
        logger.error(f"Failed to upload {local_video_path} to s3.")


def process_in_parallel(videos_to_process: List[Tuple[str, str, str]], max_workers: int = 10):
    """
    Process the videos in parallel.

    Args:
        videos_to_process (Tuple[str, str, str]): A tuple containing the following:
            - the path to the local folder of the video
            - the bucket name
            - target s3 path.
        max_workers (int): number of workers.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(upload_to_s3, videos_to_process),
                total=len(videos_to_process),
                desc="Processing videos",
            ),
        )


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
            endpoint_url=os.getenv("ENDPOINT_URL"),
        )
    except (BotoCoreError, ClientError) as exp:
        logger.error(f"Failed to create S3 client: {exp}")
        raise


load_dotenv()
s3_client = create_s3_client()


@click.command()
@click.option("--local_folder", required=True, help="Path to local folder")
@click.option("--bucket", help="S3 bucket name.")
@click.option("--s3_video_folder", default="moment_retrieval_datasets/qvhighlights/", help="Path to s3 folder.")
@click.option("--num_workers", default=1, help="Num workers")
def main(local_folder: str, bucket: str, s3_video_folder: str, num_workers: int) -> None:
    """Upload files to s3.

    Args:
        local_folder (str): Path to local folder.
        bucket (str): Bucket name.
        s3_video_folder (str): Path to s3 folder.
        num_workers (int): Number of workers.
    """
    tasks = []
    for root, _, files in os.walk(local_folder):
        for video_name in files:
            local_file_path = os.path.join(root, video_name)
            s3_key = os.path.relpath(local_file_path, local_folder)
            s3_target_path = os.path.join(s3_video_folder, s3_key)
            tasks.append((local_file_path, bucket, s3_target_path))
    logger.info(f"Going to process {len(tasks)} files.")  # noqa: WPS237
    process_in_parallel(tasks, max_workers=num_workers)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
