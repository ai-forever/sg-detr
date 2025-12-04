"""Download files from s3."""

import functools
import multiprocessing as mp  # noqa: WPS111
import os
from typing import Any

import boto3
import click
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm


# pylint: disable=broad-exception-caught
def download_file(bucket: str, key: str, target_path: str) -> None:
    """Download file from s3.

    Args:
        bucket (str): bucket name
        key (str): s3 file path
        target_path (str): target local path
    """
    count = 0
    done = False
    while not done and count < 3:
        try:
            s3_client.download_file(bucket, key, target_path)
            done = True
        except Exception as exp:
            count += 1
            logger.warning(f"Failed to download {key} to {target_path}. Try: {count} / 3")
            logger.warning(f"Exception: {exp}")
    if not done:
        logger.error(f"Failed to upload {key} to {target_path}.")


def download_folder_mp_part(s3_folder: str, output_folder: str, page: Any, bucket: str):
    """Download page content.

    Args:
        s3_folder (str): s3 path
        output_folder (str): local target path
        page (Any): page provided by paginator
        bucket (str): bucket name
    """
    for obj in page["Contents"]:
        key = obj["Key"]
        key = key.replace(s3_folder, "")
        if key == "":
            continue

        filename = os.path.basename(key)
        tmp_dir = os.path.dirname(key)
        target_dir = os.path.join(output_folder, tmp_dir)

        if tmp_dir != "" and not os.path.exists(target_dir):
            os.makedirs(target_dir)
        elif not os.path.exists(os.path.join(target_dir, filename)):
            download_file(bucket, obj["Key"], os.path.join(target_dir, filename))


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
@click.option(
    "--s3_video_folder",
    default="moment_retrieval_datasets/pretrain/custom_text/",
    help="Path to s3 folder.",
)
@click.option("--bucket", help="S3 bucket name.")
@click.option("--local_folder", required=True, help="Path to local folder")
@click.option("--num_workers", default=1, help="Num workers")
def main(s3_video_folder: str, bucket: str, local_folder: str, num_workers: int) -> None:
    """Download files from s3 to local folder.

    Args:
        s3_video_folder (str): Path to s3 folder.
        bucket (str): Bucket name.
        local_folder (str): Path to local folder.
        num_workers (int): Number of workers.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    tmp_func = functools.partial(download_folder_mp_part, s3_video_folder, local_folder)
    mp_pool = mp.Pool(processes=num_workers)
    paginator = paginator.paginate(Bucket=bucket, Prefix=s3_video_folder)
    tqdm(list(mp_pool.imap(tmp_func, paginator)))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
