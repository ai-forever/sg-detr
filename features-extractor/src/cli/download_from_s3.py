"""Download videos from s3."""

import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import boto3
import click
import numpy as np
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from loguru import logger


def get_name_itself(name_timing: str):
    """Extract vide uid.

    Args:
        name_timing (str): video uid and timings

    Returns:
        Optional[str]: matched uid.
    """
    pattern = r"(.+?)_(\d+\.\d+)_(\d+\.\d+)\.pt"
    match = re.match(pattern, name_timing)
    if match:
        return f"{match.group(1)}.mp4"  # noqa: WPS237
    return None


def check_dir_status(output_dir: str, max_num: int) -> None:
    """Count number of downloaded and not processed films in the target dir.

    Wait until len(output_dir) <= max_num

    Args:
        output_dir (str): path to output dir
        max_num (int): upper threshold.
    """
    while len(os.listdir(output_dir)) >= max_num:
        logger.info("I've done a good job and deserve a break.")
        time.sleep(10)


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


load_dotenv()
s3_client = create_s3_client()


def download_from_s3(
    input_file: str,
    output_file: str,
    monitor_dir: str,
    bucket: str,
    overwrite: bool = False,
) -> None:
    """Download a file from an S3 bucket.

    Args:
        input_file (str): S3 object key.
        output_file (str): Local file name to save the downloaded file.
        monitor_dir (str): embs dir.
        bucket (str): S3 bucket name.
        overwrite (bool): Whether to overwrite the file if it exists. Defaults to False.
    """
    if os.path.exists(output_file) and not overwrite:
        logger.info(f"Skipping download as {output_file} already exists")
        return

    already_ready = tuple(get_name_itself(file) for file in os.listdir(monitor_dir))
    target_name_format = f"{Path(input_file).stem}.mp4"  # noqa: WPS237
    if target_name_format in already_ready and not overwrite:
        logger.info(f"Skipping download as {target_name_format} already processed")
        return

    # Create temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_path = temp_file.name
    try:  # noqa: WPS229
        input_file = input_file.replace(f"{bucket}/", "")
        s3_client.download_file(bucket, input_file, temp_path)
        shutil.move(temp_path, output_file)
        logger.info(f"Successfully downloaded {output_file} from bucket {bucket}")
    except (BotoCoreError, ClientError) as exp:
        logger.error(f"Cannot download {output_file}: {exp}")
        with open("failed_download.txt", "a+", encoding="utf8") as error_logs_file:
            error_logs_file.write(f"{output_file}\n")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@click.command()
@click.option("--input_files_path", required=True, help="path to list of S3 object keys to download.")
@click.option("--output_dir", required=True, help="Local directory to save the downloaded files.")
@click.option("--monitor_dir", default="data/custom_features/video", help="Embs directory.")
@click.option("--bucket", default="rndml-stage", help="S3 bucket name.")
@click.option("--overwrite", is_flag=True, help="Overwrite the file if it exists.")
@click.option("--max_output_vol", default=1000, help="Max number of elements in target dir.")
def main(  # noqa: WPS216, WPS210
    input_files_path: str,
    output_dir: str,
    monitor_dir: str,
    bucket: str,
    overwrite: bool,
    max_output_vol: int,
) -> None:
    """Download multiple files from an S3 bucket concurrently.

    Args:
        input_files_path (str): path to list of S3 object keys to download.
        output_dir (str): Local directory to save the downloaded files.
        monitor_dir (str): Embs directory.
        bucket (str): S3 bucket name. Defaults to 'rndml-stage'.
        overwrite (bool): Overwrite the files if they exist.
        max_output_vol (int): Max number of elements in target dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    paths_to_files = pd.read_csv(input_files_path)
    already_ready = tuple(Path(get_name_itself(file)).stem for file in os.listdir(monitor_dir))  # noqa: WPS221
    paths_to_files = paths_to_files.loc[~paths_to_files["YoutubeID"].isin(already_ready)]
    input_files = paths_to_files["s3_path"].values
    output_files = paths_to_files["YoutubeID"].values
    intervals = paths_to_files.loc[:, ["start_sec", "end_sec"]].values
    logger.info(f"Going to process: {len(output_files)} samples.")  # noqa: WPS237

    for input_file, output_file, interval in zip(input_files, output_files, intervals):
        ext = Path(input_file).suffix
        start_sec, end_sec = np.round(interval, 2)
        output_file_name = f"{output_file}_{start_sec:.2f}_{end_sec:.2f}{ext}"  # noqa: WPS221
        output_file_path = os.path.join(output_dir, output_file_name)
        download_from_s3(input_file, output_file_path, monitor_dir, bucket, overwrite)
        check_dir_status(output_dir, max_output_vol)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
