"""Remove extra padding from the video embeddings."""

import os

import torch
from tqdm import tqdm

from src.utils.rw_utils import load_jsonl

MAX_VIDEO_LENGTH: int = 150


def main(
    annotation_path: str = "/data/dataset/moment_retrieval/qvhilights/annotation/highlight_test_release.jsonl",
    path_to_embeddings: str = "/data/dataset/moment_retrieval/qvhilights/custom_features/video/embeddings/",
    output_dir: str = "/data/dataset/moment_retrieval/qvhilights/custom_features_fixed/video/embeddings/",
) -> None:
    """Remove extra padding.

    Args:
        annotation_path (str): path to annotation file
        path_to_embeddings (str): path to emgeddings dir
        output_dir (str): path where to save processed embeddings
    """
    os.makedirs(output_dir, exist_ok=True)
    data = load_jsonl(annotation_path)
    durations = {vid["vid"]: vid["duration"] for vid in data}

    for stem, duration in tqdm(durations.items(), total=len(durations)):
        filename = f"{stem}.pt"
        filepath = os.path.join(path_to_embeddings, filename)
        embedding = torch.load(filepath)
        target_length = duration // 2
        embedding = embedding[:target_length]
        filepath = os.path.join(output_dir, filename)
        torch.save(embedding, filepath)


if __name__ == "__main__":
    main()
