"""Entrypoint for text features extraction."""

import os

import click
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from src.models.tokenizer_bert import BertTokenizer
from src.utils.utils import load_jsonl

MAX_SEQ_LENGTH: int = 40


# pylint: disable=too-many-arguments, too-many-locals
@click.command()
@click.option("--captions_path", type=str, default="data/positive_captions_s3path.csv")
@click.option("--checkpoint", type=str, default="weights/viclip_text_v2.pt")
@click.option("--output_folder", type=str, default="data/custom_text")
def main(  # noqa: WPS210
    captions_path: str,
    checkpoint: str,
    output_folder: str,
):
    """
    Extract video and audio features.

    Args:
        captions_path (str): Path to the feed.
        checkpoint (str): Path to the checkpoint file for the text clip extractor.
        output_folder (str): Path to folder where to save results.
    """
    logger.info(f"Started to extract features from {captions_path} file.")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(output_folder, exist_ok=True)

    # load captions
    if "jsonl" in captions_path:
        annotation = load_jsonl(captions_path)
        texts = {str(sample["qid"]): sample["query"] for sample in annotation}
    else:
        captions = pd.read_csv(captions_path)["Caption"].values
        texts = {}
        for idx, caption in enumerate(captions):
            texts[str(f"{idx}").rjust(7, "0")] = caption

    # init model
    model = torch.jit.load(checkpoint, map_location=device)  # type: ignore
    model.to(device)
    model.eval()

    # init tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", local_files_only=False)

    with torch.no_grad():
        for stem, text in tqdm(texts.items()):
            # for idx, text in enumerate(captions):
            tokens = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt",
            ).to(device)
            _, all_tfeat, _ = model(tokens.input_ids, tokens.attention_mask)

            # all features included pooled [:, 0], projected and normed
            all_tfeat = all_tfeat[tokens.attention_mask.bool()]
            all_tfeat = all_tfeat.cpu().numpy()
            np.savez(os.path.join(output_folder, stem), features=all_tfeat)  # _{idx}


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
