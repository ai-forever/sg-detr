"""Basic utils."""

import json
from typing import Any


def load_jsonl(filename: str):
    """
    Load a JSONL (JSON Lines) file.

    Args:
        filename (str): The path to the JSONL file to be loaded.

    Returns:
        list: A list of objects loaded from each line in the JSONL file.
    """
    with open(filename, "r", encoding="utf8") as file:
        return [json.loads(line.strip("\n")) for line in file.readlines()]


def save_jsonl(data: Any, filename: str):
    """
    Save data to a JSONL (JSON Lines) file.

    Args:
        data (Any): A list of objects to be saved in JSONL format.
        filename (str): The path to the file where the data will be saved.
    """
    with open(filename, "w", encoding="utf8") as file:
        file.write("\n".join([json.dumps(line) for line in data]))
