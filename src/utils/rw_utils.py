"""Module to read and write files."""

import json
import os
import pickle
import zipfile
from typing import Any, List, Optional


def load_pickle(filename: str) -> object:
    """
    Load a pickle file.

    Args:
        filename (str): The path to the file to be loaded.

    Returns:
        object: The object loaded from the pickle file.
    """
    with open(filename, "rb") as file:
        return pickle.load(file)


def save_pickle(data: object, filename: str):
    """
    Save data to a pickle file using the highest protocol available.

    Args:
        data (object): The data to be pickled and saved.
        filename (str): The path to the file where the data will be saved.
    """
    with open(filename, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(filename: str) -> dict:
    """
    Load a JSON file.

    Args:
        filename (str): The path to the JSON file to be loaded.

    Returns:
        dict: dict loaded from the JSON file.
    """
    with open(filename, "r", encoding="utf8") as file:
        return json.load(file)


def save_json(data: object, filename: str, save_pretty: bool = False, sort_keys: bool = False):
    """
    Save data to a JSON file, with optional formatting.

    Args:
        data (object): The data to be saved in JSON format.
        filename (str): The path to the file where the data will be saved.
        save_pretty (bool): If True, the output is formatted with indentation. Defaults to False.
        sort_keys (bool): If True, the keys in dictionaries will be sorted. Defaults to False.
    """
    with open(filename, "w", encoding="utf8") as file:
        if save_pretty:
            file.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, file)


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


def save_lines(list_of_str: List[str], filepath: str):
    """
    Save a list of strings to a file, each string on a new line.

    Args:
        list_of_str (List[str]): A list of strings to be saved.
        filepath (str): The path to the file where the list will be saved.
    """
    with open(filepath, "w", encoding="utf8") as file:
        file.write("\n".join(list_of_str))


def read_lines(filepath: str):
    """
    Read a file and return a list of strings, one for each line.

    Args:
        filepath (str): The path to the file to be read.

    Returns:
        list: A list of strings, one for each line in the file.
    """
    with open(filepath, "r", encoding="utf8") as file:
        return [line.strip("\n") for line in file.readlines()]


# pylint: disable=too-many-locals
def make_zipfile(
    src_dir: str,
    save_path: str,
    enclosing_dir: str = "",
    exclude_dirs: Optional[List[str]] = None,
    exclude_extensions: Optional[List[str]] = None,
    exclude_dirs_substring: Optional[str] = None,
):
    """
    Create a zip file from the contents of a source directory.

    This function zips the contents of 'src_dir', saving the zip file to 'save_path'. It allows for exclusion of
    specific directories, file extensions, and substrings within directory names. An optional 'enclosing_dir' can
    be specified to add an additional directory layer in the zip file.

    Args:
        src_dir (str): The source directory whose contents are to be zipped.
        save_path (str): The file path where the zip file will be saved.
        enclosing_dir (str): An optional directory name to enclose the zipped contents. Defaults to an empty string.
        exclude_dirs (Optional[List[str]]): Directories to exclude from the zip.
        exclude_extensions (Optional[List[str]]): File extensions to exclude from the zip (e.g., ['.jpg', '.txt']).
        exclude_dirs_substring (Optional[str]): Exclude directories that contain the given substring in their name.
    """
    abs_src = os.path.abspath(src_dir)
    with zipfile.ZipFile(save_path, "w") as file:
        for dirname, subdirs, files in os.walk(src_dir):
            if exclude_dirs is not None:
                for e_p in exclude_dirs:
                    if e_p in subdirs:
                        subdirs.remove(e_p)
            if exclude_dirs_substring is not None:
                to_rm = []
                for subdir in subdirs:
                    if exclude_dirs_substring in subdir:
                        to_rm.append(subdir)
                for exclude in to_rm:
                    subdirs.remove(exclude)
            arcname = os.path.join(enclosing_dir, dirname[len(abs_src) + 1 :])
            file.write(dirname, arcname)
            for filename in files:
                if exclude_extensions is not None:
                    if os.path.splitext(filename)[1] in exclude_extensions:
                        continue  # do not zip it
                absname = os.path.join(dirname, filename)
                arcname = os.path.join(enclosing_dir, absname[len(abs_src) + 1 :])
                file.write(absname, arcname)
