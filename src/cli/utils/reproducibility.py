"""This module provides functions to check the status of a Git repository and determine if it's reproducible."""

import re
from typing import Optional

from git import InvalidGitRepositoryError, Repo  # type: ignore


def commits_are_synced(commits_ahead) -> bool:
    """
    Check if the commits are synced with the upstream.

    Args:
        commits_ahead (re.Match): Match object from the regex search.

    Returns:
        bool: True if commits are synced, False otherwise.
    """
    return commits_ahead is not None and int(commits_ahead.group(1)) == 0


def is_valid_status_entry(entry: str) -> bool:
    """
    Determine if a single status entry is in the desired format.

    Args:
        entry (str): A single status entry string.

    Returns:
        bool: True if the entry is in the desired format, False otherwise.
    """
    return entry.startswith("# ") or entry.startswith("?")


def all_status_entries_valid(status: str) -> bool:
    """
    Check if all status entries of the repo are in the desired format.

    Args:
        status (str): The status string of the repo.

    Returns:
        bool: True if all status entries are valid, False otherwise.
    """
    status_entries = status.split("\n")
    return all(is_valid_status_entry(entry) for entry in status_entries)


def is_reproducible(repo: Repo) -> bool:
    """
    Check if the current Git repository is reproducible.

    Args:
        repo (Repo): The Git repository object.

    Returns:
        bool: True if the repository is reproducible, False otherwise.
    """
    status = repo.git.status("-s", "-b", "--porcelain=2")
    commits_ahead_pattern = r"#\sbranch\.ab\s\+(\d+)\s-\d+"  # trying to find "# branch.ab +0 -0" substring
    commits_ahead = re.search(commits_ahead_pattern, status)

    return commits_are_synced(commits_ahead) and all_status_entries_valid(status)


def git_status() -> Optional[str]:
    """
    Get the current Git branch and commit if the repository is reproducible.

    Returns:
        Tuple[Optional[str]]: commit hash, or None if the repository is not valid.
    """
    try:
        repo = Repo()
    except InvalidGitRepositoryError:
        return None
    return repo.commit().hexsha if is_reproducible(repo) else None
