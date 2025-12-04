"""Setup file."""

import os

from setuptools import find_packages, setup

PACKAGE_NAME = "src"
VERSION_FILE_PATH = os.path.join(PACKAGE_NAME, ".package-version")
README_FILE_PATH = "README.md"


def get_project_version() -> str:
    """
    Retrieve the package version from .package-version.

    Returns:
        str: Package version

    Raises:
        ValueError: File not found
    """
    with open(VERSION_FILE_PATH, encoding="utf-8") as version_file:
        for line in version_file:
            if line.startswith("__version__"):
                # Extract version using string manipulation
                return line.split("=")[-1].strip().strip('"')
    raise ValueError(f"'__version__' not found in '{VERSION_FILE_PATH}'.")


def get_content_from_readme() -> str:
    """
    Retrieve the README content.

    Returns:
        str: README content
    """
    with open(README_FILE_PATH, encoding="utf-8") as readme_file:
        return readme_file.read()


setup(
    name="Moment Retrieval",
    version=get_project_version(),
    description="DETR base Moment Retrieval",
    long_description=get_content_from_readme(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(exclude=(".github", "docs", "examples")),
)
