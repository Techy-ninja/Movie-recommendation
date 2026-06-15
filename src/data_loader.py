"""Data loading utilities for the MovieLens dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple
from urllib.request import urlretrieve
import zipfile

import pandas as pd

from .utils import ensure_dir


MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_SMALL_DIRNAME = "ml-latest-small"


def download_movielens_small(data_dir: str) -> Path:
    """Download and extract MovieLens small dataset into data/raw.

    Returns
    -------
    Path to the extracted dataset directory.
    """
    raw_dir = ensure_dir(Path(data_dir) / "raw")
    dataset_dir = raw_dir / MOVIELENS_SMALL_DIRNAME
    zip_path = raw_dir / "ml-latest-small.zip"

    if not dataset_dir.exists():
        if not zip_path.exists():
            urlretrieve(MOVIELENS_SMALL_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)

    return dataset_dir


def load_movielens_small(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load MovieLens small dataset files from the given directory.

    Returns
    -------
    ratings, movies, links, tags
    """
    dataset_dir = Path(data_dir) / "raw" / MOVIELENS_SMALL_DIRNAME
    if not dataset_dir.exists():
        dataset_dir = download_movielens_small(data_dir)

    ratings = pd.read_csv(dataset_dir / "ratings.csv")
    movies = pd.read_csv(dataset_dir / "movies.csv")
    links = pd.read_csv(dataset_dir / "links.csv")
    tags = pd.read_csv(dataset_dir / "tags.csv")

    return ratings, movies, links, tags
