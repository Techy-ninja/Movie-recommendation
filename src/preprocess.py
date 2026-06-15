"""Preprocessing routines for ratings and movie metadata."""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from .utils import ensure_dir


YEAR_PATTERN = re.compile(r"\((\d{4})\)\s*$")


def preprocess_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the ratings table."""
    required = {"userId", "movieId", "rating", "timestamp"}
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(f"Missing columns in ratings: {sorted(missing)}")

    cleaned = ratings.dropna(subset=["userId", "movieId", "rating"]).copy()
    cleaned["userId"] = cleaned["userId"].astype(int)
    cleaned["movieId"] = cleaned["movieId"].astype(int)
    cleaned["rating"] = cleaned["rating"].astype(float)
    cleaned["rating"] = cleaned["rating"].clip(lower=0.5, upper=5.0)
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], unit="s", utc=True)
    cleaned = cleaned.drop_duplicates(subset=["userId", "movieId", "timestamp"])

    return cleaned


def preprocess_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the movies table."""
    required = {"movieId", "title", "genres"}
    missing = required.difference(movies.columns)
    if missing:
        raise ValueError(f"Missing columns in movies: {sorted(missing)}")

    cleaned = movies.dropna(subset=["movieId", "title"]).copy()
    cleaned["movieId"] = cleaned["movieId"].astype(int)

    years = cleaned["title"].str.extract(YEAR_PATTERN, expand=False)
    cleaned["year"] = pd.to_numeric(years, errors="coerce")
    cleaned["title_clean"] = cleaned["title"].str.replace(YEAR_PATTERN, "", regex=True).str.strip()

    cleaned["genres"] = cleaned["genres"].fillna("(no genres listed)")
    cleaned["genres_list"] = cleaned["genres"].str.split("|")

    return cleaned


def save_processed_tables(
    data_dir: str,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    links: pd.DataFrame,
    tags: pd.DataFrame,
) -> None:
    """Save cleaned tables into data/processed as CSV files."""
    processed_dir = ensure_dir(Path(data_dir) / "processed")
    ratings.to_csv(processed_dir / "ratings_clean.csv", index=False)
    movies.to_csv(processed_dir / "movies_clean.csv", index=False)
    links.to_csv(processed_dir / "links.csv", index=False)
    tags.to_csv(processed_dir / "tags.csv", index=False)
