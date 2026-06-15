"""Exploratory data analysis utilities and plotting."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .utils import ensure_dir


def _load_processed_tables(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed_dir = Path(data_dir) / "processed"
    ratings_path = processed_dir / "ratings_clean.csv"
    movies_path = processed_dir / "movies_clean.csv"

    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError(
            "Processed files not found. Run Phase 2 before Phase 3."
        )

    ratings = pd.read_csv(ratings_path, parse_dates=["timestamp"])
    movies = pd.read_csv(movies_path)
    return ratings, movies


def _save_plot(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150)
    plt.close(fig)


def run_eda(data_dir: str, output_dir: str = "outputs") -> None:
    """Run EDA, save plots and summary tables."""
    ratings, movies = _load_processed_tables(data_dir)

    charts_dir = ensure_dir(Path(output_dir) / "charts")
    tables_dir = ensure_dir(Path(output_dir) / "tables")

    n_users = ratings["userId"].nunique()
    n_movies = ratings["movieId"].nunique()
    n_ratings = len(ratings)
    sparsity = 1.0 - (n_ratings / (n_users * n_movies))

    summary = pd.DataFrame(
        {
            "n_users": [n_users],
            "n_movies": [n_movies],
            "n_ratings": [n_ratings],
            "sparsity": [round(sparsity, 4)],
        }
    )
    summary.to_csv(tables_dir / "eda_summary.csv", index=False)

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(ratings["rating"], bins=10, kde=False, ax=ax)
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    _save_plot(fig, charts_dir, "ratings_distribution.png")

    movie_counts = ratings.groupby("movieId").size().reset_index(name="rating_count")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(movie_counts["rating_count"], bins=50, kde=False, ax=ax)
    ax.set_title("Ratings per Movie")
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("Movies")
    ax.set_xscale("log")
    _save_plot(fig, charts_dir, "ratings_per_movie.png")

    top_movies = (
        movie_counts.merge(movies[["movieId", "title"]], on="movieId", how="left")
        .sort_values("rating_count", ascending=False)
        .head(20)
    )
    top_movies.to_csv(tables_dir / "top_movies_by_count.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=top_movies,
        y="title",
        x="rating_count",
        ax=ax,
        color="#4c72b0",
    )
    ax.set_title("Top 20 Movies by Rating Count")
    ax.set_xlabel("Rating Count")
    ax.set_ylabel("Movie Title")
    _save_plot(fig, charts_dir, "top_movies_by_count.png")

    movie_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=movie_stats,
        x="count",
        y="mean",
        ax=ax,
        s=20,
        alpha=0.6,
    )
    ax.set_title("Average Rating vs Rating Count")
    ax.set_xlabel("Rating Count")
    ax.set_ylabel("Average Rating")
    ax.set_xscale("log")
    _save_plot(fig, charts_dir, "avg_rating_vs_count.png")

    ratings_by_date = ratings.set_index("timestamp").resample("ME")["rating"].count()
    fig, ax = plt.subplots(figsize=(7, 4))
    ratings_by_date.plot(ax=ax)
    ax.set_title("Ratings Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Ratings")
    _save_plot(fig, charts_dir, "ratings_over_time.png")
