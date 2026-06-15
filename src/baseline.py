"""Popularity-based baseline recommender."""
from __future__ import annotations

import pandas as pd


class PopularityRecommender:
    """Recommends popular, well-rated movies with minimum support."""

    def __init__(self, min_ratings: int = 50) -> None:
        self.min_ratings = min_ratings
        self.popularity_table: pd.DataFrame | None = None

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> "PopularityRecommender":
        """Fit the model using ratings and movie metadata."""
        required_ratings = {"movieId", "rating"}
        missing_ratings = required_ratings.difference(ratings.columns)
        if missing_ratings:
            raise ValueError(f"Missing columns in ratings: {sorted(missing_ratings)}")

        if "movieId" not in movies.columns:
            raise ValueError("Missing column in movies: movieId")

        stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
        stats = stats.rename(columns={"mean": "avg_rating", "count": "rating_count"})
        stats = stats[stats["rating_count"] >= self.min_ratings]

        merged = stats.merge(movies, on="movieId", how="left")
        merged = merged.sort_values(["avg_rating", "rating_count"], ascending=False)

        self.popularity_table = merged.reset_index(drop=True)
        return self

    def recommend(self, top_n: int = 10) -> pd.DataFrame:
        """Return top-N popular movies."""
        if self.popularity_table is None:
            raise ValueError("Model is not fit. Call fit() first.")

        if top_n <= 0:
            raise ValueError("top_n must be positive.")

        return self.popularity_table.head(top_n).copy()
