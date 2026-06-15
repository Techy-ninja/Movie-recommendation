"""Item-based collaborative filtering using cosine similarity."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity



class ItemCFRecommender:
    """Item-item similarity recommender for movies."""

    def __init__(self, min_ratings: int = 20) -> None:
        self.min_ratings = min_ratings
        self.similarity_matrix: pd.DataFrame | None = None
        self.movie_index: dict[int, int] | None = None

    def fit(self, ratings: pd.DataFrame) -> "ItemCFRecommender":
        """Build the item similarity matrix."""
        required = {"userId", "movieId", "rating"}
        missing = required.difference(ratings.columns)
        if missing:
            raise ValueError(f"Missing columns in ratings: {sorted(missing)}")

        movie_counts = ratings.groupby("movieId").size()
        eligible_movies = movie_counts[movie_counts >= self.min_ratings].index
        filtered = ratings[ratings["movieId"].isin(eligible_movies)]

        user_item = filtered.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
            aggfunc="mean",
        ).fillna(0.0)

        item_matrix = user_item.T
        similarity = cosine_similarity(item_matrix)
        movie_ids = item_matrix.index

        self.similarity_matrix = pd.DataFrame(
            similarity, index=movie_ids, columns=movie_ids
        )
        self.movie_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
        return self

    def recommend(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        """Recommend similar movies for a given movie id."""
        if self.similarity_matrix is None:
            raise ValueError("Model is not fit. Call fit() first.")

        if movie_id not in self.similarity_matrix.index:
            raise ValueError("movie_id not found in similarity matrix.")

        if top_n <= 0:
            raise ValueError("top_n must be positive.")

        sims = self.similarity_matrix.loc[movie_id].drop(movie_id)
        top = sims.sort_values(ascending=False).head(top_n)
        top.name = "similarity"
        return top.reset_index().rename(columns={"index": "movieId"})
