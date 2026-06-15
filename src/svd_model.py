"""Matrix factorization model wrapper (Surprise SVD)."""
from __future__ import annotations

import pandas as pd
from surprise import Dataset, Reader, SVD


class SVDRecommender:
    """Surprise SVD model for rating prediction."""

    def __init__(self, n_factors: int = 100, random_state: int = 42) -> None:
        self.n_factors = n_factors
        self.random_state = random_state
        self.model = None

    def fit(self, ratings: pd.DataFrame) -> "SVDRecommender":
        """Train the SVD model on ratings."""
        required = {"userId", "movieId", "rating"}
        missing = required.difference(ratings.columns)
        if missing:
            raise ValueError(f"Missing columns in ratings: {sorted(missing)}")

        min_rating = float(ratings["rating"].min())
        max_rating = float(ratings["rating"].max())
        reader = Reader(rating_scale=(min_rating, max_rating))
        data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
        trainset = data.build_full_trainset()

        self.model = SVD(n_factors=self.n_factors, random_state=self.random_state)
        self.model.fit(trainset)
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict a rating for a user-movie pair."""
        if self.model is None:
            raise ValueError("Model is not fit. Call fit() first.")

        pred = self.model.predict(uid=user_id, iid=movie_id)
        return float(pred.est)
