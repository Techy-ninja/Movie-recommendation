"""Evaluation utilities for recommender models."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

from .item_cf import ItemCFRecommender



def evaluate_rmse_mae(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute RMSE and MAE from a predictions table."""
    if "pred_rating" not in predictions.columns:
        raise ValueError("Missing column: pred_rating")

    if "rating" in predictions.columns:
        true_col = "rating"
    elif "true_rating" in predictions.columns:
        true_col = "true_rating"
    else:
        raise ValueError("Missing true rating column: rating or true_rating")

    y_true = predictions[true_col].astype(float).to_numpy()
    y_pred = predictions["pred_rating"].astype(float).to_numpy()
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    return pd.DataFrame({"rmse": [rmse], "mae": [mae]})


def evaluate_topk(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for top-K ranking evaluation."""
    if "hit" not in recommendations.columns:
        raise ValueError("Missing column: hit")

    if "k" in recommendations.columns:
        k = int(recommendations["k"].iloc[0])
    else:
        k = 10

    hits = recommendations["hit"].astype(int)
    hit_rate = float(hits.mean())
    precision_at_k = float(hits.sum() / (len(hits) * k))

    return pd.DataFrame({"hit_rate": [hit_rate], "precision_at_k": [precision_at_k], "k": [k]})


def evaluate_item_cf_hit_rate(
    ratings: pd.DataFrame,
    model: ItemCFRecommender,
    k: int = 10,
    min_user_ratings: int = 5,
    max_users: int = 200,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate item-CF with a simple hit-rate on related-item discovery.

    For each sampled user, pick one rated movie as a query and check whether
    any other movie they rated appears in the top-K similar list.
    """
    required = {"userId", "movieId"}
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(f"Missing columns in ratings: {sorted(missing)}")

    user_counts = ratings.groupby("userId").size()
    eligible_users = user_counts[user_counts >= min_user_ratings].index.to_numpy()
    if len(eligible_users) == 0:
        raise ValueError("No users with sufficient ratings for hit-rate evaluation.")

    rng = np.random.default_rng(random_state)
    if len(eligible_users) > max_users:
        eligible_users = rng.choice(eligible_users, size=max_users, replace=False)

    hits = []
    for user_id in eligible_users:
        user_movies = ratings.loc[ratings["userId"] == user_id, "movieId"].unique()
        if len(user_movies) < 2:
            continue

        query_movie = rng.choice(user_movies)
        target_set = set(user_movies) - {query_movie}

        try:
            recs = model.recommend(int(query_movie), top_n=k)
        except ValueError:
            continue

        rec_ids = set(recs["movieId"].astype(int).tolist())
        hit = int(len(rec_ids.intersection(target_set)) > 0)
        hits.append({"userId": int(user_id), "query_movieId": int(query_movie), "hit": hit, "k": k})

    hits_df = pd.DataFrame(hits)
    if hits_df.empty:
        raise ValueError("Hit-rate evaluation produced no results.")

    metrics = evaluate_topk(hits_df)
    return metrics, hits_df


def evaluate_svd_rmse_mae(
    ratings: pd.DataFrame,
    n_factors: int = 100,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train SVD on a train split and compute RMSE/MAE on the test split."""
    required = {"userId", "movieId", "rating"}
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(f"Missing columns in ratings: {sorted(missing)}")

    reader = Reader(rating_scale=(ratings["rating"].min(), ratings["rating"].max()))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)

    algo = SVD(n_factors=n_factors, random_state=random_state)
    algo.fit(trainset)
    predictions = algo.test(testset)

    pred_rows = [
        {
            "userId": int(p.uid),
            "movieId": int(p.iid),
            "true_rating": float(p.r_ui),
            "pred_rating": float(p.est),
        }
        for p in predictions
    ]
    preds_df = pd.DataFrame(pred_rows)
    metrics = evaluate_rmse_mae(preds_df)
    return metrics, preds_df
