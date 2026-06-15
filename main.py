from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import joblib

from src.baseline import PopularityRecommender
from src.data_loader import download_movielens_small, load_movielens_small
from src.eda import run_eda
from src.evaluate import evaluate_item_cf_hit_rate, evaluate_svd_rmse_mae
from src.item_cf import ItemCFRecommender
from src.preprocess import preprocess_movies, preprocess_ratings, save_processed_tables
from src.svd_model import SVDRecommender
from src.utils import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Movie recommender system project runner."
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        help="Phase to run (placeholder in Phase 1).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base data directory (default: data).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.phase == 1:
        print(
            "Phase 1 scaffold created. Future phases will add runnable steps. "
            f"Requested phase: {args.phase}."
        )
        return

    if args.phase == 2:
        print("Phase 2: downloading and preprocessing MovieLens small dataset...")
        download_movielens_small(args.data_dir)
        ratings, movies, links, tags = load_movielens_small(args.data_dir)
        ratings_clean = preprocess_ratings(ratings)
        movies_clean = preprocess_movies(movies)
        save_processed_tables(args.data_dir, ratings_clean, movies_clean, links, tags)
        print("Phase 2 complete. Cleaned files saved to data/processed.")
        return

    if args.phase == 3:
        print("Phase 3: running EDA and saving plots...")
        run_eda(args.data_dir)
        print("Phase 3 complete. Charts saved to outputs/charts and tables to outputs/tables.")
        return

    if args.phase == 4:
        print("Phase 4: building popularity-based baseline...")
        processed_dir = Path(args.data_dir) / "processed"
        ratings_path = processed_dir / "ratings_clean.csv"
        movies_path = processed_dir / "movies_clean.csv"
        if not ratings_path.exists() or not movies_path.exists():
            raise FileNotFoundError("Processed files not found. Run Phase 2 first.")

        ratings = pd.read_csv(ratings_path)
        movies = pd.read_csv(movies_path)
        model = PopularityRecommender(min_ratings=50).fit(ratings, movies)
        top_movies = model.recommend(top_n=20)

        outputs_dir = ensure_dir(Path("outputs") / "tables")
        top_movies.to_csv(outputs_dir / "baseline_top20.csv", index=False)
        print("Phase 4 complete. Saved outputs/tables/baseline_top20.csv")
        return

    if args.phase == 5:
        print("Phase 5: building item-based collaborative filtering model...")
        processed_dir = Path(args.data_dir) / "processed"
        ratings_path = processed_dir / "ratings_clean.csv"
        movies_path = processed_dir / "movies_clean.csv"
        if not ratings_path.exists() or not movies_path.exists():
            raise FileNotFoundError("Processed files not found. Run Phase 2 first.")

        ratings = pd.read_csv(ratings_path)
        movies = pd.read_csv(movies_path)

        movie_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
        movie_stats = movie_stats.rename(columns={"mean": "avg_rating", "count": "rating_count"})
        seed_movie_id = movie_stats.sort_values("rating_count", ascending=False).iloc[0]["movieId"]

        model = ItemCFRecommender(min_ratings=20).fit(ratings)
        recs = model.recommend(int(seed_movie_id), top_n=10)

        recs = (
            recs.merge(movies, on="movieId", how="left")
            .merge(movie_stats, on="movieId", how="left")
            .sort_values("similarity", ascending=False)
        )

        seed_info = movies[movies["movieId"] == seed_movie_id].merge(
            movie_stats, on="movieId", how="left"
        )

        outputs_dir = ensure_dir(Path("outputs") / "tables")
        recs.to_csv(outputs_dir / f"item_cf_recs_movie_{int(seed_movie_id)}.csv", index=False)
        seed_info.to_csv(outputs_dir / "item_cf_seed_movie.csv", index=False)
        print(
            "Phase 5 complete. Saved recommendations and seed movie info in outputs/tables."
        )
        return

    if args.phase == 6:
        print("Phase 6: training SVD matrix factorization model...")
        processed_dir = Path(args.data_dir) / "processed"
        ratings_path = processed_dir / "ratings_clean.csv"
        if not ratings_path.exists():
            raise FileNotFoundError("Processed ratings not found. Run Phase 2 first.")

        ratings = pd.read_csv(ratings_path)
        model = SVDRecommender(n_factors=100, random_state=42).fit(ratings)

        models_dir = ensure_dir(Path("outputs") / "models")
        joblib.dump(model, models_dir / "svd_model.joblib")

        sample = ratings.sample(n=min(50, len(ratings)), random_state=42)
        sample = sample.copy()
        sample["pred_rating"] = sample.apply(
            lambda row: model.predict(int(row["userId"]), int(row["movieId"])), axis=1
        )
        sample = sample[["userId", "movieId", "rating", "pred_rating"]]
        tables_dir = ensure_dir(Path("outputs") / "tables")
        sample.to_csv(tables_dir / "svd_sample_predictions.csv", index=False)

        print("Phase 6 complete. Model saved to outputs/models and sample predictions saved.")
        return

    if args.phase == 7:
        print("Phase 7: evaluating models with rating and ranking metrics...")
        processed_dir = Path(args.data_dir) / "processed"
        ratings_path = processed_dir / "ratings_clean.csv"
        if not ratings_path.exists():
            raise FileNotFoundError("Processed ratings not found. Run Phase 2 first.")

        ratings = pd.read_csv(ratings_path)
        tables_dir = ensure_dir(Path("outputs") / "tables")

        svd_metrics, svd_preds = evaluate_svd_rmse_mae(
            ratings, n_factors=100, test_size=0.2, random_state=42
        )
        svd_metrics.to_csv(tables_dir / "svd_metrics.csv", index=False)
        svd_sample = svd_preds.sample(n=min(500, len(svd_preds)), random_state=42)
        svd_sample.to_csv(tables_dir / "svd_predictions_sample_eval.csv", index=False)

        item_model = ItemCFRecommender(min_ratings=20).fit(ratings)
        item_metrics, item_hits = evaluate_item_cf_hit_rate(
            ratings, item_model, k=10, min_user_ratings=5, max_users=200, random_state=42
        )
        item_metrics.to_csv(tables_dir / "item_cf_hit_rate.csv", index=False)
        item_hits.to_csv(tables_dir / "item_cf_hit_details.csv", index=False)

        print(
            "Phase 7 complete. Saved SVD metrics and item-CF hit-rate outputs to outputs/tables."
        )
        return

    raise ValueError(f"Unsupported phase: {args.phase}")


if __name__ == "__main__":
    main()

