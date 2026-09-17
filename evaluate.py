#!/usr/bin/env python3
"""
Evaluation and Demonstration Script for MovieLens Collaborative Filtering
Compares KNN (Cosine Similarity) vs. SVD (Truncated Latent Factor Factorization).
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_raw_data, build_user_item_matrix
from src.recommender import KNNRecommender, SVDRecommender


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Computes Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return rmse, mae


def format_table(headers: list, rows: list) -> str:
    """Formats a clean ASCII table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    sep_border = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_str = "| " + " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers)) + " |"

    table_lines = [sep_border, header_str, sep_border]
    for row in rows:
        row_str = "| " + " | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(row)) + " |"
        table_lines.append(row_str)
    table_lines.append(sep_border)

    return "\n".join(table_lines)


def display_user_profile(ratings_df: pd.DataFrame, movies_df: pd.DataFrame, user_id: int, top_k: int = 5):
    """Displays top rated movies by the given user in training history."""
    user_ratings = ratings_df[ratings_df["userId"] == user_id]
    if user_ratings.empty:
        print(f"User {user_id} has no ratings in the dataset.")
        return

    merged = user_ratings.merge(movies_df, on="movieId").sort_values("rating", ascending=False)
    print(f"\nProfile for User {user_id} - Top {top_k} Favorite Rated Movies in Train Set:")
    for idx, (_, row) in enumerate(merged.head(top_k).iterrows(), 1):
        print(f"  {idx}. [{row['rating']:.1f} ★] {row['title']} ({row['genres']})")


def display_recommendations(recs: list, algorithm_name: str, user_id: int):
    """Prints formatted top recommendations."""
    print(f"\n================================================================================")
    print(f" Top-5 Unwatched Recommendations for User {user_id} via {algorithm_name}")
    print(f"================================================================================")
    for idx, item in enumerate(recs, 1):
        pred_rating = item["predicted_rating"]
        extra = f" | Neighbor Support: {item['neighbor_support']} users" if "neighbor_support" in item else ""
        print(f"  {idx}. [Pred: {pred_rating:.2f} ★] {item['title']} (ID: {item['movieId']})")
        print(f"      Genres: {item['genres']}{extra}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Collaborative Filtering Recommenders")
    parser.add_argument("--user_id", type=int, default=1, help="Target user ID for sample recommendations (default: 1)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of ratings for test set (default: 0.2)")
    parser.add_argument("--n_neighbors", type=int, default=30, help="Number of neighbors for KNN (default: 30)")
    parser.add_argument("--n_components", type=int, default=20, help="Latent components for SVD (default: 20)")
    args = parser.parse_args()

    print("=" * 80)
    print(" MovieLens 100k Collaborative Filtering: KNN vs. SVD Evaluation")
    print("=" * 80)

    # 1. Load Data
    ratings_df, movies_df = load_raw_data()
    n_ratings = len(ratings_df)
    n_users = ratings_df["userId"].nunique()
    n_movies = ratings_df["movieId"].nunique()
    sparsity = 100.0 * (1.0 - (n_ratings / (n_users * n_movies)))

    print(f"\nDataset Summary:")
    print(f"  - Total Ratings : {n_ratings:,}")
    print(f"  - Unique Users  : {n_users:,}")
    print(f"  - Unique Movies : {n_movies:,}")
    print(f"  - Matrix Sparsity: {sparsity:.2f}%\n")

    # 2. Train/Test Split
    print(f"Splitting dataset into {(1 - args.test_size) * 100:.0f}% Train and {args.test_size * 100:.0f}% Test (random_state=42)...")
    train_df, test_df = train_test_split(ratings_df, test_size=args.test_size, random_state=42)
    print(f"  - Train Set: {len(train_df):,} ratings")
    print(f"  - Test Set : {len(test_df):,} ratings\n")

    # 3. Construct Training User-Item Matrix
    train_matrix, user_to_idx, idx_to_user, movie_to_idx, idx_to_movie = build_user_item_matrix(train_df)
    test_ratings = test_df["rating"].to_numpy(dtype=np.float32)

    # 4. Fit and Evaluate KNN
    print(f"Fitting KNN Recommender (k={args.n_neighbors}, metric='cosine') ...")
    knn = KNNRecommender(n_neighbors=args.n_neighbors)
    t_start_knn_fit = time.time()
    knn.fit(train_matrix, user_to_idx, movie_to_idx, movies_df)
    t_knn_fit = time.time() - t_start_knn_fit

    t_start_knn_pred = time.time()
    knn_preds = knn.predict_batch(test_df)
    t_knn_pred = time.time() - t_start_knn_pred
    knn_rmse, knn_mae = calculate_metrics(test_ratings, knn_preds)
    print(f"  -> KNN Fit: {t_knn_fit:.3f}s | Predict: {t_knn_pred:.3f}s | RMSE: {knn_rmse:.4f} | MAE: {knn_mae:.4f}")

    # 5. Fit and Evaluate SVD
    print(f"\nFitting SVD Recommender (n_components={args.n_components}, centered) ...")
    svd = SVDRecommender(n_components=args.n_components, random_state=42)
    t_start_svd_fit = time.time()
    svd.fit(train_matrix, user_to_idx, movie_to_idx, movies_df)
    t_svd_fit = time.time() - t_start_svd_fit

    t_start_svd_pred = time.time()
    svd_preds = svd.predict_batch(test_df)
    t_svd_pred = time.time() - t_start_svd_pred
    svd_rmse, svd_mae = calculate_metrics(test_ratings, svd_preds)
    print(f"  -> SVD Fit: {t_svd_fit:.3f}s | Predict: {t_svd_pred:.3f}s | RMSE: {svd_rmse:.4f} | MAE: {svd_mae:.4f}")

    # 6. Print Comparison Table
    print("\n" + "=" * 80)
    print(" Performance Comparison Table (Test Set Evaluation)")
    print("=" * 80)
    headers = ["Algorithm", "RMSE", "MAE", "Fit Time (s)", "Predict Time (s)", "Complexity"]
    rows = [
        ["KNN (Cosine Similarity)", f"{knn_rmse:.4f}", f"{knn_mae:.4f}", f"{t_knn_fit:.3f}", f"{t_knn_pred:.3f}", "O(U * M) Memory"],
        ["SVD (Truncated Latent)", f"{svd_rmse:.4f}", f"{svd_mae:.4f}", f"{t_svd_fit:.3f}", f"{t_svd_pred:.3f}", "O((U + M) * k) Factors"],
    ]
    print(format_table(headers, rows))

    # Determine Winner
    if svd_rmse < knn_rmse:
        diff = knn_rmse - svd_rmse
        print(f"\nResult: SVD outperforms KNN by {diff:.4f} RMSE ({diff / knn_rmse * 100:.2f}% lower error).")
    else:
        diff = svd_rmse - knn_rmse
        print(f"\nResult: KNN outperforms SVD by {diff:.4f} RMSE.")

    # 7. Sample Recommendations for User 1
    display_user_profile(train_df, movies_df, user_id=args.user_id, top_k=5)

    knn_recs = knn.recommend(user_id=args.user_id, top_n=5)
    display_recommendations(knn_recs, "K-Nearest Neighbors (KNN)", user_id=args.user_id)

    svd_recs = svd.recommend(user_id=args.user_id, top_n=5)
    display_recommendations(svd_recs, "Latent Factor Matrix Factorization (SVD)", user_id=args.user_id)

    print("\n" + "=" * 80)
    print(" Recommendation Demonstration Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
