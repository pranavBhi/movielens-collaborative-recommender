import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from typing import Dict, List, Any, Optional


class KNNRecommender:
    """
    User-based Collaborative Filtering Recommender using K-Nearest Neighbors
    with Cosine Similarity.
    """

    def __init__(self, n_neighbors: int = 30, min_support: int = 2):
        """
        Args:
            n_neighbors: Number of nearest neighbors to consider.
            min_support: Minimum number of neighbor ratings required for recommendation.
        """
        self.n_neighbors = n_neighbors
        self.min_support = min_support
        self.model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.matrix: Optional[csr_matrix] = None
        self.train_csc = None
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.movie_to_idx: Dict[int, int] = {}
        self.idx_to_movie: Dict[int, int] = {}
        self.movies_dict: Dict[int, Dict[str, str]] = {}
        self.user_means: np.ndarray = np.array([])
        self.global_mean: float = 3.5
        self.neighbor_indices: Optional[np.ndarray] = None
        self.neighbor_sims: Optional[np.ndarray] = None

    def fit(
        self,
        train_matrix: csr_matrix,
        user_to_idx: Dict[int, int],
        movie_to_idx: Dict[int, int],
        movies_df: Optional[pd.DataFrame] = None
    ) -> "KNNRecommender":
        """
        Fits the NearestNeighbors model on the user-item interaction matrix.
        """
        self.matrix = train_matrix
        self.train_csc = train_matrix.tocsc()
        self.user_to_idx = user_to_idx
        self.idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
        self.movie_to_idx = movie_to_idx
        self.idx_to_movie = {idx: mid for mid, idx in movie_to_idx.items()}

        if movies_df is not None:
            self.movies_dict = movies_df.set_index("movieId").to_dict("index")

        # Global mean rating
        if train_matrix.nnz > 0:
            self.global_mean = float(train_matrix.data.mean())

        # Compute per-user mean ratings for baseline adjustments
        n_users = train_matrix.shape[0]
        row_sums = train_matrix.sum(axis=1).A1
        row_counts = np.diff(train_matrix.indptr)
        self.user_means = np.full(n_users, self.global_mean, dtype=np.float32)
        has_ratings = row_counts > 0
        self.user_means[has_ratings] = row_sums[has_ratings] / row_counts[has_ratings]

        # Fit NearestNeighbors on user vectors
        self.model.fit(train_matrix)

        # Precompute neighbors and cosine similarities for all users
        k = min(self.n_neighbors + 1, n_users)
        dists, indices = self.model.kneighbors(train_matrix, n_neighbors=k)
        # Cosine similarity = 1 - cosine distance
        self.neighbor_indices = indices[:, 1:]  # Exclude self at position 0
        self.neighbor_sims = np.clip(1.0 - dists[:, 1:], 0.0, 1.0)

        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predicts rating for a single (user_id, movie_id) pair.
        """
        if user_id not in self.user_to_idx:
            return float(self.global_mean)

        u_idx = self.user_to_idx[user_id]
        u_mean = float(self.user_means[u_idx])

        if movie_id not in self.movie_to_idx:
            return u_mean

        m_idx = self.movie_to_idx[movie_id]
        neigh_idx = self.neighbor_indices[u_idx]
        neigh_sim = self.neighbor_sims[u_idx]

        valid_s = neigh_sim > 0
        if not np.any(valid_s):
            return u_mean

        n_idx = neigh_idx[valid_s]
        weights = neigh_sim[valid_s]

        # Ratings by neighbors for movie m_idx
        ratings = self.train_csc[n_idx, m_idx].toarray().flatten()
        has_rated = ratings > 0

        if not np.any(has_rated):
            return u_mean

        denom = np.sum(weights[has_rated])
        if denom < 1e-6:
            return u_mean

        numer = np.sum(weights[has_rated] * ratings[has_rated])
        pred = numer / denom
        return float(np.clip(pred, 0.5, 5.0))

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Efficient vectorized batch rating prediction for a test set DataFrame.
        """
        test_u = test_df["userId"].to_numpy()
        test_m = test_df["movieId"].to_numpy()
        n_samples = len(test_df)
        preds = np.full(n_samples, self.global_mean, dtype=np.float32)

        # Group test samples by user for batch matrix lookups
        user_to_test_indices: Dict[int, List[int]] = {}
        for idx, uid in enumerate(test_u):
            user_to_test_indices.setdefault(uid, []).append(idx)

        for uid, test_indices in user_to_test_indices.items():
            if uid not in self.user_to_idx:
                continue

            u_idx = self.user_to_idx[uid]
            u_mean = self.user_means[u_idx]
            neigh_idx = self.neighbor_indices[u_idx]
            neigh_sim = self.neighbor_sims[u_idx]

            valid_s = neigh_sim > 0
            if not np.any(valid_s):
                preds[test_indices] = u_mean
                continue

            n_idx = neigh_idx[valid_s]
            weights = neigh_sim[valid_s, None]  # Shape (k_valid, 1)

            target_mids = [test_m[i] for i in test_indices]
            valid_mask = [mid in self.movie_to_idx for mid in target_mids]

            valid_target_indices = [test_indices[i] for i, v in enumerate(valid_mask) if v]
            valid_midxs = [self.movie_to_idx[target_mids[i]] for i, v in enumerate(valid_mask) if v]

            if not valid_midxs:
                preds[test_indices] = u_mean
                continue

            # Shape: (k_valid, len(valid_midxs))
            sub_ratings = self.train_csc[n_idx, :][:, valid_midxs].toarray()
            has_rated = sub_ratings > 0

            denom = np.sum(weights * has_rated, axis=0)
            numer = np.sum(weights * sub_ratings, axis=0)

            nonzero_denom = denom > 1e-6
            u_preds = np.full(len(valid_midxs), u_mean, dtype=np.float32)
            u_preds[nonzero_denom] = numer[nonzero_denom] / denom[nonzero_denom]

            preds[valid_target_indices] = u_preds

            invalid_target_indices = [test_indices[i] for i, v in enumerate(valid_mask) if not v]
            if invalid_target_indices:
                preds[invalid_target_indices] = u_mean

        return np.clip(preds, 0.5, 5.0)

    def recommend(self, user_id: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Recommends top_n unwatched movies for a given user_id based on KNN.
        """
        if user_id not in self.user_to_idx:
            raise ValueError(f"User ID {user_id} not found in training dataset.")

        u_idx = self.user_to_idx[user_id]
        watched_mask = self.matrix[u_idx].toarray().flatten() > 0

        neigh_idx = self.neighbor_indices[u_idx]
        neigh_sim = self.neighbor_sims[u_idx]

        valid_s = neigh_sim > 0
        n_idx = neigh_idx[valid_s]
        weights = neigh_sim[valid_s, None]

        # Ratings of neighbors for all movies: shape (k_valid, n_movies)
        neigh_ratings = self.matrix[n_idx].toarray()
        has_rated = neigh_ratings > 0

        support = np.sum(has_rated, axis=0)
        denom = np.sum(weights * has_rated, axis=0)
        numer = np.sum(weights * neigh_ratings, axis=0)

        scores = np.full(self.matrix.shape[1], -1.0, dtype=np.float32)
        valid_pred_mask = (denom > 1e-6) & (support >= self.min_support)
        scores[valid_pred_mask] = numer[valid_pred_mask] / denom[valid_pred_mask]

        # Exclude movies already watched by the target user
        scores[watched_mask] = -1.0

        top_indices = np.argsort(scores)[::-1][:top_n]

        recommendations = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < 0:
                continue
            movie_id = int(self.idx_to_movie[idx])
            info = self.movies_dict.get(movie_id, {"title": "Unknown", "genres": "Unknown"})
            recommendations.append({
                "movieId": movie_id,
                "title": info.get("title", "Unknown"),
                "genres": info.get("genres", "Unknown"),
                "predicted_rating": round(score, 2),
                "neighbor_support": int(support[idx])
            })

        return recommendations


class SVDRecommender:
    """
    Latent Factor Matrix Factorization Recommender using TruncatedSVD with
    User-Mean Centering.
    """

    def __init__(self, n_components: int = 20, random_state: int = 42):
        """
        Args:
            n_components: Number of latent factors.
            random_state: Random seed for reproducibility.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.model = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.matrix: Optional[csr_matrix] = None
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.movie_to_idx: Dict[int, int] = {}
        self.idx_to_movie: Dict[int, int] = {}
        self.movies_dict: Dict[int, Dict[str, str]] = {}
        self.user_means: np.ndarray = np.array([])
        self.global_mean: float = 3.5
        self.user_factors: Optional[np.ndarray] = None  # P matrix (n_users x k)
        self.item_factors: Optional[np.ndarray] = None  # Q matrix (k x n_movies)

    def fit(
        self,
        train_matrix: csr_matrix,
        user_to_idx: Dict[int, int],
        movie_to_idx: Dict[int, int],
        movies_df: Optional[pd.DataFrame] = None
    ) -> "SVDRecommender":
        """
        Fits TruncatedSVD on the user-mean-centered interaction matrix.
        """
        self.matrix = train_matrix
        self.user_to_idx = user_to_idx
        self.idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
        self.movie_to_idx = movie_to_idx
        self.idx_to_movie = {idx: mid for mid, idx in movie_to_idx.items()}

        if movies_df is not None:
            self.movies_dict = movies_df.set_index("movieId").to_dict("index")

        if train_matrix.nnz > 0:
            self.global_mean = float(train_matrix.data.mean())

        n_users = train_matrix.shape[0]
        row_sums = train_matrix.sum(axis=1).A1
        row_counts = np.diff(train_matrix.indptr)
        self.user_means = np.full(n_users, self.global_mean, dtype=np.float32)
        has_ratings = row_counts > 0
        self.user_means[has_ratings] = row_sums[has_ratings] / row_counts[has_ratings]

        # Mean-center observed entries so missing entries (0s) represent user baseline
        centered_matrix = train_matrix.copy()
        centered_matrix.data = centered_matrix.data - np.repeat(self.user_means, row_counts)

        # Decompose into latent factors: P (user factors) and Q (item factors)
        self.user_factors = self.model.fit_transform(centered_matrix)
        self.item_factors = self.model.components_

        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predicts rating for a single (user_id, movie_id) pair.
        """
        if user_id not in self.user_to_idx:
            return float(self.global_mean)

        u_idx = self.user_to_idx[user_id]
        u_mean = float(self.user_means[u_idx])

        if movie_id not in self.movie_to_idx:
            return u_mean

        m_idx = self.movie_to_idx[movie_id]
        pred = u_mean + float(np.dot(self.user_factors[u_idx], self.item_factors[:, m_idx]))
        return float(np.clip(pred, 0.5, 5.0))

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Fast vectorized rating prediction for a test set DataFrame.
        """
        test_u = test_df["userId"].to_numpy()
        test_m = test_df["movieId"].to_numpy()
        n_samples = len(test_df)
        preds = np.full(n_samples, self.global_mean, dtype=np.float32)

        valid_mask = np.isin(test_u, list(self.user_to_idx.keys()))
        test_u_valid = test_u[valid_mask]
        test_m_valid = test_m[valid_mask]

        valid_indices = np.where(valid_mask)[0]

        u_indices = np.array([self.user_to_idx[u] for u in test_u_valid])
        movie_in_train = np.isin(test_m_valid, list(self.movie_to_idx.keys()))

        # If movie is in training set, predict using latent factor dot product
        known_m_mask = movie_in_train
        if np.any(known_m_mask):
            sub_u_idx = u_indices[known_m_mask]
            sub_m_idx = np.array([self.movie_to_idx[m] for m in test_m_valid[known_m_mask]])

            # Row-wise dot product of user factors and item factors
            factor_dot = np.sum(self.user_factors[sub_u_idx] * self.item_factors[:, sub_m_idx].T, axis=1)
            preds[valid_indices[known_m_mask]] = self.user_means[sub_u_idx] + factor_dot

        # If movie is unseen in train, fall back to user's average rating
        unknown_m_mask = ~movie_in_train
        if np.any(unknown_m_mask):
            sub_u_idx = u_indices[unknown_m_mask]
            preds[valid_indices[unknown_m_mask]] = self.user_means[sub_u_idx]

        return np.clip(preds, 0.5, 5.0)

    def recommend(self, user_id: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Recommends top_n unwatched movies for a given user_id based on SVD latent factors.
        """
        if user_id not in self.user_to_idx:
            raise ValueError(f"User ID {user_id} not found in training dataset.")

        u_idx = self.user_to_idx[user_id]
        watched_mask = self.matrix[u_idx].toarray().flatten() > 0

        # Matrix multiplication: P_u dot Q + user_mean
        scores = self.user_means[u_idx] + np.dot(self.user_factors[u_idx], self.item_factors)
        scores[watched_mask] = -1.0  # Exclude already watched films

        top_indices = np.argsort(scores)[::-1][:top_n]

        recommendations = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < 0:
                continue
            movie_id = int(self.idx_to_movie[idx])
            info = self.movies_dict.get(movie_id, {"title": "Unknown", "genres": "Unknown"})
            recommendations.append({
                "movieId": movie_id,
                "title": info.get("title", "Unknown"),
                "genres": info.get("genres", "Unknown"),
                "predicted_rating": round(score, 2)
            })

        return recommendations
