import os
import zipfile
import urllib.request
import ssl
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple, Dict, Any, Optional

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def download_and_extract_dataset(
    url: str = DATASET_URL,
    target_dir: str = DEFAULT_DATA_DIR
) -> str:
    """
    Downloads the MovieLens dataset zip archive and extracts it to target_dir.

    Args:
        url: URL of the MovieLens zip file.
        target_dir: Destination root data directory.

    Returns:
        Path to the extracted dataset directory.
    """
    os.makedirs(target_dir, exist_ok=True)
    extracted_path = os.path.join(target_dir, "ml-latest-small")
    zip_path = os.path.join(target_dir, "ml-latest-small.zip")

    # If already extracted and files exist, skip re-downloading
    ratings_file = os.path.join(extracted_path, "ratings.csv")
    movies_file = os.path.join(extracted_path, "movies.csv")
    if os.path.exists(ratings_file) and os.path.exists(movies_file):
        print(f"[DataLoader] Dataset already exists at: {extracted_path}")
        return extracted_path

    print(f"[DataLoader] Downloading dataset from {url} ...")
    try:
        context = ssl.create_default_context()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=context) as response, open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    except Exception as e:
        print(f"[DataLoader] Standard SSL verification failed ({e}), using unverified context fallback...")
        context = ssl._create_unverified_context()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=context) as response, open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    print(f"[DataLoader] Downloaded to {zip_path}. Extracting...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    print(f"[DataLoader] Extracted successfully to {extracted_path}")
    return extracted_path


def load_raw_data(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensures dataset is downloaded and loads ratings and movies DataFrames.

    Args:
        data_dir: Path to directory containing ml-latest-small folder.

    Returns:
        ratings_df: DataFrame with columns ['userId', 'movieId', 'rating', 'timestamp']
        movies_df: DataFrame with columns ['movieId', 'title', 'genres']
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    dataset_path = download_and_extract_dataset(target_dir=data_dir)
    ratings_df = pd.read_csv(os.path.join(dataset_path, "ratings.csv"))
    movies_df = pd.read_csv(os.path.join(dataset_path, "movies.csv"))

    print(f"[DataLoader] Loaded {len(ratings_df):,} ratings and {len(movies_df):,} movies.")
    return ratings_df, movies_df


def build_user_item_matrix(
    ratings_df: pd.DataFrame,
    user_to_idx: Optional[Dict[int, int]] = None,
    movie_to_idx: Optional[Dict[int, int]] = None
) -> Tuple[csr_matrix, Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Constructs a compressed sparse row (CSR) user-item matrix from ratings DataFrame.

    Args:
        ratings_df: DataFrame with ['userId', 'movieId', 'rating']
        user_to_idx: Optional predefined mapping of userId -> row index
        movie_to_idx: Optional predefined mapping of movieId -> col index

    Returns:
        matrix: csr_matrix of shape (n_users, n_movies)
        user_to_idx: dict mapping userId -> matrix row index
        idx_to_user: dict mapping matrix row index -> userId
        movie_to_idx: dict mapping movieId -> matrix column index
        idx_to_movie: dict mapping matrix column index -> movieId
    """
    if user_to_idx is None:
        unique_users = sorted(ratings_df["userId"].unique())
        user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    idx_to_user = {i: uid for uid, i in user_to_idx.items()}

    if movie_to_idx is None:
        unique_movies = sorted(ratings_df["movieId"].unique())
        movie_to_idx = {mid: i for i, mid in enumerate(unique_movies)}
    idx_to_movie = {i: mid for mid, i in movie_to_idx.items()}

    valid_ratings = ratings_df[
        ratings_df["userId"].isin(user_to_idx) & ratings_df["movieId"].isin(movie_to_idx)
    ]

    rows = valid_ratings["userId"].map(user_to_idx).to_numpy()
    cols = valid_ratings["movieId"].map(movie_to_idx).to_numpy()
    data = valid_ratings["rating"].to_numpy(dtype=np.float32)

    n_users = len(user_to_idx)
    n_movies = len(movie_to_idx)

    sparse_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_movies), dtype=np.float32)

    return sparse_matrix, user_to_idx, idx_to_user, movie_to_idx, idx_to_movie
