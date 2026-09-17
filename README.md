# MovieLens Collaborative Filtering Recommender

A modular machine learning recommendation engine built on the **MovieLens 100k** dataset (`ml-latest-small`). This repository implements, compares, and evaluates two foundational collaborative filtering paradigms:

1. **Memory-Based Collaborative Filtering**: User-Based K-Nearest Neighbors (KNN) using Cosine Similarity.
2. **Model-Based Collaborative Filtering**: Latent Factor Matrix Factorization via Truncated Singular Value Decomposition (TruncatedSVD) with user-mean centering.

---

## Architecture & Project Structure

```text
movielens-collaborative-recommender/
├── .gitignore              # Ignores virtualenv, dataset caches, zip archives, bytecode
├── requirements.txt        # Core dependencies (pandas, numpy, scipy, scikit-learn)
├── README.md               # Project documentation and mathematical guide
├── evaluate.py             # CLI runner: train/test split, RMSE/MAE evaluation, recommendations
└── src/
    ├── __init__.py         # Package initialization
    ├── data_loader.py      # Automated dataset downloader, zip extraction, and CSR matrix builder
    └── recommender.py      # KNNRecommender and SVDRecommender implementations
```

---

## Mathematical Foundations

### 1. User-Based K-Nearest Neighbors (KNN)

KNN is a **memory-based** (or neighborhood-based) collaborative filtering method that leverages historical rating patterns across similar users.

#### Cosine Similarity Metric
For two users $\mathbf{u}$ and $\mathbf{v}$ represented by their rating vectors in the item space $\mathbb{R}^M$, the cosine similarity measures the cosine of the angle between them:

$$\text{sim}(\mathbf{u}, \mathbf{v}) = \cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^M r_{u,i} r_{v,i}}{\sqrt{\sum_{i=1}^M r_{u,i}^2} \sqrt{\sum_{i=1}^M r_{v,i}^2}}$$

`sklearn.neighbors.NearestNeighbors` computes the cosine distance:

$$d_{\text{cosine}}(\mathbf{u}, \mathbf{v}) = 1 - \text{sim}(\mathbf{u}, \mathbf{v})$$

The similarity is recovered as:

$$\text{sim}(\mathbf{u}, \mathbf{v}) = 1 - d_{\text{cosine}}(\mathbf{u}, \mathbf{v})$$

#### Rating Prediction & Neighbor Aggregation
To predict user $u$'s rating for an unseen movie $i$, the model selects the subset of user $u$'s $k$ nearest neighbors $N_k(u)$ who have rated movie $i$ ($r_{v, i} > 0$). The estimated rating $\hat{r}_{u, i}$ is computed via a similarity-weighted average:

$$\hat{r}_{u, i} = \frac{\sum_{v \in N_k(u; i)} \text{sim}(u, v) \cdot r_{v, i}}{\sum_{v \in N_k(u; i)} \text{sim}(u, v)}$$

If no neighbor has rated movie $i$, the algorithm smoothly falls back to user $u$'s historical mean rating $\bar{r}_u$, or the global mean $\mu$ if the user has no history. Predicted ratings are bounded within $[0.5, 5.0]$.

---

### 2. Latent Factor Matrix Factorization (TruncatedSVD)

TruncatedSVD is a **model-based** collaborative filtering method that maps both users and items to a shared low-dimensional latent space $\mathbb{R}^k$.

#### Matrix Decomposition
Let $R \in \mathbb{R}^{N \times M}$ denote the user-item interaction matrix ($N$ users, $M$ items). SVD decomposes $R$ into:

$$R \approx U_k \Sigma_k V_k^T$$

where:
- $U_k \in \mathbb{R}^{N \times k}$ represents orthonormal user eigenvectors.
- $\Sigma_k \in \mathbb{R}^{k \times k}$ contains the top-$k$ singular values.
- $V_k \in \mathbb{R}^{M \times k}$ contains orthonormal item eigenvectors.

We define:
- **User Latent Factor Matrix**: $P = U_k \Sigma_k \in \mathbb{R}^{N \times k}$ (computed via `TruncatedSVD.fit_transform`)
- **Item Latent Factor Matrix**: $Q = V_k^T \in \mathbb{R}^{k \times M}$ (stored in `TruncatedSVD.components_`)

#### Handling Matrix Sparsity via User-Mean Centering
The MovieLens matrix is **98.30% sparse**. Directly applying SVD to unobserved zero entries treats missing entries as zero ratings, severely skewing factor projections. To prevent this, observed ratings are centered around each user's mean $\bar{r}_u$:

$$R_{\text{centered}}[u, i] = \begin{cases} r_{u,i} - \bar{r}_u & \text{if } (u, i) \text{ is observed} \\ 0 & \text{if } (u, i) \text{ is unobserved (missing)} \end{cases}$$

Setting unobserved entries to $0$ in $R_{\text{centered}}$ represents the neutral baseline assumption that unobserved ratings deviate zero points from the user's average.

#### Reconstructed Rating Prediction
The predicted rating for user $u$ on item $i$ is calculated by projecting user factors against item factors and adding back the user baseline mean:

$$\hat{r}_{u, i} = \bar{r}_u + \mathbf{p}_u \cdot \mathbf{q}_i = \bar{r}_u + \sum_{d=1}^k P_{u, d} Q_{d, i}$$

---

## Comparison: KNN vs. SVD

| Dimension | User-Based KNN | Latent Factor TruncatedSVD |
| :--- | :--- | :--- |
| **Paradigm** | Memory-based (instance-based / lazy learning) | Model-based (dimensionality reduction) |
| **Model Size** | Stores full interaction matrix: $\mathcal{O}(N \times M)$ | Compact low-rank representation: $\mathcal{O}((N + M) \times k)$ |
| **Fit Complexity** | Fast tree/brute-force index build | Matrix decomposition via randomized SVD |
| **Inference Speed** | Slower: requires searching neighbors and looking up ratings | Extremely fast: single inner product $\mathbf{p}_u \cdot \mathbf{q}_i$ |
| **Handling Sparsity** | Suffers if users share few overlapping rated items | Captures latent semantic themes even with zero direct overlap |
| **Interpretability** | **High**: "Recommended because User X & Y also loved it" | **Moderate**: Latent dimensions represent abstract genre/thematic blends |
| **Typical RMSE** | $\approx 0.992$ | $\approx 0.923$ (Lower error) |

---

## Setup & Installation

### 1. Prerequisites
- Python 3.9+ (tested on Python 3.12)

### 2. Environment Configuration
Clone the repository and set up a virtual environment:

```bash
# Navigate to project directory
cd movielens-collaborative-recommender

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### Run Evaluation and Demo
Execute `evaluate.py` to automatically download the dataset, train both recommenders on an 80/20 train/test split, print the RMSE/MAE comparison table, and generate Top-5 recommendations for User 1:

```bash
python evaluate.py
```

### CLI Customization Options
You can configure hyperparameters, evaluation split sizes, or target users:

```bash
python evaluate.py --user_id 1 --n_neighbors 30 --n_components 20 --test_size 0.2
```

Options:
- `--user_id`: Target user ID to display recommendations for (default: `1`).
- `--test_size`: Fraction of ratings reserved for testing (default: `0.2`).
- `--n_neighbors`: Neighborhood size $k$ for KNN (default: `30`).
- `--n_components`: Number of latent dimensions for SVD (default: `20`).

---

## Benchmark Results

Evaluated on the official MovieLens 100k test split ($N_{\text{test}} = 20,168$ ratings, `random_state=42`):

```text
+-------------------------+--------+--------+--------------+------------------+------------------------+
| Algorithm               | RMSE   | MAE    | Fit Time (s) | Predict Time (s) | Complexity             |
+-------------------------+--------+--------+--------------+------------------+------------------------+
| KNN (Cosine Similarity) | 0.9924 | 0.7604 | 0.044        | 0.188            | O(U * M) Memory        |
| SVD (Truncated Latent)  | 0.9229 | 0.7117 | 0.056        | 0.011            | O((U + M) * k) Factors |
+-------------------------+--------+--------+--------------+------------------+------------------------+

Result: SVD outperforms KNN by 0.0695 RMSE (7.00% lower prediction error).
```

### Sample Recommendations for User 1

#### Target User Profile (Top Rated Films in Training Set):
- *The Terminator (1984)* - Action / Sci-Fi / Thriller (5.0 ★)
- *The Messenger: The Story of Joan of Arc (1999)* - Drama / War (5.0 ★)
- *Desperado (1995)* - Action / Romance / Western (5.0 ★)
- *The Rescuers (1977)* - Adventure / Animation (5.0 ★)

#### Top-5 Recommendations:

**KNN (Cosine Similarity)**:
1. **The Commitments (1991)** (ID: 3060) — Pred: 5.00 ★ | *Comedy, Drama, Musical*
2. **Rosencrantz and Guildenstern Are Dead (1990)** (ID: 1243) — Pred: 5.00 ★ | *Comedy, Drama*
3. **Little Big Man (1970)** (ID: 3037) — Pred: 5.00 ★ | *Western*
4. **Three Colors: Red (1994)** (ID: 306) — Pred: 5.00 ★ | *Drama*
5. **The General (1926)** (ID: 3022) — Pred: 5.00 ★ | *Comedy, War*

**SVD (Latent Factor Decomposition)**:
1. **The Shawshank Redemption (1994)** (ID: 318) — Pred: 4.61 ★ | *Crime, Drama*
2. **The Godfather (1972)** (ID: 858) — Pred: 4.54 ★ | *Crime, Drama*
3. **Terminator 2: Judgment Day (1991)** (ID: 589) — Pred: 4.54 ★ | *Action, Sci-Fi*
4. **Twelve Monkeys (1995)** (ID: 32) — Pred: 4.52 ★ | *Mystery, Sci-Fi, Thriller*
5. **Raiders of the Lost Ark (1981)** (ID: 1198) — Pred: 4.50 ★ | *Action, Adventure*

*Insight*: SVD captures latent semantic preferences, recommending *Terminator 2: Judgment Day* and *Twelve Monkeys*, reflecting User 1's demonstrated affinity for James Cameron action and sci-fi films.

---

## Programmatic API Example

```python
from src.data_loader import load_raw_data, build_user_item_matrix
from src.recommender import SVDRecommender, KNNRecommender

# 1. Load data and construct matrix
ratings_df, movies_df = load_raw_data()
matrix, user_to_idx, idx_to_user, movie_to_idx, idx_to_movie = build_user_item_matrix(ratings_df)

# 2. Train SVD
svd = SVDRecommender(n_components=20)
svd.fit(matrix, user_to_idx, movie_to_idx, movies_df)

# 3. Predict rating & recommend
pred = svd.predict_rating(user_id=1, movie_id=318)
print(f"Predicted rating for User 1 on Movie 318: {pred:.2f}")

top_recs = svd.recommend(user_id=1, top_n=5)
for movie in top_recs:
    print(f"{movie['title']} - Predicted: {movie['predicted_rating']} ★")
```
