"""
T2 - kNN Classification (Table 5)
===================================
Evaluates kNN (k=5) classification on IMDB using 15 similarity/distance
measures as custom metrics with paraphrase-MiniLM-L6-v2 embeddings.

The paper reports results on a random 5,000-sample subset (seed=42).

Output: results/KNN_results_5000.csv

Usage:
    python run_knn.py
"""

import os, sys
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Allow importing measures from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from measures import (
    compute_kl_divergence, compute_js_divergence, compute_jaccard_similarity,
    compute_dice_similarity, compute_lin_similarity, compute_sentic_path,
    compute_cosine, compute_euclidean, compute_manhattan, compute_chebyshev,
    compute_canberra, compute_bray_curtis, compute_pearson, compute_spearman,
    compute_kulczynski, compute_dot_product,
)

# ── Configuration ────────────────────────────────────────────────────────────
SAMPLE_SIZE = 5000
SEED = 42
K = 5
TEST_RATIO = 0.3

METRICS = [
    compute_cosine, compute_euclidean, compute_manhattan, compute_chebyshev,
    compute_canberra, compute_bray_curtis, compute_pearson, compute_spearman,
    compute_jaccard_similarity, compute_dice_similarity, compute_lin_similarity,
    compute_kl_divergence, compute_js_divergence, compute_dot_product,
    compute_kulczynski, compute_sentic_path,
]

# ── Data preparation (done once, shared across workers) ──────────────────────
dataset = load_dataset("imdb")
imdb = pd.DataFrame({"text": dataset["train"]["text"],
                      "label": dataset["train"]["label"]})
imdb = imdb.sample(SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    imdb["text"], imdb["label"], test_size=TEST_RATIO, random_state=SEED
)


def process_metric(metric_func):
    """Encode with paraphrase-MiniLM-L6-v2 and run kNN for one metric."""
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    train_emb = model.encode(train_texts.tolist(), convert_to_tensor=False)
    test_emb = model.encode(test_texts.tolist(), convert_to_tensor=False)

    knn = KNeighborsClassifier(n_neighbors=K, metric=metric_func, n_jobs=8)
    knn.fit(train_emb, train_labels)
    preds = knn.predict(test_emb)

    return {
        "Metric": metric_func.__name__,
        "Accuracy": round(accuracy_score(test_labels, preds), 2),
        "Precision": round(precision_score(test_labels, preds, average="binary"), 2),
        "Recall": round(recall_score(test_labels, preds, average="binary"), 2),
        "F1 Score": round(f1_score(test_labels, preds, average="binary"), 2),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    results = []

    with ProcessPoolExecutor() as pool:
        futures = {pool.submit(process_metric, m): m for m in METRICS}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="kNN metrics"):
            results.append(future.result())

    df = pd.DataFrame(results)
    df.to_csv(f"results/KNN_results_{SAMPLE_SIZE}.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSaved  results/KNN_results_{SAMPLE_SIZE}.csv")
