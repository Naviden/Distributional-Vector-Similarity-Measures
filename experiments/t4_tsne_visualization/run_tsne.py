"""
T4 - t-SNE Visualisation (Figures 3-4)
========================================
Generates t-SNE plots using precomputed pairwise distance matrices for 15
similarity/distance measures on the 20 Newsgroups and Reuters-21578 corpora.

Embeddings: all-MiniLM-L6-v2 (CLS token)

Output: results/{dataset}_{metric}.pdf  +  results/silhouette_scores.csv

Usage:
    python run_tsne.py                       # runs both datasets
    python run_tsne.py --dataset 20newsgroups
    python run_tsne.py --dataset reuters

Note:
    Requires MulticoreTSNE (pip install MulticoreTSNE).
    Computing pairwise distance matrices is CPU-intensive; the script
    caches embeddings as .pkl files and skips already-generated plots.
"""

import os, sys, csv, pickle, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from itertools import combinations_with_replacement
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import silhouette_score
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset as hf_load

# Allow importing measures from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from measures import (
    compute_kl_divergence, compute_js_divergence, compute_jaccard_similarity,
    compute_dice_similarity, compute_lin_similarity, compute_sentic_path,
    compute_cosine, compute_euclidean, compute_manhattan, compute_chebyshev,
    compute_canberra, compute_bray_curtis, compute_pearson, compute_spearman,
    compute_kulczynski, compute_dot_product,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

METRICS = [
    compute_kl_divergence, compute_js_divergence, compute_jaccard_similarity,
    compute_dice_similarity, compute_lin_similarity, compute_sentic_path,
    compute_cosine, compute_euclidean, compute_manhattan, compute_chebyshev,
    compute_canberra, compute_bray_curtis, compute_pearson, compute_spearman,
    compute_kulczynski, compute_dot_product,
]

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N_JOBS = 8  # parallel workers for pairwise distance computation


# ── Embedding loaders ────────────────────────────────────────────────────────

def _encode_texts(texts, tokenizer, model):
    """CLS-token embeddings via Hugging Face transformer."""
    embeddings = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, padding=True, truncation=True,
                               return_tensors="pt", max_length=512)
            out = model(**inputs)
            embeddings.append(out.last_hidden_state[:, 0, :].squeeze().numpy())
    return np.array(embeddings)


def load_20ng(samples_per_category=1000, cache="20NG_embeddings.pkl"):
    if os.path.exists(cache):
        print(f"Loading cached embeddings: {cache}")
        with open(cache, "rb") as f:
            d = pickle.load(f)
        return d["embeddings"], d["labels"], d["categories"]

    print("Fetching 20 Newsgroups...")
    ng = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    categories = ng.target_names
    texts, labels = [], []
    for idx, cat in enumerate(categories):
        idxs = np.where(ng.target == idx)[0][:samples_per_category]
        texts.extend([ng.data[i] for i in idxs])
        labels.extend([cat] * len(idxs))

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModel.from_pretrained(HF_MODEL)
    embeddings = _encode_texts(texts, tokenizer, model)

    with open(cache, "wb") as f:
        pickle.dump({"embeddings": embeddings, "labels": labels, "categories": categories}, f)
    return embeddings, labels, categories


def load_reuters(cache="reuters_embeddings.pkl"):
    if os.path.exists(cache):
        print(f"Loading cached embeddings: {cache}")
        with open(cache, "rb") as f:
            d = pickle.load(f)
        return d["embeddings"], d["labels"], d["categories"]

    print("Fetching Reuters-21578...")
    ds = hf_load("yangwang825/reuters-21578")
    texts = ds["train"]["text"]
    labels = ds["train"]["label"]
    categories = sorted(set(labels))

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModel.from_pretrained(HF_MODEL)
    embeddings = _encode_texts(texts, tokenizer, model)

    with open(cache, "wb") as f:
        pickle.dump({"embeddings": embeddings, "labels": labels, "categories": categories}, f)
    return embeddings, labels, categories


# ── Distance computation ─────────────────────────────────────────────────────

def _compute_one(args):
    i, j, emb_i, emb_j, func = args
    return i, j, func(emb_i, emb_j)


def parallel_distance_matrix(embeddings, func, n_jobs=N_JOBS):
    n = len(embeddings)
    mat = np.zeros((n, n), dtype=np.float32)
    pairs = [(i, j, embeddings[i], embeddings[j], func)
             for i, j in combinations_with_replacement(range(n), 2)]
    with multiprocessing.Pool(processes=n_jobs) as pool:
        for i, j, d in pool.map(_compute_one, pairs):
            mat[i, j] = d
            mat[j, i] = d
    return mat


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_tsne(embeddings, labels, dataset_name, n_palette=10):
    """Generate per-metric t-SNE PDFs and silhouette scores."""
    from MulticoreTSNE import MulticoreTSNE

    os.makedirs("results", exist_ok=True)
    csv_path = "results/silhouette_scores.csv"
    file_exists = os.path.isfile(csv_path)
    palette = sns.color_palette("tab10", n_palette)
    perplexity = min(len(labels) - 1, 30)

    with open(csv_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(["dataset_name", "measure", "score"])

        for func in METRICS:
            name = func.__name__.replace("compute_", "")
            save_path = f"results/{dataset_name}_{name}.pdf"

            if os.path.exists(save_path):
                print(f"  [skip] {name} (exists)")
                continue

            print(f"  {name} ...", end=" ", flush=True)
            try:
                dist = parallel_distance_matrix(embeddings, func)
                tsne = MulticoreTSNE(n_components=2, metric="precomputed",
                                     init="random", n_jobs=-1, random_state=42)
                red = tsne.fit_transform(dist)

                sil = silhouette_score(red, labels)
                writer.writerow([dataset_name, name, sil])
                print(f"silhouette={sil:.4f}")

                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=red[:, 0], y=red[:, 1], hue=labels,
                                palette=palette, alpha=0.3)
                plt.legend([], [], frameon=False)
                plt.xticks([], [])
                plt.yticks([], [])
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0)
                plt.close()

            except Exception as e:
                print(f"ERROR: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["20newsgroups", "reuters", "both"],
                        default="both")
    args = parser.parse_args()

    if args.dataset in ("20newsgroups", "both"):
        print("\n=== 20 Newsgroups ===")
        emb, lab, cats = load_20ng()
        plot_tsne(emb, lab, "20newsgroups", n_palette=10)

    if args.dataset in ("reuters", "both"):
        print("\n=== Reuters-21578 ===")
        emb, lab, cats = load_reuters()
        plot_tsne(emb, lab, "reuters", n_palette=8)

    print("\nDone.")
