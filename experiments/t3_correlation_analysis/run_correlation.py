"""
T3 - Correlation Analysis (Figure 2)
======================================
Computes pairwise Pearson correlation between similarity/distance values
produced by 15 measures on three word-similarity benchmarks:
  - SimLex-999
  - MEN
  - WordSim-353

Embeddings: paraphrase-MiniLM-L6-v2

Output: results/{dataset}_corr.pdf  (lower-triangle heatmaps)

Usage:
    python run_correlation.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

# Allow importing measures from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from measures import (
    compute_cosine, compute_euclidean, compute_manhattan, compute_chebyshev,
    compute_canberra, compute_bray_curtis, compute_pearson, compute_spearman,
    compute_jaccard_similarity, compute_dice_similarity, compute_lin_similarity,
    compute_kl_divergence, compute_js_divergence, compute_dot_product,
    compute_sentic_path,
)

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"

METRICS = [
    compute_cosine, compute_euclidean, compute_manhattan, compute_chebyshev,
    compute_canberra, compute_bray_curtis, compute_pearson, compute_spearman,
    compute_jaccard_similarity, compute_dice_similarity, compute_lin_similarity,
    compute_kl_divergence, compute_js_divergence, compute_dot_product,
    compute_sentic_path,
]

DATASETS = [
    {
        "name": "SimLex",
        "file": "data/SimLex-999.txt",
        "w1": "word1", "w2": "word2", "sim": "SimLex999",
        "sep": "\t",
    },
    {
        "name": "MEN",
        "file": "data/MEN_dataset_lemma_form_full.csv",
        "w1": "Word1", "w2": "Word2", "sim": "sim",
        "sep": ",",
    },
    {
        "name": "WordSim353",
        "file": "data/wordsim353crowd.csv",
        "w1": "Word 1", "w2": "Word 2", "sim": "Human (Mean)",
        "sep": ",",
    },
]


def run_dataset(cfg, model):
    """Compute all pairwise similarities and produce a correlation heatmap."""
    df = pd.read_csv(cfg["file"], sep=cfg["sep"])
    df = df[[cfg["w1"], cfg["w2"], cfg["sim"]]]

    # Encode unique words once
    unique_words = set(df[cfg["w1"]]).union(set(df[cfg["w2"]]))
    embeddings = {w: model.encode(w) for w in unique_words}

    # Compute similarity values per metric
    similarity_funcs = {v.__name__.replace("compute_", ""): v for v in METRICS}
    results = {name: [] for name in similarity_funcs}

    for name, func in similarity_funcs.items():
        for _, row in df.iterrows():
            vec1, vec2 = embeddings[row[cfg["w1"]]], embeddings[row[cfg["w2"]]]
            results[name].append(func(vec1, vec2))

    sim_df = pd.DataFrame(results)
    corr = sim_df.corr()

    # Plot lower-triangle heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, cbar=False, mask=mask)

    labels = corr.columns.tolist()
    ticks = np.arange(len(labels)) + 0.5
    plt.xticks(ticks=ticks, labels=labels, rotation=90, ha="right", fontsize=18)
    plt.yticks(ticks=ticks, labels=labels, rotation=0, fontsize=18)
    plt.tight_layout()

    out = f"results/{cfg['name']}_corr.pdf"
    plt.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"  Saved {out}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)

    for cfg in DATASETS:
        print(f"Processing {cfg['name']}...")
        run_dataset(cfg, model)

    print("\nDone.")
