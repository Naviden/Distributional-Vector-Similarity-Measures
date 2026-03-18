"""
T1 - Sentence Similarity (Table 4)
====================================
Evaluates 15 similarity/distance measures on the STS Benchmark test set
across 6 sentence-transformer embedding models.

Output: results/STS-B_results.xlsx   (one sheet per model)

Usage:
    python run_sts.py
"""

import os, sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# Allow importing measures from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import measures

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Configuration ────────────────────────────────────────────────────────────
EMB_MODELS = [
    "all-mpnet-base-v2",
    "all-distilroberta-v1",
    "all-MiniLM-L12-v2",
    "all-MiniLM-L6-v2",
    "multi-qa-mpnet-base-dot-v1",
    "paraphrase-MiniLM-L6-v2",
]

METRIC_NAMES = [
    "cosine", "euclidean", "manhattan", "chebyshev", "canberra",
    "bray_curtis", "pearson", "spearman", "jaccard", "dice",
    "lin", "kl_divergence", "js_divergence", "dot_product",
    "kulczynski", "sentic_path",
]


def encode_sentence(sentence, tokenizer, model):
    """Mean-pooling sentence embedding."""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()


def compute_similarity(v1, v2, metric):
    """Return a scalar similarity score (higher = more similar)."""
    dispatch = {
        "cosine":        lambda: -measures.compute_cosine(v1, v2),
        "euclidean":     lambda: -measures.compute_euclidean(v1, v2),
        "manhattan":     lambda: -measures.compute_manhattan(v1, v2),
        "chebyshev":     lambda: -measures.compute_chebyshev(v1, v2),
        "canberra":      lambda: -measures.compute_canberra(v1, v2),
        "bray_curtis":   lambda: -measures.compute_bray_curtis(v1, v2),
        "jaccard":       lambda:  measures.compute_jaccard_similarity(v1, v2),
        "dice":          lambda:  measures.compute_dice_similarity(v1, v2),
        "lin":           lambda:  measures.compute_lin_similarity(v1, v2),
        "kl_divergence": lambda: -measures.compute_kl_divergence(v1, v2),
        "js_divergence": lambda: -measures.compute_js_divergence(v1, v2),
        "pearson":       lambda:  measures.compute_pearson(v1, v2),
        "spearman":      lambda:  measures.compute_spearman(v1, v2),
        "dot_product":   lambda:  measures.compute_dot_product(v1, v2),
        "kulczynski":    lambda:  measures.compute_kulczynski(v1, v2),
        "sentic_path":   lambda:  measures.compute_sentic_path(v1, v2),
    }
    return dispatch[metric]()


def evaluate_similarity(data, tokenizer, model, metric):
    """Return (Spearman, Pearson) correlation with gold scores."""
    predictions, labels = [], []
    for sent1, sent2, label in zip(data["sentence1"], data["sentence2"], data["score"]):
        if not sent1 or not sent2:
            continue
        vec1 = encode_sentence(sent1, tokenizer, model)
        vec2 = encode_sentence(sent2, tokenizer, model)
        try:
            sim = compute_similarity(vec1, vec2, metric)
        except Exception as e:
            print(f"  [skip] {metric}: {e}")
            continue
        predictions.append(sim)
        labels.append(label)

    predictions = np.array(predictions)
    labels = np.array(labels)
    valid = ~np.isnan(predictions) & ~np.isinf(predictions)
    predictions, labels = predictions[valid], labels[valid]

    return spearmanr(predictions, labels).correlation, pearsonr(predictions, labels)[0]


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dataset = load_dataset("sentence-transformers/stsb")
    subset = dataset["test"]

    os.makedirs("results", exist_ok=True)
    writer = pd.ExcelWriter("results/STS-B_results.xlsx")

    for emb_model in EMB_MODELS:
        print(f"\n{'='*60}\nModel: {emb_model}\n{'='*60}")
        model_name = f"sentence-transformers/{emb_model}"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        rows = []
        for metric in METRIC_NAMES:
            sp, pr = evaluate_similarity(subset, tokenizer, model, metric)
            print(f"  {metric:<20s}  Spearman={sp:.4f}  Pearson={pr:.4f}")
            rows.append({"Metric": metric, "Spearman_corr": sp, "Pearson_corr": pr})

        pd.DataFrame(rows).to_excel(writer, sheet_name=emb_model, index=False)

    writer.close()
    print("\nSaved  results/STS-B_results.xlsx")
