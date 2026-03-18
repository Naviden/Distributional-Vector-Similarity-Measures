"""
Vector Similarity and Distance Measures
========================================
Implementations of 15 similarity/distance measures evaluated in:

"Benchmarking Distributional Vector Similarity Measures: A Survey"

"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import (
    cosine, euclidean, cityblock, chebyshev, canberra, braycurtis, kulczynski1
)


# ── Distance-based measures ──────────────────────────────────────────────────

def compute_cosine(v1, v2):
    """Cosine distance (scipy): 1 - cos(v1, v2)."""
    return cosine(v1, v2)


def compute_euclidean(v1, v2):
    """Euclidean (L2) distance."""
    return euclidean(v1, v2)


def compute_manhattan(v1, v2):
    """Manhattan (L1 / city-block) distance."""
    return cityblock(v1, v2)


def compute_chebyshev(v1, v2):
    """Chebyshev (L-infinity) distance."""
    return chebyshev(v1, v2)


def compute_canberra(v1, v2):
    """Canberra distance."""
    return canberra(v1, v2)


def compute_bray_curtis(v1, v2):
    """Bray-Curtis dissimilarity."""
    return braycurtis(v1, v2)


# ── Projection-based measures ────────────────────────────────────────────────

def compute_dot_product(vec1, vec2):
    """Dot product similarity."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same shape")
    return np.dot(vec1, vec2)


# ── Correlation-based measures ───────────────────────────────────────────────

def compute_spearman(v1, v2):
    """Spearman rank correlation."""
    return spearmanr(v1, v2).correlation


def compute_pearson(v1, v2):
    """Pearson correlation."""
    return pearsonr(v1, v2).correlation


# ── Set-based measures ───────────────────────────────────────────────────────

def compute_jaccard_similarity(v1, v2):
    """Jaccard (Tanimoto) coefficient for continuous vectors."""
    vec1 = np.array(v1, dtype=np.float64)
    vec2 = np.array(v2, dtype=np.float64)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.dot(vec1, vec1)
    norm_vec2 = np.dot(vec2, vec2)
    return dot_product / (norm_vec1 + norm_vec2 - dot_product)


def compute_dice_similarity(v1, v2):
    """Dice coefficient for continuous vectors."""
    vec1 = np.array(v1, dtype=np.float64)
    vec2 = np.array(v2, dtype=np.float64)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.dot(vec1, vec1)
    norm_vec2 = np.dot(vec2, vec2)
    return 2 * dot_product / (norm_vec1 + norm_vec2)


def compute_lin_similarity(v1, v2):
    """Lin similarity between two dense vectors."""
    u = np.array(v1)
    v = np.array(v2)
    if u.shape != v.shape:
        raise ValueError("Vectors u and v must have the same dimensions.")
    numerator = np.sum(np.minimum(u, v))
    denominator = np.sum(u) + np.sum(v)
    if denominator == 0:
        return 0.0
    return (2 * numerator) / denominator


def compute_sentic_path(v1, v2, penalty_function=None):
    """Sentic Path similarity (inverse of penalised Euclidean distance)."""
    vector1 = np.array(v1)
    vector2 = np.array(v2)
    raw_distance = euclidean(vector1, vector2)
    if penalty_function:
        penalized_distance = penalty_function(vector1, vector2, raw_distance)
    else:
        penalized_distance = raw_distance
    return 1 / (1 + penalized_distance)


def compute_kulczynski(v1, v2):
    """Kulczynski dissimilarity (scipy)."""
    return kulczynski1(v1, v2)


# ── Information-theoretic measures ───────────────────────────────────────────

def compute_kl_divergence(v1, v2):
    """Kullback-Leibler divergence (after normalisation to probability distributions)."""
    p = np.array(v1, dtype=np.float64)
    q = np.array(v2, dtype=np.float64)
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    p /= p.sum()
    q /= q.sum()
    return np.sum(p * np.log(p / q))


def compute_js_divergence(v1, v2):
    """Jensen-Shannon divergence."""
    m = 0.5 * (np.array(v1) + np.array(v2))
    return 0.5 * (compute_kl_divergence(v1, m) + compute_kl_divergence(v2, m))
