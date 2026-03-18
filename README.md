# Benchmarking Distributional Vector Similarity Measures: A Survey

Code for reproducing the experimental results in:

> **Benchmarking Distributional Vector Similarity Measures: A Survey**
> Erik Cambria, Navid Nobani, Filippo Pallucchini, Fabio Mercurio

## Repository structure

```
.
├── measures.py                          # 15 similarity/distance measure implementations
├── requirements.txt
├── experiments/
│   ├── t1_sentence_similarity/          # Table 4  – STS Benchmark
│   │   └── run_sts.py
│   ├── t2_knn_classification/           # Table 5  – kNN on IMDB
│   │   └── run_knn.py
│   ├── t3_correlation_analysis/         # Figure 2 – Measure correlation heatmaps
│   │   ├── run_correlation.py
│   │   └── data/                        # SimLex-999, MEN, WordSim-353
│   └── t4_tsne_visualization/           # Figures 3-4 – t-SNE plots
│       └── run_tsne.py
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

**Note:** The t-SNE experiment (T4) requires [`MulticoreTSNE`](https://github.com/DmitryUlyanov/Multicore-TSNE). On some systems you may need `cmake` installed first.

## Reproducing results

All scripts download datasets automatically via Hugging Face `datasets`.

### T1 – Sentence Similarity (Table 4)

Evaluates 15 measures on the STS-B test set across 6 embedding models. Reports Spearman and Pearson correlations.

```bash
cd experiments/t1_sentence_similarity
python run_sts.py
# Output: results/STS-B_results.xlsx
```

### T2 – kNN Classification (Table 5)

kNN (k=5) classification on a 5,000-sample subset of IMDB with `paraphrase-MiniLM-L6-v2`.

```bash
cd experiments/t2_knn_classification
python run_knn.py
# Output: results/KNN_results_5000.csv
```

### T3 – Correlation Analysis (Figure 2)

Pairwise Pearson correlation heatmaps of measure outputs on SimLex-999, MEN, and WordSim-353.

```bash
cd experiments/t3_correlation_analysis
python run_correlation.py
# Output: results/{SimLex,MEN,WordSim353}_corr.pdf
```

### T4 – t-SNE Visualisation (Figures 3-4)

t-SNE scatter plots of 20 Newsgroups and Reuters-21578 using each measure as a precomputed distance metric. Also outputs silhouette scores.

```bash
cd experiments/t4_tsne_visualization
python run_tsne.py                        # both datasets
python run_tsne.py --dataset 20newsgroups # or individually
python run_tsne.py --dataset reuters
# Output: results/{dataset}_{metric}.pdf, results/silhouette_scores.csv
```

**Note:** The t-SNE experiments compute O(n^2) pairwise distances and are CPU-intensive. Embedding caches (`.pkl`) are created on first run to speed up reruns.

## Measures implemented

| Family | Measure | Function |
|--------|---------|----------|
| Distance | Cosine, Euclidean, Manhattan, Chebyshev, Canberra, Bray-Curtis | `compute_cosine`, ... |
| Projection | Dot Product | `compute_dot_product` |
| Correlation | Pearson, Spearman | `compute_pearson`, `compute_spearman` |
| Set-based | Jaccard, Dice, Lin, Sentic Path, Kulczynski | `compute_jaccard_similarity`, ... |
| Information-theoretic | KL Divergence, JS Divergence | `compute_kl_divergence`, `compute_js_divergence` |


## License

This code is released for academic and research purposes.
