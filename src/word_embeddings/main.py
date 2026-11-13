"""End-to-end pipeline: BBC News -> TF-IDF embeddings -> PCA from scratch.

Run with:  python -m word_embeddings.main  [--max-features 1000] [--data-path FILE]
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write PNGs without a display
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .cleaning import clean_dataset
from .data import CLEAN_CSV, KAGGLE_DATASET, load_raw
from .pca import PCA
from .tfidf import TfidfVectorizer

log = logging.getLogger("word_embeddings")

CATEGORY_COLORS = {
    "business": "#1f77b4",
    "entertainment": "#ff7f0e",
    "politics": "#2ca02c",
    "sport": "#d62728",
    "tech": "#9467bd",
}


def top_terms(component: np.ndarray, terms: np.ndarray, k: int = 10) -> dict[str, list]:
    """Terms with the largest positive and negative loadings on one component."""
    order = np.argsort(component)
    return {
        "positive": [[str(terms[i]), round(float(component[i]), 4)] for i in order[::-1][:k]],
        "negative": [[str(terms[i]), round(float(component[i]), 4)] for i in order[:k]],
    }


def classification_by_dimension(
    documents: Sequence[str], labels: np.ndarray,
    dims: tuple[int, ...] = (2, 5, 10, 50), seed: int = 42,
    max_features: int = 1000, min_df: int = 2,
) -> tuple[dict[str, float], dict]:
    """Topic-classification accuracy on a held-out 20% split after reducing to k dims.

    Split document text BEFORE fitting either TF-IDF or PCA. Vocabulary, IDF,
    centering and principal directions are learned from training articles only.
    Besides ``dims`` it also tests the k that keeps 50% of the training variance.
    """
    docs_tr, docs_te, y_tr, y_te = train_test_split(
        list(documents), labels, test_size=0.2, stratify=labels, random_state=seed)
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df)
    X_tr = vectorizer.fit_transform(docs_tr)
    X_te = vectorizer.transform(docs_te)
    results: dict[str, float] = {}
    pca = PCA(n_components=1).fit(X_tr)  # eigenvectors_ holds every direction
    k50 = pca.n_components_for_variance(0.5)
    retained_variance: dict[str, float] = {}
    for k in sorted(set(dims) | {k50}):
        if not 1 <= k <= X_tr.shape[1]:
            raise ValueError(f"Requested dimension {k} exceeds the training vocabulary")
        W = pca.eigenvectors_[:, :k]
        Z_tr, Z_te = (X_tr - pca.mean_) @ W, (X_te - pca.mean_) @ W
        clf = LogisticRegression(max_iter=2000).fit(Z_tr, y_tr)
        key = f"k={k} (50% of variance)" if k == k50 else f"k={k}"
        results[key] = round(float(clf.score(Z_te, y_te)), 4)
        retained_variance[key] = float(pca.cumulative_variance_ratio()[k - 1])
    clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    results[f"full ({X_tr.shape[1]} terms)"] = round(float(clf.score(X_te, y_te)), 4)
    classes, counts = np.unique(y_tr, return_counts=True)
    majority = classes[np.argmax(counts)]
    protocol = {
        "split": "stratified 80/20 split of cleaned document text",
        "random_state": seed,
        "n_train": len(docs_tr), "n_test": len(docs_te),
        "tfidf_fit_on": "training documents only (vocabulary and IDF)",
        "pca_fit_on": "training TF-IDF only (mean, covariance and eigenvectors)",
        "n_training_terms": X_tr.shape[1],
        "majority_class": str(majority),
        "majority_baseline_accuracy": float(np.mean(y_te == majority)),
        "components_for_50pct_training_variance": k50,
        "training_variance_retained_by_dimension": retained_variance,
        "classifier": "LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000)",
        "limitations": "One split; no hyperparameter tuning or uncertainty estimate. "
                      "Near-duplicate filtering only compares identical headlines.",
    }
    return results, protocol


def plot_scree(pca: PCA, out: Path, n_show: int = 30) -> None:
    n_show = min(n_show, len(pca.eigenvalues_))
    ratios = pca.eigenvalues_ / pca.eigenvalues_.sum()
    cumulative = pca.cumulative_variance_ratio()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.bar(np.arange(1, n_show + 1), ratios[:n_show] * 100, color="#4c72b0")
    ax1.set_xlabel("Principal component (sorted by eigenvalue)")
    ax1.set_ylabel("Variance explained (%)")
    ax1.set_title(f"Scree plot: first {n_show} components")

    ax2.plot(np.arange(1, len(cumulative) + 1), cumulative * 100, color="#dd8452")
    ax2.set_xlabel("Number of components kept")
    ax2.set_ylabel("Cumulative variance explained (%)")
    ax2.set_ylim(0, 101)
    ax2.set_xlim(0, len(cumulative))
    ax2.set_title("Cumulative explained variance")
    for thr in (0.5, 0.8, 0.9):
        k = pca.n_components_for_variance(thr)
        ax2.plot([k, k], [0, thr * 100], color="grey", lw=0.7, ls="--")
        ax2.plot([0, k], [thr * 100, thr * 100], color="grey", lw=0.7, ls="--")
        ax2.annotate(f"{int(thr * 100)}% at k={k}", (k, thr * 100),
                     textcoords="offset points", xytext=(6, -14), fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_projection(Z: np.ndarray, labels: np.ndarray, pca: PCA, out: Path) -> None:
    if Z.shape[1] < 2:
        raise ValueError("The 2-D plot needs at least two retained components")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for cat, color in CATEGORY_COLORS.items():
        mask = labels == cat
        ax.scatter(Z[mask, 0], Z[mask, 1], s=10, alpha=0.6, color=color, label=cat)
    r = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({r[0] * 100:.2f}% of variance)")
    ax.set_ylabel(f"PC2 ({r[1] * 100:.2f}% of variance)")
    ax.set_title("BBC News articles: TF-IDF vectors projected onto first two PCs")
    ax.legend(title="Category")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run(max_features: int, n_components: int, data_path: str | None, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Data: load raw file and clean it --------------------------------
    df, cleaning = clean_dataset(load_raw(data_path))
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns="text").to_csv(CLEAN_CSV, sep="\t", index=False, encoding="utf-8")
    (CLEAN_CSV.with_name("cleaning_report.json")).write_text(
        json.dumps(asdict(cleaning), indent=2, ensure_ascii=False), encoding="utf-8")
    labels = df["category"].to_numpy()
    log.info("Cleaned %d -> %d articles (exact dups %d, near dups %d): %s",
             cleaning.rows_in, cleaning.rows_out, cleaning.exact_duplicates_removed,
             cleaning.near_duplicates_removed, cleaning.category_counts_after)

    # ---- 2. TF-IDF embeddings ---------------------------------------------
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=2)
    X = vectorizer.fit_transform(df["text"].tolist())
    terms = vectorizer.get_feature_names_out()
    log.info("TF-IDF matrix X: %s (documents x terms)", X.shape)

    # ---- 3. PCA from scratch (fit on ALL components to study the spectrum) -
    pca = PCA(n_components=n_components).fit(X)
    Z = pca.transform(X)

    mean_order = np.argsort(pca.mean_)[::-1][:10]
    log.info("Step 1  mean vector shape %s; highest-mean terms: %s",
             pca.mean_.shape, [str(terms[i]) for i in mean_order])
    log.info("Step 2  centred data column means max |.| = %.2e",
             np.abs(pca.X_centered_.mean(axis=0)).max())
    log.info("Step 3  covariance matrix shape %s, symmetric=%s",
             pca.covariance_.shape, np.allclose(pca.covariance_, pca.covariance_.T))
    log.info("Step 4  top-10 eigenvalues: %s", np.round(pca.eigenvalues_[:10], 6))

    # ---- 4. Validate against NumPy and scikit-learn -----------------------
    cov_diff = float(np.abs(pca.covariance_ - np.cov(X, rowvar=False)).max())
    sk = SklearnPCA(n_components=n_components, svd_solver="full").fit(X)
    eig_diff = float(np.abs(pca.explained_variance_ - sk.explained_variance_).max())
    # Components may differ by sign only; compare |cosine| between matching PCs.
    cosines = np.abs(np.sum(pca.components_ * sk.components_, axis=1))
    log.info("Check   max|cov - np.cov| = %.2e, max|eigval - sklearn| = %.2e, "
             "min |cos(PC_ours, PC_sklearn)| = %.6f", cov_diff, eig_diff, cosines.min())

    # ---- 5. Information preserved -----------------------------------------
    accuracy, evaluation_protocol = classification_by_dimension(
        df["text"].tolist(), labels, max_features=max_features)
    log.info("Held-out topic accuracy by PCA dimension: %s", accuracy)
    total_var = float(pca.eigenvalues_.sum())
    reconstruction_mse = float(np.mean((X - pca.inverse_transform(Z)) ** 2))
    summary = {
        "dataset": {
            "source": f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}",
            "n_documents_raw": cleaning.rows_in,
            "n_documents": int(len(df)),
            "category_counts": {k: int(v) for k, v in df["category"].value_counts().items()},
            "median_words_per_article": int(df["content"].str.split().str.len().median()),
        },
        "tfidf": {"n_terms": int(X.shape[1]), "min_df": 2,
                  "matrix_shape": list(X.shape),
                  "sparsity": round(float((X == 0).mean()), 4)},
        "pca": {
            "n_components_kept": n_components,
            "total_variance": round(total_var, 6),
            "top10_eigenvalues": [round(float(v), 6) for v in pca.eigenvalues_[:10]],
            "explained_variance_ratio_kept": [round(float(v), 6)
                                              for v in pca.explained_variance_ratio_],
            "cumulative_variance_kept": round(float(pca.explained_variance_ratio_.sum()), 6),
            "components_for_50pct": pca.n_components_for_variance(0.5),
            "components_for_80pct": pca.n_components_for_variance(0.8),
            "components_for_90pct": pca.n_components_for_variance(0.9),
            "reconstruction_mse": reconstruction_mse,
            "highest_mean_terms": [str(terms[i]) for i in mean_order],
        },
        "heldout_topic_accuracy_by_dimension": accuracy,
        "evaluation_protocol": evaluation_protocol,
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "validation": {"max_abs_cov_diff_vs_numpy": cov_diff,
                       "max_abs_eigenvalue_diff_vs_sklearn": eig_diff,
                       "min_abs_cosine_vs_sklearn_components": float(cosines.min())},
        "top_terms": {f"PC{i + 1}": top_terms(pca.components_[i], terms)
                      for i in range(min(3, n_components))},
    }

    plot_scree(pca, out_dir / "scree_plot.png")
    plot_projection(Z, labels, pca, out_dir / "pca_2d_projection.png")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote results to %s", out_dir.resolve())
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-features", type=int, default=1000,
                        help="vocabulary size (number of TF-IDF features)")
    parser.add_argument("--n-components", type=int, default=2,
                        help="number of principal components to keep")
    parser.add_argument("--data-path", default=None,
                        help="local bbc-news-data.csv; downloads from Kaggle if omitted")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = run(args.max_features, args.n_components, args.data_path, args.out_dir)
    print(json.dumps(summary["pca"], indent=2))


if __name__ == "__main__":
    main()
