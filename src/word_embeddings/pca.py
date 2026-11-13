"""Principal Component Analysis (PCA) implemented from scratch with NumPy.

Every step the assignment asks for is a separate, inspectable attribute:

1. ``mean_``                 - mean of every feature (column)
2. ``X_centered_``           - data with the mean subtracted from each feature
3. ``covariance_``           - covariance matrix  C = Xc^T Xc / (n - 1)
4. ``eigenvalues_`` /
   ``eigenvectors_``         - eigen-decomposition of C, sorted largest -> smallest
5. ``components_``           - the top-k eigenvectors (principal directions)
6. ``transform``             - projection of data onto those directions
"""

from __future__ import annotations

import numpy as np


class PCA:
    """Reduce data to ``n_components`` dimensions while keeping maximum variance."""

    def __init__(self, n_components: int = 2) -> None:
        if (
            isinstance(n_components, (bool, np.bool_))
            or not isinstance(n_components, (int, np.integer))
            or n_components < 1
        ):
            raise ValueError("n_components must be a positive integer")
        self.n_components = n_components

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected a 2-D array, got shape {X.shape}")
        n_samples, n_features = X.shape
        if not np.isfinite(X).all():
            raise ValueError("PCA requires finite data (no NaN or infinity)")
        if n_samples < 2:
            raise ValueError("PCA needs at least two samples")
        if self.n_components > n_features:
            raise ValueError(
                f"n_components={self.n_components} exceeds n_features={n_features}"
            )

        # Step 1: mean of each feature.
        self.mean_ = X.mean(axis=0)

        # Step 2: centre the data so every feature has mean 0.
        self.X_centered_ = X - self.mean_

        # Step 3: covariance matrix (n_features x n_features), unbiased (n - 1).
        self.covariance_ = (self.X_centered_.T @ self.X_centered_) / (n_samples - 1)

        # Step 4: eigen-decomposition. eigh is the right solver for a symmetric
        # matrix: it guarantees real eigenvalues and orthonormal eigenvectors.
        # It returns them in ASCENDING order, so we re-sort descending.
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Round-off can make the tiniest eigenvalues slightly negative; variance
        # cannot be negative, so clip them to zero.
        eigenvalues = np.clip(eigenvalues, 0.0, None)

        # An eigenvector v and -v are equally valid. Fix the sign so the entry
        # with the largest magnitude is positive. Repeated eigenvalues can still
        # have different orthonormal bases across eigensolvers.
        max_abs_rows = np.argmax(np.abs(eigenvectors), axis=0)
        signs = np.sign(eigenvectors[max_abs_rows, np.arange(n_features)])
        signs[signs == 0] = 1.0
        eigenvectors = eigenvectors * signs

        self.eigenvalues_ = eigenvalues            # shape (n_features,)
        self.eigenvectors_ = eigenvectors          # columns are eigenvectors

        # Step 5: keep the top-k directions. Rows of components_ are the PCs.
        self.components_ = eigenvectors[:, : self.n_components].T
        self.explained_variance_ = eigenvalues[: self.n_components]
        total_variance = eigenvalues.sum()
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_variance
            if total_variance > 0
            else np.zeros(self.n_components)
        )
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project ``X`` onto the principal components: Z = (X - mean) W."""
        if not hasattr(self, "components_"):
            raise RuntimeError("PCA must be fitted before calling transform")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Map reduced data back to the original space (lossy reconstruction)."""
        return np.asarray(Z) @ self.components_ + self.mean_

    def cumulative_variance_ratio(self) -> np.ndarray:
        """Cumulative share of total variance explained by the first 1..d components."""
        total = self.eigenvalues_.sum()
        return np.cumsum(self.eigenvalues_) / total if total > 0 else self.eigenvalues_ * 0

    def n_components_for_variance(self, threshold: float) -> int:
        """Smallest number of components that explain at least ``threshold`` of the variance."""
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        cumulative = self.cumulative_variance_ratio()
        if self.eigenvalues_.sum() == 0:
            raise ValueError("Explained variance is undefined for zero-variance data")
        if threshold == 1.0:
            # Avoid an unattainable 1.0 due to floating-point summation error.
            return int(np.count_nonzero(self.eigenvalues_))
        return int(np.searchsorted(cumulative, threshold) + 1)
