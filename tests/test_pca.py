import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA

from word_embeddings.pca import PCA


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    # Correlated features so the spectrum is clearly non-uniform.
    latent = rng.normal(size=(200, 3))
    mixing = rng.normal(size=(3, 8))
    return latent @ mixing + 0.1 * rng.normal(size=(200, 8)) + 5.0


def test_mean_and_centering(data):
    pca = PCA(n_components=2).fit(data)
    np.testing.assert_allclose(pca.mean_, data.mean(axis=0))
    np.testing.assert_allclose(pca.X_centered_.mean(axis=0), 0.0, atol=1e-12)


def test_covariance_matches_numpy(data):
    pca = PCA(n_components=2).fit(data)
    np.testing.assert_allclose(pca.covariance_, np.cov(data, rowvar=False))
    np.testing.assert_allclose(pca.covariance_, pca.covariance_.T)


def test_eigen_pairs_sorted_and_valid(data):
    pca = PCA(n_components=3).fit(data)
    vals, vecs = pca.eigenvalues_, pca.eigenvectors_
    assert np.all(np.diff(vals) <= 1e-12), "eigenvalues must be in descending order"
    # C v = lambda v for every pair
    np.testing.assert_allclose(pca.covariance_ @ vecs, vecs * vals, atol=1e-9)
    # eigenvectors are orthonormal
    np.testing.assert_allclose(vecs.T @ vecs, np.eye(vecs.shape[1]), atol=1e-10)
    # eigenvalues sum to the total variance (trace of C)
    np.testing.assert_allclose(vals.sum(), np.trace(pca.covariance_))


def test_matches_sklearn(data):
    ours = PCA(n_components=3).fit(data)
    ref = SklearnPCA(n_components=3, svd_solver="full").fit(data)
    np.testing.assert_allclose(ours.explained_variance_, ref.explained_variance_, rtol=1e-8)
    np.testing.assert_allclose(ours.explained_variance_ratio_,
                               ref.explained_variance_ratio_, rtol=1e-8)
    # Same directions up to sign.
    np.testing.assert_allclose(np.abs(ours.components_), np.abs(ref.components_), atol=1e-8)
    np.testing.assert_allclose(np.abs(ours.transform(data)), np.abs(ref.transform(data)),
                               atol=1e-8)


def test_projection_variance_equals_eigenvalues(data):
    pca = PCA(n_components=3).fit(data)
    Z = pca.transform(data)
    np.testing.assert_allclose(Z.var(axis=0, ddof=1), pca.explained_variance_, rtol=1e-10)


def test_full_rank_reconstruction_is_exact(data):
    pca = PCA(n_components=data.shape[1]).fit(data)
    np.testing.assert_allclose(pca.inverse_transform(pca.transform(data)), data, atol=1e-9)


def test_n_components_for_variance(data):
    pca = PCA(n_components=1).fit(data)
    k = pca.n_components_for_variance(0.95)
    cum = pca.cumulative_variance_ratio()
    assert cum[k - 1] >= 0.95 and (k == 1 or cum[k - 2] < 0.95)


@pytest.mark.parametrize("kwargs, X", [
    ({"n_components": 0}, None),
    ({"n_components": 5}, np.ones((10, 3))),
    ({"n_components": 1}, np.ones(4)),
    ({"n_components": 1}, np.ones((1, 3))),
])
def test_invalid_input(kwargs, X):
    with pytest.raises(ValueError):
        PCA(**kwargs).fit(X if X is not None else np.ones((3, 3)))


def test_transform_before_fit():
    with pytest.raises(RuntimeError):
        PCA().transform(np.ones((2, 2)))


def test_zero_variance_has_no_variance_threshold():
    pca = PCA(n_components=1).fit(np.ones((5, 3)))
    np.testing.assert_array_equal(pca.explained_variance_ratio_, [0.0])
    with pytest.raises(ValueError, match="zero-variance"):
        pca.n_components_for_variance(0.5)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        PCA(n_components=1).fit([[1, 2], [value, 3]])


@pytest.mark.parametrize("value", [1.5, True])
def test_noninteger_component_count_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        PCA(n_components=value)


def test_retained_variance_is_maximal_and_matches_reconstruction_error(data):
    pca = PCA(n_components=3).fit(data)
    residual = data - pca.inverse_transform(pca.transform(data))
    np.testing.assert_allclose(
        np.sum(residual ** 2) / (len(data) - 1),
        pca.eigenvalues_[3:].sum(), rtol=1e-9)
    # An unrelated orthonormal projection cannot capture more sample variance.
    q, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(data.shape[1], 3)))
    random_variance = np.trace(q.T @ pca.covariance_ @ q)
    assert pca.explained_variance_.sum() >= random_variance
