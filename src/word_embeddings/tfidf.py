"""TF-IDF vectorizer implemented from scratch with NumPy.

The weighting scheme matches scikit-learn's ``TfidfVectorizer`` defaults so the
two can be compared directly in the tests:

    tf(t, d)  = raw count of term t in document d
    idf(t)    = ln((1 + n) / (1 + df(t))) + 1          (smoothed IDF)
    tfidf     = tf * idf, then each document row is L2-normalised
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Same default token pattern as scikit-learn: words of 2+ alphanumeric characters.
_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def tokenize(text: str, stop_words: frozenset[str] = ENGLISH_STOP_WORDS) -> list[str]:
    """Lower-case ``text``, split it into word tokens and drop stop words."""
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in stop_words]


class TfidfVectorizer:
    """Turn a list of documents into an (n_documents x n_terms) TF-IDF matrix.

    Parameters
    ----------
    max_features:
        Keep only the ``max_features`` terms with the highest total count across
        the corpus. ``None`` keeps every term.
    min_df:
        Ignore terms that appear in fewer than ``min_df`` documents.
    """

    def __init__(self, max_features: int | None = 1000, min_df: int = 1) -> None:
        if max_features is not None and max_features < 1:
            raise ValueError("max_features must be a positive integer or None")
        if min_df < 1:
            raise ValueError("min_df must be >= 1")
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, documents: Sequence[str]) -> "TfidfVectorizer":
        tokenized = [tokenize(doc) for doc in documents]
        if not tokenized:
            raise ValueError("Cannot fit on an empty corpus")

        total_counts: Counter[str] = Counter()
        doc_freq: Counter[str] = Counter()
        for tokens in tokenized:
            total_counts.update(tokens)
            doc_freq.update(set(tokens))

        candidates = [t for t in total_counts if doc_freq[t] >= self.min_df]
        # Sort by corpus frequency (desc), then alphabetically for deterministic ties.
        candidates.sort(key=lambda t: (-total_counts[t], t))
        if self.max_features is not None:
            candidates = candidates[: self.max_features]
        if not candidates:
            raise ValueError("Vocabulary is empty; lower min_df or check the input")

        terms = sorted(candidates)  # alphabetical column order, like scikit-learn
        self.vocabulary_ = {term: idx for idx, term in enumerate(terms)}

        n_docs = len(tokenized)
        df = np.array([doc_freq[t] for t in terms], dtype=np.float64)
        self.idf_ = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        return self

    # ------------------------------------------------------------ transform
    def transform(self, documents: Iterable[str]) -> np.ndarray:
        if self.idf_ is None:
            raise RuntimeError("TfidfVectorizer must be fitted before calling transform")

        docs = list(documents)
        tf = np.zeros((len(docs), len(self.vocabulary_)), dtype=np.float64)
        for row, doc in enumerate(docs):
            for term, count in Counter(tokenize(doc)).items():
                col = self.vocabulary_.get(term)
                if col is not None:
                    tf[row, col] = count

        tfidf = tf * self.idf_
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0  # leave all-zero rows untouched
        return tfidf / norms

    def fit_transform(self, documents: Sequence[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)

    def get_feature_names_out(self) -> np.ndarray:
        return np.array(sorted(self.vocabulary_, key=self.vocabulary_.__getitem__))
