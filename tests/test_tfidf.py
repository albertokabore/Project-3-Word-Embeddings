import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidf

from word_embeddings.tfidf import TfidfVectorizer, tokenize

DOCS = [
    "The stock market rallied as shares in banks rose.",
    "The striker scored twice and the football club won the match.",
    "Shares fell after the bank reported lower profits.",
    "The club signed a new striker before the match.",
    "Parliament debated the new election law.",
]


def test_tokenize_lowercases_and_drops_stop_words():
    assert tokenize("The Market ROSE, and a bank fell") == ["market", "rose", "bank", "fell"]


def test_matches_sklearn_given_same_vocabulary():
    ours = TfidfVectorizer(max_features=None)
    X = ours.fit_transform(DOCS)
    ref = SklearnTfidf(stop_words="english", vocabulary=ours.vocabulary_)
    np.testing.assert_allclose(X, ref.fit_transform(DOCS).toarray(), atol=1e-12)
    np.testing.assert_allclose(ours.idf_, ref.idf_)


def test_rows_are_unit_length():
    X = TfidfVectorizer(max_features=None).fit_transform(DOCS)
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0)


def test_max_features_and_min_df():
    vec = TfidfVectorizer(max_features=3, min_df=2).fit(DOCS)
    assert len(vec.vocabulary_) == 3
    assert set(vec.vocabulary_) <= {"shares", "club", "striker", "match", "new"}


def test_unknown_words_give_zero_row():
    vec = TfidfVectorizer(max_features=None).fit(DOCS)
    assert not vec.transform(["zzz qqq"]).any()


def test_errors():
    with pytest.raises(RuntimeError):
        TfidfVectorizer().transform(DOCS)
    with pytest.raises(ValueError):
        TfidfVectorizer(max_features=0)
    with pytest.raises(ValueError):
        TfidfVectorizer(min_df=10).fit(DOCS)
