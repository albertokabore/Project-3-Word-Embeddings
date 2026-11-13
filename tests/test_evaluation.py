import numpy as np
from sklearn.model_selection import train_test_split

from word_embeddings.main import classification_by_dimension
from word_embeddings.tfidf import TfidfVectorizer


def test_test_documents_do_not_influence_vocabulary_or_idf(monkeypatch):
    labels = np.array(["sport"] * 10 + ["business"] * 10)
    documents = [
        f"{'football goal match' if y == 'sport' else 'bank market shares'} marker{i}"
        for i, y in enumerate(labels)
    ]
    train_idx, test_idx = train_test_split(
        np.arange(20), test_size=0.2, stratify=labels, random_state=42)
    fitted = []
    original_fit = TfidfVectorizer.fit

    def record_fit(self, docs):
        original_fit(self, docs)
        fitted.append(self)
        return self

    monkeypatch.setattr(TfidfVectorizer, "fit", record_fit)
    scores, protocol = classification_by_dimension(
        documents, labels, dims=(1, 2), min_df=1)
    assert len(fitted) == 1
    vec = fitted[0]
    assert all(f"marker{i}" in vec.vocabulary_ for i in train_idx)
    assert all(f"marker{i}" not in vec.vocabulary_ for i in test_idx)
    # Every retained marker occurs once among the 16 training articles.
    marker_col = vec.vocabulary_[f"marker{train_idx[0]}"]
    np.testing.assert_allclose(vec.idf_[marker_col], np.log(17 / 2) + 1)
    assert protocol["n_train"] == 16 and protocol["n_test"] == 4
    assert scores and all(0 <= value <= 1 for value in scores.values())
