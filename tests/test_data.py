import pandas as pd
import pytest

from word_embeddings.cleaning import clean_dataset, jaccard
from word_embeddings.data import load_bbc_news

LONG = " ".join(f"word{i}" for i in range(60))  # passes the 50-word minimum


def _df(rows):
    return pd.DataFrame(rows, columns=["category", "filename", "title", "content"])


def test_whitespace_and_replacement_char():
    df, rep = clean_dataset(_df([
        ["Sport ", "1.txt", " Win ", f"  A �15m  deal.   {LONG} "],
    ]))
    assert df.loc[0, "content"] == f"A £15m deal. {LONG}"
    assert df.loc[0, "title"] == "Win" and df.loc[0, "category"] == "sport"
    assert df.loc[0, "text"].startswith("Win. A £15m")
    assert rep.whitespace_normalised == 1 and rep.replacement_chars_repaired == 1


def test_exact_and_near_duplicates():
    near = LONG + " extra"
    df, rep = clean_dataset(_df([
        ["tech", "1.txt", "Phones", LONG],
        ["tech", "2.txt", "Phones again", " " + LONG + "  "],  # exact after spacing
        ["tech", "3.txt", "Phones", near],                       # near duplicate
        ["tech", "4.txt", "Other", "totally different " + LONG.replace("word", "x")],
    ]))
    assert rep.exact_duplicates_removed == 1
    assert rep.near_duplicates_removed == 1
    assert list(df["filename"]) == ["1.txt", "4.txt"]


def test_same_title_different_story_is_kept():
    other = " ".join(f"other{i}" for i in range(60))
    df, rep = clean_dataset(_df([
        ["politics", "1.txt", "Budget", LONG], ["politics", "2.txt", "Budget", other],
    ]))
    assert rep.near_duplicates_removed == 0 and len(df) == 2


def test_missing_short_and_unknown_rows_dropped():
    df, rep = clean_dataset(_df([
        ["sport", "1.txt", "A", LONG], ["sport", "2.txt", "B", None],
        ["sport", "3.txt", "C", "too short"], ["weather", "4.txt", "D", LONG + " z"],
    ]))
    assert len(df) == 1
    assert rep.missing_values["content"] == 1 and rep.too_short_removed == 1
    assert rep.unknown_categories == ["weather"]


def test_jaccard():
    assert jaccard("a b c", "a b c") == 1.0
    assert jaccard("a b", "c d") == 0.0


def test_load_from_file_and_errors(tmp_path):
    path = tmp_path / "bbc.csv"
    _df([["sport", "1.txt", "Win", LONG]]).to_csv(path, sep="\t", index=False)
    assert len(load_bbc_news(path)) == 1
    with pytest.raises(FileNotFoundError):
        load_bbc_news(tmp_path / "nope.csv")
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(bad, sep="\t", index=False)
    with pytest.raises(ValueError):
        load_bbc_news(bad)
