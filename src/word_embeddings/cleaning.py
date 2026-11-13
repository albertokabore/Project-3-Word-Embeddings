"""Data cleaning for the BBC News Archive dataset.

Every step records how many rows it touched, so the cleaning is auditable:

1. Validate schema   - required columns present, no missing values, known categories
2. Normalise spacing - strip leading/trailing whitespace, collapse runs of spaces
3. Repair characters - restore the "£" that was replaced by U+FFFD before a number
4. Exact duplicates  - drop articles whose normalised text is identical
5. Near duplicates   - drop re-published versions of a story (same title and
                       >= 80% word overlap, measured with Jaccard similarity)
6. Length filter     - drop articles too short to give reliable term statistics

Run ``python -m word_embeddings.cleaning`` to write the cleaned CSV and a report.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

EXPECTED_CATEGORIES = {"business", "entertainment", "politics", "sport", "tech"}
REQUIRED_COLUMNS = ["category", "filename", "title", "content"]
NEAR_DUPLICATE_THRESHOLD = 0.8
MIN_WORDS = 50
_WORD_RE = re.compile(r"\w+")


@dataclass
class CleaningReport:
    rows_in: int = 0
    missing_values: dict = field(default_factory=dict)
    unknown_categories: list = field(default_factory=list)
    whitespace_normalised: int = 0
    replacement_chars_repaired: int = 0
    replacement_chars_remaining: int = 0
    exact_duplicates_removed: int = 0
    near_duplicates_removed: int = 0
    near_duplicate_titles: list = field(default_factory=list)
    too_short_removed: int = 0
    rows_out: int = 0
    category_counts_before: dict = field(default_factory=dict)
    category_counts_after: dict = field(default_factory=dict)


def _normalise_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def jaccard(a: str, b: str) -> float:
    """Share of distinct words two texts have in common (|A ∩ B| / |A ∪ B|)."""
    wa, wb = set(_WORD_RE.findall(a.lower())), set(_WORD_RE.findall(b.lower()))
    return len(wa & wb) / len(wa | wb) if wa | wb else 1.0


def clean_dataset(
    raw: pd.DataFrame,
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD,
    min_words: int = MIN_WORDS,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Return the cleaned DataFrame (with an added ``text`` column) and a report."""
    rep = CleaningReport(rows_in=len(raw))

    # 1. Validate schema ------------------------------------------------------
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing columns: {missing_cols}")
    df = raw[REQUIRED_COLUMNS].copy()
    rep.missing_values = {c: int(n) for c, n in df.isna().sum().items()}
    df = df.dropna(subset=["category", "content"])
    df["title"] = df["title"].fillna("")
    df["category"] = df["category"].str.strip().str.lower()
    rep.unknown_categories = sorted(set(df["category"]) - EXPECTED_CATEGORIES)
    df = df[df["category"].isin(EXPECTED_CATEGORIES)]
    rep.category_counts_before = df["category"].value_counts().to_dict()

    # 2. Normalise whitespace -------------------------------------------------
    for col in ("title", "content"):
        cleaned = df[col].map(_normalise_space)
        if col == "content":
            rep.whitespace_normalised = int((cleaned != df[col]).sum())
        df[col] = cleaned

    # 3. Repair the Unicode replacement character ----------------------------
    # In this file U+FFFD only appears where a "£" was lost before an amount.
    has_bad = df["content"].str.contains("�(?=\\d)", regex=True)
    rep.replacement_chars_repaired = int(has_bad.sum())
    df["content"] = df["content"].str.replace("�(?=\\d)", "£", regex=True)
    rep.replacement_chars_remaining = int(df["content"].str.contains("�").sum())

    # 4. Exact duplicates (compared after whitespace normalisation) ----------
    before = len(df)
    df = df.drop_duplicates(subset="content", keep="first")
    rep.exact_duplicates_removed = before - len(df)

    # 5. Near duplicates: same headline and mostly the same words ------------
    drop_idx = []
    for title, group in df[df["title"].duplicated(keep=False)].groupby("title"):
        kept: list[str] = []
        for idx, text in group["content"].items():
            if any(jaccard(text, k) >= near_duplicate_threshold for k in kept):
                drop_idx.append(idx)
                rep.near_duplicate_titles.append(title)
            else:
                kept.append(text)
    df = df.drop(index=drop_idx)
    rep.near_duplicates_removed = len(drop_idx)

    # 6. Very short articles --------------------------------------------------
    n_words = df["content"].str.split().str.len()
    rep.too_short_removed = int((n_words < min_words).sum())
    df = df[n_words >= min_words]

    df = df.reset_index(drop=True)
    df["text"] = df["title"] + ". " + df["content"]
    rep.rows_out = len(df)
    rep.category_counts_after = df["category"].value_counts().to_dict()
    return df, rep


def main() -> None:
    from .data import CLEAN_CSV, load_raw

    df, rep = clean_dataset(load_raw())
    df.drop(columns="text").to_csv(CLEAN_CSV, sep="\t", index=False, encoding="utf-8")
    report_path = CLEAN_CSV.with_name("cleaning_report.json")
    report_path.write_text(json.dumps(asdict(rep), indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"{rep.rows_in} -> {rep.rows_out} articles; wrote {CLEAN_CSV} and {report_path}")


if __name__ == "__main__":
    main()
