"""Load the BBC News Archive dataset from Kaggle.

Dataset: "BBC News Archive" by Kaggle user hgultekin
https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive
2,225 BBC news articles (2004-2005) in five categories: business, entertainment,
politics, sport, tech. Originally published by Greene & Cunningham (2006).

Files in ``data/``:
    bbc-news-data.csv          raw file exactly as downloaded from Kaggle
    bbc-news-cleaned.csv       output of the cleaning step (see cleaning.py)
    cleaning_report.json       what each cleaning step changed
If the raw file is missing it is downloaded with ``kagglehub`` and copied there.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

KAGGLE_DATASET = "hgultekin/bbcnewsarchive"
CSV_NAME = "bbc-news-data.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / CSV_NAME
CLEAN_CSV = DATA_DIR / "bbc-news-cleaned.csv"


def download_dataset(dest: Path = RAW_CSV) -> Path:
    """Download the dataset from Kaggle (if needed) and copy the CSV to ``dest``."""
    if dest.is_file():
        return dest
    import kagglehub  # imported lazily so tests do not need network access

    downloaded = Path(kagglehub.dataset_download(KAGGLE_DATASET)) / CSV_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, dest)
    return dest


def load_raw(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Read the raw tab-separated file without changing anything."""
    path = Path(csv_path) if csv_path else download_dataset()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_csv(path, sep="\t", encoding="utf-8")


def load_bbc_news(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw file and return the cleaned articles with a ``text`` column."""
    from .cleaning import clean_dataset

    df, _ = clean_dataset(load_raw(csv_path))
    return df
