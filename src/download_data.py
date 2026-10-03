"""
Dataset Downloader and Loader for Fake News Detection.
Source: ISOT Fake News Dataset (Ahmed, Traore & Saad, 2017) / Kaggle Fake and Real News Dataset.
License: Research and Educational Use.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Optional
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("download_data")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

TRUE_NEWS_URL = "https://raw.githubusercontent.com/laxmimerit/fake-real-news-dataset/master/data/True.csv"
FAKE_NEWS_URL = "https://raw.githubusercontent.com/laxmimerit/fake-real-news-dataset/master/data/Fake.csv"

DATASET_INFO = {
    "name": "ISOT Fake and Real News Dataset",
    "authors": "H. Ahmed, I. Traore, S. Saad (University of Victoria, 2017)",
    "source_url": "https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php",
    "mirror_url": "https://github.com/laxmimerit/fake-real-news-dataset",
    "description": "Contains ~21,417 real news articles from Reuters and ~23,481 fake news articles flagged by politifact.com and various fact-checking organizations.",
    "license": "Creative Commons Attribution-NonCommercial-ShareAlike (Academic/Research Use)"
}


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download a file via HTTP streaming with size tracking."""
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        logger.info(f"File already exists at {dest_path} ({dest_path.stat().st_size / (1024 * 1024):.2f} MB). Skipping download.")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading from {url} to {dest_path}...")
    
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    
    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    logger.debug(f"Progress: {percent:.1f}% ({downloaded / (1024 * 1024):.2f} MB)")

    logger.info(f"Successfully downloaded {dest_path.name} ({dest_path.stat().st_size / (1024 * 1024):.2f} MB)")


def download_dataset(force: bool = False) -> Tuple[Path, Path]:
    """Download both True and Fake news CSV files."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    true_path = RAW_DIR / "True.csv"
    fake_path = RAW_DIR / "Fake.csv"

    if force:
        if true_path.exists():
            true_path.unlink()
        if fake_path.exists():
            fake_path.unlink()

    download_file(TRUE_NEWS_URL, true_path)
    download_file(FAKE_NEWS_URL, fake_path)

    return true_path, fake_path


def load_raw_dataset(sample_size: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    """
    Load raw True and Fake news CSVs, assign binary labels, and return combined DataFrame.
    Label 1 = Fake, Label 0 = Real.
    """
    true_path = RAW_DIR / "True.csv"
    fake_path = RAW_DIR / "Fake.csv"

    if not true_path.exists() or not fake_path.exists():
        logger.info("Raw files missing. Triggering automated download...")
        download_dataset()

    logger.info("Reading raw datasets...")
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    df_true["label"] = 0
    df_true["label_name"] = "REAL"
    df_fake["label"] = 1
    df_fake["label_name"] = "FAKE"

    logger.info(f"Loaded {len(df_true)} REAL articles and {len(df_fake)} FAKE articles.")

    if sample_size and sample_size < (len(df_true) + len(df_fake)):
        per_class = sample_size // 2
        df_true = df_true.sample(n=min(per_class, len(df_true)), random_state=random_state)
        df_fake = df_fake.sample(n=min(per_class, len(df_fake)), random_state=random_state)
        logger.info(f"Subsampled to {len(df_true)} REAL and {len(df_fake)} FAKE articles (total {len(df_true) + len(df_fake)}).")

    combined_df = pd.concat([df_true, df_fake], ignore_index=True)
    combined_df = combined_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return combined_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download and verify Fake News dataset.")
    parser.add_argument("--force", action="store_true", help="Force redownload even if files exist.")
    parser.add_argument("--sample", type=int, default=None, help="Optionally subsample rows.")
    args = parser.parse_args()

    logger.info(f"Dataset info: {DATASET_INFO}")
    download_dataset(force=args.force)
    df = load_raw_dataset(sample_size=args.sample)
    logger.info(f"Dataset ready with shape {df.shape}. Columns: {list(df.columns)}")
