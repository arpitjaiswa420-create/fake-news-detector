"""
Text Preprocessing and Data Cleaning Module for Fake News Detection.
Implements:
- Lowercasing, HTML/URL stripping, punctuation and special character cleaning
- Wire service / publisher artifact removal (e.g., '(Reuters) - ' datelines)
- Stopword removal and WordNet lemmatization
- Missing value imputation and deduplication
- Stratified splitting into train, validation, and test sets
"""

import re
import html
import string
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger("preprocessing")

# Ensure required NLTK resources
def _ensure_nltk_resources():
    for res in ["stopwords", "wordnet", "punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"corpora/{res}" if res in ["stopwords", "wordnet"] else f"tokenizers/{res}")
        except LookupError:
            nltk.download(res, quiet=True)

_ensure_nltk_resources()

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Known news agency wire patterns to prevent trivial overfitting to publisher signatures
REUTERS_PATTERN = re.compile(
    r"^(?:[A-Za-z\s,]+)?\s*\((?:Reuters|REUTERS)\)\s*[-—–:]\s*",
    flags=re.IGNORECASE
)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_PATTERN = re.compile(r"<.*?>")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
SPECIAL_CHARS_PATTERN = re.compile(r"[^a-zA-Z\s]")
MULTIPLE_SPACES_PATTERN = re.compile(r"\s+")


class TextCleaner:
    """
    Configurable text cleaning and normalization pipeline.
    """

    def __init__(
        self,
        remove_wire_prefixes: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_word_length: int = 2
    ):
        self.remove_wire_prefixes = remove_wire_prefixes
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_word_length = min_word_length
        self.stop_words = STOP_WORDS
        self.lemmatizer = LEMMATIZER

    def clean_raw_text(self, text: str) -> str:
        """
        Clean raw text string:
        1. Decode HTML entities and strip HTML tags
        2. Remove publisher wire prefixes (e.g. 'WASHINGTON (Reuters) - ')
        3. Remove URLs and emails
        4. Lowercase
        5. Remove special characters and digits
        6. Lemmatize and remove stopwords
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # Unescape HTML entities
        text = html.unescape(text)

        # Remove wire datelines from beginning
        if self.remove_wire_prefixes:
            text = REUTERS_PATTERN.sub("", text)

        # Remove HTML tags
        text = HTML_TAG_PATTERN.sub(" ", text)

        # Remove URLs and emails
        text = URL_PATTERN.sub(" ", text)
        text = EMAIL_PATTERN.sub(" ", text)

        # Lowercase
        text = text.lower()

        # Remove special characters, punctuation, and numbers
        text = SPECIAL_CHARS_PATTERN.sub(" ", text)

        # Tokenization via whitespace / regex for speed and stability
        tokens = text.split()

        cleaned_tokens = []
        for token in tokens:
            if len(token) < self.min_word_length:
                continue
            if self.remove_stopwords and token in self.stop_words:
                continue
            if self.lemmatize:
                token = self.lemmatizer.lemmatize(token)
            cleaned_tokens.append(token)

        return " ".join(cleaned_tokens)

    def clean_series(self, series: pd.Series) -> pd.Series:
        """Apply cleaning to a pandas Series of strings."""
        return series.fillna("").astype(str).apply(self.clean_raw_text)


def preprocess_dataframe(
    df: pd.DataFrame,
    cleaner: Optional[TextCleaner] = None,
    drop_duplicates: bool = True,
    min_text_length: int = 10
) -> pd.DataFrame:
    """
    Process full dataset:
    - Handle missing title/text
    - Optionally drop duplicates
    - Create combined raw text
    - Apply TextCleaner
    - Filter empty or near-empty articles
    """
    if cleaner is None:
        cleaner = TextCleaner()

    df = df.copy()

    # Fill missing values
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    # Combined full raw text (useful for metadata feature extraction prior to cleaning)
    df["raw_full_text"] = df["title"].str.strip() + ". " + df["text"].str.strip()

    if drop_duplicates:
        initial_len = len(df)
        df = df.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)
        dropped = initial_len - len(df)
        logger.info(f"Dropped {dropped} duplicate articles (Remaining: {len(df)}).")

    # Clean title and text
    logger.info("Cleaning article titles...")
    df["cleaned_title"] = cleaner.clean_series(df["title"])

    logger.info("Cleaning article body texts...")
    df["cleaned_text"] = cleaner.clean_series(df["text"])

    # Combine cleaned text: give prominence to title words by prepending
    df["cleaned_full_text"] = (
        df["cleaned_title"] + " " + df["cleaned_text"]
    ).str.strip()

    # Remove articles that resulted in empty or extremely short cleaned text
    initial_valid = len(df)
    df = df[df["cleaned_full_text"].str.len() >= min_text_length].reset_index(drop=True)
    logger.info(
        f"Filtered {initial_valid - len(df)} empty/too-short articles. Final count: {len(df)}"
    )

    return df


def split_and_save_data(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified train / val / test split and save to CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    val_test_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=val_test_ratio,
        stratify=df["label"],
        random_state=random_state
    )

    test_share_of_temp = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_share_of_temp,
        stratify=temp_df["label"],
        random_state=random_state
    )

    logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    logger.info(f"Train class balance: {train_df['label'].value_counts().to_dict()}")
    logger.info(f"Val class balance:   {val_df['label'].value_counts().to_dict()}")
    logger.info(f"Test class balance:  {test_df['label'].value_counts().to_dict()}")

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    logger.info(f"Successfully saved processed datasets to {output_dir}")
    return train_df, val_df, test_df


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.download_data import load_raw_dataset

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Preprocess and split fake news dataset.")
    parser.add_argument("--sample", type=int, default=None, help="Optionally subsample rows.")
    parser.add_argument("--keep-wire-prefixes", action="store_true", help="Don't strip Reuters datelines.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"

    raw_df = load_raw_dataset(sample_size=args.sample)
    cleaner = TextCleaner(remove_wire_prefixes=not args.keep_wire_prefixes)
    cleaned_df = preprocess_dataframe(raw_df, cleaner=cleaner)
    split_and_save_data(cleaned_df, output_dir=processed_dir)
