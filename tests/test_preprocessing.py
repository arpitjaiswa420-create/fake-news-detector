"""
Unit tests for text preprocessing and data cleaning.
"""

import pytest
import pandas as pd
from src.preprocessing import TextCleaner, preprocess_dataframe, split_and_save_data


def test_clean_html_and_urls():
    cleaner = TextCleaner()
    raw = "<p>Check this out at https://example.com/breaking-news! Contact info@news.org.</p>"
    cleaned = cleaner.clean_raw_text(raw)
    assert "https" not in cleaned
    assert "example" not in cleaned
    assert "info@news.org" not in cleaned
    assert "<p>" not in cleaned
    assert "check" in cleaned


def test_remove_wire_prefixes():
    cleaner = TextCleaner(remove_wire_prefixes=True)
    raw = "WASHINGTON (Reuters) - The Senate approved the federal budget measure on Tuesday."
    cleaned = cleaner.clean_raw_text(raw)
    assert "reuters" not in cleaned
    assert "washington" not in cleaned or "senate approved federal budget" in cleaned


def test_lemmatization_and_stopwords():
    cleaner = TextCleaner(remove_stopwords=True, lemmatize=True)
    raw = "The ministers were discussing critical policies and running tests."
    cleaned = cleaner.clean_raw_text(raw)
    words = cleaned.split()
    assert "were" not in words
    assert "and" not in words
    assert "the" not in words
    # 'policies' should be lemmatized to 'policy', 'running' to 'running' or 'run'
    assert "minister" in words or "ministers" in words
    assert "policy" in words


def test_empty_and_special_characters():
    cleaner = TextCleaner()
    assert cleaner.clean_raw_text("") == ""
    assert cleaner.clean_raw_text("   !!! ??? $$$   ") == ""
    assert cleaner.clean_raw_text(None) == ""


def test_preprocess_dataframe_dedup():
    data = {
        "title": ["Breaking News Headline", "Breaking News Headline", "Different Story"],
        "text": ["Full body article text here.", "Full body article text here.", "Another body text."],
        "label": [1, 1, 0]
    }
    df = pd.DataFrame(data)
    processed = preprocess_dataframe(df, drop_duplicates=True)
    assert len(processed) == 2
    assert "cleaned_title" in processed.columns
    assert "cleaned_text" in processed.columns
    assert "cleaned_full_text" in processed.columns


def test_split_and_save_data(tmp_path):
    data = {
        "title": [f"Title {i}" for i in range(100)],
        "text": [f"Body text for news item {i}" for i in range(100)],
        "cleaned_full_text": [f"title body news item {i}" for i in range(100)],
        "label": [0] * 50 + [1] * 50
    }
    df = pd.DataFrame(data)
    train_df, val_df, test_df = split_and_save_data(
        df, output_dir=tmp_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "val.csv").exists()
    assert (tmp_path / "test.csv").exists()
