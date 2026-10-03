"""
Unit tests for feature extraction and engineering.
"""

import pytest
import numpy as np
import pandas as pd
from src.features import (
    count_syllables,
    estimate_flesch_reading_ease,
    compute_lexical_sentiment,
    MetadataFeatureExtractor,
    DenseEmbeddingReducer,
    build_tfidf_vectorizer,
    TextMetadataCombinedPipeline
)


def test_syllables_and_reading_ease():
    assert count_syllables("cat") == 1
    assert count_syllables("computer") == 3
    assert count_syllables("unbelievable") >= 4

    simple_text = "The cat sat on the mat. It was a good day."
    score = estimate_flesch_reading_ease(simple_text)
    assert 60.0 <= score <= 100.0


def test_sentiment_lexicon():
    pos_tokens = ["great", "progress", "peace", "success"]
    pol, subj = compute_lexical_sentiment(pos_tokens)
    assert pol > 0.5
    assert subj > 0.0

    neg_tokens = ["terrible", "scandal", "crisis", "disaster"]
    pol, subj = compute_lexical_sentiment(neg_tokens)
    assert pol < -0.5


def test_metadata_extractor():
    data = pd.DataFrame([
        {
            "title": "BREAKING: HUGE SCANDAL EXPOSED!!",
            "text": "You will not believe what happened today! Shocking details inside."
        },
        {
            "title": "City council approves routine budget",
            "text": "The local council met on Monday to finalize annual expenditure plans."
        }
    ])

    extractor = MetadataFeatureExtractor()
    features = extractor.transform(data)

    assert features.shape == (2, len(MetadataFeatureExtractor.FEATURE_NAMES))
    assert not np.isnan(features).any()
    # Clickbait title should have higher caps ratio and exclamations
    assert features[0, 9] > features[1, 9]  # title_caps_ratio
    assert features[0, 10] > features[1, 10]  # title_exclamation_count


def test_combined_pipeline():
    df = pd.DataFrame({
        "title": ["Breaking News Headline", "Standard Official Report", "Another Fake Report"],
        "text": ["Full text story details.", "Standard verified announcement.", "Unverified shocking news."],
        "cleaned_full_text": ["breaking news headline story", "standard official report announcement", "fake report shocking news"]
    })

    pipeline = TextMetadataCombinedPipeline(
        tfidf_max_features=50,
        min_df=1,
        max_df=1.0,
        include_metadata=True
    )
    matrix = pipeline.fit_transform(df)

    assert matrix.shape[0] == 3
    assert matrix.shape[1] > 0
    assert not np.isnan(matrix.data).any()
