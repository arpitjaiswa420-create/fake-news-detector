"""
Feature Extraction and Engineering Module for Fake News Detection.
Implements:
- TF-IDF and N-gram feature extractors
- Metadata / Stylometric feature extractor:
    - Article & title length, word counts
    - Capitalization ratios (clickbait signal)
    - Punctuation densities (sensationalism signal: !, ?)
    - Syllable estimation & Flesch Reading Ease score
    - Lexical sentiment polarity & subjectivity
- Dense semantic representation (LSA / SVD dense embeddings)
- Unified feature pipelines compatible with scikit-learn
"""

import re
import string
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger("features")

# Common sentiment lexicon tokens for fast, robust polarity calculation without heavy dependencies
POSITIVE_WORDS = {
    "good", "great", "excellent", "positive", "fortunate", "correct", "superior",
    "honor", "praise", "truth", "genuine", "honest", "reliable", "peace", "hope",
    "success", "triumph", "progress", "benefit", "growth", "agree", "support"
}
NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "negative", "unfortunate", "wrong", "inferior",
    "scandal", "lie", "hoax", "corrupt", "fake", "disaster", "crisis", "outrage",
    "threat", "danger", "fraud", "criminal", "shame", "betrayal", "panic", "guilty",
    "evil", "destroy", "horrific", "shocking", "conspiracy"
}


def count_syllables(word: str) -> int:
    """Estimate syllable count in an English word using vowel grouping heuristics."""
    word = word.lower().strip()
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    # Count vowel groups
    count = len(re.findall(r"[aeiouy]+", word))
    # Deduct trailing 'e' if preceded by consonant
    if word.endswith("e") and not word.endswith("le") and count > 1:
        count -= 1
    return max(1, count)


def estimate_flesch_reading_ease(text: str) -> float:
    """
    Compute Flesch Reading Ease Score:
    Score = 206.835 - 1.015 * (total words / total sentences) - 84.6 * (total syllables / total words)
    Higher score (60-100) = plain English, easier to read.
    Lower score (0-30) = complex / academic text.
    """
    if not text or not text.strip():
        return 60.0

    sentences = max(1, len(re.split(r"[.!?]+", text)))
    words = re.findall(r"\b[A-Za-z]+\b", text)
    word_count = len(words)
    if word_count == 0:
        return 60.0

    syllables = sum(count_syllables(w) for w in words)
    asl = word_count / sentences
    asw = syllables / word_count
    score = 206.835 - (1.015 * asl) - (84.6 * asw)
    return float(np.clip(score, 0.0, 100.0))


def compute_lexical_sentiment(tokens: List[str]) -> Tuple[float, float]:
    """
    Compute sentiment polarity (-1.0 to 1.0) and subjectivity (0.0 to 1.0)
    using rule-based lexical counting.
    """
    if not tokens:
        return 0.0, 0.0
    pos_count = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg_count = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    total_sentiment_tokens = pos_count + neg_count

    if total_sentiment_tokens == 0:
        polarity = 0.0
        subjectivity = 0.0
    else:
        polarity = (pos_count - neg_count) / total_sentiment_tokens
        subjectivity = min(1.0, total_sentiment_tokens / (len(tokens) + 1) * 5.0)

    return float(polarity), float(subjectivity)


class MetadataFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts stylometric and metadata features from raw article text and headlines.
    Fake news articles frequently demonstrate:
    - High capital letter ratios (all-caps shouting)
    - High exclamation/question mark density (sensationalism)
    - Lower readability scores or extreme short/long texts
    - Higher negative or extreme emotional polarity
    """

    FEATURE_NAMES = [
        "char_count",
        "word_count",
        "avg_word_length",
        "caps_ratio",
        "exclamation_count",
        "question_count",
        "punct_density",
        "title_char_count",
        "title_word_count",
        "title_caps_ratio",
        "title_exclamation_count",
        "readability_score",
        "sentiment_polarity",
        "sentiment_subjectivity"
    ]

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def _extract_single(self, row: Union[pd.Series, Dict[str, str]]) -> np.ndarray:
        title = str(row.get("title", "") or "")
        text = str(row.get("text", "") or "")
        full_text = f"{title}. {text}".strip()

        # Text level metrics
        char_count = len(full_text)
        words = full_text.split()
        word_count = len(words)
        avg_word_len = (char_count / word_count) if word_count > 0 else 0.0

        # Capital letters ratio (excluding whitespace and digits)
        letters_count = sum(1 for c in full_text if c.isalpha())
        caps_count = sum(1 for c in full_text if c.isupper())
        caps_ratio = (caps_count / letters_count) if letters_count > 0 else 0.0

        # Punctuation counts
        exclamations = full_text.count("!")
        questions = full_text.count("?")
        punct_count = sum(1 for c in full_text if c in string.punctuation)
        punct_density = (punct_count / char_count) if char_count > 0 else 0.0

        # Title specific metrics (strong clickbait indicators)
        title_chars = len(title)
        title_words = len(title.split())
        title_letters = sum(1 for c in title if c.isalpha())
        title_caps = sum(1 for c in title if c.isupper())
        title_caps_ratio = (title_caps / title_letters) if title_letters > 0 else 0.0
        title_exclamations = title.count("!")

        # Readability & Sentiment
        readability = estimate_flesch_reading_ease(full_text)
        tokens_lower = [w.lower() for w in re.findall(r"\b[A-Za-z]+\b", full_text)]
        polarity, subjectivity = compute_lexical_sentiment(tokens_lower)

        return np.array([
            char_count,
            word_count,
            avg_word_len,
            caps_ratio,
            exclamations,
            questions,
            punct_density,
            title_chars,
            title_words,
            title_caps_ratio,
            title_exclamations,
            readability,
            polarity,
            subjectivity
        ], dtype=np.float32)

    def transform(self, X: Union[pd.DataFrame, List[Dict[str, str]]]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            features = [self._extract_single(row) for _, row in X.iterrows()]
        elif isinstance(X, list):
            features = [self._extract_single(item) for item in X]
        else:
            raise ValueError("Input to MetadataFeatureExtractor must be a DataFrame or list of dicts.")

        features_arr = np.array(features, dtype=np.float32)
        # Replace any NaN or Inf with 0.0
        features_arr = np.nan_to_num(features_arr, nan=0.0, posinf=0.0, neginf=0.0)
        return features_arr

    def get_feature_names_out(self, input_features=None):
        return np.array(self.FEATURE_NAMES)


class DenseEmbeddingReducer(BaseEstimator, TransformerMixin):
    """
    Transforms sparse TF-IDF vectors into a dense semantic latent representation
    using Truncated SVD (Latent Semantic Analysis / dense document embeddings).
    Provides an embedding-based pipeline for distance-based and deep neural models.
    """

    def __init__(self, n_components: int = 128, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)

    def fit(self, X, y=None):
        self.svd.fit(X)
        return self

    def transform(self, X):
        return self.svd.transform(X)


def build_tfidf_vectorizer(
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: Union[int, float] = 2,
    max_df: Union[int, float] = 0.95
) -> TfidfVectorizer:
    """Construct an optimized TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        dtype=np.float32
    )


class TextMetadataCombinedPipeline:
    """
    Combined feature pipeline that extracts:
    1. TF-IDF features from cleaned text
    2. Scaled Metadata features from raw text/title
    Returns a unified scipy sparse matrix or numpy array.
    """

    def __init__(
        self,
        tfidf_max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: Union[int, float] = 2,
        max_df: Union[int, float] = 0.95,
        include_metadata: bool = True
    ):
        self.tfidf = build_tfidf_vectorizer(
            max_features=tfidf_max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df
        )
        self.include_metadata = include_metadata
        self.meta_extractor = MetadataFeatureExtractor() if include_metadata else None
        self.scaler = StandardScaler() if include_metadata else None

    def fit(self, df: pd.DataFrame, y=None):
        logger.info("Fitting TF-IDF vectorizer on cleaned text...")
        self.tfidf.fit(df["cleaned_full_text"])

        if self.include_metadata:
            logger.info("Fitting metadata extractor and scaler...")
            meta_feats = self.meta_extractor.transform(df)
            self.scaler.fit(meta_feats)

        return self

    def transform(self, df: pd.DataFrame):
        from scipy.sparse import hstack, csr_matrix

        tfidf_feats = self.tfidf.transform(df["cleaned_full_text"])

        if not self.include_metadata:
            return tfidf_feats

        meta_feats = self.meta_extractor.transform(df)
        meta_scaled = self.scaler.transform(meta_feats)
        meta_sparse = csr_matrix(meta_scaled)

        combined = hstack([tfidf_feats, meta_sparse], format="csr")
        return combined

    def fit_transform(self, df: pd.DataFrame, y=None):
        return self.fit(df, y).transform(df)
