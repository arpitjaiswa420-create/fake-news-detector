"""
Inference and URL Scraping Module for Fake News Detection.
Implements:
- Production inference engine with confidence scoring
- Robust URL scraping and article extraction via BeautifulSoup
- Fast batch prediction for tabular datasets
- Integration with explainability and stylometric analysis
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import TextCleaner
from src.features import MetadataFeatureExtractor

logger = logging.getLogger("predict")

MODELS_DIR = PROJECT_ROOT / "models"


class ArticleScraper:
    """Extracts headline and clean body content from news URLs."""

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    @classmethod
    def scrape(cls, url: str, timeout: int = 10) -> Dict[str, str]:
        """Fetch and extract title and body text from a webpage."""
        headers = {"User-Agent": cls.USER_AGENT}
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to fetch URL {url}: {e}")
            raise ValueError(f"Could not fetch URL: {e}")

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove irrelevant elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            element.decompose()

        # Extract title
        title = ""
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text().strip()
        elif soup.title:
            title = soup.title.get_text().strip()

        # Extract main text
        article_elem = soup.find("article")
        if article_elem:
            paragraphs = article_elem.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        text_content = " ".join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20])

        if not text_content:
            text_content = soup.get_text(separator=" ", strip=True)

        return {
            "title": title,
            "text": text_content,
            "url": url
        }


class NewsPredictor:
    """Production inference engine for fake news classification."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        pipeline_path: Optional[Path] = None,
        enable_lime: bool = True
    ):
        self.model_path = model_path or (MODELS_DIR / "best_model.joblib")
        self.pipeline_path = pipeline_path or (MODELS_DIR / "feature_pipeline.joblib")

        if not self.model_path.exists() or not self.pipeline_path.exists():
            raise FileNotFoundError(
                f"Model or pipeline not found at {self.model_path} / {self.pipeline_path}. "
                "Train the models first using src/train.py"
            )

        self.model = joblib.load(self.model_path)
        self.pipeline = joblib.load(self.pipeline_path)
        self.cleaner = TextCleaner()
        self.meta_extractor = MetadataFeatureExtractor()
        self.enable_lime = enable_lime
        self._explainer = None

    def _get_explainer(self):
        if self._explainer is None and self.enable_lime:
            from src.explain import NewsExplainer
            self._explainer = NewsExplainer(model_path=self.model_path, pipeline_path=self.pipeline_path)
        return self._explainer

    def predict(
        self,
        text: str,
        title: str = "",
        explain: bool = False,
        num_explanation_words: int = 8
    ) -> Dict[str, Any]:
        """
        Predict whether a single article is FAKE or REAL.
        Returns label, confidence score, class probabilities, metadata summary, and explanation.
        """
        title = (title or "").strip()
        text = (text or "").strip()

        if not text and not title:
            return {
                "label": "UNKNOWN",
                "confidence": 0.0,
                "probabilities": {"REAL": 0.5, "FAKE": 0.5},
                "error": "Both title and text are empty."
            }

        cleaned_title = self.cleaner.clean_raw_text(title)
        cleaned_text = self.cleaner.clean_raw_text(text)
        cleaned_full_text = f"{cleaned_title} {cleaned_text}".strip()

        df_single = pd.DataFrame([{
            "title": title,
            "text": text,
            "cleaned_full_text": cleaned_full_text
        }])

        X = self.pipeline.transform(df_single)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            prob_real = float(probs[0])
            prob_fake = float(probs[1])
        else:
            dec = float(self.model.decision_function(X)[0])
            prob_fake = float(1 / (1 + np.exp(-dec)))
            prob_real = 1.0 - prob_fake

        label = "FAKE" if prob_fake >= 0.5 else "REAL"
        confidence = prob_fake if label == "FAKE" else prob_real

        # Extract stylometric metadata
        meta_features = self.meta_extractor.transform(df_single)[0]
        meta_names = MetadataFeatureExtractor.FEATURE_NAMES
        metadata_dict = {name: float(val) for name, val in zip(meta_names, meta_features)}

        result = {
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "REAL": round(prob_real, 4),
                "FAKE": round(prob_fake, 4)
            },
            "metadata_signals": {
                "word_count": int(metadata_dict.get("word_count", 0)),
                "caps_ratio_percent": round(metadata_dict.get("caps_ratio", 0.0) * 100, 2),
                "title_caps_percent": round(metadata_dict.get("title_caps_ratio", 0.0) * 100, 2),
                "exclamation_count": int(metadata_dict.get("exclamation_count", 0)),
                "readability_score": round(metadata_dict.get("readability_score", 60.0), 1),
                "sentiment_polarity": round(metadata_dict.get("sentiment_polarity", 0.0), 3)
            }
        }

        if explain and self.enable_lime:
            explainer = self._get_explainer()
            if explainer:
                exp_res = explainer.explain(
                    text=text,
                    title=title,
                    num_features=num_explanation_words,
                    num_samples=150
                )
                result["explanation"] = {
                    "influencing_features": exp_res["influencing_features"],
                    "html_snippet": exp_res["html_snippet"]
                }

        return result

    def predict_url(self, url: str, explain: bool = False) -> Dict[str, Any]:
        """Fetch article from URL and predict."""
        scraped = ArticleScraper.scrape(url)
        prediction = self.predict(
            text=scraped["text"],
            title=scraped["title"],
            explain=explain
        )
        prediction["scraped_article"] = {
            "title": scraped["title"],
            "url": url,
            "preview_snippet": scraped["text"][:300] + ("..." if len(scraped["text"]) > 300 else "")
        }
        return prediction

    def predict_batch_dataframe(
        self,
        df: pd.DataFrame,
        title_col: str = "title",
        text_col: str = "text"
    ) -> pd.DataFrame:
        """Vectorized batch prediction on DataFrame with multiple articles."""
        df_out = df.copy()

        t_col = df_out[title_col].fillna("").astype(str) if title_col in df_out.columns else pd.Series([""] * len(df_out))
        b_col = df_out[text_col].fillna("").astype(str) if text_col in df_out.columns else pd.Series([""] * len(df_out))

        cleaned_titles = self.cleaner.clean_series(t_col)
        cleaned_texts = self.cleaner.clean_series(b_col)
        cleaned_full = (cleaned_titles + " " + cleaned_texts).str.strip()

        prep_df = pd.DataFrame({
            "title": t_col,
            "text": b_col,
            "cleaned_full_text": cleaned_full
        })

        X = self.pipeline.transform(prep_df)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            probs_real = probs[:, 0]
            probs_fake = probs[:, 1]
        else:
            dec = self.model.decision_function(X)
            probs_fake = 1 / (1 + np.exp(-dec))
            probs_real = 1.0 - probs_fake

        labels = np.where(probs_fake >= 0.5, "FAKE", "REAL")
        confidences = np.where(labels == "FAKE", probs_fake, probs_real)

        df_out["predicted_label"] = labels
        df_out["confidence"] = np.round(confidences, 4)
        df_out["fake_probability"] = np.round(probs_fake, 4)
        df_out["real_probability"] = np.round(probs_real, 4)

        return df_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict if a news article or URL is Fake or Real.")
    parser.add_argument("--text", type=str, default=None, help="Article body text.")
    parser.add_argument("--title", type=str, default="", help="Article title.")
    parser.add_argument("--url", type=str, default=None, help="URL of the news article to scrape and check.")
    parser.add_argument("--explain", action="store_true", help="Generate LIME explanation.")
    args = parser.parse_args()

    predictor = NewsPredictor()

    if args.url:
        print(f"Scraping and analyzing URL: {args.url}")
        res = predictor.predict_url(args.url, explain=args.explain)
    elif args.text or args.title:
        res = predictor.predict(text=args.text or "", title=args.title, explain=args.explain)
    else:
        sample_title = "Senate passes bipartisan infrastructure plan with overwhelming majority"
        sample_text = (
            "The United States Senate voted on Tuesday to approve the sweeping infrastructure bill, "
            "authorizing $1 trillion in federal funding for roads, bridges, and public transit. "
            "Lawmakers from both parties praised the compromise reached after months of negotiation."
        )
        print(f"Running on sample article:\nTitle: {sample_title}")
        res = predictor.predict(text=sample_text, title=sample_title, explain=args.explain)

    print("\n--- Prediction Result ---")
    print(f"Label: {res['label']} (Confidence: {res['confidence'] * 100:.2f}%)")
    print(f"Probabilities: {res['probabilities']}")
    print(f"Metadata Signals: {res.get('metadata_signals')}")
    if "explanation" in res:
        print(f"Top Influencing Words: {res['explanation']['influencing_features']}")
