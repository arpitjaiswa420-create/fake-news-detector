"""
Model Explainability Module using LIME (Local Interpretable Model-agnostic Explanations).
Explains why an article was classified as FAKE or REAL by revealing
which words, n-grams, and phrases push the decision in either direction.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib
from lime.lime_text import LimeTextExplainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import TextCleaner

logger = logging.getLogger("explain")

MODELS_DIR = PROJECT_ROOT / "models"


class NewsExplainer:
    """
    Explainer class leveraging LIME to provide token-level attribution
    for fake news predictions.
    """

    def __init__(self, model_path: Optional[Path] = None, pipeline_path: Optional[Path] = None):
        self.model_path = model_path or (MODELS_DIR / "best_model.joblib")
        self.pipeline_path = pipeline_path or (MODELS_DIR / "feature_pipeline.joblib")

        logger.info(f"Loading model from {self.model_path}...")
        self.model = joblib.load(self.model_path)
        logger.info(f"Loading feature pipeline from {self.pipeline_path}...")
        self.pipeline = joblib.load(self.pipeline_path)

        self.cleaner = TextCleaner()
        self.explainer = LimeTextExplainer(
            class_names=["REAL", "FAKE"],
            split_expression=r"\W+",
            bow=False,
            random_state=42
        )

    def _predict_proba_for_lime(self, raw_texts: List[str]) -> np.ndarray:
        """
        Callable prediction wrapper for LIME:
        Accepts raw string instances, cleans and extracts features,
        and returns (n_samples, 2) prediction probabilities.
        """
        cleaned_texts = [self.cleaner.clean_raw_text(t) for t in raw_texts]
        temp_df = pd.DataFrame({
            "title": ["" for _ in raw_texts],
            "text": raw_texts,
            "cleaned_full_text": cleaned_texts
        })

        features = self.pipeline.transform(temp_df)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)
        else:
            # Fallback for models without predict_proba (e.g. LinearSVC)
            dec = self.model.decision_function(features)
            p1 = 1 / (1 + np.exp(-dec))
            probs = np.column_stack([1 - p1, p1])

        return probs

    def explain(
        self,
        text: str,
        title: str = "",
        num_features: int = 10,
        num_samples: int = 300
    ) -> Dict[str, Any]:
        """
        Generate LIME explanation for an article.
        Returns:
        - prediction label ('REAL' or 'FAKE')
        - probabilities
        - list of influencing features with directional impact
        - HTML rendering snippet
        """
        full_text = f"{title}. {text}".strip() if title else text.strip()
        if not full_text:
            return {
                "label": "UNKNOWN",
                "confidence": 0.0,
                "probabilities": {"REAL": 0.5, "FAKE": 0.5},
                "influencing_words": [],
                "html_snippet": "<p>Empty text provided.</p>"
            }

        # Predict first
        probs = self._predict_proba_for_lime([full_text])[0]
        prob_real = float(probs[0])
        prob_fake = float(probs[1])
        predicted_label = "FAKE" if prob_fake >= 0.5 else "REAL"
        confidence = prob_fake if predicted_label == "FAKE" else prob_real

        # Generate LIME explanation
        exp = self.explainer.explain_instance(
            text_instance=full_text,
            classifier_fn=self._predict_proba_for_lime,
            num_features=num_features,
            num_samples=num_samples,
            labels=(1,)  # Explain class 1 (FAKE)
        )

        explanation_list = exp.as_list(label=1)
        influencing_features = []

        for word, weight in explanation_list:
            pushes = "FAKE" if weight > 0 else "REAL"
            influencing_features.append({
                "word": str(word),
                "weight": round(float(weight), 4),
                "direction": pushes
            })

        # Generate custom styled HTML for clean embedding in Streamlit/UI
        html_highlighted = self._generate_html_badge_view(full_text, influencing_features)

        return {
            "label": predicted_label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "REAL": round(prob_real, 4),
                "FAKE": round(prob_fake, 4)
            },
            "influencing_features": influencing_features,
            "html_snippet": html_highlighted
        }

    def _generate_html_badge_view(self, text: str, features: List[Dict[str, Any]]) -> str:
        """Create visual highlighted HTML snippet of top tokens."""
        badges = []
        for feat in features:
            color = "#ff4d4f" if feat["direction"] == "FAKE" else "#52c41a"
            sign = "+" if feat["weight"] > 0 else ""
            badge = (
                f'<span style="display:inline-block; margin:4px; padding:4px 8px; border-radius:4px; '
                f'background-color:{color}; color:white; font-size:12px; font-weight:bold;">'
                f'{feat["word"]} ({sign}{feat["weight"]:.3f} &rarr; {feat["direction"]})'
                f'</span>'
            )
            badges.append(badge)

        html_out = (
            '<div style="font-family:sans-serif; margin-top:10px;">'
            '<h4>Top Influencing Keywords (LIME):</h4>'
            f'<div style="margin-bottom:12px;">{" ".join(badges)}</div>'
            '</div>'
        )
        return html_out


if __name__ == "__main__":
    explainer = NewsExplainer()
    sample_fake = (
        "BREAKING: Huge scandal exposed as secret documents reveal shocking treason in the government! "
        "Citizens are in an outrage over the corrupt establishment's horrific lies."
    )
    result = explainer.explain(sample_fake, num_features=6, num_samples=150)
    print("Explanation output:")
    print(f"Prediction: {result['label']} (Confidence: {result['confidence']})")
    print("Influencing words:", result["influencing_features"])
