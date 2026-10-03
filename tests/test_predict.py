"""
Unit tests for NewsPredictor inference engine and batch prediction.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.predict import NewsPredictor, ArticleScraper


@pytest.fixture(scope="module")
def predictor():
    return NewsPredictor(enable_lime=False)


def test_predict_real_sample(predictor):
    title = "Congress votes to pass annual federal budget extension"
    text = (
        "The Senate passed a temporary spending measure on Wednesday, averting a government shutdown. "
        "The bill passed with bipartisan support following committee hearings."
    )
    res = predictor.predict(text=text, title=title)
    assert res["label"] in ["REAL", "FAKE"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert "probabilities" in res
    assert "metadata_signals" in res
    assert res["metadata_signals"]["word_count"] > 10


def test_predict_empty_text(predictor):
    res = predictor.predict(text="", title="")
    assert res["label"] == "UNKNOWN"
    assert res["confidence"] == 0.0


def test_predict_batch_dataframe(predictor):
    df = pd.DataFrame({
        "title": [
            "Federal Reserve maintains interest rate benchmark",
            "SHOCKING CONSPIRACY: Secret alien base discovered under White House!!"
        ],
        "text": [
            "The central bank concluded its two-day policy meeting by keeping the target rate unchanged.",
            "You won't believe what whistleblowers have revealed about top politicians!"
        ]
    })
    preds_df = predictor.predict_batch_dataframe(df)
    assert "predicted_label" in preds_df.columns
    assert "confidence" in preds_df.columns
    assert len(preds_df) == 2
    assert preds_df["predicted_label"].iloc[0] in ["REAL", "FAKE"]


def test_scraper_mock():
    mock_html = """
    <html>
        <head><title>Test News Article</title></head>
        <body>
            <h1>Test Headline</h1>
            <article>
                <p>This is the first paragraph with substantial words describing the event.</p>
                <p>This is the second paragraph providing background details on the story.</p>
            </article>
        </body>
    </html>
    """
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.text = mock_html
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        data = ArticleScraper.scrape("https://example.com/test-news")
        assert data["title"] == "Test Headline"
        assert "first paragraph" in data["text"]
        assert "second paragraph" in data["text"]
