"""
FastAPI REST API for Fake News Detection.
Endpoints:
- GET  /              : API metadata and sitemap
- GET  /health        : Health check
- GET  /metrics       : Benchmark evaluation metrics
- POST /predict       : Single article or URL classification with confidence & LIME explanation
- POST /batch-predict : CSV batch upload classification
"""

import sys
import io
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import NewsPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api")

from contextlib import asynccontextmanager

# Global predictor instance
predictor: Optional[NewsPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    try:
        logger.info("Initializing NewsPredictor on API startup...")
        predictor = NewsPredictor(enable_lime=True)
        logger.info("NewsPredictor successfully initialized.")
    except Exception as e:
        logger.error(f"Error loading model on startup: {e}")
        predictor = None
    yield


app = FastAPI(
    title="Fake News Detector API",
    description="Production-ready REST API for fake and real news classification using machine learning, stylometric NLP, and LIME explainability.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for web UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Request / Response Models ---

class PredictRequest(BaseModel):
    title: Optional[str] = Field(default="", description="Headline of the article.")
    text: Optional[str] = Field(default="", description="Body text of the article.")
    url: Optional[str] = Field(default=None, description="Direct URL of a news article to scrape and analyze.")
    explain: bool = Field(default=False, description="Whether to include token-level LIME feature attribution.")
    num_explanation_words: int = Field(default=8, ge=1, le=25, description="Number of top influencing words.")


class MetadataSignals(BaseModel):
    word_count: int
    caps_ratio_percent: float
    title_caps_percent: float
    exclamation_count: int
    readability_score: float
    sentiment_polarity: float


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]
    metadata_signals: Optional[MetadataSignals] = None
    explanation: Optional[Dict[str, Any]] = None
    scraped_article: Optional[Dict[str, Any]] = None
    disclaimer: str = (
        "Notice: This prediction is based on statistical machine learning patterns "
        "and stylometric signatures. It does not replace independent journalistic fact-checking."
    )


from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Request
from fastapi.responses import FileResponse

# --- Endpoints ---

@app.get("/")
def root(request: Request):
    accept = request.headers.get("accept", "")
    html_path = PROJECT_ROOT / "public" / "index.html"
    if "application/json" not in accept and html_path.exists():
        return FileResponse(html_path)
    return {
        "service": "Fake News Detector API",
        "status": "online",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "predict": "POST /predict",
            "batch_predict": "POST /batch-predict",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to initialize.")
    return {
        "status": "healthy",
        "model_loaded": True,
        "best_model": type(predictor.model).__name__
    }


@app.get("/metrics")
def get_metrics():
    metrics_path = PROJECT_ROOT / "models" / "evaluation_results.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics report not found. Run training first.")
    with open(metrics_path, "r") as f:
        data = json.load(f)
    return data


@app.post("/predict", response_model=PredictResponse)
def predict_article(req: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")

    try:
        if req.url and req.url.strip():
            result = predictor.predict_url(
                url=req.url.strip(),
                explain=req.explain
            )
        elif (req.text and req.text.strip()) or (req.title and req.title.strip()):
            result = predictor.predict(
                text=req.text or "",
                title=req.title or "",
                explain=req.explain,
                num_explanation_words=req.num_explanation_words
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either article 'text', 'title', or a valid 'url'."
            )

        return PredictResponse(**result)

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(
    file: UploadFile = File(...),
    title_col: str = Query("title", description="Name of the title column in CSV"),
    text_col: str = Query("text", description="Name of the text column in CSV")
):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        if text_col not in df.columns and title_col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain at least a '{title_col}' or '{text_col}' column."
            )

        results_df = predictor.predict_batch_dataframe(df, title_col=title_col, text_col=text_col)

        summary = {
            "total_articles": len(results_df),
            "predicted_fake": int((results_df["predicted_label"] == "FAKE").sum()),
            "predicted_real": int((results_df["predicted_label"] == "REAL").sum()),
            "avg_confidence": round(float(results_df["confidence"].mean()), 4),
            "preview_results": results_df.head(20).to_dict(orient="records")
        }
        return summary

    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process CSV: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=False)
