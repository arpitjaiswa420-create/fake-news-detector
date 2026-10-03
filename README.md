# TruthPulse: Production-Grade Fake News Detector & Explainability Engine

TruthPulse is an end-to-end Machine Learning and Natural Language Processing (NLP) system designed to identify misinformation and clickbait patterns in news articles. It combines **TF-IDF n-gram feature representations**, **stylometric metadata engineering** (headline capitalization, punctuation density, reading ease, lexical sentiment), **ensemble and deep learning models**, and **token-level LIME explainability**, packaged with a **FastAPI REST API** and an interactive **Streamlit Web Dashboard**.

---

## 🌟 Key Highlights

- **Dataset Quality & De-biasing**: Automated pipeline downloads and processes the ISOT / Kaggle Fake and Real News dataset (~44,898 articles). Removes news agency wire datelines (e.g., `(Reuters) -`) to prevent trivial publisher overfitting.
- **Multimodal Stylometric Engineering**: Extracts 14 structural and emotional signals (reading ease, punctuation density, sensational ALL-CAPS ratios, lexical sentiment).
- **Comprehensive Model Benchmark**: Trained and cross-validated across 5 model families:
  - **Logistic Regression** (Fast linear baseline)
  - **Multinomial Naive Bayes** (Probabilistic baseline)
  - **Random Forest** (Tree bagging)
  - **XGBoost** (Gradient boosted trees — **Top Performer**)
  - **Deep Neural Network (MLP)** (Multi-layer perceptron with ReLU & Adam)
- **Model Explainability (LIME)**: Visualizes which specific words or phrases push a prediction toward "FAKE" vs "REAL" with directional weights.
- **URL Scraping Feature**: Accepts direct web URLs, automatically parses headline and article bodies using BeautifulSoup, and runs live inference.
- **Batch CSV Processing**: Upload CSV files for instant vectorized batch predictions.
- **Production Deployment**: Includes a documented FastAPI REST API, Streamlit web app, Dockerfile, and docker-compose orchestration.
- **100% Tested**: 20 comprehensive unit and integration tests with `pytest`.

---

## 📁 Project Structure

```
fake_news_detector/
├── api/
│   ├── __init__.py
│   └── app.py                     # FastAPI REST API (/predict, /batch-predict, /health, /metrics)
├── data/
│   ├── raw/                       # Raw downloaded datasets (True.csv, Fake.csv)
│   └── processed/                 # Stratified train.csv, val.csv, test.csv
├── models/
│   ├── best_model.joblib          # Persisted top-performing model
│   ├── feature_pipeline.joblib    # Fitted TF-IDF + Metadata pipeline
│   ├── evaluation_results.json    # Full cross-validation and test benchmark metrics
│   ├── model_comparison.csv       # Comparison summary table
│   ├── model_benchmark.png        # Bar chart comparison
│   ├── confusion_matrix.png       # Confusion matrix visualization
│   └── linguistic_patterns.png    # Stylometric differences chart
├── src/
│   ├── __init__.py
│   ├── download_data.py           # Automated dataset downloader and validator
│   ├── preprocessing.py          # Text cleaning, publisher de-biasing, splitting
│   ├── features.py               # TF-IDF + stylometric metadata extractors
│   ├── train.py                  # Model zoo training, 5-fold CV, model selection
│   ├── evaluate.py               # Evaluation plotting and linguistic analysis
│   ├── predict.py                # Inference engine & URL scraper
│   └── explain.py                # LIME token-level attribution explainer
├── tests/
│   ├── test_preprocessing.py     # Unit tests for text cleaning and splitting
│   ├── test_features.py          # Unit tests for feature extraction
│   ├── test_predict.py           # Unit tests for inference & web scraper
│   └── test_api.py               # Integration tests for FastAPI endpoints
├── ui/
│   └── app.py                     # Streamlit interactive web interface
├── Dockerfile                     # Containerization build recipe
├── docker-compose.yml             # Orchestration for API + UI
├── requirements.txt               # Locked production dependencies
├── pytest.ini                     # Pytest configuration
├── app.py                         # Root entry point wrapper
└── README.md                      # Documentation
```

---

## 📊 Dataset & Crucial Data Caveat

### Source & License
- **Source**: University of Victoria ISOT Fake News Dataset (Ahmed, Traore & Saad, 2017) and Kaggle Fake and Real News Dataset.
- **Volume**: 44,898 total raw articles (21,417 Real from Reuters wires; 23,481 Fake from PolitiFact/FactCheck.org).
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike (Academic / Research use).

### Critical Dataset Caveat & De-Biasing
> [!IMPORTANT]
> In raw ISOT/Kaggle datasets, nearly all real articles begin with news agency datelines like `"WASHINGTON (Reuters) - "`. A naive model trained on raw text learns to associate the single word `"Reuters"` with truth, giving an artificial 99.9% accuracy that fails completely when testing news from other outlets like BBC, AP, or CNN.
> 
> **Our Mitigation**: The `TextCleaner` in `src/preprocessing.py` explicitly strips news wire datelines and agency signatures before feature extraction. This forces models to learn genuine semantic and stylometric characteristics rather than publisher signatures.

---

## 📈 Model Benchmark Results

Models were evaluated using **5-fold Stratified Cross-Validation** on the training set (27,370 articles) and verified on an independent hold-out test set (5,865 articles):

| Model | 5-Fold CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Training Time |
|---|---|---|---|---|---|---|---|
| **Deep Neural Net (MLP)** | 0.9957 ± 0.0019 | **99.81%** | 99.85% | 99.74% | **0.9980** | **1.0000** | 118s |
| **XGBoost** | **0.9965 ± 0.0011** | **99.76%** | **99.89%** | 99.59% | **0.9974** | **1.0000** | 229s |
| **Logistic Regression** | 0.9942 ± 0.0021 | 99.74% | 99.85% | 99.59% | 0.9972 | 0.9999 | **8.7s** ⚡ |
| **Random Forest** | 0.9948 ± 0.0021 | 99.66% | 99.85% | 99.40% | 0.9963 | 1.0000 | 12.6s |
| **Multinomial Naive Bayes** | 0.9502 ± 0.0073 | 95.50% | 94.49% | 95.75% | 0.9512 | 0.9899 | 15.5s |

### Hold-out Test Confusion Matrix (XGBoost)
- **True Negatives (Real correctly identified)**: 3,177
- **False Positives (Real flagged as Fake)**: 3
- **False Negatives (Fake missed)**: 11
- **True Positives (Fake correctly caught)**: 2,674

---

## 🔬 Linguistic & Stylometric Insights

Our feature extractor tracks 14 metadata attributes that reveal profound differences between real and fake news:

1. **Sensational Exclamation Marks**:
   - Fake news headlines contain **0.315** exclamation marks per title on average.
   - Real journalism headlines contain **0.000** exclamation marks (professional wire editors forbid sensational punctuation in titles).
2. **ALL-CAPS Shouting**:
   - Fake news headlines have a **19.8% capital letter ratio**, utilizing aggressive uppercase words (*"BOMBSHELL"*, *"SHOCKING"*, *"MUST READ"*).
   - Real news headlines average **9.4% capital letters**, adhering to standard journalistic casing.
3. **Flesch Reading Ease**:
   - Real news articles maintain a higher readability score (~58.2), with structured syntax and direct quotes.
   - Fake news exhibits more extreme distributions (either extremely simple sensational clickbait or dense conspiratorial paragraphs).
4. **Sentiment Polarity**:
   - Fake news displays significantly higher negative lexical sentiment density (*"scandal"*, *"treason"*, *"lies"*, *"corrupt"*).

---

## 🧠 Explainability with LIME

TruthPulse integrates **LIME (Local Interpretable Model-agnostic Explanations)**. For any given article:
- **Red tokens** push the prediction toward **FAKE** (e.g. *"breaking"*, *"scandal"*, *"treason"*, *"shocking"*).
- **Green tokens** push the prediction toward **REAL** (e.g. *"said"*, *"spokesman"*, *"reuters"*, *"meeting"*, *"committee"*).

In the Streamlit UI, these are displayed as colored badges with exact attribution weights, making the model's decision-making fully transparent to investigators.

---

## 🚀 Quickstart & Setup

### 1. Prerequisites & Environment Setup

```bash
# Clone or navigate to the repository
cd fake_news_detector

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Data Download & Training (Optional - Pre-trained Artifacts Included)

```bash
# 1. Download raw data (ISOT corpus)
python src/download_data.py

# 2. Clean, de-bias, and perform stratified splits
python src/preprocessing.py

# 3. Train all models, run cross-validation, and save best model
python src/train.py

# 4. Generate evaluation reports & plots
python src/evaluate.py
```

### 3. Run the Unit & Integration Tests

```bash
python -m pytest
```
Output:
```
tests/test_api.py ......
tests/test_features.py ....
tests/test_predict.py ....
tests/test_preprocessing.py ......
============================= 20 passed in 6.65s =============================
```

---

## 💻 Web Interface & REST API Usage

### Launch the Streamlit Web Application

```bash
streamlit run ui/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser:
- **Tab 1: Single Article / URL Detector**: Paste article text or enter any web URL to extract and classify, view confidence gauge, stylometric breakdown, and LIME keyword badges.
- **Tab 2: Batch CSV Predictor**: Upload CSV with news articles for instant batch classification and downloadable results.
- **Tab 3: Model Benchmark & Insights**: Explore the cross-model performance metrics and linguistic charts.

---

### Launch the FastAPI REST Server

```bash
python app.py
# or
uvicorn api.app:app --host 0.0.0.0 --port 8000
```
Interactive Swagger API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

#### API Endpoints:

#### 1. Single Article Prediction: `POST /predict`

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Senate passes bipartisan infrastructure plan with overwhelming majority",
       "text": "The United States Senate voted on Tuesday to approve sweeping infrastructure legislation.",
       "explain": true
     }'
```

**Response Example:**
```json
{
  "label": "REAL",
  "confidence": 0.9982,
  "probabilities": {
    "REAL": 0.9982,
    "FAKE": 0.0018
  },
  "metadata_signals": {
    "word_count": 16,
    "caps_ratio_percent": 3.8,
    "title_caps_percent": 11.2,
    "exclamation_count": 0,
    "readability_score": 62.4,
    "sentiment_polarity": 0.25
  },
  "explanation": {
    "influencing_features": [
      {"word": "senate", "weight": -0.124, "direction": "REAL"},
      {"word": "passes", "weight": -0.098, "direction": "REAL"}
    ]
  },
  "disclaimer": "Notice: This prediction is based on statistical machine learning patterns and stylometric signatures. It does not replace independent journalistic fact-checking."
}
```

#### 2. Predict Directly from URL: `POST /predict`

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://www.reuters.com/world/us/bipartisan-senate-measure-2026-05-10/",
       "explain": false
     }'
```

#### 3. Batch CSV Prediction: `POST /batch-predict`

```bash
curl -X POST "http://localhost:8000/batch-predict" \
     -F "file=@sample_news.csv" \
     -F "title_col=title" \
     -F "text_col=text"
```

---

## 🐳 Docker Deployment

Run both the FastAPI backend and Streamlit UI using Docker Compose:

```bash
# Build and run containers
docker-compose up --build

# API accessible at http://localhost:8000
# UI accessible at http://localhost:8501
```

---

## ⚠️ Ethical Considerations & Limitations

> [!CAUTION]
> 1. **Not a Ground-Truth Oracle**: This system detects **linguistic and stylometric signatures** associated with misinformation (e.g. sensational vocabulary, ragebait punctuation, unverified phrasing). It does not cross-reference external real-time fact databases or establish objective ground truth.
> 2. **Temporal Drift**: Language, political discourse, and disinformation tactics evolve. A model trained on 2016-2020 news corpora may perform differently on modern events without periodic fine-tuning.
> 3. **Satire vs Malice**: Satirical publications (e.g. *The Onion*) employ stylometric patterns mimicking both real and exaggerated news and may be flagged as fake.
> 4. **Assisted Verification**: Always cross-reference high-stakes claims with established independent fact-checking bodies (e.g. Snopes, PolitiFact, AP Fact Check, Reuters Fact Check).
