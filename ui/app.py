"""
Streamlit Web UI for Fake News Detection, Explainability, and Linguistic Analytics.
Features:
- Single article & URL news verification with confidence score
- Token-level LIME explainability badge highlights
- Batch CSV upload and instant classification
- Interactive model performance benchmarks & linguistic analytics
- Ethical disclaimer and dataset transparency
"""

import sys
import io
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import NewsPredictor, ArticleScraper

st.set_page_config(
    page_title="TruthPulse | Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 14px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .badge-fake {
        background-color: #EF4444;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-real {
        background-color: #10B981;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 12px;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #92400E;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading machine learning models and feature pipeline...")
def load_predictor():
    return NewsPredictor(enable_lime=True)


try:
    predictor = load_predictor()
    model_ready = True
except Exception as e:
    st.error(f"Failed to load trained model: {e}. Please ensure you ran `python src/train.py`.")
    model_ready = False


# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/news.png", width=64)
    st.title("TruthPulse AI")
    st.markdown("**Production Fake News Detector**")
    st.markdown("Combines TF-IDF N-grams, stylometric metadata, gradient boosted trees, and LIME explainability.")
    
    st.divider()
    st.subheader("Model Status")
    if model_ready:
        st.success(f"Loaded: `{type(predictor.model).__name__}`")
        st.caption("Fitted on 27,370 balanced news articles.")
    else:
        st.error("Model offline")

    st.divider()
    st.subheader("Dataset & Methodology")
    st.markdown("""
    - **Corpus**: ISOT Fake News Dataset (~44,898 articles)
    - **Real Source**: Reuters International Wires
    - **Fake Source**: PolitiFact / FactCheck.org flagged items
    - **Caveat Mitigation**: Stripped Reuters publisher wire tags to prevent publication bias.
    """)

    st.divider()
    st.subheader("Ethical Disclaimer")
    st.caption("""
    This system evaluates statistical, lexical, and stylometric patterns in text.
    It does not independently verify ground truth facts and should never be relied
    upon as a sole definitive arbiter of factual authenticity.
    """)


# --- Header ---
st.markdown('<div class="main-header">TruthPulse: Fake News Detector & Explainability</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyze articles for linguistic markers of misinformation with model transparency</div>', unsafe_allow_html=True)

# --- Navigation Tabs ---
tab1, tab2, tab3 = st.tabs([
    "🔍 Article & URL Detector",
    "📁 Batch CSV Predictor",
    "📊 Model Benchmark & Linguistic Insights"
])


# ==========================================
# TAB 1: Single Article & URL Detector
# ==========================================
with tab1:
    st.subheader("Analyze News Content")
    
    input_mode = st.radio("Choose Input Mode:", ["Paste Article Text", "Analyze Web URL"], horizontal=True)

    sample_real_title = "Senate passes bipartisan infrastructure plan with overwhelming majority"
    sample_real_text = (
        "The United States Senate voted on Tuesday to approve the sweeping infrastructure legislation, "
        "authorizing $1 trillion in federal funding for highways, bridges, clean water, and broadband. "
        "Lawmakers from both parties praised the committee agreement reached after months of negotiation."
    )

    sample_fake_title = "SHOCKING BOMBSHELL: Secret documents expose corrupt treason in government!!"
    sample_fake_text = (
        "You will not believe the horrific lies being covered up by the establishment media! "
        "Shocking leaks prove the corrupt officials have been conspiring against ordinary citizens. "
        "Share this urgent truth before they delete it immediately!"
    )

    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        load_real = st.button("Load Real News Sample", use_container_width=True)
    with col_btn2:
        load_fake = st.button("Load Fake News Sample", use_container_width=True)

    default_title = sample_fake_title if load_fake else (sample_real_title if load_real else "")
    default_text = sample_fake_text if load_fake else (sample_real_text if load_real else "")

    if input_mode == "Paste Article Text":
        title_input = st.text_input("Article Headline / Title (optional but recommended):", value=default_title, placeholder="e.g. Senate passes budget measure...")
        text_input = st.text_area("Article Body Text:", value=default_text, height=180, placeholder="Paste the full body of the article here...")
        url_input = None
    else:
        url_input = st.text_input("News Article URL:", placeholder="https://www.reuters.com/... or https://example.com/news/...")
        title_input = None
        text_input = None

    col_opt1, col_opt2 = st.columns([1, 1])
    with col_opt1:
        enable_lime = st.checkbox("Generate Token Explainability (LIME)", value=True, help="Computes word-by-word importance weights showing what pulled toward Fake vs Real.")
    with col_opt2:
        num_lime_tokens = st.slider("Number of Influencing Words to Show", min_value=4, max_value=15, value=8)

    predict_btn = st.button("⚡ Run Fake News Analysis", type="primary", use_container_width=True)

    if predict_btn and model_ready:
        with st.spinner("Analyzing text patterns, calculating stylometric metrics, and evaluating model..."):
            try:
                if input_mode == "Analyze Web URL":
                    if not url_input or not url_input.strip():
                        st.warning("Please provide a valid URL.")
                        st.stop()
                    res = predictor.predict_url(url=url_input.strip(), explain=enable_lime)
                    st.info(f"**Extracted Article Title:** {res.get('scraped_article', {}).get('title', 'N/A')}")
                else:
                    if not text_input and not title_input:
                        st.warning("Please enter either an article title or body text.")
                        st.stop()
                    res = predictor.predict(
                        text=text_input or "",
                        title=title_input or "",
                        explain=enable_lime,
                        num_explanation_words=num_lime_tokens
                    )

                st.divider()

                # --- Results Display ---
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    is_fake = res["label"] == "FAKE"
                    badge_class = "badge-fake" if is_fake else "badge-real"
                    st.markdown(f'<div class="{badge_class}">Predicted: {res["label"]}</div>', unsafe_allow_html=True)
                    st.metric("Confidence Score", f"{res['confidence'] * 100:.1f}%")

                    st.write("**Class Probability Distribution:**")
                    fake_p = float(np.clip(res["probabilities"]["FAKE"], 0.0, 1.0))
                    real_p = float(np.clip(res["probabilities"]["REAL"], 0.0, 1.0))
                    st.progress(fake_p, text=f"Fake: {fake_p*100:.1f}%")
                    st.progress(real_p, text=f"Real: {real_p*100:.1f}%")

                with res_col2:
                    st.markdown("##### Stylometric & Linguistic Signals")
                    meta = res.get("metadata_signals", {})
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.metric("Word Count", meta.get("word_count", 0))
                        st.metric("Exclamation Marks", meta.get("exclamation_count", 0))
                    with m_col2:
                        st.metric("Body ALL-CAPS", f"{meta.get('caps_ratio_percent', 0.0):.1f}%")
                        st.metric("Title ALL-CAPS", f"{meta.get('title_caps_percent', 0.0):.1f}%")
                    with m_col3:
                        st.metric("Reading Ease", f"{meta.get('readability_score', 60.0):.1f}")
                        st.metric("Sentiment Polarity", f"{meta.get('sentiment_polarity', 0.0):.2f}")

                # --- LIME Explainability ---
                if "explanation" in res and res["explanation"]:
                    st.divider()
                    st.markdown("#### 🧠 Model Explainability (LIME)")
                    st.caption("Red tokens push the prediction toward **FAKE**; Green tokens push toward **REAL**.")
                    st.markdown(res["explanation"]["html_snippet"], unsafe_allow_html=True)

                    # Table breakdown
                    exp_df = pd.DataFrame(res["explanation"]["influencing_features"])
                    if not exp_df.empty:
                        exp_df.columns = ["Word / Token", "Attribution Weight", "Pushes Toward"]
                        st.dataframe(exp_df, use_container_width=True)

            except Exception as ex:
                st.error(f"Error during prediction: {ex}")


# ==========================================
# TAB 2: Batch CSV Predictor
# ==========================================
with tab2:
    st.subheader("Batch CSV Prediction")
    st.markdown("Upload a CSV file containing multiple articles for fast vectorized batch classification.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None and model_ready:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(df_batch):,} rows**. Columns found: `{list(df_batch.columns)}`")

            col_t, col_b = st.columns(2)
            with col_t:
                title_col_sel = st.selectbox("Select Title Column (or None)", ["None"] + list(df_batch.columns))
            with col_b:
                text_col_sel = st.selectbox("Select Article Text Column", list(df_batch.columns))

            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Processing batch predictions..."):
                    title_arg = "" if title_col_sel == "None" else title_col_sel
                    results_df = predictor.predict_batch_dataframe(
                        df_batch,
                        title_col=title_arg,
                        text_col=text_col_sel
                    )

                    fake_count = int((results_df["predicted_label"] == "FAKE").sum())
                    real_count = int((results_df["predicted_label"] == "REAL").sum())

                    st.success(f"Batch prediction complete for {len(results_df):,} articles!")

                    b_c1, b_c2, b_c3 = st.columns(3)
                    b_c1.metric("Predicted FAKE", f"{fake_count:,} ({fake_count/len(results_df)*100:.1f}%)")
                    b_c2.metric("Predicted REAL", f"{real_count:,} ({real_count/len(results_df)*100:.1f}%)")
                    b_c3.metric("Average Confidence", f"{results_df['confidence'].mean()*100:.1f}%")

                    st.dataframe(results_df.head(50), use_container_width=True)

                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Full Classified CSV",
                        data=csv_buffer.getvalue(),
                        file_name="classified_news_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as batch_ex:
            st.error(f"Error processing CSV: {batch_ex}")


# ==========================================
# TAB 3: Model Benchmark & Insights
# ==========================================
with tab3:
    st.subheader("Model Benchmark & Evaluation")
    
    comp_csv_path = PROJECT_ROOT / "models" / "model_comparison.csv"
    if comp_csv_path.exists():
        comp_df = pd.read_csv(comp_csv_path)
        st.dataframe(comp_df, use_container_width=True)
    else:
        st.info("Benchmark summary table not yet generated. Run `python src/train.py`.")

    chart1_path = PROJECT_ROOT / "models" / "model_benchmark.png"
    chart2_path = PROJECT_ROOT / "models" / "confusion_matrix.png"
    chart3_path = PROJECT_ROOT / "models" / "linguistic_patterns.png"

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if chart1_path.exists():
            st.image(str(chart1_path), caption="Cross-Model Accuracy & F1-Score Benchmark", use_container_width=True)
    with b_col2:
        if chart2_path.exists():
            st.image(str(chart2_path), caption="Hold-Out Test Confusion Matrix (Best Model)", use_container_width=True)

    if chart3_path.exists():
        st.divider()
        st.subheader("Linguistic Signatures of Misinformation")
        st.image(str(chart3_path), caption="Stylometric Differences: Fake vs Real Articles", use_container_width=True)

    st.markdown("""
    ### 🔬 Key Scientific Observations
    1. **Sensationalism & Punctuation**: Fake news titles contain substantial exclamation marks and question marks, whereas verified news agencies adhere to strict journalistic neutrality guidelines.
    2. **Headline Capitalization**: Fake news titles frequently deploy ALL-CAPS words (*"SHOCKING"*, *"MUST SEE"*, *"BREAKING"*), aiming to evoke emotional reactions.
    3. **Publisher Bias Control**: Explicit news wire prefixes (e.g., `WASHINGTON (Reuters) -`) were explicitly stripped prior to model training to ensure the model generalizes across independent news outlets.
    """)

# --- Footer Disclaimer ---
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Limitations & Ethical Disclaimer:</strong>
    This machine learning model was trained on historical news corpora (including ISOT and PolitiFact). 
    While it demonstrates high discriminative power on stylometric and linguistic patterns, language changes over time.
    Automated NLP models should be used as assistive screening tools, not authoritative judges of objective reality.
</div>
""", unsafe_allow_html=True)
