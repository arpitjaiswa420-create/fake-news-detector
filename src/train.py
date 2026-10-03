"""
Model Training and Comparison Module for Fake News Detection.
Implements:
- Training and cross-validation across multiple model families:
    1. Logistic Regression (Linear baseline)
    2. Multinomial Naive Bayes (Probabilistic)
    3. Random Forest (Bagging ensemble)
    4. XGBoost (Gradient boosting)
    5. Neural Network / MLP (Deep Learning architecture)
- Stratified cross-validation and evaluation on hold-out validation/test splits
- Tracking Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix
- Automated model selection and artifact persistence (.joblib & .json)
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.features import TextMetadataCombinedPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train")

MODELS_DIR = PROJECT_ROOT / "models"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def get_model_zoo(random_state: int = 42) -> Dict[str, Any]:
    """Define the collection of model architectures for comparison."""
    return {
        "Logistic Regression": LogisticRegression(
            C=2.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state
        ),
        "Multinomial Naive Bayes": MultinomialNB(
            alpha=0.1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=25,
            n_jobs=-1,
            random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.15,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        ),
        "Deep Neural Net (MLP)": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=30,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state
        )
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def train_and_evaluate_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tfidf_max_features: int = 10000,
    include_metadata: bool = True,
    cv_folds: int = 5,
    sample_for_cv: Optional[int] = 10000
) -> Tuple[Dict[str, Any], Any, str]:
    """
    Fit feature pipeline, train and cross-validate each model, evaluate on test set,
    and return results dictionary and best performing model.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing and fitting feature pipeline (TF-IDF + Metadata)...")
    feature_pipeline = TextMetadataCombinedPipeline(
        tfidf_max_features=tfidf_max_features,
        include_metadata=include_metadata
    )

    X_train = feature_pipeline.fit_transform(train_df)
    y_train = train_df["label"].values

    X_val = feature_pipeline.transform(val_df)
    y_val = val_df["label"].values

    X_test = feature_pipeline.transform(test_df)
    y_test = test_df["label"].values

    logger.info(f"Feature matrix dimensions: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # For Naive Bayes, ensure non-negative inputs (metadata standard scaling can produce negative values)
    # MultinomialNB requires X >= 0, so we create a min-max shifted or tfidf-only view if needed
    model_zoo = get_model_zoo()
    results = {}

    best_model_name = None
    best_f1_score = -1.0
    best_model_obj = None

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, model in model_zoo.items():
        logger.info(f"--- Training model: {name} ---")
        start_time = time.time()

        # Handle Naive Bayes requirement for non-negative feature matrix
        is_nb = "Naive Bayes" in name
        if is_nb and include_metadata:
            # Train NB on pure TF-IDF sparse features (strictly non-negative)
            X_tr_model = feature_pipeline.tfidf.transform(train_df["cleaned_full_text"])
            X_va_model = feature_pipeline.tfidf.transform(val_df["cleaned_full_text"])
            X_te_model = feature_pipeline.tfidf.transform(test_df["cleaned_full_text"])
        else:
            X_tr_model = X_train
            X_va_model = X_val
            X_te_model = X_test

        # 5-fold Stratified Cross-validation on subset/full for efficiency
        if sample_for_cv and len(y_train) > sample_for_cv:
            cv_indices = np.random.RandomState(42).choice(len(y_train), size=sample_for_cv, replace=False)
            X_cv = X_tr_model[cv_indices]
            y_cv = y_train[cv_indices]
        else:
            X_cv = X_tr_model
            y_cv = y_train

        logger.info(f"Computing {cv_folds}-fold stratified cross-validation for {name}...")
        try:
            cv_scores = cross_val_score(model, X_cv, y_cv, cv=skf, scoring="f1", n_jobs=-1)
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
        except Exception as e:
            logger.warning(f"Cross-validation error on {name}: {e}. Falling back to single-fold.")
            cv_mean, cv_std = 0.0, 0.0

        # Fit on full training set
        logger.info(f"Fitting {name} on full train set ({X_tr_model.shape[0]} samples)...")
        model.fit(X_tr_model, y_train)
        training_time = time.time() - start_time

        # Validation set evaluation
        y_val_pred = model.predict(X_va_model)
        y_val_prob = model.predict_proba(X_va_model)[:, 1] if hasattr(model, "predict_proba") else None
        val_metrics = evaluate_predictions(y_val, y_val_pred, y_val_prob)

        # Test set evaluation
        y_test_pred = model.predict(X_te_model)
        y_test_prob = model.predict_proba(X_te_model)[:, 1] if hasattr(model, "predict_proba") else None
        test_metrics = evaluate_predictions(y_test, y_test_pred, y_test_prob)

        logger.info(
            f"[{name}] Val F1: {val_metrics['f1']:.4f} | Test F1: {test_metrics['f1']:.4f} | "
            f"Test Acc: {test_metrics['accuracy']:.4f} | Test ROC-AUC: {test_metrics['roc_auc']:.4f} | "
            f"CV F1: {cv_mean:.4f} (±{cv_std:.4f}) | Time: {training_time:.2f}s"
        )

        results[name] = {
            "training_time_seconds": round(training_time, 2),
            "cv_f1_mean": round(cv_mean, 4),
            "cv_f1_std": round(cv_std, 4),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

        # Track best model on Validation F1
        if val_metrics["f1"] > best_f1_score:
            best_f1_score = val_metrics["f1"]
            best_model_name = name
            best_model_obj = model

    logger.info(f"=== BEST MODEL SELECTED: {best_model_name} (Val F1: {best_f1_score:.4f}) ===")

    # Persist best model and feature pipeline
    best_model_path = MODELS_DIR / "best_model.joblib"
    pipeline_path = MODELS_DIR / "feature_pipeline.joblib"
    results_path = MODELS_DIR / "evaluation_results.json"
    comparison_csv_path = MODELS_DIR / "model_comparison.csv"

    logger.info(f"Saving best model to {best_model_path}...")
    joblib.dump(best_model_obj, best_model_path)

    logger.info(f"Saving feature pipeline to {pipeline_path}...")
    joblib.dump(feature_pipeline, pipeline_path)

    # Save summary dataframe
    comparison_rows = []
    for m_name, m_res in results.items():
        tm = m_res["test_metrics"]
        comparison_rows.append({
            "Model": m_name,
            "CV F1": f"{m_res['cv_f1_mean']:.4f} ± {m_res['cv_f1_std']:.4f}",
            "Test Accuracy": round(tm["accuracy"], 4),
            "Test Precision": round(tm["precision"], 4),
            "Test Recall": round(tm["recall"], 4),
            "Test F1-Score": round(tm["f1"], 4),
            "Test ROC-AUC": round(tm["roc_auc"], 4) if tm["roc_auc"] is not None else "N/A",
            "Train Time (s)": m_res["training_time_seconds"]
        })

    comparison_df = pd.DataFrame(comparison_rows).sort_values(by="Test F1-Score", ascending=False)
    comparison_df.to_csv(comparison_csv_path, index=False)
    logger.info(f"\nModel Comparison Summary:\n{comparison_df.to_string(index=False)}")

    # Save full JSON report
    with open(results_path, "w") as f:
        json.dump({
            "best_model_name": best_model_name,
            "best_val_f1": best_f1_score,
            "models": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    return results, best_model_obj, best_model_name


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and benchmark fake news detection models.")
    parser.add_argument("--sample", type=int, default=None, help="Train on a subsample for rapid benchmarking.")
    parser.add_argument("--max-features", type=int, default=8000, help="Max TF-IDF features.")
    parser.add_argument("--no-metadata", action="store_true", help="Disable engineered metadata features.")
    args = parser.parse_args()

    train_path = DATA_PROCESSED_DIR / "train.csv"
    val_path = DATA_PROCESSED_DIR / "val.csv"
    test_path = DATA_PROCESSED_DIR / "test.csv"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        logger.error("Processed data missing. Run src/preprocessing.py first!")
        sys.exit(1)

    logger.info("Loading processed datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    if args.sample and args.sample < len(train_df):
        logger.info(f"Subsampling training set to {args.sample} rows...")
        train_df = train_df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        val_sample = min(len(val_df), args.sample // 4)
        val_df = val_df.sample(n=val_sample, random_state=42).reset_index(drop=True)
        test_sample = min(len(test_df), args.sample // 4)
        test_df = test_df.sample(n=test_sample, random_state=42).reset_index(drop=True)

    train_and_evaluate_models(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tfidf_max_features=args.max_features,
        include_metadata=not args.no_metadata
    )
