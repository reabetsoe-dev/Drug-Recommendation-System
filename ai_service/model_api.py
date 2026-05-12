"""
AI service API for model inference and drug recommendations.
"""

from __future__ import annotations

import os
import pickle
import re
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from nltk.corpus import stopwords
from nltk.data import find as nltk_find
from nltk.tokenize import word_tokenize
from pydantic import BaseModel, Field

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import settings


app = FastAPI(title="AIDrugReview AI Service", version="1.0.0")


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "ai_service",
        "status": "running",
        "health": f"{settings.ai_service_base_url}/health",
        "docs": f"{settings.ai_service_base_url}/docs",
        "analyze_endpoint": f"{settings.ai_service_base_url}/analyze",
    }


CONDITION_PATTERNS = {
    "Depression": "Depression",
    "High Blood Pressure": "High Blood Pressure",
    "Type 2 Diabetes": "Diabetes, Type 2",
}

POSITIVE_WORDS = {
    "better",
    "best",
    "effective",
    "improved",
    "relief",
    "stable",
    "good",
    "great",
    "helped",
    "manageable",
}

NEGATIVE_WORDS = {
    "worse",
    "worst",
    "pain",
    "anxious",
    "sad",
    "hopeless",
    "dizzy",
    "nausea",
    "fatigue",
    "tired",
    "severe",
    "bad",
}

INVALID_REVIEW_MESSAGE = (
    "Please enter a clear symptom or medication experience using meaningful words. "
    "The current text does not provide enough clinical context for a reliable prediction."
)

model: Any = None
vectorizer: Any = None
label_encoder: Any = None
train_df: Optional[pd.DataFrame] = None
load_error: Optional[str] = None


class AnalyzeRequest(BaseModel):
    text: Optional[str] = None
    condition: Optional[str] = None
    top_n: int = Field(default=5, ge=1, le=25)
    min_reviews: int = Field(default=5, ge=1, le=200)


def configure_runtime_environment() -> None:
    os.environ.setdefault("NLTK_DATA", str(settings.nltk_data_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(settings.mplconfig_dir))
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    settings.mplconfig_dir.mkdir(parents=True, exist_ok=True)


def ensure_nltk_resources() -> None:
    import nltk

    if str(settings.nltk_data_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(settings.nltk_data_dir))

    missing_resources = []
    for resource in ("corpora/stopwords", "tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk_find(resource)
        except LookupError:
            missing_resources.append(resource)

    if missing_resources:
        raise RuntimeError(
            "Missing NLTK resources in local nltk_data directory: "
            + ", ".join(missing_resources)
            + f". Expected path: {settings.nltk_data_dir}"
        )


def load_artifacts() -> None:
    global model, vectorizer, label_encoder, train_df

    required_files = [
        settings.model_path,
        settings.vectorizer_path,
        settings.label_encoder_path,
        settings.train_dataset_path,
    ]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Required files are missing: " + ", ".join(missing_files)
        )

    with settings.model_path.open("rb") as model_file:
        model = pickle.load(model_file)

    with settings.vectorizer_path.open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    with settings.label_encoder_path.open("rb") as label_encoder_file:
        label_encoder = pickle.load(label_encoder_file)

    train_df = pd.read_csv(settings.train_dataset_path)


def clean_text(text: str) -> str:
    if text is None or text == "":
        return ""

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    try:
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        text = " ".join(tokens)
    except Exception:
        words = text.split()
        stop_words = set(stopwords.words("english"))
        words = [word for word in words if word not in stop_words and len(word) > 2]
        text = " ".join(words)

    return text


def predict_sentiment(cleaned_text: str) -> Dict[str, Any]:
    if not cleaned_text:
        return {"label": "neutral", "score": 0}

    tokens = cleaned_text.split()
    positive_hits = sum(1 for token in tokens if token in POSITIVE_WORDS)
    negative_hits = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    score = positive_hits - negative_hits

    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    else:
        label = "neutral"

    return {"label": label, "score": score}


def estimate_rating_prediction(condition: str, confidence: float) -> Optional[float]:
    if train_df is None:
        return None

    actual_pattern = CONDITION_PATTERNS.get(condition, condition)
    subset = train_df[train_df["condition"].str.contains(actual_pattern, case=False, na=False)]
    if subset.empty:
        return None

    base_rating = float(subset["rating"].mean())
    adjusted_rating = base_rating + ((confidence - 0.5) * 1.5)
    adjusted_rating = max(1.0, min(10.0, adjusted_rating))
    return round(adjusted_rating, 2)


def recommend_drugs(
    dataframe: pd.DataFrame,
    condition: str,
    top_n: int = 5,
    min_reviews: int = 5,
) -> List[Dict[str, Any]]:
    actual_condition = CONDITION_PATTERNS.get(condition, condition)
    condition_drugs = dataframe[
        dataframe["condition"].str.contains(actual_condition, case=False, na=False)
    ]

    if condition_drugs.empty:
        return []

    drug_stats = (
        condition_drugs.groupby("drugName")
        .agg({"rating": ["mean", "count"], "usefulCount": "mean"})
        .round(2)
    )
    drug_stats.columns = ["avg_rating", "review_count", "avg_useful_count"]
    drug_stats = drug_stats.reset_index()
    drug_stats = drug_stats[drug_stats["review_count"] >= min_reviews]
    drug_stats = drug_stats.sort_values(
        ["avg_rating", "review_count"], ascending=[False, False]
    )

    return drug_stats.head(top_n).to_dict(orient="records")


def run_prediction(text: str) -> Dict[str, Any]:
    cleaned_text = clean_text(text)
    if cleaned_text == "":
        raise ValueError(INVALID_REVIEW_MESSAGE)

    text_vector = vectorizer.transform([cleaned_text])
    if getattr(text_vector, "nnz", 0) == 0:
        raise ValueError(INVALID_REVIEW_MESSAGE)

    prediction = model.predict(text_vector)[0]
    prediction_proba = model.predict_proba(text_vector)[0]

    condition = label_encoder.inverse_transform([prediction])[0]
    confidence = float(prediction_proba[int(prediction)])
    proba_dict = {
        class_name: float(prob)
        for class_name, prob in zip(label_encoder.classes_, prediction_proba)
    }

    sentiment_prediction = predict_sentiment(cleaned_text)
    rating_prediction = estimate_rating_prediction(condition, confidence)

    return {
        "cleaned_text": cleaned_text,
        "predicted_condition": condition,
        "confidence": confidence,
        "probabilities": proba_dict,
        "sentiment_prediction": sentiment_prediction,
        "rating_prediction": rating_prediction,
    }


@app.on_event("startup")
def startup_event() -> None:
    global load_error
    try:
        configure_runtime_environment()
        ensure_nltk_resources()
        load_artifacts()
        load_error = None
    except Exception as exc:
        load_error = str(exc)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "service": "ai_service",
        "status": "ok" if load_error is None else "error",
        "ready": load_error is None,
        "error": load_error,
        "paths": {
            "model": str(settings.model_path),
            "vectorizer": str(settings.vectorizer_path),
            "label_encoder": str(settings.label_encoder_path),
            "dataset": str(settings.train_dataset_path),
            "nltk_data": str(settings.nltk_data_dir),
        },
    }


@app.post("/analyze")
def analyze(payload: AnalyzeRequest) -> Dict[str, Any]:
    if load_error is not None:
        raise HTTPException(status_code=503, detail=f"AI service is not ready: {load_error}")

    if model is None or vectorizer is None or label_encoder is None or train_df is None:
        raise HTTPException(status_code=503, detail="AI service artifacts are not loaded.")

    if not payload.text and not payload.condition:
        raise HTTPException(
            status_code=400,
            detail="Provide either review text or condition to analyze.",
        )

    prediction_result: Dict[str, Any] = {
        "cleaned_text": None,
        "predicted_condition": None,
        "confidence": None,
        "probabilities": {},
        "sentiment_prediction": None,
        "rating_prediction": None,
    }

    if payload.text and payload.text.strip():
        try:
            prediction_result = run_prediction(payload.text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    effective_condition = payload.condition or prediction_result["predicted_condition"]
    recommendations = []
    if effective_condition:
        recommendations = recommend_drugs(
            train_df,
            effective_condition,
            top_n=payload.top_n,
            min_reviews=payload.min_reviews,
        )

    return {
        "predicted_condition": prediction_result["predicted_condition"],
        "confidence": prediction_result["confidence"],
        "probabilities": prediction_result["probabilities"],
        "sentiment_prediction": prediction_result["sentiment_prediction"],
        "rating_prediction": prediction_result["rating_prediction"],
        "recommendations": recommendations,
        "condition_used_for_recommendation": effective_condition,
        "top_n": payload.top_n,
        "min_reviews": payload.min_reviews,
    }
