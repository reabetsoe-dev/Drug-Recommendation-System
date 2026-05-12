"""
Model creation pipeline for AIDrugReview assignment.

This script covers:
1) Data gathering and cleaning
2) Data preprocessing and feature extraction
3) Model determination and reasoning by comparison
4) Model training
5) Model tuning
6) Artifact and plot generation
"""

from __future__ import annotations

import os
import pickle
import re
import string
import sys
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.data import find as nltk_find
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import settings


TARGET_CONDITIONS = ["Depression", "High Blood Pressure", "Type 2 Diabetes"]
CONDITION_PATTERNS = {
    "Depression": "Depression",
    "High Blood Pressure": "High Blood Pressure",
    "Type 2 Diabetes": "Diabetes, Type 2",
}


def setup_environment() -> None:
    os.environ.setdefault("NLTK_DATA", str(settings.nltk_data_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(settings.mplconfig_dir))
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

    settings.mplconfig_dir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "artifacts" / "plots").mkdir(parents=True, exist_ok=True)


def ensure_nltk_resources() -> None:
    import nltk

    if str(settings.nltk_data_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(settings.nltk_data_dir))

    missing = []
    for resource in ("corpora/stopwords", "tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk_find(resource)
        except LookupError:
            missing.append(resource)
    if missing:
        raise RuntimeError(
            "Missing NLTK resources: "
            + ", ".join(missing)
            + f" in {settings.nltk_data_dir}"
        )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=["condition", "review", "rating"])
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned[(cleaned["rating"] >= 1) & (cleaned["rating"] <= 10)]
    if "date" in cleaned.columns:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    return cleaned


def iqr_outlier_summary(series: pd.Series) -> Dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {"lower_bound": 0.0, "upper_bound": 0.0, "outlier_count": 0}

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)
    count = int(((numeric < lower) | (numeric > upper)).sum())
    return {"lower_bound": float(lower), "upper_bound": float(upper), "outlier_count": count}


def save_data_quality_summary(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    clean_train: pd.DataFrame,
    clean_test: pd.DataFrame,
    artifacts_dir: Path,
) -> None:
    rows = []
    for name, raw, cleaned in (
        ("training", raw_train, clean_train),
        ("testing", raw_test, clean_test),
    ):
        rating_outliers = iqr_outlier_summary(raw["rating"]) if "rating" in raw.columns else {}
        useful_outliers = (
            iqr_outlier_summary(raw["usefulCount"]) if "usefulCount" in raw.columns else {}
        )
        rows.append(
            {
                "dataset": name,
                "raw_rows": len(raw),
                "cleaned_rows": len(cleaned),
                "removed_rows": len(raw) - len(cleaned),
                "duplicate_rows": int(raw.duplicated().sum()),
                "missing_condition": int(raw["condition"].isna().sum())
                if "condition" in raw.columns
                else 0,
                "missing_review": int(raw["review"].isna().sum()) if "review" in raw.columns else 0,
                "missing_rating": int(raw["rating"].isna().sum()) if "rating" in raw.columns else 0,
                "rating_outlier_count_iqr": rating_outliers.get("outlier_count", 0),
                "useful_count_outlier_count_iqr": useful_outliers.get("outlier_count", 0),
                "rating_min": float(pd.to_numeric(cleaned["rating"], errors="coerce").min()),
                "rating_mean": float(pd.to_numeric(cleaned["rating"], errors="coerce").mean()),
                "rating_max": float(pd.to_numeric(cleaned["rating"], errors="coerce").max()),
            }
        )

    pd.DataFrame(rows).to_csv(artifacts_dir / "data_quality_summary.csv", index=False)


def filter_conditions(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["condition"].str.contains(
        "|".join(CONDITION_PATTERNS.values()), case=False, na=False
    )
    filtered = df[mask].copy()
    for target, pattern in CONDITION_PATTERNS.items():
        filtered.loc[
            filtered["condition"].str.contains(pattern, case=False, na=False), "condition"
        ] = target
    return filtered


def clean_text(text: str) -> str:
    if pd.isna(text) or text == "":
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    try:
        tokens = word_tokenize(text)
        sw = set(stopwords.words("english"))
        tokens = [w for w in tokens if w not in sw and len(w) > 2]
        return " ".join(tokens)
    except Exception:
        words = text.split()
        sw = set(stopwords.words("english"))
        words = [w for w in words if w not in sw and len(w) > 2]
        return " ".join(words)


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    processed = df.copy()
    processed["cleaned_review"] = processed["review"].apply(clean_text)
    processed = processed[processed["cleaned_review"].str.len() > 0]
    return processed


def plot_condition_distribution(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(10, 6))
    df["condition"].value_counts().plot(kind="bar")
    plt.title("Condition Distribution")
    plt.xlabel("Condition")
    plt.ylabel("Count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_rating_distribution(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(df["rating"], bins=10, kde=True)
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_rating_boxplot(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["rating"])
    plt.title("Rating Boxplot")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_useful_count_distribution(df: pd.DataFrame, path: Path) -> None:
    if "usefulCount" not in df.columns:
        return
    plt.figure(figsize=(10, 6))
    useful_values = pd.to_numeric(df["usefulCount"], errors="coerce").dropna()
    sns.histplot(useful_values.clip(upper=useful_values.quantile(0.98)), bins=30, kde=True)
    plt.title("Useful Count Distribution (Clipped at 98th Percentile)")
    plt.xlabel("Useful Count")
    plt.ylabel("Review Count")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_date_trends(df: pd.DataFrame, path: Path) -> None:
    if "date" not in df.columns:
        return
    trend_df = df.dropna(subset=["date"]).copy()
    if trend_df.empty:
        return

    trend_df["month"] = trend_df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        trend_df.groupby(["month", "condition"])
        .agg(review_count=("review", "count"), avg_rating=("rating", "mean"))
        .reset_index()
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    sns.lineplot(data=monthly, x="month", y="review_count", hue="condition", ax=axes[0])
    axes[0].set_title("Monthly Review Count by Condition")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Reviews")

    sns.lineplot(data=monthly, x="month", y="avg_rating", hue="condition", ax=axes[1], legend=False)
    axes[1].set_title("Monthly Average Rating by Condition")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Average Rating")

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_drugs_by_usefulness(df: pd.DataFrame, path: Path) -> None:
    if "usefulCount" not in df.columns:
        return
    grouped = (
        df.groupby("drugName")
        .agg(avg_useful_count=("usefulCount", "mean"), review_count=("review", "count"))
        .reset_index()
    )
    grouped = grouped[grouped["review_count"] >= 5].nlargest(10, "avg_useful_count")
    if grouped.empty:
        return

    plt.figure(figsize=(12, 7))
    sns.barplot(data=grouped, y="drugName", x="avg_useful_count", color="#60a5fa")
    plt.title("Top Drugs by Average Useful Count")
    plt.xlabel("Average Useful Count")
    plt.ylabel("Drug")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_wordcloud(text_series: pd.Series, path: Path) -> None:
    text_blob = " ".join(text_series.tolist())
    wc = WordCloud(width=900, height=450, background_color="white", max_words=120).generate(
        text_blob
    )
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Reviews")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def summarize(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print("Shape:", df.shape)
    print("Missing values:")
    print(df.isnull().sum())
    if "rating" in df.columns:
        print("Rating describe:")
        print(df["rating"].describe())


def create_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int,
    ngram_range: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TfidfVectorizer, LabelEncoder]:
    label_encoder = LabelEncoder()
    y_train_full = label_encoder.fit_transform(train_df["condition"])
    y_test = label_encoder.transform(test_df["condition"])

    x_train_text, x_val_text, y_train, y_val = train_test_split(
        train_df["cleaned_review"],
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full,
    )

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    x_train = vectorizer.fit_transform(x_train_text)
    x_val = vectorizer.transform(x_val_text)
    x_test = vectorizer.transform(test_df["cleaned_review"])
    return x_train, x_val, y_train, y_val, x_test, y_test, vectorizer, label_encoder


def tune_feature_config(
    train_df: pd.DataFrame,
    plots_dir: Path,
) -> Tuple[int, Tuple[int, int]]:
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_df["condition"])
    x_train_text, x_val_text, y_train, y_val = train_test_split(
        train_df["cleaned_review"],
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    configs = [
        {"name": "3000_unigram", "max_features": 3000, "ngram_range": (1, 1)},
        {"name": "5000_unigram", "max_features": 5000, "ngram_range": (1, 1)},
        {"name": "5000_unigram_bigram", "max_features": 5000, "ngram_range": (1, 2)},
        {"name": "8000_unigram_bigram", "max_features": 8000, "ngram_range": (1, 2)},
    ]

    rows = []
    for config in configs:
        vectorizer = TfidfVectorizer(
            max_features=config["max_features"],
            ngram_range=config["ngram_range"],
        )
        x_train = vectorizer.fit_transform(x_train_text)
        x_val = vectorizer.transform(x_val_text)
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(x_train, y_train)
        accuracy = accuracy_score(y_val, model.predict(x_val))
        rows.append({**config, "validation_accuracy": accuracy})

    results_df = pd.DataFrame(rows)
    results_df["ngram_range"] = results_df["ngram_range"].astype(str)
    results_df.to_csv(plots_dir.parent / "feature_tuning_results.csv", index=False)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(results_df["name"], results_df["validation_accuracy"], color="#2563eb")
    for bar, score in zip(bars, results_df["validation_accuracy"]):
        plt.text(bar.get_x() + bar.get_width() / 2, score + 0.003, f"{score:.3f}", ha="center")
    plt.ylim(0, 1)
    plt.title("Feature Tuning Comparison")
    plt.ylabel("Validation Accuracy")
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_tuning_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    best = max(rows, key=lambda row: row["validation_accuracy"])
    print(
        "\nBest feature configuration: "
        f"{best['name']} (validation accuracy {best['validation_accuracy']:.4f})"
    )
    return int(best["max_features"]), best["ngram_range"]


def evaluate_model(
    name: str,
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    labels,
    plots_dir: Path,
) -> Dict[str, float]:
    model.fit(x_train, y_train)
    y_val_pred = model.predict(x_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    cv_mean = cross_val_score(model, x_train, y_train, cv=5, scoring="accuracy").mean()

    y_test_pred = model.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\n{name}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Cross-Validation Accuracy: {cv_mean:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(classification_report(y_test, y_test_pred, target_names=labels))
    report = classification_report(y_test, y_test_pred, target_names=labels, output_dict=True)
    safe_name = name.lower().replace(" ", "_")
    pd.DataFrame(report).transpose().to_csv(
        plots_dir.parent / f"classification_report_{safe_name}.csv"
    )

    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(plots_dir / f"confusion_matrix_{safe_name}.png", dpi=300)
    plt.close()

    return {"val_accuracy": val_acc, "cv_accuracy": cv_mean, "test_accuracy": test_acc}


def main() -> None:
    setup_environment()
    ensure_nltk_resources()

    print("=" * 80)
    print("AIDrugReview - Model Creation Pipeline")
    print("=" * 80)

    plots_dir = PROJECT_ROOT / "artifacts" / "plots"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    plots_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    if not settings.train_dataset_path.exists() or not settings.test_dataset_path.exists():
        raise FileNotFoundError(
            "Expected datasets in data folder:\n"
            f"- {settings.train_dataset_path}\n"
            f"- {settings.test_dataset_path}"
        )

    train_df = pd.read_csv(settings.train_dataset_path)
    test_df = pd.read_csv(settings.test_dataset_path)
    summarize(train_df, "Raw Training Dataset")
    summarize(test_df, "Raw Testing Dataset")

    train_clean = clean_data(train_df)
    test_clean = clean_data(test_df)
    save_data_quality_summary(train_df, test_df, train_clean, test_clean, artifacts_dir)
    summarize(train_clean, "Cleaned Training Dataset")
    summarize(test_clean, "Cleaned Testing Dataset")

    plot_rating_boxplot(train_clean, plots_dir / "training_rating_boxplot.png")
    plot_rating_boxplot(test_clean, plots_dir / "testing_rating_boxplot.png")

    train_filtered = filter_conditions(train_clean)
    test_filtered = filter_conditions(test_clean)
    summarize(train_filtered, "Filtered Training Dataset")
    summarize(test_filtered, "Filtered Testing Dataset")

    train_processed = preprocess_text(train_filtered)
    test_processed = preprocess_text(test_filtered)

    plot_condition_distribution(train_processed, plots_dir / "condition_distribution.png")
    plot_rating_distribution(train_processed, plots_dir / "rating_distribution.png")
    plot_useful_count_distribution(train_processed, plots_dir / "useful_count_distribution.png")
    plot_date_trends(train_processed, plots_dir / "date_trends.png")
    plot_top_drugs_by_usefulness(train_processed, plots_dir / "top_drugs_by_usefulness.png")
    plot_wordcloud(train_processed["cleaned_review"], plots_dir / "wordcloud.png")

    max_features, ngram_range = tune_feature_config(train_processed, plots_dir)
    print(f"Using TF-IDF max_features={max_features}, ngram_range={ngram_range}")

    (
        x_train,
        x_val,
        y_train,
        y_val,
        x_test,
        y_test,
        vectorizer,
        label_encoder,
    ) = create_features(train_processed, test_processed, max_features, ngram_range)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Multinomial Naive Bayes": MultinomialNB(),
    }

    results = {}
    trained_models = {}
    for name, model in models.items():
        results[name] = evaluate_model(
            name,
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            label_encoder.classes_,
            plots_dir,
        )
        trained_models[name] = model
    pd.DataFrame.from_dict(results, orient="index").to_csv(artifacts_dir / "model_results.csv")

    plt.figure(figsize=(8, 5))
    names = list(results.keys())
    scores = [results[n]["test_accuracy"] for n in names]
    bars = plt.bar(names, scores, color=["#60a5fa", "#34d399"])
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, score + 0.005, f"{score:.3f}", ha="center")
    plt.ylim(0, 1)
    plt.title("Model Comparison (Test Accuracy)")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(plots_dir / "model_comparison.png", dpi=300)
    plt.close()

    best_name = max(results.keys(), key=lambda n: results[n]["test_accuracy"])
    print(f"\nBest baseline model: {best_name} ({results[best_name]['test_accuracy']:.4f})")

    param_grids = {
        "Logistic Regression": {"C": [0.1, 1, 10], "solver": ["lbfgs"], "penalty": ["l2"]},
        "Multinomial Naive Bayes": {"alpha": [0.1, 0.5, 1.0]},
    }

    tuned_scores = {}
    tuned_models = {}
    tuned_rows = []
    for name, model in trained_models.items():
        print(f"\nTuning {name}...")
        gs = GridSearchCV(model, param_grids[name], scoring="accuracy", cv=5, n_jobs=1, verbose=1)
        gs.fit(x_train, y_train)
        tuned_model = gs.best_estimator_
        y_pred = tuned_model.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        print(f"Best params: {gs.best_params_}")
        print(f"Tuned test accuracy: {score:.4f}")
        tuned_scores[name] = score
        tuned_models[name] = tuned_model
        tuned_rows.append({"model": name, "test_accuracy": score, "best_params": str(gs.best_params_)})

        safe_name = f"tuned_{name.lower().replace(' ', '_')}"
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        pd.DataFrame(
            classification_report(
                y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
            )
        ).transpose().to_csv(artifacts_dir / f"classification_report_{safe_name}.csv")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
        )
        plt.title(f"Confusion Matrix - Tuned {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(plots_dir / f"confusion_matrix_{safe_name}.png", dpi=300)
        plt.close()

    pd.DataFrame(tuned_rows).to_csv(artifacts_dir / "tuned_model_results.csv", index=False)

    best_tuned_name = max(tuned_scores.keys(), key=lambda n: tuned_scores[n])
    best_model = tuned_models[best_tuned_name]
    print(f"\nBest tuned model: {best_tuned_name} ({tuned_scores[best_tuned_name]:.4f})")

    with settings.model_path.open("wb") as f:
        pickle.dump(best_model, f)
    with settings.vectorizer_path.open("wb") as f:
        pickle.dump(vectorizer, f)
    with settings.label_encoder_path.open("wb") as f:
        pickle.dump(label_encoder, f)

    print("\nSaved artifacts:")
    print("-", settings.model_path)
    print("-", settings.vectorizer_path)
    print("-", settings.label_encoder_path)
    print("-", plots_dir)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
