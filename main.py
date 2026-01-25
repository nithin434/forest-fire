
"""
Runnable training script for the Forest Fire Prediction model.

What it does
- Loads forestfires.csv
- Builds a binary target (fire occurred vs. no fire) from the area column
- Trains a probabilistic classifier on the 8 numeric features used by the web form
- Evaluates with ROC-AUC, confusion matrix, and classification report
- Saves plots and metrics to outputs/
- Saves the trained model pipeline to forestfiremodel.pkl for the Flask app

Run:
    python main.py
Outputs:
    outputs/roc_curve.png
    outputs/confusion_matrix.png
    outputs/feature_importance.png
    outputs/metrics.json
    forestfiremodel.pkl
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib

# Use a non-GUI backend so plots render in headless / server environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Global configuration
RANDOM_STATE = 42
DATA_PATH = Path("forestfires.csv")
MODEL_PATH = Path("forestfiremodel.pkl")
OUTPUT_DIR = Path("outputs")
FEATURE_COLUMNS = [
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "temp",
    "RH",
    "wind",
    "rain",
]


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find data file: {path}")
    df = pd.read_csv(path)
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fire"] = (df["area"] > 0).astype(int)
    return df


def train_model(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df["fire"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Pipeline: scale numeric features then classify
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "positive_rate_test": float(y_test.mean()),
    }

    return model, (X_test, y_test, y_pred, y_proba), metrics


def plot_roc(y_true, y_score) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label="ROC curve", color="#4caf50", linewidth=2.5)
    ax.plot([0, 1], [0, 1], "--", color="#9e9e9e", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    output_path = OUTPUT_DIR / "roc_curve.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_confusion(y_true, y_pred) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Fire", "Fire"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(cmap="Greens", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    output_path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_feature_importance(model: Pipeline) -> Path:
    clf = model.named_steps["clf"]
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(range(len(importances)), importances[indices], color="#66bb6a")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(np.array(FEATURE_COLUMNS)[indices], rotation=45, ha="right")
    ax.set_title("Feature Importance (Random Forest)")
    ax.set_ylabel("Importance")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "feature_importance.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_metrics(metrics: dict) -> Path:
    output_path = OUTPUT_DIR / "metrics.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return output_path


def save_model(model: Pipeline) -> Path:
    joblib.dump(model, MODEL_PATH)
    return MODEL_PATH


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    ensure_output_dir()

    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    df = build_target(df)

    print("Training model...")
    model, eval_data, metrics = train_model(df)
    X_test, y_test, y_pred, y_proba = eval_data
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")

    print("Saving artifacts...")
    save_metrics(metrics)
    plot_roc(y_test, y_proba)
    plot_confusion(y_test, y_pred)
    plot_feature_importance(model)
    save_model(model)

    print("Artifacts written to outputs/ and forestfiremodel.pkl")


if __name__ == "__main__":
    main()