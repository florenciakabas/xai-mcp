"""Train a toy XGBoost model on breast cancer data and save artifacts.

Run once to populate models/ and data/ directories:
    uv run python scripts/train_toy_model.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# --- Paths ---
ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"


def main() -> None:
    # Load sklearn's breast cancer dataset
    bunch = load_breast_cancer()
    X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    y = pd.Series(bunch.target, name="target")

    # Split: 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")

    # Save model
    joblib.dump(model, MODELS_DIR / "xgboost_breast_cancer.joblib")
    print(f"Model saved to {MODELS_DIR / 'xgboost_breast_cancer.joblib'}")

    # Save test data (this is what we'll explain against)
    X_test.to_csv(DATA_DIR / "breast_cancer_test_X.csv", index=False)
    y_test.to_csv(DATA_DIR / "breast_cancer_test_y.csv", index=False)
    print(f"Test data saved to {DATA_DIR}")

    # Save metadata for the registry
    metadata = {
        "model_id": "xgboost_breast_cancer",
        "model_type": "XGBClassifier",
        "dataset_name": "breast_cancer",
        "feature_names": list(bunch.feature_names),
        "target_names": list(bunch.target_names),  # ['malignant', 'benign']
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "accuracy": float(accuracy),
    }
    with open(MODELS_DIR / "xgboost_breast_cancer_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {MODELS_DIR / 'xgboost_breast_cancer_meta.json'}")


if __name__ == "__main__":
    main()
