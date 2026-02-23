"""Train a RandomForest classifier on breast cancer data and save artifacts.

This is the second model type (alongside XGBoost) that proves the toolkit's
framework-agnostic design. All 7 MCP tools should work with this model
without any code changes — the same Pydantic schemas, narrators, and
explainers handle both model types transparently.

Run once after training XGBoost:
    uv run python scripts/train_rf_model.py

Design note: We use the exact same train/test split (random_state=42)
as the XGBoost model so both models explain the same test samples.
This enables direct comparison (Day 4, D4-S2).
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"


def main() -> None:
    # Same dataset and split as XGBoost — same test samples, comparable results
    bunch = load_breast_cancer()
    X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    y = pd.Series(bunch.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    # Train RandomForest — deliberately different hyperparameters from XGBoost
    # to produce meaningfully different feature importances for comparison
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")

    # Save model
    model_path = MODELS_DIR / "rf_breast_cancer.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Test data already exists from XGBoost training — both models share it.
    # We still save it under the rf name so the registry can find it by
    # dataset_name convention without coupling to the XGBoost artifact.
    X_test.to_csv(DATA_DIR / "breast_cancer_test_X.csv", index=False)
    y_test.to_csv(DATA_DIR / "breast_cancer_test_y.csv", index=False)
    print(f"Test data confirmed at {DATA_DIR}")

    metadata = {
        "model_id": "rf_breast_cancer",
        "model_type": "RandomForestClassifier",
        "dataset_name": "breast_cancer",
        "feature_names": list(bunch.feature_names),
        "target_names": list(bunch.target_names),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "accuracy": float(accuracy),
    }
    meta_path = MODELS_DIR / "rf_breast_cancer_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")
    print("\nDone. Register with: registry.load_from_disk('rf_breast_cancer', ...)")


if __name__ == "__main__":
    main()
