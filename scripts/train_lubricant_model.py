"""Train a LightGBM lubricant quality classifier and save artifacts.

Run once to populate models/ and data/ directories:
    uv run python scripts/train_lubricant_model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

# --- Paths ---
ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

FEATURE_NAMES = [
    "viscosity_40c",
    "viscosity_100c",
    "total_acid_number",
    "water_content_ppm",
    "flash_point_c",
    "oxidation_stability",
    "particle_count_4um",
    "iron_ppm",
    "copper_ppm",
    "foam_tendency_ml",
    "color_astm",
    "demulsibility_min",
]

TARGET_NAMES = ["in_spec", "degraded"]


def generate_samples(n_samples: int, rng: np.random.RandomState, summer: bool = False):
    """Generate synthetic lubricant quality data.

    Args:
        n_samples: Number of samples to generate.
        rng: Random state for reproducibility.
        summer: If True, apply summer environmental shifts (drift source).
    """
    degraded_fraction = 0.30
    n_degraded = int(n_samples * degraded_fraction)
    labels = np.array([0] * (n_samples - n_degraded) + [1] * n_degraded)
    rng.shuffle(labels)

    # Base values (winter / nominal conditions)
    data = {
        "viscosity_40c": rng.normal(68.0, 3.0, n_samples),
        "viscosity_100c": rng.normal(10.5, 0.8, n_samples),
        "total_acid_number": rng.normal(0.5, 0.1, n_samples),
        "water_content_ppm": rng.normal(80, 20, n_samples),
        "flash_point_c": rng.normal(220, 8, n_samples),
        "oxidation_stability": rng.normal(300, 30, n_samples),
        "particle_count_4um": rng.normal(50, 15, n_samples),
        "iron_ppm": rng.normal(10, 3, n_samples),
        "copper_ppm": rng.normal(2, 0.8, n_samples),
        "foam_tendency_ml": rng.normal(20, 8, n_samples),
        "color_astm": rng.normal(2.0, 0.5, n_samples),
        "demulsibility_min": rng.normal(15, 4, n_samples),
    }

    # Apply degradation effects
    for i in range(n_samples):
        if labels[i] == 1:
            severity = rng.uniform(0.5, 1.0)
            data["total_acid_number"][i] += severity * rng.uniform(1.0, 3.0)
            data["water_content_ppm"][i] += severity * rng.uniform(100, 400)
            data["flash_point_c"][i] -= severity * rng.uniform(15, 40)
            data["oxidation_stability"][i] -= severity * rng.uniform(80, 180)
            data["particle_count_4um"][i] += severity * rng.uniform(40, 120)
            data["iron_ppm"][i] += severity * rng.uniform(10, 40)
            data["copper_ppm"][i] += severity * rng.uniform(3, 10)
            data["foam_tendency_ml"][i] += severity * rng.uniform(15, 50)
            data["color_astm"][i] += severity * rng.uniform(1.5, 4.0)
            data["demulsibility_min"][i] += severity * rng.uniform(10, 30)
            # Viscosity shifts both directions under degradation
            direction = rng.choice([-1, 1])
            data["viscosity_40c"][i] += direction * severity * rng.uniform(5, 15)
            data["viscosity_100c"][i] += direction * severity * rng.uniform(1, 4)

    # Summer drift: temperature-sensitive shifts applied to ALL samples
    if summer:
        data["viscosity_40c"] -= rng.normal(3.0, 1.0, n_samples)  # thinner in heat
        data["viscosity_100c"] -= rng.normal(0.5, 0.2, n_samples)
        data["water_content_ppm"] += rng.normal(40, 10, n_samples)  # humidity
        data["particle_count_4um"] += rng.normal(12, 4, n_samples)  # equipment wear

    X = pd.DataFrame(data, columns=FEATURE_NAMES)
    y = pd.Series(labels, name="target")
    return X, y


def main() -> None:
    rng = np.random.RandomState(42)

    # Training data: winter baseline
    X_train, y_train = generate_samples(500, rng, summer=False)
    # Test data: summer with environmental drift
    X_test, y_test = generate_samples(150, rng, summer=True)

    # Train LightGBM
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")

    # Save model
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "lgbm_lubricant_quality.joblib")
    print(f"Model saved to {MODELS_DIR / 'lgbm_lubricant_quality.joblib'}")

    # Save data
    DATA_DIR.mkdir(exist_ok=True)
    X_train.to_csv(DATA_DIR / "lubricant_quality_train_X.csv", index=False)
    X_test.to_csv(DATA_DIR / "lubricant_quality_test_X.csv", index=False)
    y_test.to_csv(DATA_DIR / "lubricant_quality_test_y.csv", index=False)
    print(f"Train/test data saved to {DATA_DIR}")

    # Save metadata for the registry
    metadata = {
        "model_id": "lgbm_lubricant_quality",
        "model_type": "LGBMClassifier",
        "dataset_name": "lubricant_quality",
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "accuracy": float(accuracy),
    }
    with open(MODELS_DIR / "lgbm_lubricant_quality_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {MODELS_DIR / 'lgbm_lubricant_quality_meta.json'}")


if __name__ == "__main__":
    main()
