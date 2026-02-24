"""Pipeline compatibility — functions adapted from the Kedro explainability pipeline.

This module bridges our MCP toolkit with the existing Kedro-based
explainability pipeline (xai-xgboost-clf repo) developed by Tamas.

The key function here is detect_model_type(), adapted from the pipeline's
_detect_model_type() in nodes_explainability.py. It uses the same detection
logic so that our toolkit's metadata is compatible with pipeline artifacts.

Why this matters:
  - When the pipeline saves shap_metadata.json, it includes 'detected_type'
  - When our toolkit detects a model, it should produce the SAME type string
  - This means: pipeline artifacts + toolkit outputs use a shared vocabulary
  - An auditor can compare detected_type across both systems

Origin:
  Repository: EMOrg-Prd/xai-xgboost-clf
  File: src/xgboost_clf/pipelines/model_explanation/nodes_explainability.py
  Function: _detect_model_type (lines 46-100)
  Author: Tamas Toth

Adaptation notes:
  - Made public (removed underscore prefix) since it's part of our API
  - Added comprehensive docstring with examples
  - Kept the same detection order and logic for compatibility
  - Kept the lazy import pattern (_try) for optional dependencies
"""

import importlib
from typing import Any

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports for optional ML frameworks
# ---------------------------------------------------------------------------
# This pattern comes directly from the pipeline codebase. It avoids hard
# dependencies on frameworks the user may not have installed. If xgboost
# isn't installed, _xgb is None and the xgboost checks are skipped.
# ---------------------------------------------------------------------------


def _try_import(name: str):
    """Attempt to import a module; return it if available, else None.

    This avoids ImportError at module load time for optional dependencies.

    Args:
        name: Module name to import (e.g., "xgboost", "lightgbm").

    Returns:
        The imported module, or None if not installed.
    """
    if importlib.util.find_spec(name):
        return importlib.import_module(name)
    return None


_xgb = _try_import("xgboost")
_lgb = _try_import("lightgbm")
_tf = _try_import("tensorflow")
_torch = _try_import("torch")


# ---------------------------------------------------------------------------
# Model type detection — adapted from Tamas's pipeline
# ---------------------------------------------------------------------------


def detect_model_type(model: Any) -> str:
    """Detect the type of a fitted ML model using runtime introspection.

    Adapted from the Kedro explainability pipeline's _detect_model_type().
    Uses the same detection order and string type checking, plus duck-typing
    via hasattr() for cases where the type string alone is ambiguous.

    The detection order matters — it goes from most specific to most general:
      1. XGBoost (checks module + 'get_booster' attribute)
      2. LightGBM (checks module + 'booster_' attribute)
      3. Tree-based (RandomForest, etc. — checks 'feature_importances_')
      4. Sklearn (checks 'predict_proba')
      5. Keras/TensorFlow (checks module + 'fit' attribute)
      6. PyTorch (checks module)
      7. MLflow PyFunc (checks '_model_impl' attribute)
      8. "unknown" fallback

    This ordering is intentional: XGBoost and LightGBM models also have
    'feature_importances_' and 'predict_proba', so they must be checked
    BEFORE the generic tree/sklearn checks.

    Args:
        model: A fitted model object (any framework).

    Returns:
        A string identifier compatible with the pipeline's metadata format.
        One of: "xgboost", "lightgbm", "tree", "sklearn", "keras",
                "pytorch", "pyfunc", "unknown".

    Examples:
        >>> from xgboost import XGBClassifier
        >>> detect_model_type(XGBClassifier())
        'xgboost'

        >>> from sklearn.ensemble import RandomForestClassifier
        >>> detect_model_type(RandomForestClassifier())
        'tree'

        >>> from sklearn.linear_model import LogisticRegression
        >>> detect_model_type(LogisticRegression())
        'sklearn'
    """
    t = str(type(model)).lower()

    # --- XGBoost ---
    if _xgb and (
        "xgboost" in t or "xgb" in t or hasattr(model, "get_booster")
    ):
        return "xgboost"

    # --- LightGBM ---
    if _lgb and (
        "lightgbm" in t or "lgbm" in t or hasattr(model, "booster_")
    ):
        return "lightgbm"

    # --- Tree-based (RandomForest, GradientBoosting, etc.) ---
    if "randomforest" in t or "forest" in t or hasattr(model, "feature_importances_"):
        return "tree"

    # --- Generic sklearn ---
    if "sklearn" in t or hasattr(model, "predict_proba"):
        return "sklearn"

    # --- Keras / TensorFlow ---
    if _tf and (
        "tensorflow" in t or "keras" in t or hasattr(model, "fit")
    ):
        return "keras"

    # --- PyTorch ---
    if _torch and (
        "torch" in t or "pytorch" in t
    ):
        return "pytorch"

    # --- MLflow PyFunc wrapper ---
    if "pyfunc" in t or (
        hasattr(model, "predict") and hasattr(model, "_model_impl")
    ):
        return "pyfunc"

    return "unknown"
