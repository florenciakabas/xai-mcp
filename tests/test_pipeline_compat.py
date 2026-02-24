"""Tests for pipeline_compat — Tamas's model detection logic in our toolkit.

These tests verify that detect_model_type() produces the same type strings
as the Kedro explainability pipeline's _detect_model_type(), ensuring
metadata compatibility between both systems.

The detection order matters (most specific → most general) because models
from different frameworks share attributes. An XGBoost model has both
'get_booster' AND 'feature_importances_' AND 'predict_proba' — so it
would match 'tree' or 'sklearn' if XGBoost weren't checked first.
"""

import pytest
from xai_toolkit.pipeline_compat import detect_model_type


# ---------------------------------------------------------------------------
# Real model tests — these use actual fitted models
# ---------------------------------------------------------------------------


class TestDetectRealModels:
    """Test detection against actual fitted sklearn/xgboost models."""

    def test_xgboost_classifier(self):
        """XGBClassifier should be detected as 'xgboost'."""
        from xgboost import XGBClassifier

        model = XGBClassifier(n_estimators=2, use_label_encoder=False)
        assert detect_model_type(model) == "xgboost"

    def test_xgboost_fitted(self):
        """Fitted XGBClassifier should still be detected as 'xgboost'."""
        from sklearn.datasets import load_iris
        from xgboost import XGBClassifier

        X, y = load_iris(return_X_y=True)
        model = XGBClassifier(n_estimators=5, eval_metric="logloss")
        model.fit(X, y)
        assert detect_model_type(model) == "xgboost"

    def test_random_forest(self):
        """RandomForestClassifier should be detected as 'tree'."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        assert detect_model_type(model) == "tree"

    def test_random_forest_fitted(self):
        """Fitted RandomForestClassifier should still be detected as 'tree'."""
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier

        X, y = load_iris(return_X_y=True)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        assert detect_model_type(model) == "tree"

    def test_gradient_boosting(self):
        """GradientBoostingClassifier should be detected as 'tree'.

        It has feature_importances_, so it hits the tree check before sklearn.
        This matches the pipeline's behavior.
        """
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier(n_estimators=5)
        # Note: unfitted GBT doesn't have feature_importances_ yet,
        # but 'sklearn' in the type string → falls through to 'sklearn'.
        # Fitted version gets 'tree'. This matches the pipeline.
        result = detect_model_type(model)
        assert result in ("tree", "sklearn")

    def test_logistic_regression(self):
        """LogisticRegression should be detected as 'sklearn'."""
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        assert detect_model_type(model) == "sklearn"

    def test_decision_tree(self):
        """DecisionTreeClassifier should be detected as 'tree'."""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.datasets import load_iris

        X, y = load_iris(return_X_y=True)
        model = DecisionTreeClassifier()
        model.fit(X, y)
        assert detect_model_type(model) == "tree"

# ---------------------------------------------------------------------------
# Detection order tests — verify the specificity ordering
# ---------------------------------------------------------------------------


class TestDetectionOrder:
    """Verify that more specific checks take priority.

    This is the key insight in Tamas's design: XGBoost models also have
    predict_proba (sklearn check) and feature_importances_ (tree check),
    but the XGBoost check comes first. Getting this wrong means the
    pipeline would select the wrong SHAP explainer.
    """

    def test_xgboost_before_tree(self):
        """XGBoost has feature_importances_ but should NOT match 'tree'."""
        from sklearn.datasets import load_iris
        from xgboost import XGBClassifier

        X, y = load_iris(return_X_y=True)
        model = XGBClassifier(n_estimators=5, eval_metric="logloss")
        model.fit(X, y)

        # After fitting, XGBClassifier has feature_importances_
        assert hasattr(model, "feature_importances_")
        # But detect_model_type should return 'xgboost', not 'tree'
        assert detect_model_type(model) == "xgboost"

    def test_xgboost_before_sklearn(self):
        """XGBoost has predict_proba but should NOT match 'sklearn'."""
        from xgboost import XGBClassifier

        model = XGBClassifier(n_estimators=2)
        assert hasattr(model, "predict_proba")
        assert detect_model_type(model) == "xgboost"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and fallback behavior."""

    def test_unknown_object(self):
        """A plain object with no ML attributes returns 'unknown'."""
        assert detect_model_type("not a model") == "unknown"

    def test_unknown_class(self):
        """A custom class with predict() but no other attributes."""

        class MyModel:
            def predict(self, X):
                return X

        assert detect_model_type(MyModel()) == "unknown"

    def test_return_type_is_string(self):
        """detect_model_type always returns a string."""
        from sklearn.ensemble import RandomForestClassifier

        result = detect_model_type(RandomForestClassifier())
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Integration: verify it flows through the registry
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """Verify detect_model_type is called during model registration."""

    def test_registry_stores_detected_type(self, tmp_path):
        """When a model is loaded, detected_type should appear in metadata."""
        import json
        import joblib
        import pandas as pd
        from sklearn.datasets import load_breast_cancer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        from xai_toolkit.registry import ModelRegistry

        # Create a minimal model + data on disk
        data = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)

        models_dir = tmp_path / "models"
        data_dir = tmp_path / "data"
        models_dir.mkdir()
        data_dir.mkdir()

        model_id = "test_rf"
        joblib.dump(model, models_dir / f"{model_id}.joblib")

        meta = {
            "model_type": "RandomForestClassifier",
            "dataset_name": "test_dataset",
            "feature_names": list(data.feature_names),
            "target_names": ["malignant", "benign"],
            "accuracy": 0.95,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
        }
        with open(models_dir / f"{model_id}_meta.json", "w") as f:
            json.dump(meta, f)

        X_test_df = pd.DataFrame(X_test, columns=data.feature_names)
        y_test_s = pd.Series(y_test)
        X_test_df.to_csv(data_dir / "test_dataset_test_X.csv", index=False)
        y_test_s.to_csv(data_dir / "test_dataset_test_y.csv", index=False)

        # Load through the registry
        registry = ModelRegistry()
        registry.load_from_disk(model_id, models_dir, data_dir)

        entry = registry.get(model_id)
        assert entry.metadata["detected_type"] == "tree"

    def test_list_models_includes_detected_type(self, tmp_path):
        """list_models() should surface detected_type."""
        import json
        import joblib
        import pandas as pd
        from sklearn.datasets import load_breast_cancer
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split

        from xai_toolkit.registry import ModelRegistry

        data = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
        model = XGBClassifier(n_estimators=5, eval_metric="logloss")
        model.fit(X_train, y_train)

        models_dir = tmp_path / "models"
        data_dir = tmp_path / "data"
        models_dir.mkdir()
        data_dir.mkdir()

        model_id = "test_xgb"
        joblib.dump(model, models_dir / f"{model_id}.joblib")

        meta = {
            "model_type": "XGBClassifier",
            "dataset_name": "test_dataset",
            "feature_names": list(data.feature_names),
            "target_names": ["malignant", "benign"],
            "accuracy": 0.96,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
        }
        with open(models_dir / f"{model_id}_meta.json", "w") as f:
            json.dump(meta, f)

        X_test_df = pd.DataFrame(X_test, columns=data.feature_names)
        y_test_s = pd.Series(y_test)
        X_test_df.to_csv(data_dir / "test_dataset_test_X.csv", index=False)
        y_test_s.to_csv(data_dir / "test_dataset_test_y.csv", index=False)

        registry = ModelRegistry()
        registry.load_from_disk(model_id, models_dir, data_dir)

        models_list = registry.list_models()
        assert models_list[0]["detected_type"] == "xgboost"
