"""Tests for plots.py — verify plot generation produces valid base64 PNGs."""

import base64

from xai_toolkit.plots import plot_pdp_ice, plot_shap_bar, plot_shap_waterfall
from xai_toolkit.schemas import PartialDependenceResult, ShapResult


# --- Fixtures ---


def _make_shap_result() -> ShapResult:
    return ShapResult(
        prediction=1,
        prediction_label="malignant",
        probability=0.91,
        base_value=0.42,
        shap_values={
            "worst_radius": 0.28,
            "worst_concave_points": 0.19,
            "mean_concavity": 0.14,
            "mean_smoothness": -0.06,
            "mean_texture": 0.03,
        },
        feature_values={
            "worst_radius": 23.4,
            "worst_concave_points": 0.18,
            "mean_concavity": 0.15,
            "mean_smoothness": 0.08,
            "mean_texture": 19.2,
        },
        feature_names=[
            "worst_radius",
            "worst_concave_points",
            "mean_concavity",
            "mean_smoothness",
            "mean_texture",
        ],
    )


def _make_pdp_result() -> PartialDependenceResult:
    return PartialDependenceResult(
        feature_name="mean radius",
        feature_values=[6.98, 10.0, 15.0, 20.0, 28.11],
        predictions=[0.12, 0.25, 0.55, 0.78, 0.89],
        ice_curves=[
            [0.10, 0.22, 0.50, 0.75, 0.88],
            [0.15, 0.30, 0.60, 0.80, 0.90],
            [0.08, 0.18, 0.45, 0.70, 0.85],
        ],
        feature_min=6.98,
        feature_max=28.11,
        prediction_min=0.12,
        prediction_max=0.89,
    )


def _is_valid_png_base64(b64_string: str) -> bool:
    """Check if a base64 string decodes to a valid PNG."""
    try:
        data = base64.b64decode(b64_string)
        # PNG magic bytes: \x89PNG\r\n\x1a\n
        return data[:4] == b"\x89PNG"
    except Exception:
        return False


# --- Tests ---


class TestPlotShapBar:
    def test_returns_string(self):
        result = plot_shap_bar(_make_shap_result())
        assert isinstance(result, str)

    def test_returns_valid_png(self):
        result = plot_shap_bar(_make_shap_result())
        assert _is_valid_png_base64(result), "Output is not a valid PNG"

    def test_nonempty(self):
        result = plot_shap_bar(_make_shap_result())
        assert len(result) > 100, "PNG seems too small to be a real plot"

    def test_top_n_parameter(self):
        result_3 = plot_shap_bar(_make_shap_result(), top_n=3)
        result_5 = plot_shap_bar(_make_shap_result(), top_n=5)
        # Both should be valid, possibly different sizes
        assert _is_valid_png_base64(result_3)
        assert _is_valid_png_base64(result_5)

    def test_is_deterministic(self):
        r1 = plot_shap_bar(_make_shap_result())
        r2 = plot_shap_bar(_make_shap_result())
        assert r1 == r2


class TestPlotShapWaterfall:
    def test_returns_valid_png(self):
        result = plot_shap_waterfall(_make_shap_result())
        assert _is_valid_png_base64(result)

    def test_nonempty(self):
        result = plot_shap_waterfall(_make_shap_result())
        assert len(result) > 100

    def test_is_deterministic(self):
        r1 = plot_shap_waterfall(_make_shap_result())
        r2 = plot_shap_waterfall(_make_shap_result())
        assert r1 == r2


class TestPlotPdpIce:
    def test_returns_valid_png(self):
        result = plot_pdp_ice(_make_pdp_result())
        assert _is_valid_png_base64(result)

    def test_nonempty(self):
        result = plot_pdp_ice(_make_pdp_result())
        assert len(result) > 100

    def test_works_without_ice_curves(self):
        """PDP-only (no ICE data) should still produce a valid plot."""
        pdp = _make_pdp_result()
        pdp.ice_curves = []
        result = plot_pdp_ice(pdp)
        assert _is_valid_png_base64(result)

    def test_is_deterministic(self):
        r1 = plot_pdp_ice(_make_pdp_result())
        r2 = plot_pdp_ice(_make_pdp_result())
        assert r1 == r2
