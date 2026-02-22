"""Visualization module — matplotlib → base64 PNG (ADR-001, ADR-005).

Pure functions that take structured data from explainers and produce
static matplotlib plots encoded as base64 PNGs. No MCP imports.

The plots serve as *supporting evidence* for the narrative — they are
optional supplements, not the primary output.

Three plot types:
1. PDP + ICE overlay — model-agnostic (not SHAP), shows feature effect
2. SHAP bar chart (tornado) — ranked feature contributions for one sample
3. SHAP waterfall — cumulative buildup from base value to prediction
"""

import base64
import io
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from xai_toolkit.schemas import PartialDependenceResult, ShapResult

# Use non-interactive backend for server environments
matplotlib.use("Agg")

# --- Shared styling ---

# Clean, professional color palette
COLORS = {
    "positive": "#d94a4a",    # red — pushes toward positive class
    "negative": "#4a90d9",    # blue — pushes toward negative class
    "pdp_line": "#1a1a1a",    # black — PDP average line
    "ice_line": "#b0b0b0",    # light gray — individual ICE curves
    "highlight": "#e8913a",   # orange — accent / highlighted sample
    "background": "#fafafa",  # off-white background
}


def _fig_to_base64(fig: plt.Figure, dpi: int = 150) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string.

    This is the shared serialization step — every plot function calls this.

    Args:
        fig: A matplotlib Figure object.
        dpi: Resolution in dots per inch.

    Returns:
        Base64-encoded PNG string (no data URI prefix).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_pdp_ice(
    pdp_result: PartialDependenceResult,
    max_ice_curves: int = 50,
    figsize: tuple[float, float] = (8, 5),
) -> str:
    """Plot Partial Dependence (PDP) with Individual Conditional Expectation (ICE) overlay.

    This is a model-agnostic visualization (NOT SHAP-based):
    - Gray lines: ICE curves showing how each individual sample's prediction
      changes as the feature varies. Reveals heterogeneity.
    - Bold black line: PDP — the average across all ICE curves.

    The gap between ICE curves shows whether the feature's effect is
    consistent (tight bundle) or varies by context (wide spread).

    Args:
        pdp_result: Output from compute_partial_dependence().
        max_ice_curves: Maximum number of ICE curves to draw (too many = clutter).
        figsize: Figure dimensions in inches.

    Returns:
        Base64-encoded PNG string.

    Example:
        >>> png_b64 = plot_pdp_ice(pdp_result)
        >>> len(png_b64) > 0
        True
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(COLORS["background"])

    x = pdp_result.feature_values

    # Plot ICE curves (individual sample effects)
    if pdp_result.ice_curves:
        n_curves = min(len(pdp_result.ice_curves), max_ice_curves)
        for i in range(n_curves):
            ax.plot(
                x,
                pdp_result.ice_curves[i],
                color=COLORS["ice_line"],
                alpha=0.3,
                linewidth=0.5,
            )

    # Plot PDP (average effect) on top
    ax.plot(
        x,
        pdp_result.predictions,
        color=COLORS["pdp_line"],
        linewidth=2.5,
        label="PDP (average effect)",
        zorder=10,
    )

    ax.set_xlabel(pdp_result.feature_name, fontsize=12)
    ax.set_ylabel("Predicted probability", fontsize=12)
    ax.set_title(
        f"Partial Dependence + ICE: {pdp_result.feature_name}",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return _fig_to_base64(fig)


def plot_shap_bar(
    shap_result: ShapResult,
    top_n: int = 10,
    figsize: tuple[float, float] = (8, 6),
) -> str:
    """Plot horizontal bar chart of SHAP values for a single prediction (tornado plot).

    Features are ranked by absolute SHAP magnitude. Red bars push toward
    the positive class; blue bars push away. This is the classic "which
    features mattered for THIS prediction" visualization.

    Args:
        shap_result: Output from compute_shap_values().
        top_n: Number of top features to display.
        figsize: Figure dimensions in inches.

    Returns:
        Base64-encoded PNG string.

    Example:
        >>> png_b64 = plot_shap_bar(shap_result, top_n=10)
        >>> len(png_b64) > 0
        True
    """
    # Sort by absolute SHAP value, take top N
    sorted_features = sorted(
        shap_result.shap_values.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:top_n]

    # Reverse so largest is on top in horizontal bar chart
    sorted_features = sorted_features[::-1]

    names = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]
    colors = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in values]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(COLORS["background"])

    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        x_pos = val + (0.002 if val > 0 else -0.002)
        ha = "left" if val > 0 else "right"
        ax.text(
            x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center", ha=ha, fontsize=9, color="#333333",
        )

    # Zero line
    ax.axvline(x=0, color="#666666", linewidth=0.8, linestyle="-")

    ax.set_xlabel("SHAP value (impact on prediction)", fontsize=11)
    ax.set_title(
        f"Feature Contributions — {shap_result.prediction_label} "
        f"(p={shap_result.probability:.2f})",
        fontsize=13,
        fontweight="bold",
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["positive"], label="Pushes toward positive class"),
        Patch(facecolor=COLORS["negative"], label="Pushes away from positive class"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.grid(True, axis="x", alpha=0.3)

    return _fig_to_base64(fig)


def plot_shap_waterfall(
    shap_result: ShapResult,
    top_n: int = 10,
    figsize: tuple[float, float] = (8, 7),
) -> str:
    """Plot SHAP waterfall showing cumulative buildup from base value to prediction.

    Reads left-to-right: starting from the base value (average model output),
    each feature pushes the prediction up (red) or down (blue) until arriving
    at the final predicted probability. This is the single most informative
    SHAP plot for understanding a specific prediction.

    Args:
        shap_result: Output from compute_shap_values().
        top_n: Number of features to show individually (rest grouped as "other").
        figsize: Figure dimensions in inches.

    Returns:
        Base64-encoded PNG string.

    Example:
        >>> png_b64 = plot_shap_waterfall(shap_result, top_n=10)
        >>> len(png_b64) > 0
        True
    """
    # Sort by absolute SHAP value
    sorted_features = sorted(
        shap_result.shap_values.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    # Take top N, group the rest as "other features"
    top_features = sorted_features[:top_n]
    remaining = sorted_features[top_n:]
    other_sum = sum(val for _, val in remaining)

    # Build the waterfall entries (bottom-up: base → features → prediction)
    entries = []
    if remaining:
        entries.append(("other features", other_sum))

    # Add features in reverse importance order (least important at bottom)
    for name, val in reversed(top_features):
        entries.append((name, val))

    names = [e[0] for e in entries]
    values = [e[1] for e in entries]

    # Calculate cumulative positions for the waterfall
    base_val = shap_result.base_value
    cumulative = base_val
    lefts = []
    for val in values:
        if val >= 0:
            lefts.append(cumulative)
            cumulative += val
        else:
            cumulative += val
            lefts.append(cumulative)

    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in values]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(COLORS["background"])

    # Draw bars
    bars = ax.barh(
        range(len(names)),
        [abs(v) for v in values],
        left=lefts,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.6,
    )

    # Connector lines between bars
    running = base_val
    for i, val in enumerate(values):
        running += val
        if i < len(values) - 1:
            ax.plot(
                [running, running],
                [i - 0.3, i + 1.3],
                color="#aaaaaa",
                linewidth=0.7,
                linestyle="--",
            )

    # Value labels
    for i, (val, left) in enumerate(zip(values, lefts)):
        x_pos = left + abs(val) / 2
        ax.text(
            x_pos, i,
            f"{val:+.4f}",
            va="center", ha="center", fontsize=8,
            color="white", fontweight="bold",
        )

    # Y-axis labels
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)

    # Base value and prediction markers
    ax.axvline(x=base_val, color="#888888", linewidth=1, linestyle=":", label=f"Base value: {base_val:.4f}")
    final_val = base_val + sum(values)
    ax.axvline(x=final_val, color=COLORS["highlight"], linewidth=1.5, linestyle="-", label=f"Prediction: {shap_result.probability:.4f}")

    ax.set_xlabel("Model output (probability)", fontsize=11)
    ax.set_title(
        f"SHAP Waterfall — {shap_result.prediction_label} "
        f"(p={shap_result.probability:.2f})",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)

    return _fig_to_base64(fig)
