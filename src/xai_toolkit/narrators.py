"""Deterministic narrative generation — data → English (ADR-002).

This module converts structured explainability data into plain-English
paragraphs. NO LLM calls. All output is deterministic: same input
always produces the exact same English text.

Design pattern: Template Method — each narrator function follows the same
structure (extract data, rank features, format template) but the template
content varies by explanation type.
"""

from xai_toolkit.schemas import ShapResult


def narrate_prediction(shap_result: ShapResult, top_n: int = 3) -> str:
    """Convert SHAP result for a single prediction into an English paragraph.

    Args:
        shap_result: Output from compute_shap_values().
        top_n: Number of top contributing features to highlight.

    Returns:
        A complete English paragraph explaining the prediction.

    Example:
        >>> narrative = narrate_prediction(shap_result, top_n=3)
        >>> print(narrative)
        "The model classified this sample as benign (probability: 0.91)..."
    """
    # Sort features by absolute SHAP magnitude (largest impact first)
    sorted_features = sorted(
        shap_result.shap_values.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    # Split into drivers (pushing toward prediction) and opposing factors
    top_drivers = []
    top_opposing = []

    for name, shap_val in sorted_features:
        if len(top_drivers) >= top_n and len(top_opposing) >= 1:
            break

        # Determine direction relative to positive class
        if shap_val > 0:
            direction = "pushing toward"
            direction_label = "positive class"
        else:
            direction = "pushing away from"
            direction_label = "positive class"

        entry = {
            "name": name,
            "shap_value": shap_val,
            "feature_value": shap_result.feature_values.get(name),
            "direction": direction,
        }

        # A "driver" aligns with the prediction; "opposing" works against it.
        # For positive class predictions, positive SHAP = driver.
        # For negative class predictions, negative SHAP = driver.
        is_positive_prediction = shap_result.prediction == 1
        is_positive_shap = shap_val > 0

        if is_positive_prediction == is_positive_shap:
            if len(top_drivers) < top_n:
                top_drivers.append(entry)
        else:
            if len(top_opposing) < 1:
                top_opposing.append(entry)

    # --- Build the narrative ---
    label = shap_result.prediction_label
    prob = shap_result.probability

    # Opening sentence
    narrative = (
        f"The model classified this sample as {label} "
        f"(probability: {prob:.2f})"
    )

    # Driver features
    if top_drivers:
        narrative += " primarily because of "
        if len(top_drivers) == 1:
            narrative += "one factor: "
        else:
            narrative += f"{len(top_drivers)} factors: "

        driver_descriptions = []
        for i, d in enumerate(top_drivers):
            sign = "+" if d["shap_value"] > 0 else ""
            desc = (
                f"{d['name']} = {d['feature_value']:.4f} "
                f"({d['direction']} the positive class by "
                f"{sign}{d['shap_value']:.4f})"
            )
            driver_descriptions.append(desc)

        # Join with commas and "and"
        if len(driver_descriptions) == 1:
            narrative += driver_descriptions[0]
        elif len(driver_descriptions) == 2:
            narrative += f"{driver_descriptions[0]} and {driver_descriptions[1]}"
        else:
            narrative += (
                ", ".join(driver_descriptions[:-1])
                + f", and {driver_descriptions[-1]}"
            )
        narrative += "."

    # Opposing factor
    if top_opposing:
        opp = top_opposing[0]
        sign = "+" if opp["shap_value"] > 0 else ""
        narrative += (
            f" The top opposing factor is {opp['name']} = "
            f"{opp['feature_value']:.4f} ({opp['direction']} the positive "
            f"class by {sign}{opp['shap_value']:.4f})."
        )

    return narrative
