"""Deterministic narrative generation — data → English (ADR-002).

This module converts structured explainability data into plain-English
paragraphs. NO LLM calls. All output is deterministic: same input
always produces the exact same English text.

Design pattern: Template Method — each narrator function follows the same
structure (extract data, rank features, format template) but the template
content varies by explanation type.
"""

from xai_toolkit.schemas import (
    DatasetDescription,
    DatasetDriftResult,
    FeatureImportance,
    FeatureDriftResult,
    ModelSummary,
    PartialDependenceResult,
    PredictionComparison,
    ShapResult,
)


def narrate_intrinsic_importance(
    importances: list[FeatureImportance],
    model_type: str,
    n_features_total: int,
    top_n: int = 5,
    source_attr: str = "feature_importances_",
) -> str:
    """Generate a deterministic narrative for intrinsic feature importances.

    Produces different narratives for coefficient-based (linear) vs
    importance-based (tree) models, since they have different interpretations.

    Args:
        importances: Sorted list from extract_intrinsic_importances().
        model_type: Model type string (e.g., "LogisticRegression", "RandomForestClassifier").
        n_features_total: Total number of features in the model.
        top_n: Number of top features to describe.
        source_attr: Which model attribute the importances came from
            ("coef_" or "feature_importances_").

    Returns:
        A deterministic English paragraph describing intrinsic importances.
    """
    if not importances:
        return "No intrinsic feature importances are available for this model."

    features = importances[:top_n]
    is_coefficient_based = source_attr == "coef_"

    if is_coefficient_based:
        narrative = (
            f"This {model_type} model exposes its coefficients directly "
            f"(intrinsic interpretability). Out of {n_features_total} features, "
            f"the top {len(features)} by absolute coefficient magnitude are: "
        )
        parts = []
        for i, feat in enumerate(features, 1):
            sign = "+" if feat.mean_shap >= 0 else ""
            parts.append(
                f"#{i} {feat.name} (coefficient: {sign}{feat.mean_shap:.4f} — "
                f"each unit increase shifts the prediction by this amount)"
            )
    else:
        narrative = (
            f"This {model_type} model exposes its feature importances directly "
            f"(intrinsic interpretability). Out of {n_features_total} features, "
            f"the top {len(features)} by importance are: "
        )
        parts = []
        total_importance = sum(f.importance for f in importances)
        for i, feat in enumerate(features, 1):
            pct = (feat.importance / total_importance * 100) if total_importance > 0 else 0
            parts.append(
                f"#{i} {feat.name} (importance: {feat.importance:.4f}, "
                f"accounts for {pct:.1f}% of the model's decision splits)"
            )

    narrative += "; ".join(parts) + "."
    return narrative


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
        else:
            direction = "pushing away from"

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
        for d in top_drivers:
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


def narrate_model_summary(summary: ModelSummary) -> str:
    """Convert a model summary into a plain-English overview.

    Args:
        summary: Output from compute_model_summary().

    Returns:
        An English paragraph describing the model and its top features.

    Example:
        >>> narrate_model_summary(summary)
        "This is an XGBClassifier trained to distinguish between malignant
         and benign cases. It uses 30 features and achieves 95.6% accuracy..."
    """
    # Target names as a readable list
    targets = " and ".join(summary.target_names)

    narrative = (
        f"This is a {summary.model_type} model trained to distinguish "
        f"between {targets} cases. It uses {summary.n_features} features "
        f"and achieves {summary.accuracy:.1%} accuracy on the test set "
        f"({summary.n_test_samples} samples)."
    )

    # Top features
    if summary.top_features:
        narrative += (
            " The most influential features are: "
        )
        feature_parts = []
        for i, feat in enumerate(summary.top_features):
            rank_word = ["most", "second most", "third most", "fourth most", "fifth most"]
            rank = rank_word[i] if i < len(rank_word) else f"#{i+1}"
            feature_parts.append(
                f"{feat.name} ({rank} important, average impact "
                f"{feat.importance:.4f}, generally pushes {feat.direction})"
            )

        if len(feature_parts) == 1:
            narrative += feature_parts[0]
        elif len(feature_parts) == 2:
            narrative += f"{feature_parts[0]} and {feature_parts[1]}"
        else:
            narrative += (
                ", ".join(feature_parts[:-1]) + f", and {feature_parts[-1]}"
            )
        narrative += "."

    return narrative


def narrate_feature_comparison(
    importances: list[FeatureImportance],
    top_n: int = 10,
) -> str:
    """Convert ranked feature importances into comparative English.

    Args:
        importances: Sorted list from compute_global_feature_importance().
        top_n: How many features to describe.

    Returns:
        An English paragraph ranking features by importance.

    Example:
        >>> narrate_feature_comparison(importances)
        "The most important feature is worst_radius (mean |SHAP| = 0.15)..."
    """
    features = importances[:top_n]

    if not features:
        return "No feature importance data is available."

    narrative = (
        f"Ranked by importance (mean absolute SHAP value across all samples), "
        f"the top {len(features)} features are: "
    )

    parts = []
    for i, feat in enumerate(features, 1):
        sign_desc = (
            "tends to increase risk" if feat.direction == "positive"
            else "tends to decrease risk"
        )
        parts.append(
            f"#{i} {feat.name} (importance: {feat.importance:.4f}, "
            f"{sign_desc})"
        )

    narrative += "; ".join(parts) + "."

    # Add a summary comparison
    if len(features) >= 2:
        ratio = features[0].importance / features[1].importance
        if ratio > 1.5:
            narrative += (
                f" Notably, {features[0].name} is {ratio:.1f}× more "
                f"influential than the next most important feature."
            )

    return narrative


def narrate_partial_dependence(pdp: PartialDependenceResult) -> str:
    """Convert partial dependence data into an English description.

    Args:
        pdp: Output from compute_partial_dependence().

    Returns:
        An English paragraph describing how the feature affects predictions.

    Example:
        >>> narrate_partial_dependence(pdp)
        "As mean radius increases from 6.98 to 28.11, the predicted
         probability changes from 12.3% to 89.1%..."
    """
    # Determine the overall trend
    pred_start = pdp.predictions[0]
    pred_end = pdp.predictions[-1]
    change = pred_end - pred_start

    if change > 0.05:
        trend = "increases"
    elif change < -0.05:
        trend = "decreases"
    else:
        trend = "remains relatively stable"

    narrative = (
        f"As {pdp.feature_name} increases from {pdp.feature_min:.2f} to "
        f"{pdp.feature_max:.2f}, the predicted probability {trend} "
        f"from {pdp.prediction_min:.1%} to {pdp.prediction_max:.1%}."
    )

    # Add detail about the range of effect
    pred_range = pdp.prediction_max - pdp.prediction_min
    narrative += (
        f" The total effect range is {pred_range:.1%} "
        f"(from {pdp.prediction_min:.1%} to {pdp.prediction_max:.1%})."
    )

    # Find the steepest change region (approximate inflection point)
    if len(pdp.predictions) > 2:
        diffs = [
            abs(pdp.predictions[i + 1] - pdp.predictions[i])
            for i in range(len(pdp.predictions) - 1)
        ]
        max_change_idx = diffs.index(max(diffs))
        steep_start = pdp.feature_values[max_change_idx]
        steep_end = pdp.feature_values[max_change_idx + 1]
        narrative += (
            f" The steepest change occurs between "
            f"{pdp.feature_name} = {steep_start:.2f} and {steep_end:.2f}."
        )

    return narrative


def narrate_dataset(description: DatasetDescription) -> str:
    """Convert dataset statistics into an English overview.

    Args:
        description: Output from compute_dataset_description().

    Returns:
        An English paragraph describing the dataset.

    Example:
        >>> narrate_dataset(description)
        "The dataset contains 114 samples with 30 features..."
    """
    narrative = (
        f"The dataset contains {description.n_samples} samples with "
        f"{description.n_features} features."
    )

    # Class distribution
    class_parts = [
        f"{name}: {count} ({count / description.n_samples:.1%})"
        for name, count in description.class_distribution.items()
    ]
    narrative += f" Class distribution: {', '.join(class_parts)}."

    # Missing values
    if description.missing_values == 0:
        narrative += " There are no missing values in the dataset."
    else:
        narrative += (
            f" There are {description.missing_values} missing values "
            f"across all features."
        )

    return narrative


def narrate_prediction_comparison(comparison: PredictionComparison) -> str:
    """Convert a prediction comparison into an English narrative.

    Covers agreement/disagreement, confidence gap, and shared/divergent
    feature drivers between two models on the same sample.

    Args:
        comparison: Output from compute_prediction_comparison().

    Returns:
        A complete English paragraph comparing the two models' predictions.

    Example:
        >>> narrate_prediction_comparison(comparison)
        "Both models agree: they classified this sample as benign..."
    """
    m1 = comparison.per_model[0]
    m2 = comparison.per_model[1]

    # --- Opening: agreement or disagreement ---
    if comparison.agreement:
        narrative = (
            f"Both models agree: they classified this sample as "
            f"{m1.predicted_label}. "
            f"{m1.model_id} ({m1.model_type}) assigns probability {m1.probability:.2f}, "
            f"while {m2.model_id} ({m2.model_type}) assigns {m2.probability:.2f}"
        )
        if comparison.confidence_gap < 0.05:
            narrative += " — their confidence levels are very close."
        else:
            narrative += (
                f" — a confidence gap of {comparison.confidence_gap:.2f}."
            )
    else:
        narrative = (
            f"The models disagree on this sample. "
            f"{m1.model_id} ({m1.model_type}) classifies it as "
            f"{m1.predicted_label} (probability: {m1.probability:.2f}), "
            f"while {m2.model_id} ({m2.model_type}) classifies it as "
            f"{m2.predicted_label} (probability: {m2.probability:.2f})."
        )

    # --- Shared drivers ---
    if comparison.shared_top_features:
        shared = comparison.shared_top_features
        if len(shared) == 1:
            narrative += (
                f" Both models rely on {shared[0]} as a top driver."
            )
        else:
            joined = ", ".join(shared[:-1]) + f" and {shared[-1]}"
            narrative += (
                f" Both models share {len(shared)} top drivers: {joined}."
            )
    else:
        narrative += (
            " The models rely on entirely different features in their "
            "top drivers, suggesting fundamentally different reasoning paths."
        )

    # --- Divergent drivers ---
    has_divergent = any(
        len(feats) > 0 for feats in comparison.divergent_features.values()
    )
    if has_divergent:
        parts = []
        for model_id, feats in comparison.divergent_features.items():
            if feats:
                parts.append(
                    f"{model_id} uniquely relies on {', '.join(feats)}"
                )
        if parts:
            narrative += " " + "; ".join(parts) + "."

    # --- Disagreement explanation ---
    if not comparison.agreement:
        # Identify which features push the models apart
        m1_top_names = [f["name"] for f in m1.top_features]
        m2_top_names = [f["name"] for f in m2.top_features]
        m1_directions = {
            f["name"]: f["direction"] for f in m1.top_features
        }
        m2_directions = {
            f["name"]: f["direction"] for f in m2.top_features
        }

        # Find features where both models look at them but in opposite directions
        opposing = []
        for feat in comparison.shared_top_features:
            if feat in m1_directions and feat in m2_directions:
                if m1_directions[feat] != m2_directions[feat]:
                    opposing.append(feat)

        if opposing:
            narrative += (
                f" The disagreement may be driven by {', '.join(opposing)}, "
                f"where the models interpret the feature's effect in "
                f"opposite directions."
            )
        else:
            narrative += (
                " The disagreement stems from the models weighting "
                "different features rather than interpreting the same "
                "features differently."
            )

    return narrative


# --- Drift narrators ---


def narrate_feature_drift(result: FeatureDriftResult) -> str:
    """Convert a per-feature drift result into an English paragraph.

    Leads with severity, describes shift direction and magnitude,
    includes supporting statistical evidence. Does not speculate
    about causes (epistemic honesty — ADR-002).

    Args:
        result: Output from detect_feature_drift().

    Returns:
        A deterministic English paragraph describing drift for one feature.
    """
    name = result.feature_name

    if not result.drift_detected:
        return (
            f"No significant drift detected in feature `{name}`. "
            f"The distribution remains stable between reference and current data "
            f"({result.test_name} statistic = {result.statistic:.4f}"
            + (f", p-value = {result.p_value:.4f}" if result.p_value is not None else "")
            + ")."
        )

    # --- Drifted feature: severity first ---
    test_label = {
        "psi": "PSI",
        "ks": "KS",
        "chi_squared": "chi-squared",
    }[result.test_name]

    narrative = (
        f"Significant drift detected in feature `{name}` "
        f"({test_label} = {result.statistic:.4f}, {result.severity})"
    )

    # Supporting p-value for PSI (KS p-value is secondary evidence)
    if result.p_value is not None and result.test_name == "psi":
        narrative += f", KS p-value = {result.p_value:.4f}"

    narrative += "."

    # --- Shift direction (numeric features) ---
    ref = result.reference_summary
    cur = result.current_summary

    median_diff = cur.median - ref.median
    if result.test_name in ("psi", "ks"):
        if abs(median_diff) > 1e-6:
            direction = "rightward (increased)" if median_diff > 0 else "leftward (decreased)"
            narrative += (
                f" The distribution shifted {direction} — "
                f"median moved from {ref.median:.2f} to {cur.median:.2f}."
            )

        # Mean and std context
        mean_diff = cur.mean - ref.mean
        std_change = cur.std - ref.std
        parts = []
        if abs(mean_diff) > 1e-6:
            parts.append(f"mean: {ref.mean:.2f} → {cur.mean:.2f}")
        if abs(std_change) > 0.01:
            direction_std = "widened" if std_change > 0 else "narrowed"
            parts.append(f"spread {direction_std} (std: {ref.std:.2f} → {cur.std:.2f})")
        if parts:
            narrative += f" Additional context: {'; '.join(parts)}."
    else:
        # Categorical: less meaningful to describe shift direction
        narrative += (
            f" The category distribution has changed significantly "
            f"(p-value = {result.p_value:.4f})."
        )

    # Sample sizes
    narrative += (
        f" Reference: {ref.n_samples} samples, current: {cur.n_samples} samples."
    )

    return narrative


def narrate_dataset_drift(result: DatasetDriftResult) -> str:
    """Convert aggregate drift results into an English summary.

    Ranks drifted features by severity, states overall drift level,
    and frames whether the drift warrants investigation. Does not
    speculate about causes (epistemic honesty — ADR-002).

    Args:
        result: Output from detect_drift().

    Returns:
        A deterministic English paragraph summarizing dataset-level drift.
    """
    if result.n_drifted == 0:
        return (
            f"No significant drift detected across any of the "
            f"{result.n_features} features. The input distributions "
            f"remain stable between reference and current data."
        )

    # --- Opening: count and severity ---
    narrative = (
        f"{result.n_drifted} of {result.n_features} features show "
        f"significant drift (overall severity: {result.overall_severity})."
    )

    # --- Ranked list of drifted features ---
    drifted = [f for f in result.features if f.drift_detected]
    severity_order = {"severe": 0, "moderate": 1, "none": 2}
    drifted.sort(key=lambda f: (severity_order[f.severity], -f.statistic))

    test_label = {"psi": "PSI", "ks": "KS", "chi_squared": "chi-squared"}

    parts = []
    for f in drifted:
        label = test_label.get(f.test_name, f.test_name)
        parts.append(f"`{f.feature_name}` ({label} {f.statistic:.2f}, {f.severity})")

    narrative += " The most affected features are: " + ", ".join(parts) + "."

    # --- "So what" framing ---
    if result.overall_severity == "severe":
        narrative += (
            " This level of drift suggests the model's input data has changed "
            "substantially. Investigation is recommended to determine whether "
            "model retraining or recalibration is needed."
        )
    elif result.overall_severity == "moderate":
        narrative += (
            " Moderate drift has been detected. The affected features should "
            "be monitored; if drift persists or worsens, model performance "
            "may degrade."
        )

    return narrative
