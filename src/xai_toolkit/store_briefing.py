"""Pure formatting helpers for discovery and standard briefing outputs."""

from __future__ import annotations

from collections import Counter
import json

from xai_toolkit.schemas import StoredDriftResult, StoredExplanation


def build_drift_alerts_payload(
    drift_results: list[StoredDriftResult],
    model_id: str | None = None,
    severity: str | None = None,
) -> tuple[str, dict]:
    """Build narrative/evidence payload for drift alert discovery."""
    filtered = drift_results
    if severity is not None:
        filtered = [r for r in filtered if r.severity == severity]

    if not filtered:
        narrative = (
            "No batch drift results available"
            + (f" for model '{model_id}'" if model_id else "")
            + (f" with severity='{severity}'" if severity else "")
            + "."
        )
    else:
        n_drifted = sum(1 for r in filtered if r.drift_detected)
        narrative = (
            f"Found {len(filtered)} drift results"
            + (f" for model '{model_id}'" if model_id else "")
            + f": {n_drifted} features with detected drift."
        )

    alerts = [
        {
            "model_id": r.model_id,
            "feature_name": r.feature_name,
            "severity": r.severity,
            "test_name": r.test_name,
            "statistic": r.statistic,
            "drift_detected": r.drift_detected,
            "narrative": r.narrative,
            "run_id": r.run_id,
            "computed_at": r.computed_at,
        }
        for r in filtered
    ]
    return narrative, {"alerts": alerts, "count": len(alerts)}


def build_explained_samples_payload(
    model_id: str,
    explanations: list[StoredExplanation],
) -> tuple[str, dict]:
    """Build narrative/evidence payload for explained-sample discovery."""
    if not explanations:
        return (
            f"No precomputed explanations available for model '{model_id}'.",
            {"samples": [], "count": 0},
        )

    narrative = (
        f"Found {len(explanations)} precomputed explanations for model "
        f"'{model_id}' (run: {explanations[0].run_id})."
    )
    samples = [
        {
            "sample_index": e.sample_index,
            "prediction": e.prediction,
            "prediction_label": e.prediction_label,
            "probability": e.probability,
            "run_id": e.run_id,
            "computed_at": e.computed_at,
        }
        for e in explanations
    ]
    return narrative, {"samples": samples, "count": len(samples)}


def build_standard_briefing_payload(
    per_model_results: list[tuple[str, list[StoredExplanation], list[StoredDriftResult]]],
    model_id: str | None = None,
    run_id: str | None = None,
    top_cases: int = 5,
) -> tuple[str, dict]:
    """Build narrative/evidence payload for a predefined standard briefing."""
    per_model: list[dict] = []
    for mid, explanations, drift_rows in per_model_results:
        if not explanations and not drift_rows:
            continue

        if explanations:
            run_for_model = explanations[0].run_id
            highlighted = sorted(explanations, key=lambda e: e.probability, reverse=True)[:top_cases]
            top_explained_cases = [
                {
                    "sample_index": e.sample_index,
                    "prediction_label": e.prediction_label,
                    "probability": e.probability,
                    "short_narrative": e.narrative.split(". ")[0].strip(),
                }
                for e in highlighted
            ]
        else:
            run_for_model = drift_rows[0].run_id
            top_explained_cases = []

        n_drifted = sum(1 for r in drift_rows if r.drift_detected)
        overall_severity = drift_rows[0].overall_severity if drift_rows else "none"
        severe_features = [r.feature_name for r in drift_rows if r.severity == "severe"][:3]

        top_driver_names: list[str] = []
        if explanations:
            feature_counter: Counter[str] = Counter()
            for exp in explanations:
                try:
                    for feat in json.loads(exp.top_features):
                        name = feat.get("name")
                        if isinstance(name, str) and name:
                            feature_counter[name] += 1
                except Exception:
                    continue
            top_driver_names = [name for name, _ in feature_counter.most_common(3)]

        per_model.append(
            {
                "model_id": mid,
                "run_id": run_for_model,
                "n_precomputed_explanations": len(explanations),
                "n_drift_features": len(drift_rows),
                "n_drifted_features": n_drifted,
                "overall_drift_severity": overall_severity,
                "severe_drift_features": severe_features,
                "top_recurrent_drivers": top_driver_names,
                "top_explained_cases": top_explained_cases,
            }
        )

    if not per_model:
        narrative = (
            "No precomputed batch results available for a standard briefing"
            + (f" (model='{model_id}')" if model_id else "")
            + (f" (run='{run_id}')" if run_id else "")
            + "."
        )
        return narrative, {"models": [], "count_models": 0}

    total_explanations = sum(item["n_precomputed_explanations"] for item in per_model)
    total_drifted = sum(item["n_drifted_features"] for item in per_model)
    severe_models = [item["model_id"] for item in per_model if item["overall_drift_severity"] == "severe"]

    narrative_parts = [
        f"Standard briefing covers {len(per_model)} model(s)",
        f"with {total_explanations} precomputed explanations",
        f"and {total_drifted} drifted features in total",
    ]
    if severe_models:
        narrative_parts.append(f"Severe drift is present for: {', '.join(severe_models)}")
    narrative = ". ".join(narrative_parts) + "."

    evidence = {
        "models": per_model,
        "count_models": len(per_model),
        "summary": {
            "total_precomputed_explanations": total_explanations,
            "total_drifted_features": total_drifted,
            "models_with_severe_drift": severe_models,
        },
    }
    return narrative, evidence
