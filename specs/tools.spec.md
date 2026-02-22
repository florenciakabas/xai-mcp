# Tool Catalog — xai-toolkit MCP Server

## explain_prediction

**Purpose:** Explain why a single sample received its classification.

**When to use:** User asks "Why was sample X classified as Y?" or
"What drove the prediction for this patient/well/record?"

### Input Parameters

| Parameter      | Type   | Required | Description                                    |
|---------------|--------|----------|------------------------------------------------|
| model_id      | string | yes      | ID of a registered model (e.g., "xgboost_breast_cancer") |
| sample_index  | int    | yes      | Row index in the dataset to explain            |

### Output Structure

```json
{
  "narrative": "The model classified sample 42 as malignant (probability: 0.91) primarily because of three factors: worst_radius is 2.1× above average (pushing risk up by +0.28), worst_concave_points is elevated (contributing +0.19), and mean_concavity exceeds the norm (+0.14). The top protective factor is mean_smoothness, which is below average (−0.06).",
  "evidence": {
    "prediction": 1,
    "prediction_label": "malignant",
    "probability": 0.91,
    "base_value": 0.42,
    "shap_values": {
      "worst_radius": 0.28,
      "worst_concave_points": 0.19,
      "mean_concavity": 0.14,
      "mean_smoothness": -0.06
    },
    "feature_values": {
      "worst_radius": 23.4,
      "worst_concave_points": 0.18,
      "mean_concavity": 0.15,
      "mean_smoothness": 0.08
    },
    "feature_names": ["worst_radius", "worst_concave_points", "..."],
    "top_n_features": 3
  },
  "metadata": {
    "model_id": "xgboost_breast_cancer",
    "model_type": "XGBClassifier",
    "timestamp": "2025-02-21T10:30:00Z",
    "tool_version": "0.1.0",
    "sample_index": 42,
    "dataset_size": 569
  }
}
```

### Narrative Requirements

The narrative MUST:
- State the predicted class and probability
- Name the top 3 contributing features (by absolute SHAP magnitude)
- For each feature, state the direction ("pushing toward" / "pushing away from")
  and the magnitude (e.g., "+0.23")
- Mention at least one protective/opposing factor if one exists
- Be a complete, readable English paragraph (not bullet points)

### Error Cases

| Condition              | Response                                          |
|-----------------------|---------------------------------------------------|
| model_id not found    | Error listing available models                    |
| sample_index < 0      | Error: "Sample index must be non-negative"        |
| sample_index >= N     | Error: "Dataset has N samples, index M is out of range" |

---

## Tools Planned for Day 2+

### summarize_model
Provide an overview of what a model does and what drives its decisions.

### compare_features
Rank features by importance with magnitude and direction in English.

### get_partial_dependence
Describe how a single feature affects predictions across its range.

### list_models
List all registered models with their metadata.

### describe_dataset
Provide data shape, feature names, types, missing values, and basic stats.
