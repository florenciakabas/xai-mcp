# Tool Catalog — xai-toolkit MCP Server

All tools return a consistent `ToolResponse` structure (see ADR-005):
```json
{
  "narrative":   "Plain English interpretation...",
  "evidence":    { ... structured data ... },
  "metadata":    { "model_id": ..., "timestamp": ..., "tool_version": ..., "data_hash": ... },
  "plot_base64": null,
  "grounded":    true
}
```

`grounded: true` is present on every tool response. It signals to the LLM that
this answer was computed deterministically from a registered model and is
audit-ready. If the LLM answers a question without calling a tool (general
knowledge), it is instructed to prepend a ⚠️ disclaimer — because no
`grounded: true` field will be present in that case.

---

## explain_prediction

**Purpose:** Explain why a single sample received its classification.
**Trigger:** "Why was sample X classified as Y?" / "What drove this prediction?"

| Parameter    | Type   | Required | Description                              |
|-------------|--------|----------|------------------------------------------|
| model_id    | string | yes      | ID of a registered model                 |
| sample_index| int    | yes      | Row index in the test dataset to explain |

**Narrative includes:** predicted class, probability, top 3 contributing
features with direction and magnitude, top opposing factor.

---

## summarize_model

**Purpose:** Provide a high-level overview of what a model does.
**Trigger:** "What does this model do?" / "Summarize the model"

| Parameter | Type   | Required | Description              |
|----------|--------|----------|--------------------------|
| model_id | string | yes      | ID of a registered model |

**Narrative includes:** model type, accuracy, feature count, target classes,
top 5 features by importance.

---

## compare_features

**Purpose:** Rank features by global importance.
**Trigger:** "Which features matter most?" / "Compare feature importance"

| Parameter | Type   | Required | Default | Description              |
|----------|--------|----------|---------|--------------------------|
| model_id | string | yes      |         | ID of a registered model |
| top_n    | int    | no       | 10      | Number of features to rank |

**Narrative includes:** ranked list with magnitude, direction, comparative
language (e.g., "1.5× more influential").

---

## get_partial_dependence

**Purpose:** Show how a single feature affects predictions.
**Trigger:** "How does [feature] affect predictions?" / "What happens when [feature] changes?"

| Parameter    | Type   | Required | Description                    |
|-------------|--------|----------|--------------------------------|
| model_id    | string | yes      | ID of a registered model       |
| feature_name| string | yes      | Feature to analyze             |

**Narrative includes:** feature range, prediction range, overall trend,
steepest change region.

**Error handling:** Typos in feature_name return suggestions via fuzzy matching.

---

## list_models

**Purpose:** Discover what models are available.
**Trigger:** "What models are available?" / "List models"

No parameters required.

**Narrative includes:** count, model IDs, types, feature counts, accuracy.

---

## describe_dataset

**Purpose:** Describe the data associated with a model.
**Trigger:** "Tell me about the data" / "Describe the dataset"

| Parameter | Type   | Required | Description              |
|----------|--------|----------|--------------------------|
| model_id | string | yes      | ID of a registered model |

**Narrative includes:** sample count, feature count, class distribution,
missing value count.
