# XAI Methodology Guide

> **⚠️ Deprecated:** The canonical version of this workflow now lives in
> `skills/xai-workflow/SKILL.md`. This file is kept for reference but may
> fall out of sync. Update the skill, not this file.

> Grounded in Christoph Molnar's *Interpretable Machine Learning* (2023).
> Maps the global-to-local explanation funnel to this toolkit's MCP tools.

## When to Use This Guide

Follow this workflow when a user asks for a thorough model explanation — or
when you judge that a single tool call would give an incomplete picture.
Each step builds context for the next. Skip steps only if the user explicitly
asks for something narrow.

## The Explanation Funnel

### Step 1: Orient — Understand the data and model context

**Tools:** `describe_dataset`, `summarize_model`

Why first: You cannot interpret feature contributions without knowing what the
features are, their ranges, and what the model is trying to predict.
Before explaining predictions, understand the data generating process.

*Reference: Molnar, Ch. 2 — "Interpretability" and the role of data context.*

### Step 2: Global picture — What does the model rely on overall?

**Tools:** `compare_features`

Why second: Global feature importance tells you which features the model uses
most across ALL predictions. This is the baseline. Individual predictions
may deviate from this pattern, and those deviations are often the most
interesting findings.

*Reference: Molnar, Ch. 9.5 — "SHAP Feature Importance" at the global level.*

### Step 3: Feature effects — How does changing a feature affect predictions?

**Tools:** `get_partial_dependence`

Call this for the top 2-3 features from Step 2. Partial Dependence Plots show
the *shape* of the relationship (linear? threshold? U-shaped?) which SHAP
importance alone cannot reveal. Importance tells you a feature matters; PDP
tells you *how* it matters.

*Reference: Molnar, Ch. 8.1 — "Partial Dependence Plot (PDP)".*

### Step 4: Local explanation — Why did THIS sample get THIS prediction?

**Tools:** `explain_prediction`, `explain_prediction_waterfall`

Why last: Now the user has context. When you say "worst texture pushed the
prediction toward malignant", they already know from Steps 2-3 that texture
is globally important and has a monotonic relationship with risk.

*Reference: Molnar, Ch. 9.6 — "SHAP values for individual predictions".*

## When to Abbreviate

Not every question needs all four steps. Match depth to intent:

- **"Why was sample X classified as Y?"** → Step 4, but briefly cite the top
  global feature from Step 2 for context.
- **"What features matter?"** → Steps 1-2 only.
- **"How does feature Z affect predictions?"** → Step 3 for that feature.
- **"Give me a full analysis"** → All four steps in order.