---
name: xai-workflow
description: >
  Guide for conducting ML model explainability analyses using the XAI toolkit's
  MCP tools. Use this skill whenever a user asks for a model explanation, prediction
  analysis, feature investigation, or any question about why a model made a specific
  decision. Also use when the user asks for a "full analysis", "thorough explanation",
  or "deep dive" into model behavior. This skill defines the correct tool sequencing
  (global-to-local funnel), Glass Floor protocol for separating deterministic
  explanations from business context, and abbreviation rules for narrower questions.
---

# XAI Explainability Workflow

> Grounded in Christoph Molnar's *Interpretable Machine Learning* (2023).
> Maps the global-to-local explanation funnel to this toolkit's MCP tools.

## The Golden Rule

**You are the presenter, not the analyst.**

All computation is done by the toolkit's pure Python functions. Your job is to
choose the right tool, call it, and present the pre-computed `narrative` field
conversationally. The narrative is deterministic — same question + same data =
same answer, always. Do not reinterpret, paraphrase, or supplement SHAP values
beyond what the narrative states.

If none of the available tools answer the user's question, say so. Admitting
uncertainty is always better than guessing.


## The Explanation Funnel

Follow this sequence when a user asks for a thorough model explanation. Each
step builds context for the next. Skip steps only when the user explicitly
asks for something narrow (see "When to Abbreviate" below).

### Step 1 — Orient: Understand the data and model context

**Tools:** `describe_dataset` → `summarize_model`

Why first: You cannot interpret feature contributions without knowing what the
features represent, their ranges, and what the model predicts. Before explaining
predictions, establish the data generating process.

Present both tool narratives before moving on. The user now knows: what the
data looks like, what the model does, and how accurate it is.

*Molnar, Ch. 2 — "Interpretability" and the role of data context.*

### Step 2 — Global picture: What does the model rely on overall?

**Tool:** `compare_features`

Why second: Global feature importance tells you which features the model weighs
most across ALL predictions. This is the baseline. Individual predictions may
deviate from this pattern — those deviations are often the most interesting
findings.

*Molnar, Ch. 9.5 — "SHAP Feature Importance" at the global level.*

### Step 3 — Feature effects: How does changing a feature affect predictions?

**Tool:** `get_partial_dependence`

Call this for the top 2–3 features from Step 2. Partial Dependence Plots reveal
the *shape* of the relationship (linear? threshold? U-shaped?) that importance
alone cannot show. Importance tells you a feature matters; PDP tells you *how*
it matters.

*Molnar, Ch. 8.1 — "Partial Dependence Plot (PDP)".*

### Step 4 — Local explanation: Why did THIS sample get THIS prediction?

**Tools:** `explain_prediction`, `explain_prediction_waterfall`

Why last: Now the user has context. When you say "worst texture pushed the
prediction toward malignant", they already know from Steps 2–3 that texture is
globally important and has a monotonic relationship with risk.

Use `explain_prediction` for the narrative + bar chart. Use
`explain_prediction_waterfall` if the user wants the full SHAP force breakdown
or asks for a waterfall plot.

*Molnar, Ch. 9.6 — "SHAP values for individual predictions".*


## When to Abbreviate

Not every question requires the full funnel. Match depth to intent:

| User intent | Steps | Tools |
|---|---|---|
| "Why was sample X classified as Y?" | Step 4, cite top global feature for context | `explain_prediction` |
| "What features matter most?" | Steps 1–2 | `describe_dataset` → `compare_features` |
| "How does feature Z affect predictions?" | Step 3 for that feature | `get_partial_dependence` |
| "Compare these two models" | Steps 1–2 for each model | `summarize_model` (×2) → `compare_features` |
| "Give me a full analysis" | All four steps in order | Full funnel |
| "What should I do about this result?" | Step 4 + Glass Floor Layer 2 | `explain_prediction` → `retrieve_business_context` |


## Glass Floor Protocol (ADR-009)

When an analysis calls for business context — clinical protocols, operational
thresholds, recommended actions — use the two-layer separation:

**Layer 1 (always first):** Present the deterministic tool narrative exactly as
returned. This is computed, reproducible, and audit-ready. Label it:

> 📊 **Model Explanation** (deterministic, grounded)

**Layer 2 (always second, always separate):** Present business context from
`retrieve_business_context`. Always prefix with source attribution:

> 📋 **Business Context** (AI-interpreted from [source_document], [section])

Rules:
- Never blend Layer 1 and Layer 2 into a single paragraph.
- Never modify the deterministic narrative based on business context.
- Layer 2 content should be verified by the user before acting on it.
- If `retrieve_business_context` returns low-relevance results, say so rather
  than presenting weakly matched content as authoritative.

*The name "Glass Floor" means the separation is transparent: users can always
see which layer produced which content and verify accordingly.*


## Common Anti-Patterns

Avoid these — they undermine the toolkit's value proposition:

- **Inventing SHAP explanations.** If the tool errors or returns unexpected
  results, present the error. Never fabricate feature contributions.
- **Skipping the orient step.** Jumping to `explain_prediction` without context
  means neither you nor the user knows what the features represent.
- **Blending layers.** Saying "the model flagged this because texture is high,
  and per hospital protocol patients should be referred for biopsy" in one
  sentence destroys the audit trail. Keep layers visually distinct.
- **Ignoring the `grounded` flag.** If you answer without calling a tool,
  prepend a ⚠️ disclaimer. Your response is not grounded in computation.
