# ADR-007: Single-Agent Architecture — LLM as Natural Router

## Status: Accepted
## Date: 2025-02-21

## Context

Multi-agent architectures (supervisor + specialist agents) are frequently
proposed for complex tool ecosystems. The question is whether this system
needs that complexity now, or whether a simpler approach suffices.

## Decision

Use a **single-agent architecture** where one LLM (Sonnet 4.5 in VS Code
Copilot) routes user intent to tools via MCP tool descriptions. No
supervisor agent, no orchestration layer, no agent-to-agent communication.

The routing works because MCP tool descriptions are precise enough for the
LLM to select the correct tool from natural language alone:

| User asks | LLM calls |
|---|---|
| "Why was patient 42 flagged?" | `explain_prediction` |
| "Which features matter most?" | `compare_features` |
| "How does mean radius affect risk?" | `get_partial_dependence` |
| "What models are available?" | `list_models` |

This is not a limitation — it is the correct architecture for this scale.

## Consequences

- ✅ Zero orchestration overhead — no supervisor model, no routing latency
- ✅ Simpler to debug — one LLM, one tool call, one response
- ✅ Easier to test — each tool is independently testable with pytest
- ✅ MCP protocol already handles the intent → tool routing problem
- ✅ Tool descriptions are the configuration layer — no code needed to adjust routing
- ⚠️ Multi-step workflows (e.g., "compare these 3 models and recommend one") require
     the LLM to chain tool calls — acceptable given Copilot's agentic mode supports this
- ⚠️ If the tool count grows beyond ~15, tool selection accuracy may degrade —
     at that point, consider grouping into domain-scoped MCP servers

## When to Revisit This Decision

Upgrade to multi-agent when ANY of these conditions are true:
1. Tool count exceeds 15 in a single server (routing accuracy degrades)
2. Workflows require parallel tool execution (e.g., explain 10 samples simultaneously)
3. Different tools require different security contexts (separate auth domains)
4. A dedicated orchestration layer adds measurable business value

## Production Evolution Path

```
Current:  Single MCP server (7 tools) → Sonnet 4.5 routes via descriptions
Phase 2:  2-3 domain-scoped MCP servers (explainability, data access, compliance)
          All connected to Copilot simultaneously — still single-agent routing
Phase 3:  Autonomous multi-step workflows with LangGraph or similar orchestration
          Only if Phase 2 proves insufficient
```

## Alternatives Considered

- **LangGraph supervisor**: Adds significant complexity (agent definitions, state management,
  inter-agent messages). No benefit at this scale. Rejected for now.
- **Separate specialist agents per model type**: Over-engineered. The Registry pattern
  already handles multi-model without agent separation. Rejected.
