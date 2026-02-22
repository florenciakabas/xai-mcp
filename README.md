# xai-toolkit

ML model explainability as plain-English narratives, exposed via MCP.

## What It Does

Ask a question in VS Code Copilot:
> "Why was sample 42 classified as malignant?"

Get back a deterministic English explanation:
> "The model classified sample 42 as malignant (probability: 0.91) primarily
> because of three factors: worst_radius is 2.1× above average (pushing risk
> up by +0.28), worst_concave_points is elevated (+0.19), and mean_concavity
> exceeds the norm (+0.14)."

No plots to interpret. No code to run. English that a decision-maker can act on.

## Quick Start

```bash
uv sync                                    # Install dependencies
uv run pytest                              # Run tests
uv run python -m xai_toolkit.server        # Start MCP server
```

## Architecture

See [AGENTS.md](AGENTS.md) for full project structure and design decisions.
See [docs/decisions/](docs/decisions/) for Architecture Decision Records.

## Design Principle

**The LLM is the presenter, not the analyst.** All computation and narrative
generation is done deterministically in Python. The LLM chooses the right tool
and presents the result — nothing more.
