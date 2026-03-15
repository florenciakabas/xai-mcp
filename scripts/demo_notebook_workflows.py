"""Generate a Monday-demo artifact pack from notebook-style workflows.

This script runs local wrapper flows and writes JSON outputs under:
  scenarios/demo_outputs/
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from xai_toolkit.notebook_wrappers import ask_xai, drift_summary, explain_sample
from xai_toolkit.server import init_server


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "scenarios" / "demo_outputs"
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat().replace(":", "-")
    session_dir = output_dir / f"session-{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)

    init_server()
    model_id = "xgboost_breast_cancer"

    run_summary = ask_xai(
        question="What changed in latest run for this model?",
        intent="standard_briefing",
        model_id=model_id,
    )
    drift = drift_summary(model_id=model_id)
    sample = explain_sample(model_id=model_id, sample_index=0, include_plot=False)
    fallback = explain_sample(model_id=model_id, sample_index=113, include_plot=False)

    _write_json(session_dir / "01_standard_briefing.json", run_summary)
    _write_json(session_dir / "02_drift_summary.json", drift)
    _write_json(session_dir / "03_explain_sample_0.json", sample)
    _write_json(session_dir / "04_explain_sample_fallback.json", fallback)

    print("Demo artifacts written to:")
    print(f"  {session_dir}")
    print("Generated files:")
    for name in sorted(p.name for p in session_dir.iterdir()):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
