"""Versioned skill registry with lightweight guardrails."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, TypedDict

from xai_toolkit.instructions import get_glass_floor_principles, get_methodology_content


class RegisteredSkill(TypedDict):
    """A versioned context skill exposed to tool callers."""

    skill_id: str
    title: str
    version: str
    intent_tags: list[str]
    max_scope: Literal["minimal", "standard", "extended"]
    checksum: str
    content: str


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_skill_registry(root_dir: Path) -> dict[str, RegisteredSkill]:
    """Load all built-in skills and compute immutable checksums."""
    methodology = get_methodology_content(root_dir)
    glass_floor = get_glass_floor_principles()
    return {
        "xai_methodology": {
            "skill_id": "xai_methodology",
            "title": "XAI Methodology",
            "version": "1.0.0",
            "intent_tags": ["analysis_workflow", "explainability", "sequencing"],
            "max_scope": "extended",
            "checksum": _checksum(methodology),
            "content": methodology,
        },
        "glass_floor": {
            "skill_id": "glass_floor",
            "title": "Glass Floor Protocol",
            "version": "1.0.0",
            "intent_tags": ["provenance", "communication", "governance"],
            "max_scope": "standard",
            "checksum": _checksum(glass_floor),
            "content": glass_floor,
        },
    }


def resolve_skill(
    registry: dict[str, RegisteredSkill],
    *,
    skill_id: str,
    version: str | None = None,
) -> RegisteredSkill:
    """Resolve a skill with guardrails for unknown IDs and version mismatches."""
    if skill_id not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown skill_id '{skill_id}'. Available: {available}")

    skill = registry[skill_id]
    if version is not None and version != skill["version"]:
        raise ValueError(
            f"Unsupported version '{version}' for skill '{skill_id}'. "
            f"Available version: {skill['version']}"
        )
    return skill
