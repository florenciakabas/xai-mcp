"""Tests for instruction-serving tools and pure functions.

Covers:
- get_methodology_content: reads SKILL.md, handles missing file
- get_glass_floor_principles: returns curated content
- Server tool wrappers: get_xai_methodology, get_glass_floor
"""

import pytest
from pathlib import Path

from xai_toolkit.instructions import get_glass_floor_principles, get_methodology_content


ROOT = Path(__file__).parent.parent  # xai-mcp/


# ---------------------------------------------------------------------------
# Pure function: get_methodology_content
# ---------------------------------------------------------------------------


class TestGetMethodologyContent:

    def test_returns_string(self):
        result = get_methodology_content(ROOT)
        assert isinstance(result, str)

    def test_contains_explanation_funnel(self):
        result = get_methodology_content(ROOT)
        assert "Explanation Funnel" in result

    def test_contains_glass_floor_protocol(self):
        result = get_methodology_content(ROOT)
        assert "Glass Floor Protocol" in result

    def test_contains_when_to_abbreviate(self):
        result = get_methodology_content(ROOT)
        assert "When to Abbreviate" in result

    def test_length_is_substantial(self):
        result = get_methodology_content(ROOT)
        assert len(result) > 500

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_methodology_content(tmp_path)


# ---------------------------------------------------------------------------
# Pure function: get_glass_floor_principles
# ---------------------------------------------------------------------------


class TestGetGlassFloorPrinciples:

    def test_returns_string(self):
        result = get_glass_floor_principles()
        assert isinstance(result, str)

    def test_contains_layer_1(self):
        result = get_glass_floor_principles()
        assert "Layer 1" in result

    def test_contains_layer_2(self):
        result = get_glass_floor_principles()
        assert "Layer 2" in result

    def test_contains_deterministic(self):
        result = get_glass_floor_principles()
        assert "deterministic" in result.lower()

    def test_contains_ai_interpreted(self):
        result = get_glass_floor_principles()
        assert "ai-interpreted" in result.lower()

    def test_contains_provenance(self):
        result = get_glass_floor_principles()
        assert "provenance" in result.lower()


# ---------------------------------------------------------------------------
# Server tool wrappers (called as plain functions)
# ---------------------------------------------------------------------------


class TestServerTools:

    def test_get_xai_methodology_returns_content(self):
        from xai_toolkit.server import get_xai_methodology

        result = get_xai_methodology()
        assert isinstance(result, str)
        assert "Explanation Funnel" in result

    def test_get_glass_floor_returns_content(self):
        from xai_toolkit.server import get_glass_floor

        result = get_glass_floor()
        assert isinstance(result, str)
        assert "Layer 1" in result
        assert "Layer 2" in result
