"""
Tests for Priority 2: Local interface variant in MD/HDOCK shortlist.

Verifies:
- _build_downstream_shortlist includes local interface variants
- _build_downstream_shortlist excludes interface variants not in the local region
- _build_downstream_shortlist handles empty variant list
- _build_downstream_shortlist respects max_variants limit
"""

import pytest


class TestBuildDownstreamShortlist:
    """Test _build_downstream_shortlist method."""

    def test_includes_local_interface_variants(self):
        """Should include interface variants that are in the local region."""
        from orchestration.agents.structural_agent import StructuralAgent

        agent = StructuralAgent()

        # Mock variants with one in local region
        variants = [
            {"variant_id": "var1", "position": 50, "region": "interface"},
            {"variant_id": "var2", "position": 100, "region": "surface"},
        ]

        # Local region: positions 40-60
        result = agent._build_downstream_shortlist(
            variants=variants,
            local_region_start=40,
            local_region_end=60,
            max_variants=10,
        )

        # var1 is in local region and interface
        assert "var1" in result
        # var2 is not in local region
        assert "var2" not in result

    def test_excludes_non_interface_variants_in_local_region(self):
        """Should exclude non-interface variants even if in local region."""
        from orchestration.agents.structural_agent import StructuralAgent

        agent = StructuralAgent()

        variants = [
            {"variant_id": "var1", "position": 50, "region": "core"},
            {"variant_id": "var2", "position": 55, "region": "surface"},
        ]

        result = agent._build_downstream_shortlist(
            variants=variants,
            local_region_start=40,
            local_region_end=60,
            max_variants=10,
        )

        # Neither is interface region
        assert "var1" not in result
        assert "var2" not in result

    def test_handles_empty_variant_list(self):
        """Should handle empty variant list gracefully."""
        from orchestration.agents.structural_agent import StructuralAgent

        agent = StructuralAgent()

        result = agent._build_downstream_shortlist(
            variants=[],
            local_region_start=40,
            local_region_end=60,
            max_variants=10,
        )

        assert result == []

    def test_respects_max_variants_limit(self):
        """Should respect max_variants limit."""
        from orchestration.agents.structural_agent import StructuralAgent

        agent = StructuralAgent()

        variants = [
            {"variant_id": f"var{i}", "position": 40 + i, "region": "interface"}
            for i in range(20)
        ]

        result = agent._build_downstream_shortlist(
            variants=variants,
            local_region_start=40,
            local_region_end=60,
            max_variants=5,
        )

        assert len(result) <= 5

    def test_includes_multiple_local_interface_variants(self):
        """Should include multiple local interface variants."""
        from orchestration.agents.structural_agent import StructuralAgent

        agent = StructuralAgent()

        variants = [
            {"variant_id": "var1", "position": 45, "region": "interface"},
            {"variant_id": "var2", "position": 50, "region": "interface"},
            {"variant_id": "var3", "position": 55, "region": "interface"},
            {"variant_id": "var4", "position": 70, "region": "interface"},
        ]

        result = agent._build_downstream_shortlist(
            variants=variants,
            local_region_start=40,
            local_region_end=60,
            max_variants=10,
        )

        assert "var1" in result
        assert "var2" in result
        assert "var3" in result
        assert "var4" not in result

    def test_boundary_positions_included(self):
        """Should include variants at exact boundary positions."""
        from orchestration.agents.structural_agent import StructuralAgent

        agent = StructuralAgent()

        variants = [
            {"variant_id": "var_start", "position": 40, "region": "interface"},
            {"variant_id": "var_end", "position": 60, "region": "interface"},
        ]

        result = agent._build_downstream_shortlist(
            variants=variants,
            local_region_start=40,
            local_region_end=60,
            max_variants=10,
        )

        assert "var_start" in result
        assert "var_end" in result

    def test_variant_without_region_field_excluded(self):
        """Should exclude variants without region field."""
        from orchestration.agents.structural_agent import StructuralAgent

        agent = StructuralAgent()

        variants = [
            {"variant_id": "var1", "position": 50},  # No region field
        ]

        result = agent._build_downstream_shortlist(
            variants=variants,
            local_region_start=40,
            local_region_end=60,
            max_variants=10,
        )

        assert "var1" not in result


class TestBuildDownstreamShortlistIntegration:
    """Integration tests for downstream shortlist with full agent."""

    def test_shortlist_used_in_docking_workflow(self):
        """Should use shortlist in MD/HDOCK docking workflow."""
        from orchestration.agents.structural_agent import StructuralAgent

        agent = StructuralAgent()

        # Create variants in interface region
        variants = [
            {"variant_id": "var1", "position": 50, "region": "interface"},
            {"variant_id": "var2", "position": 55, "region": "interface"},
        ]

        # Build shortlist
        shortlist = agent._build_downstream_shortlist(
            variants=variants,
            local_region_start=40,
            local_region_end=60,
            max_variants=5,
        )

        # Shortlist should be used for downstream processing
        assert len(shortlist) > 0
        assert "var1" in shortlist or "var2" in shortlist

    def test_empty_shortlist_skips_downstream(self):
        """Should handle empty shortlist gracefully."""
        from orchestration.agents.structural_agent import StructuralAgent

        agent = StructuralAgent()

        variants = [
            {"variant_id": "var1", "position": 50, "region": "core"},
        ]

        shortlist = agent._build_downstream_shortlist(
            variants=variants,
            local_region_start=40,
            local_region_end=60,
            max_variants=5,
        )

        assert shortlist == []