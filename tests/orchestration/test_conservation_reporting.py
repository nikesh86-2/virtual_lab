"""
Tests for Priority 4: Conservation reporting separation.

Verifies:
- Structural metrics reported separately from sequence metrics
- Structural metrics include: secondary structure, solvent accessibility, B-factors
- Sequence metrics include: Shannon entropy, conservation scores, gap frequency
- Report contains both metric types
"""

import pytest


class TestConservationReportingSeparation:
    """Test conservation reporting separation."""

    def test_structural_metrics_separate_from_sequence(self):
        """Structural metrics should be reported separately from sequence metrics."""
        from orchestration.conservation_reporter import ConservationReporter

        reporter = ConservationReporter()

        report = reporter.generate_report(
            sequence_data={"positions": list(range(100))},
            structural_data={"helix_positions": [10, 20, 30]},
        )

        # Should have separate sections
        assert "structural_metrics" in report
        assert "sequence_metrics" in report

    def test_structural_metrics_include_secondary_structure(self):
        """Structural metrics should include secondary structure info."""
        from orchestration.conservation_reporter import ConservationReporter

        reporter = ConservationReporter()

        structural_data = {
            "secondary_structure": "HHHEEE",
            "helix_positions": [0, 1, 2],
            "sheet_positions": [3, 4, 5],
            "coil_positions": [6, 7, 8],
        }

        report = reporter.generate_report(
            sequence_data={},
            structural_data=structural_data,
        )

        structural = report["structural_metrics"]
        assert "secondary_structure" in structural
        assert "helix_positions" in structural
        assert "sheet_positions" in structural

    def test_structural_metrics_include_solvent_accessibility(self):
        """Structural metrics should include solvent accessibility."""
        from orchestration.conservation_reporter import ConservationReporter

        reporter = ConservationReporter()

        structural_data = {
            "solvent_accessibility": [10.5, 25.3, 5.2, 100.0, 45.6],
            "exposed_positions": [1, 3, 4],
            "buried_positions": [0, 2],
        }

        report = reporter.generate_report(
            sequence_data={},
            structural_data=structural_data,
        )

        structural = report["structural_metrics"]
        assert "solvent_accessibility" in structural
        assert "exposed_positions" in structural
        assert "buried_positions" in structural

    def test_structural_metrics_include_bfactors(self):
        """Structural metrics should include B-factors."""
        from orchestration.conservation_reporter import ConservationReporter

        reporter = ConservationReporter()

        structural_data = {
            "bfactors": [25.0, 30.5, 45.2, 15.0, 20.8],
            "flexible_positions": [2],
            "rigid_positions": [0, 1, 3, 4],
        }

        report = reporter.generate_report(
            sequence_data={},
            structural_data=structural_data,
        )

        structural = report["structural_metrics"]
        assert "bfactors" in structural
        assert "flexible_positions" in structural
        assert "rigid_positions" in structural

    def test_sequence_metrics_include_shannon_entropy(self):
        """Sequence metrics should include Shannon entropy."""
        from orchestration.conservation_reporter import ConservationReporter

        reporter = ConservationReporter()

        sequence_data = {
            "shannon_entropy": [0.5, 0.2, 1.0, 0.8, 0.1],
            "positions": list(range(5)),
        }

        report = reporter.generate_report(
            sequence_data=sequence_data,
            structural_data={},
        )

        sequence = report["sequence_metrics"]
        assert "shannon_entropy" in sequence

    def test_sequence_metrics_include_conservation_scores(self):
        """Sequence metrics should include conservation scores."""
        from orchestration.conservation_reporter import ConservationReporter

        reporter = ConservationReporter()

        sequence_data = {
            "conservation_scores": [0.9, 0.7, 0.3, 0.8, 0.95],
            "highly_conserved": [0, 1, 3, 4],
            "variable": [2],
        }

        report = reporter.generate_report(
            sequence_data=sequence_data,
            structural_data={},
        )

        sequence = report["sequence_metrics"]
        assert "conservation_scores" in sequence
        assert "highly_conserved" in sequence
        assert "variable" in sequence

    def test_sequence_metrics_include_gap_frequency(self):
        """Sequence metrics should include gap frequency."""
        from orchestration.conservation_reporter import ConservationReporter

        reporter = ConservationReporter()

        sequence_data = {
            "gap_frequency": [0.0, 0.1, 0.5, 0.0, 0.2],
            "gapped_positions": [1, 2, 4],
        }

        report = reporter.generate_report(
            sequence_data=sequence_data,
            structural_data={},
        )

        sequence = report["sequence_metrics"]
        assert "gap_frequency" in sequence
        assert "gapped_positions" in sequence

    def test_report_contains_both_metric_types(self):
        """Report should contain both structural and sequence metrics."""
        from orchestration.conservation_reporter import ConservationReporter

        reporter = ConservationReporter()

        report = reporter.generate_report(
            sequence_data={
                "shannon_entropy": [0.5],
                "conservation_scores": [0.8],
            },
            structural_data={
                "secondary_structure": "H",
                "solvent_accessibility": [50.0],
            },
        )

        assert "structural_metrics" in report
        assert "sequence_metrics" in report
        assert len(report["structural_metrics"]) > 0
        assert len(report["sequence_metrics"]) > 0


class TestConservationReportingFormat:
    """Test conservation report format."""

    def test_report_is_dict(self):
        """Report should be a dictionary."""
        from orchestration.conservation_reporter import ConservationReporter

        reporter = ConservationReporter()

        report = reporter.generate_report(
            sequence_data={},
            structural_data={},
        )

        assert isinstance(report, dict)

    def test_report_has_timestamp(self):
        """Report should have a timestamp."""
        from orchestration.conservation_reporter import ConservationReporter

        reporter = ConservationReporter()

        report = reporter.generate_report(
            sequence_data={},
            structural_data={},
        )

        assert "timestamp" in report

    def test_report_handles_empty_data(self):
        """Report should handle empty data gracefully."""
        from orchestration.conservation_reporter import ConservationReporter

        reporter = ConservationReporter()

        report = reporter.generate_report(
            sequence_data={},
            structural_data={},
        )

        assert "structural_metrics" in report
        assert "sequence_metrics" in report