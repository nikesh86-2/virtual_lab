"""
test_motif_coordinate_mapping.py
================================

Regression tests for motif alignment-to-sequence coordinate mapping.

Tests the exact issue from run 7110842 where motif length was being
shortened due to incorrect half-open interval mapping.
"""

from __future__ import annotations

import pytest

from orchestration.utils.sequence_utils import (
    alignment_boundary_to_sequence_offset,
    map_alignment_interval_to_sequence,
    validate_motif_mapping,
)


class TestBoundaryToSequenceOffset:
    """Test the fundamental boundary conversion function."""

    def test_ungapped_offset_at_boundary(self):
        """Without gaps, sequence offset equals alignment position."""
        aligned = "AACUGAGUCC"

        assert alignment_boundary_to_sequence_offset(aligned, 0) == 0
        assert alignment_boundary_to_sequence_offset(aligned, 5) == 5
        assert alignment_boundary_to_sequence_offset(aligned, 10) == 10

    def test_gapped_offset_counts_only_non_gaps(self):
        """Gaps are not counted when computing sequence offset."""
        aligned = "AA-CUG-AGUCC"

        # Positions: 0  1  2  3  4  5  6  7  8  9  10 11
        # Aligned:   A  A  -  C  U  G  -  A  G  U  C  C
        # Sequence:  0  1     2  3  4     5  6  7  8  9

        assert alignment_boundary_to_sequence_offset(aligned, 0) == 0
        assert alignment_boundary_to_sequence_offset(aligned, 2) == 2  # Before gap
        assert alignment_boundary_to_sequence_offset(aligned, 3) == 2  # After gap, same count
        assert alignment_boundary_to_sequence_offset(aligned, 7) == 5  # After second gap
        assert alignment_boundary_to_sequence_offset(aligned, 12) == 10

    def test_offset_at_zero_boundary(self):
        """Offset at position 0 is always 0."""
        assert alignment_boundary_to_sequence_offset("ACGU", 0) == 0
        assert alignment_boundary_to_sequence_offset("---ACGU", 0) == 0
        assert alignment_boundary_to_sequence_offset("-", 0) == 0

    def test_offset_out_of_range_returns_none(self):
        """Out-of-range boundaries return None."""
        aligned = "ACGU"

        assert alignment_boundary_to_sequence_offset(aligned, -1) is None
        assert alignment_boundary_to_sequence_offset(aligned, 5) is None


class TestHalfOpenIntervalMapping:
    """Test mapping of half-open intervals from alignment to sequence."""

    def test_ungapped_interval_preserves_length(self):
        """Without gaps, interval length is preserved."""
        aligned = "AACUGAGUCC"

        result = map_alignment_interval_to_sequence(aligned, 4, 7)

        assert result == (4, 7)
        assert result[1] - result[0] == 3  # Length preserved

    def test_gapped_interval_accounts_for_gaps(self):
        """Gaps before interval shorten the sequence start."""
        aligned = "AA-CUGA-GUCC"

        # Alignment [3,7) covers: C U G -
        # Sequence coverage: 2 3 4 (skip the gap)
        result = map_alignment_interval_to_sequence(aligned, 3, 7)

        assert result == (2, 5)
        assert result[1] - result[0] == 3

    def test_gag_motif_mapping_does_not_shorten(self):
        """
        Regression test for run 7110842 GAG motif.

        Mapped GAG must preserve length 3, not become length 2.
        """
        aligned = "AAAAAGAGCCCC"

        result = map_alignment_interval_to_sequence(aligned, 5, 8)

        assert result == (5, 8), f"Expected (5, 8), got {result}"
        assert result[1] - result[0] == 3, "GAG length must be 3"

    def test_invalid_boundaries_return_none(self):
        """Invalid interval boundaries return None."""
        aligned = "ACGU"

        # Start >= end
        assert map_alignment_interval_to_sequence(aligned, 2, 2) is None

        # Out of range
        assert map_alignment_interval_to_sequence(aligned, 0, 10) is None

        # Negative start
        assert map_alignment_interval_to_sequence(aligned, -1, 2) is None

    def test_non_integer_boundaries_return_none(self):
        """Non-integer boundaries return None."""
        aligned = "ACGU"

        assert map_alignment_interval_to_sequence(aligned, "0", 2) is None
        assert map_alignment_interval_to_sequence(aligned, 0, 2.5) is None
        assert map_alignment_interval_to_sequence(aligned, None, 2) is None

    def test_all_gap_interval_returns_none(self):
        """Interval covering only gaps returns None (zero-length sequence)."""
        aligned = "AA---ACGU"

        # [2, 5) covers only gaps
        result = map_alignment_interval_to_sequence(aligned, 2, 5)

        assert result is None


class TestMotifMappingValidation:
    """Test validation of mapped motif intervals."""

    def test_valid_mapped_motif_passes_validation(self):
        """A correctly mapped motif passes validation."""
        is_valid, reason = validate_motif_mapping(
            matched_motif="GAG",
            sequence_start=5,
            sequence_end=8,
            reference_sequence="AAAAAGAGCCCC",
        )

        assert is_valid is True
        assert reason is None

    def test_missing_sequence_start_fails_validation(self):
        """Missing sequence start fails validation."""
        is_valid, reason = validate_motif_mapping(
            matched_motif="GAG",
            sequence_start=None,
            sequence_end=8,
            reference_sequence="AAAAAGAGCCCC",
        )

        assert is_valid is False
        assert "missing_sequence_interval" in reason

    def test_missing_sequence_end_fails_validation(self):
        """Missing sequence end fails validation."""
        is_valid, reason = validate_motif_mapping(
            matched_motif="GAG",
            sequence_start=5,
            sequence_end=None,
            reference_sequence="AAAAAGAGCCCC",
        )

        assert is_valid is False
        assert "missing_sequence_interval" in reason

    def test_out_of_bounds_interval_fails_validation(self):
        """Out-of-bounds interval fails validation."""
        is_valid, reason = validate_motif_mapping(
            matched_motif="GAG",
            sequence_start=5,
            sequence_end=100,  # Beyond reference length
            reference_sequence="AAAAAGAGCCCC",
        )

        assert is_valid is False
        assert "interval_out_of_bounds" in reason

    def test_mapped_length_mismatch_fails_validation(self):
        """
        Mapped length not matching motif length fails validation.

        This is the core regression test for the shortening bug.
        """
        is_valid, reason = validate_motif_mapping(
            matched_motif="GAG",  # Expected length 3
            sequence_start=5,
            sequence_end=7,  # Actual length 2
            reference_sequence="AAAAAGAGCCCC",
        )

        assert is_valid is False
        assert "mapped_length_mismatch" in reason
        assert "expected=3" in reason
        assert "observed=2" in reason

    def test_mapped_sequence_mismatch_fails_validation(self):
        """Mapped sequence not matching motif fails validation."""
        is_valid, reason = validate_motif_mapping(
            matched_motif="GAG",
            sequence_start=6,  # Points to "AGC" not "GAG"
            sequence_end=9,
            reference_sequence="AAAAAGAGCCCC",
        )

        assert is_valid is False
        assert "mapped_sequence_mismatch" in reason

    def test_rgag_motif_with_concrete_sequence(self):
        """
        Motif with ambiguity code (R) validates against concrete sequence.

        R = A or G, but we validate against the concrete matched sequence.
        """
        is_valid, reason = validate_motif_mapping(
            matched_motif="AGAG",  # Concrete sequence (R matched as A)
            sequence_start=4,
            sequence_end=8,
            reference_sequence="CCCCAGAGCCCC",
        )

        assert is_valid is True
        assert reason is None

    def test_raag_motif_mapping(self):
        """Another motif variant from run 7110842."""
        is_valid, reason = validate_motif_mapping(
            matched_motif="AAAG",  # Concrete (R matched as A)
            sequence_start=2,
            sequence_end=6,
            reference_sequence="CCAAAGCCCCCC",
        )

        assert is_valid is True
        assert reason is None


class TestRegressionScenariosFrom7110842:
    """Direct regression tests from VLAB run 7110842 defects."""

    def test_gag_not_shortened_to_two_bases(self):
        """
        Main regression: GAG must not be shortened to length 2.

        Original defect showed:
        motif=GAG  msa=[5,8) sequence=[5,7)  <- WRONG: length 2
        Should be:
        motif=GAG  msa=[5,8) sequence=[5,8)  <- CORRECT: length 3
        """
        aligned = "AAAAAGAGCCCC"
        sequence_start, sequence_end = map_alignment_interval_to_sequence(
            aligned, 5, 8
        )

        expected_motif = "GAG"
        actual_motif = aligned.replace("-", "")[sequence_start:sequence_end]

        assert len(actual_motif) == len(expected_motif), (
            f"Motif length mismatch: expected {len(expected_motif)}, "
            f"got {len(actual_motif)}"
        )
        assert actual_motif == expected_motif

    def test_multiple_motifs_preserve_length(self):
        """All three motifs from the run should preserve their lengths."""
        cases = [
            ("AAAAAGAGCCCC", 5, 8, "GAG"),
            ("AAAAARGAGCCCC", 4, 8, "RGAG"),
            ("AAAAARAAGCCCCCC", 2, 6, "RAAG"),
        ]

        for aligned, start, end, expected_concrete in cases:
            seq_start, seq_end = map_alignment_interval_to_sequence(
                aligned, start, end
            )

            if seq_start is not None and seq_end is not None:
                mapped = aligned.replace("-", "")[seq_start:seq_end]
                assert (
                    len(mapped) == len(expected_concrete)
                ), f"Motif {expected_concrete} shortened from length {len(expected_concrete)} to {len(mapped)}"
