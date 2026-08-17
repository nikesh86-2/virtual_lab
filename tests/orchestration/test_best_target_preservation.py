"""
test_best_target_preservation.py
================================

Regression tests for target state separation: latest batch vs best validated.

Tests the exact scenario from run 7110842 where a later partial batch should
not demote an earlier accepted target.
"""

from __future__ import annotations

import pytest

from orchestration.utils.target_state_utils import (
    should_promote_best_target,
    is_validated_target_record,
    target_quality_key,
)


def make_target_record(
    target_pdb: str = "8K75",
    status: str = "accepted_target",
    iteration: int = 0,
    clean: int = 3,
    clashes: int = 0,
    valid: int = 3,
    required: int = 2,
    min_clean: int = 2,
    quality: float = 7.8,
    hdock_score: float | None = -8.5,
) -> dict:
    """
    Construct a test target evaluation record with sensible defaults.
    """
    return {
        "target_pdb": target_pdb,
        "status": status,
        "status_reason": f"{status}_reason",
        "iteration": iteration,
        "batch_id": f"batch_{target_pdb}_{iteration}",
        "evaluated_at": "2026-08-14T12:00:00Z",
        "docking_valid_count": valid,
        "docking_required_count": required,
        "clean_interface_count": clean,
        "clean_interface_required_count": min_clean,
        "steric_clash_count": clashes,
        "best_hdock_relative_score": hdock_score,
        "hdock_score_spread": 1.2 if hdock_score else None,
        "interface_quality_score": 0.85,
        "aggregate_quality_score": quality,
        "accepted": status == "accepted_target",
        "interface_validated": clean >= min_clean,
        "exploratory": False,
        "binding_result_ids": [f"result_{i}" for i in range(clean)],
        "clean_sequence_ids": [f"seq_clean_{i}" for i in range(clean)],
        "clash_sequence_ids": [f"seq_clash_{i}" for i in range(clashes)],
        "docking_summary_json": None,
        "docking_summary_csv": None,
        "docking_summary_md": None,
        "metadata": {
            "binding_units": "hdock_relative_score",
            "backend": "hdock",
        },
    }


class TestTargetRecordValidation:
    """Test the validation predicate for accepted targets."""

    def test_valid_accepted_target_passes_validation(self):
        """An accepted target with metrics > required passes validation."""
        record = make_target_record(
            status="accepted_target",
            clean=3,
            min_clean=2,
            valid=3,
            required=2,
        )

        assert is_validated_target_record(record) is True

    def test_partial_target_fails_validation(self):
        """A partial_success target does not pass validation."""
        record = make_target_record(
            status="partial_success_target",
            clean=1,
            min_clean=2,
        )

        assert is_validated_target_record(record) is False

    def test_accepted_status_with_insufficient_clean_fails_validation(self):
        """An accepted_target with insufficient clean interfaces fails."""
        record = make_target_record(
            status="accepted_target",
            clean=1,
            min_clean=2,
        )

        assert is_validated_target_record(record) is False

    def test_none_record_fails_validation(self):
        """None record fails validation."""
        assert is_validated_target_record(None) is False

    def test_empty_dict_fails_validation(self):
        """Empty dict fails validation."""
        assert is_validated_target_record({}) is False


class TestQualityKeyOrdering:
    """Test the deterministic quality comparison function."""

    def test_accepted_ranks_higher_than_partial(self):
        """Accepted target ranks higher than partial regardless of metrics."""
        accepted = make_target_record(status="accepted_target", clean=1, clashes=10)
        partial = make_target_record(status="partial_success_target", clean=10, clashes=0)

        assert target_quality_key(accepted) > target_quality_key(partial)

    def test_more_clean_interfaces_rank_higher(self):
        """Among accepted targets, more clean interfaces ranks higher."""
        three_clean = make_target_record(status="accepted_target", clean=3, clashes=0)
        two_clean = make_target_record(status="accepted_target", clean=2, clashes=0)

        assert target_quality_key(three_clean) > target_quality_key(two_clean)

    def test_fewer_clashes_rank_higher(self):
        """Among accepted targets with same clean count, fewer clashes ranks higher."""
        no_clash = make_target_record(
            status="accepted_target",
            clean=3,
            clashes=0,
        )
        one_clash = make_target_record(
            status="accepted_target",
            clean=3,
            clashes=1,
        )

        assert target_quality_key(no_clash) > target_quality_key(one_clash)

    def test_better_quality_score_ranks_higher(self):
        """Among otherwise equal targets, higher quality_score ranks higher."""
        high_quality = make_target_record(
            status="accepted_target",
            clean=3,
            clashes=0,
            quality=10.0,
        )
        low_quality = make_target_record(
            status="accepted_target",
            clean=3,
            clashes=0,
            quality=5.0,
        )

        assert target_quality_key(high_quality) > target_quality_key(low_quality)

    def test_earlier_iteration_wins_exact_tie(self):
        """In exact tie, earlier iteration (lower iteration number) wins."""
        early = make_target_record(
            status="accepted_target",
            clean=3,
            clashes=0,
            quality=7.8,
            iteration=0,
        )
        late = make_target_record(
            status="accepted_target",
            clean=3,
            clashes=0,
            quality=7.8,
            iteration=2,
        )

        # Same quality key up to iteration tie-breaker
        # Early should be > late because -0 > -2
        assert target_quality_key(early) > target_quality_key(late)


class TestTargetPromotion:
    """Test the promotion logic."""

    def test_later_partial_does_not_demote_accepted(self):
        """
        Regression test for run 7110842.

        A later partial batch should NOT replace an earlier accepted batch
        as the best validated target.
        """
        accepted = make_target_record(
            target_pdb="8K75",
            status="accepted_target",
            iteration=0,
            clean=3,
            clashes=0,
            valid=3,
            required=2,
            min_clean=2,
            quality=7.8,
        )

        partial = make_target_record(
            target_pdb="8K75",
            status="partial_success_target",
            iteration=2,
            clean=1,
            clashes=2,
            valid=3,
            required=2,
            min_clean=2,
            quality=1.1,
        )

        # Partial should NOT promote over accepted
        assert should_promote_best_target(partial, accepted) is False

        # Accepted should be retained
        assert is_validated_target_record(accepted) is True
        assert is_validated_target_record(partial) is False

    def test_stronger_accepted_can_promote_weaker_accepted(self):
        """A stronger accepted record can replace a weaker accepted record."""
        weak_accepted = make_target_record(
            status="accepted_target",
            iteration=0,
            clean=2,
            clashes=1,
            quality=5.0,
        )

        strong_accepted = make_target_record(
            status="accepted_target",
            iteration=1,
            clean=3,
            clashes=0,
            quality=8.0,
        )

        # Strong should promote over weak
        assert should_promote_best_target(strong_accepted, weak_accepted) is True

    def test_any_validated_promotes_over_unvalidated(self):
        """Any validated record promotes over an unvalidated incumbent."""
        unvalidated = make_target_record(
            status="partial_success_target",
            clean=1,
            min_clean=3,  # Insufficient
        )

        validated = make_target_record(
            status="accepted_target",
            clean=2,
            min_clean=2,
        )

        assert should_promote_best_target(validated, unvalidated) is True

    def test_promotion_against_none_incumbent(self):
        """A validated record promotes when there is no incumbent."""
        validated = make_target_record(status="accepted_target")

        assert should_promote_best_target(validated, None) is True

    def test_partial_cannot_promote_at_all(self):
        """A partial target can never be promoted, even when incumbent is None."""
        partial = make_target_record(status="partial_success_target")

        assert should_promote_best_target(partial, None) is False
        assert should_promote_best_target(partial, {}) is False

    def test_equal_quality_incumbent_retained(self):
        """When quality keys are exactly equal, incumbent is retained."""
        incumbent = make_target_record(
            status="accepted_target",
            iteration=0,
            clean=3,
            clashes=0,
            quality=7.8,
        )

        candidate = make_target_record(
            status="accepted_target",
            iteration=1,  # Later iteration, but same quality otherwise
            clean=3,
            clashes=0,
            quality=7.8,
        )

        # Later identical candidate should NOT promote (earlier wins tie)
        assert should_promote_best_target(candidate, incumbent) is False


class TestTargetRecordFields:
    """Test that records maintain required fields."""

    def test_record_has_all_required_fields(self):
        """A proper record has all required fields for quality comparison."""
        record = make_target_record()

        required_fields = [
            "target_pdb",
            "status",
            "iteration",
            "clean_interface_count",
            "steric_clash_count",
            "aggregate_quality_score",
            "best_hdock_relative_score",
        ]

        for field in required_fields:
            assert field in record, f"Missing field: {field}"
