"""
Tests for Priority 1: Pareto analysis validation.

Verifies:
- Pareto front identification is correct
- Invalid individuals don't affect Pareto front
- Empty and single-individual populations handled
"""

import numpy as np
import pytest


class TestParetoAnalysis:
    """Test Pareto front analysis."""

    def test_pareto_front_identified_correctly(self):
        """Should correctly identify Pareto front."""
        from utils.pareto_analysis import identify_pareto_front

        objectives = np.array(
            [
                [1.0, 5.0],  # Dominated by 2
                [2.0, 4.0],  # Dominated by 2
                [3.0, 3.0],  # Pareto optimal
                [4.0, 2.0],  # Pareto optimal
                [5.0, 1.0],  # Pareto optimal
            ]
        )

        pareto_mask = identify_pareto_front(objectives, minimize=True)

        assert pareto_mask[0] is False
        assert pareto_mask[1] is False
        assert pareto_mask[2] is True
        assert pareto_mask[3] is True
        assert pareto_mask[4] is True

    def test_pareto_front_maximize(self):
        """Should correctly identify Pareto front for maximization."""
        from utils.pareto_analysis import identify_pareto_front

        objectives = np.array(
            [
                [1.0, 5.0],  # Pareto optimal
                [2.0, 4.0],  # Pareto optimal
                [3.0, 3.0],  # Dominated
                [4.0, 2.0],  # Dominated
                [5.0, 1.0],  # Dominated
            ]
        )

        pareto_mask = identify_pareto_front(objectives, minimize=False)

        assert pareto_mask[0] is True
        assert pareto_mask[1] is True
        assert pareto_mask[2] is False
        assert pareto_mask[3] is False
        assert pareto_mask[4] is False

    def test_empty_population(self):
        """Should handle empty population."""
        from utils.pareto_analysis import identify_pareto_front

        objectives = np.array([]).reshape(0, 2)

        pareto_mask = identify_pareto_front(objectives, minimize=True)

        assert len(pareto_mask) == 0

    def test_single_individual_is_pareto(self):
        """Single individual should always be Pareto optimal."""
        from utils.pareto_analysis import identify_pareto_front

        objectives = np.array([[1.0, 2.0]])

        pareto_mask = identify_pareto_front(objectives, minimize=True)

        assert pareto_mask[0] is True

    def test_invalid_individuals_excluded(self):
        """Invalid individuals should be excluded from Pareto front."""
        from utils.pareto_analysis import identify_pareto_front

        objectives = np.array(
            [
                [1.0, 5.0],  # Valid, Pareto
                [np.nan, 4.0],  # Invalid
                [3.0, 3.0],  # Valid, Pareto
                [4.0, 2.0],  # Valid, Pareto
            ]
        )

        pareto_mask = identify_pareto_front(objectives, minimize=True)

        # Only valid Pareto individuals should be marked
        assert pareto_mask[0] is True
        assert pareto_mask[1] is False  # Invalid
        assert pareto_mask[2] is True
        assert pareto_mask[3] is True

    def test_pareto_front_size_limited(self):
        """Pareto front size should be limited."""
        from utils.pareto_analysis import identify_pareto_front

        # Create many non-dominated individuals
        objectives = np.array([[i, 10 - i] for i in range(20)])

        pareto_mask = identify_pareto_front(objectives, minimize=True, max_front_size=5)

        assert np.sum(pareto_mask) <= 5

    def test_tied_objectives(self):
        """Should handle tied objective values."""
        from utils.pareto_analysis import identify_pareto_front

        objectives = np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],  # Tied
                [2.0, 2.0],
            ]
        )

        pareto_mask = identify_pareto_front(objectives, minimize=True)

        # All should be Pareto (no one dominates due to ties)
        assert np.all(pareto_mask)


class TestParetoRanking:
    """Test Pareto ranking for NSGA-II."""

    def test_pareto_ranking_assigns_correct_ranks(self):
        """Should assign correct ranks to individuals."""
        from utils.pareto_analysis import pareto_rank

        objectives = np.array(
            [
                [1.0, 5.0],  # Rank 0
                [2.0, 4.0],  # Rank 0
                [3.0, 3.0],  # Rank 1
                [4.0, 2.0],  # Rank 2
                [5.0, 1.0],  # Rank 3
            ]
        )

        ranks = pareto_rank(objectives, minimize=True)

        assert ranks[0] == 0
        assert ranks[1] == 0
        assert ranks[2] == 1
        assert ranks[3] == 2
        assert ranks[4] == 3

    def test_crowding_distance(self):
        """Should calculate crowding distance correctly."""
        from utils.pareto_analysis import crowding_distance

        front = np.array(
            [
                [1.0, 5.0],
                [2.0, 4.0],
                [3.0, 3.0],
                [4.0, 2.0],
                [5.0, 1.0],
            ]
        )

        distances = crowding_distance(front, minimize=True)

        # Boundary individuals should have infinite distance
        assert np.isinf(distances[0])
        assert np.isinf(distances[-1])
        # Middle individuals should have finite distances
        assert not np.isinf(distances[1])
        assert not np.isinf(distances[2])
        assert not np.isinf(distances[3])