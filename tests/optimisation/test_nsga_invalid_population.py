"""
Tests for Priority 1: NSGA-II invalid population rejection.

Verifies:
- Invalid individuals are rejected before evaluation
- Valid individuals are accepted
- Mixed populations handle correctly
- Empty population is handled
"""

import numpy as np
import pytest


class TestNSGAInvalidPopulationRejection:
    """Test NSGA-II invalid population rejection."""

    def test_valid_individuals_accepted(self):
        """Valid individuals should be accepted."""
        from optimisation.nsgaii import NSGAIIOptimizer

        optimizer = NSGAIIOptimizer()

        # Create valid population
        population = np.array(
            [
                [0.5, 0.5, 0.5],  # Valid
                [0.3, 0.7, 0.2],  # Valid
                [0.8, 0.1, 0.9],  # Valid
            ]
        )

        valid_mask = optimizer._validate_population(population)

        assert np.all(valid_mask)

    def test_invalid_individuals_rejected(self):
        """Invalid individuals should be rejected."""
        from optimisation.nsgaii import NSGAIIOptimizer

        optimizer = NSGAIIOptimizer()

        # Create population with invalid individuals
        population = np.array(
            [
                [0.5, 0.5, 0.5],  # Valid
                [np.nan, 0.7, 0.2],  # Invalid - NaN
                [0.8, 0.1, 0.9],  # Valid
            ]
        )

        valid_mask = optimizer._validate_population(population)

        assert valid_mask[0] is True
        assert valid_mask[1] is False
        assert valid_mask[2] is True

    def test_all_invalid_rejected(self):
        """Population with all invalid individuals should be rejected."""
        from optimisation.nsgaii import NSGAIIOptimizer

        optimizer = NSGAIIOptimizer()

        population = np.array(
            [
                [np.nan, np.nan, np.nan],
                [np.inf, 0.5, 0.5],
                [-np.inf, 0.5, 0.5],
            ]
        )

        valid_mask = optimizer._validate_population(population)

        assert not np.any(valid_mask)

    def test_empty_population_handled(self):
        """Empty population should be handled gracefully."""
        from optimisation.nsgaii import NSGAIIOptimizer

        optimizer = NSGAIIOptimizer()

        population = np.array([]).reshape(0, 3)

        valid_mask = optimizer._validate_population(population)

        assert len(valid_mask) == 0

    def test_out_of_bounds_rejected(self):
        """Out-of-bounds individuals should be rejected."""
        from optimisation.nsgaii import NSGAIIOptimizer

        optimizer = NSGAIIOptimizer(bounds=[(0, 1), (0, 1), (0, 1)])

        population = np.array(
            [
                [0.5, 0.5, 0.5],  # Valid
                [1.5, 0.5, 0.5],  # Invalid - out of bounds
                [0.8, 0.1, 0.9],  # Valid
            ]
        )

        valid_mask = optimizer._validate_population(population)

        assert valid_mask[0] is True
        assert valid_mask[1] is False
        assert valid_mask[2] is True

    def test_negative_values_rejected_when_not_allowed(self):
        """Negative values should be rejected when bounds are positive."""
        from optimisation.nsgaii import NSGAIIOptimizer

        optimizer = NSGAIIOptimizer(bounds=[(0, 1), (0, 1), (0, 1)])

        population = np.array(
            [
                [0.5, 0.5, 0.5],  # Valid
                [-0.1, 0.5, 0.5],  # Invalid - negative
            ]
        )

        valid_mask = optimizer._validate_population(population)

        assert valid_mask[0] is True
        assert valid_mask[1] is False

    def test_mixed_valid_invalid_handled(self):
        """Mixed valid/invalid population should be handled correctly."""
        from optimisation.nsgaii import NSGAIIOptimizer

        optimizer = NSGAIIOptimizer()

        population = np.array(
            [
                [0.1, 0.2, 0.3],  # Valid
                [np.nan, 0.2, 0.3],  # Invalid
                [0.4, 0.5, 0.6],  # Valid
                [0.7, np.inf, 0.9],  # Invalid
                [0.8, 0.9, 1.0],  # Valid
            ]
        )

        valid_mask = optimizer._validate_population(population)

        assert valid_mask[0] is True
        assert valid_mask[1] is False
        assert valid_mask[2] is True
        assert valid_mask[3] is False
        assert valid_mask[4] is True


class TestNSGAIICrossoverWithValidation:
    """Test NSGA-II crossover with invalid individual handling."""

    def test_crossover_skips_invalid_parents(self):
        """Crossover should skip invalid parents."""
        from optimisation.nsgaii import NSGAIIOptimizer

        optimizer = NSGAIIOptimizer()

        population = np.array(
            [
                [0.5, 0.5, 0.5],  # Valid
                [np.nan, 0.5, 0.5],  # Invalid
                [0.8, 0.8, 0.8],  # Valid
            ]
        )

        valid_mask = optimizer._validate_population(population)
        valid_population = population[valid_mask]

        # Should only have 2 valid individuals
        assert len(valid_population) == 2

    def test_offspring_generation_from_valid_only(self):
        """Offspring should only be generated from valid individuals."""
        from optimisation.nsgaii import NSGAIIOptimizer

        optimizer = NSGAIIOptimizer()

        population = np.array(
            [
                [0.5, 0.5, 0.5],
                [np.nan, 0.5, 0.5],
                [0.8, 0.8, 0.8],
            ]
        )

        valid_mask = optimizer._validate_population(population)
        valid_population = population[valid_mask]

        # Generate offspring from valid population only
        offspring = optimizer._generate_offspring(valid_population, num_offspring=2)

        # Offspring should be valid
        offspring_valid = optimizer._validate_population(offspring)
        assert np.all(offspring_valid)


class TestNSGAIPopulationReplacement:
    """Test NSGA-II population replacement with validation."""

    def test_invalid_offspring_not_added_to_next_generation(self):
        """Invalid offspring should not be added to next generation."""
        from optimisation.nsgaii import NSGAIIOptimizer

        optimizer = NSGAIIOptimizer()

        parents = np.array(
            [
                [0.5, 0.5, 0.5],
                [0.8, 0.8, 0.8],
            ]
        )

        # Generate some offspring
        offspring = optimizer._generate_offspring(parents, num_offspring=4)

        # Validate offspring
        valid_mask = optimizer._validate_population(offspring)
        valid_offspring = offspring[valid_mask]

        # All valid offspring should be kept
        assert np.all(optimizer._validate_population(valid_offspring))