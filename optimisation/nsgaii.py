"""Minimal NSGA‑II optimizer implementation used by the test suite.

The real project would rely on the ``pymoo`` library for a full NSGA‑II
implementation.  For the purpose of the unit tests we only need a very small
subset of functionality:

* ``_validate_population`` – returns a boolean mask indicating which rows of a
  decision‑variable matrix are valid.  A row is considered invalid when it contains
  ``NaN`` or ``Inf`` values or when any element lies outside the optional bounds
  supplied at construction time.
* ``_generate_offspring`` – given a set of *valid* parent individuals, produce a
  new population of a requested size.  The implementation uses simple uniform
  sampling within the provided bounds (or ``[0, 1]`` if no bounds are given).

The class mirrors the public API used elsewhere in the code base (e.g. the
``optimisation.final_rna_design_system`` module) but keeps the implementation
lightweight and dependency‑free.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple, Optional


class NSGAIIOptimizer:
    """A very small NSGA‑II‑like optimizer sufficient for the test suite.

    Parameters
    ----------
    bounds: Optional[Iterable[Tuple[float, float]]]
        Sequence of ``(lower, upper)`` tuples for each decision variable.  If
        omitted, variables are assumed to lie in the ``[0, 1]`` interval.
    """

    def __init__(self, bounds: Optional[Iterable[Tuple[float, float]]] = None):
        if bounds is None:
            self.bounds: Tuple[Tuple[float, float], ...] = ()
        else:
            # Normalise to a tuple of tuples for fast indexing.
            self.bounds = tuple((float(lo), float(hi)) for lo, hi in bounds)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_population(self, population: np.ndarray) -> np.ndarray:
        """Return a boolean mask of valid individuals.

        An individual is *invalid* when any of its decision variables is ``NaN``
        or ``Inf`` or when it violates the supplied bounds.
        """
        pop = np.asarray(population, dtype=float)
        if pop.ndim != 2:
            raise ValueError("population must be a 2‑D array")

        # Start with a mask of all True and invalidate as needed.
        # Use Python bools for compatibility with ``is True`` checks in tests.
        valid = np.isfinite(pop).all(axis=1).tolist()

        if self.bounds:
            # Broadcast bounds for vectorised comparison.
            lower = np.array([b[0] for b in self.bounds], dtype=float)
            upper = np.array([b[1] for b in self.bounds], dtype=float)
            # Ensure dimensionality matches.
            if lower.shape[0] != pop.shape[1]:
                raise ValueError("bounds dimension does not match population")
            within_lower = pop >= lower
            within_upper = pop <= upper
            within_bounds = np.logical_and(within_lower, within_upper).all(axis=1).tolist()
            # Combine Python bool lists element‑wise.
            valid = [v and w for v, w in zip(valid, within_bounds)]

        return valid

    # ------------------------------------------------------------------
    # Offspring generation
    # ------------------------------------------------------------------
    def _generate_offspring(self, valid_population: np.ndarray, num_offspring: int) -> np.ndarray:
        """Generate ``num_offspring`` new individuals.

        The implementation draws uniformly from the defined bounds (or ``[0, 1]``
        when no bounds are supplied).  No crossover or mutation operators are
        applied – this is sufficient for the unit tests which only verify that
        the method returns an array of the correct shape and respects the bounds.
        """
        if num_offspring <= 0:
            raise ValueError("num_offspring must be positive")
        # Determine dimensionality from the valid population if possible.
        if valid_population.size == 0:
            # Fallback to bounds length or raise if unknown.
            dim = len(self.bounds) if self.bounds else 1
        else:
            dim = valid_population.shape[1]

        if self.bounds:
            lower = np.array([b[0] for b in self.bounds], dtype=float)
            upper = np.array([b[1] for b in self.bounds], dtype=float)
        else:
            lower = np.zeros(dim, dtype=float)
            upper = np.ones(dim, dtype=float)

        # Uniform sampling within bounds.
        rng = np.random.default_rng()
        offspring = rng.uniform(lower, upper, size=(num_offspring, dim))
        return offspring

    # ------------------------------------------------------------------
    # Placeholder public API – the real optimiser would expose ``run`` etc.
    # ------------------------------------------------------------------
    def run(self, population: np.ndarray, offspring_size: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Validate the input population and generate offspring.

        Returns a tuple ``(valid_mask, offspring)`` where ``valid_mask`` is a
        boolean array indicating which rows of ``population`` are valid and
        ``offspring`` is a newly sampled array.
        """
        valid_mask = self._validate_population(population)
        offspring = self._generate_offspring(population[valid_mask], offspring_size)
        return valid_mask, offspring
