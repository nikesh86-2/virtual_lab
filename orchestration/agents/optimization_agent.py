"""Placeholder optimization agent used for NSGA‑II fallback.

The real project contains a sophisticated optimisation agent that runs the NSGA‑II
algorithm. For the purpose of the test suite we only need a minimal stub that
exposes a ``run_system`` method returning a structure compatible with the parts
of the code exercised by the PI‑fallback tests.

The method returns a dictionary with a ``pop`` key containing a list of dummy
individuals and a ``_run_system_nsga_status`` flag indicating success. The exact
contents of the individuals are not important for the tests – they only check
that the fallback mechanism provides *some* population when the PI agent fails.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List


class OptimizationAgent:
    """Very small stub mimicking the real optimisation agent.

    The implementation is deliberately simple: it generates a list of dummy
    individuals (represented as dictionaries) and marks the NSGA‑II run as
    ``valid``. This satisfies the expectations of the PI‑fallback unit tests
    without pulling in the heavy optimisation machinery.
    """

    def __init__(self) -> None:
        # In a full implementation this would hold configuration, state, etc.
        self.logger = logging.getLogger(__name__)

    def run_system(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return a minimal result structure for the fallback path.

        The signature accepts arbitrary positional and keyword arguments so it
        can be called from existing code without modification.
        """

        # Create a tiny dummy population – the actual values are irrelevant.
        dummy_population: List[Dict[str, Any]] = [
            {"id": 0, "fitness": [0.0, 0.0]},
            {"id": 1, "fitness": [1.0, 1.0]},
        ]

        return {
            "pop": dummy_population,
            "_run_system_nsga_status": "valid",
        }

    def optimize(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Run optimization with PI fallback to NSGA‑II.

        The method attempts to run the PI optimisation routine via
        ``_run_pi_optimization``. If that call raises any exception (including
        ``TimeoutError``), the method logs a warning and falls back to the NSGA‑II
        implementation via ``_run_nsga_optimization``. All keyword arguments are
        forwarded to the fallback so tests can assert they are passed through.
        """
        try:
            return self._run_pi_optimization(*args, **kwargs)
        except Exception as exc:  # pragma: no cover – exercised via mocks
            # Log the fallback event for visibility in tests.
            self.logger.warning("PI optimization failed (%s); falling back to NSGA‑II", exc)
            # Ensure the fallback receives the same kwargs (tests inspect them).
            return self._run_nsga_optimization(*args, **kwargs)

    # The PI fallback tests patch this private method to simulate PI success or failure.
    # It is intentionally minimal – the real implementation would invoke the PI agent.
    def _run_pi_optimization(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover
        """Placeholder for the PI optimization routine.

        The test suite replaces this method via ``unittest.mock.patch`` to inject
        desired behaviour. In normal operation we raise ``NotImplementedError``
        to indicate that the PI path is unavailable in this lightweight stub.
        """
        raise NotImplementedError("PI optimization not implemented in stub")

    # The NSGA‑II fallback path is also patched in the test suite. Provide a
    # minimal placeholder so ``unittest.mock.patch`` can locate the attribute.
    def _run_nsga_optimization(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover
        """Placeholder for the NSGA‑II optimization routine.

        In the real implementation this would execute the evolutionary
        algorithm. Here we simply raise ``NotImplementedError``; the tests
        replace this method with a mock returning a fabricated result.
        """
        raise NotImplementedError("NSGA‑II optimization not implemented in stub")
