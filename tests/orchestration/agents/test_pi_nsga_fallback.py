"""
Tests for Priority 2: PI-NSGA fallback mechanism.

Verifies:
- Falls back to NSGA-II when PI fails
- Falls back to NSGA-II when PI times out
- PI success continues normally
- Fallback is logged appropriately
"""

import pytest
from unittest.mock import MagicMock, patch


class TestPINFALLBACK:
    """Test PI-NSGA fallback mechanism."""

    def test_falls_back_to_nsga_on_pi_failure(self):
        """Should fall back to NSGA-II when PI fails."""
        from orchestration.agents.optimization_agent import OptimizationAgent

        agent = OptimizationAgent()

        # Mock PI to fail
        with patch.object(agent, "_run_pi_optimization", side_effect=Exception("PI failed")):
            with patch.object(agent, "_run_nsga_optimization") as mock_nsga:
                mock_nsga.return_value = {"solutions": [], "pareto_front": []}

                result = agent.optimize(
                    objectives=["binding_affinity", "selectivity"],
                    population_size=50,
                )

                # Should have called NSGA as fallback
                mock_nsga.assert_called_once()

    def test_falls_back_to_nsga_on_pi_timeout(self):
        """Should fall back to NSGA-II when PI times out."""
        from orchestration.agents.optimization_agent import OptimizationAgent

        agent = OptimizationAgent()

        # Mock PI to timeout
        with patch.object(
            agent,
            "_run_pi_optimization",
            side_effect=TimeoutError("PI timed out"),
        ):
            with patch.object(agent, "_run_nsga_optimization") as mock_nsga:
                mock_nsga.return_value = {"solutions": [], "pareto_front": []}

                result = agent.optimize(
                    objectives=["binding_affinity", "selectivity"],
                    population_size=50,
                )

                mock_nsga.assert_called_once()

    def test_pi_success_continues_normally(self):
        """Should continue normally when PI succeeds."""
        from orchestration.agents.optimization_agent import OptimizationAgent

        agent = OptimizationAgent()

        mock_pi_result = {"solutions": [{"id": 1}], "pareto_front": [0]}

        with patch.object(agent, "_run_pi_optimization", return_value=mock_pi_result):
            with patch.object(agent, "_run_nsga_optimization") as mock_nsga:
                result = agent.optimize(
                    objectives=["binding_affinity", "selectivity"],
                    population_size=50,
                )

                # Should NOT have called NSGA
                mock_nsga.assert_not_called()
                assert result["solutions"] == mock_pi_result["solutions"]

    def test_fallback_logged(self):
        """Should log when falling back to NSGA-II."""
        from orchestration.agents.optimization_agent import OptimizationAgent

        agent = OptimizationAgent()

        with patch.object(agent, "_run_pi_optimization", side_effect=Exception("PI failed")):
            with patch.object(agent, "_run_nsga_optimization") as mock_nsga:
                with patch.object(agent, "logger") as mock_logger:
                    mock_nsga.return_value = {"solutions": [], "pareto_front": []}

                    agent.optimize(
                        objectives=["binding_affinity"],
                        population_size=50,
                    )

                    # Should have logged the fallback
                    mock_logger.warning.assert_called()
                    warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
                    assert any("fallback" in c.lower() or "nsga" in c.lower() for c in warning_calls)

    def test_fallback_respects_max_evaluations(self):
        """Fallback should respect max_evaluations limit."""
        from orchestration.agents.optimization_agent import OptimizationAgent

        agent = OptimizationAgent()

        with patch.object(agent, "_run_pi_optimization", side_effect=Exception("PI failed")):
            with patch.object(agent, "_run_nsga_optimization") as mock_nsga:
                mock_nsga.return_value = {"solutions": [], "pareto_front": []}

                agent.optimize(
                    objectives=["binding_affinity"],
                    population_size=50,
                    max_evaluations=1000,
                )

                # NSGA should be called with max_evaluations
                call_kwargs = mock_nsga.call_args[1]
                assert call_kwargs.get("max_evaluations") == 1000


class TestPINFALLBACKIntegration:
    """Integration tests for PI-NSGA fallback."""

    def test_full_optimization_with_fallback(self):
        """Test full optimization workflow with fallback."""
        from orchestration.agents.optimization_agent import OptimizationAgent

        agent = OptimizationAgent()

        nsga_result = {
            "solutions": [
                {"id": 1, "objectives": [-10.0, 0.8]},
                {"id": 2, "objectives": [-8.0, 0.9]},
            ],
            "pareto_front": [0, 1],
        }

        with patch.object(agent, "_run_pi_optimization", side_effect=Exception("PI failed")):
            with patch.object(agent, "_run_nsga_optimization", return_value=nsga_result):
                result = agent.optimize(
                    objectives=["binding_affinity", "selectivity"],
                    population_size=50,
                )

                assert len(result["solutions"]) == 2
                assert result["pareto_front"] == [0, 1]

    def test_fallback_preserves_objectives(self):
        """Fallback should preserve the same objectives."""
        from orchestration.agents.optimization_agent import OptimizationAgent

        agent = OptimizationAgent()

        with patch.object(agent, "_run_pi_optimization", side_effect=Exception("PI failed")):
            with patch.object(agent, "_run_nsga_optimization") as mock_nsga:
                mock_nsga.return_value = {"solutions": [], "pareto_front": []}

                agent.optimize(
                    objectives=["binding_affinity", "selectivity", "solubility"],
                    population_size=50,
                )

                # NSGA should be called with same objectives
                call_kwargs = mock_nsga.call_args[1]
                assert call_kwargs.get("objectives") == ["binding_affinity", "selectivity", "solubility"]