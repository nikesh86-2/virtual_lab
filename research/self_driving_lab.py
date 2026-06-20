"""
self_driving_lab.py

Autonomous research layer:
- generates hypotheses
- proposes experiments (design runs)
- evaluates outcomes
- updates research strategy

This is the "scientist brain"
"""

from typing import Dict, List
import random

from VLAB2.research.research_agent_loop import agent_step


# ----------------------------------------------------------------------
# HYPOTHESIS GENERATION
# ----------------------------------------------------------------------
def generate_hypothesis(weights: Dict) -> str:
    """
    Translate current weights into a hypothesis
    """

    dominant = max(weights, key=weights.get)

    hypotheses = {
        "thermo": "Higher GC content improves RNA structural stability",
        "structure": "Secondary structure density drives functional behaviour",
        "motif": "RGAG/RAAG motifs increase functional efficiency",
        "binding": "Binding affinity motifs dominate performance",
        "kinetic": "Folding kinetics constrain optimal structures",
    }

    return hypotheses.get(dominant, "RNA performance depends on mixed constraints")


# ----------------------------------------------------------------------
# EXPERIMENT PLANNER
# ----------------------------------------------------------------------
def plan_experiment(weights: Dict) -> Dict:
    """
    Decide how to explore sequence space next
    """

    focus = max(weights, key=weights.get)

    plan = {
        "focus": focus,
        "mutation_rate": 0.1 + weights[focus],
        "population_size": 40 + int(60 * weights[focus]),
        "exploration_bias": random.random(),
    }

    return plan


# ----------------------------------------------------------------------
# RESULT INTERPRETATION
# ----------------------------------------------------------------------
def interpret_results(scores: List[Dict]) -> Dict:
    """
    Summarise experimental outcomes
    """

    if not scores:
        return {}

    avg = {k: sum(s[k] for s in scores) / len(scores) for k in scores[0] if not k.startswith("_")}

    dominant = max(avg, key=avg.get)

    return {
        "average_scores": avg,
        "dominant_signal": dominant,
    }


# ----------------------------------------------------------------------
# STRATEGY UPDATE
# ----------------------------------------------------------------------
def update_strategy(
    scores: List[Dict],
    weights: Dict,
    generation: int,
) -> Dict:
    """
    Full reasoning step of lab
    """

    summary = interpret_results(scores)

    hypothesis = generate_hypothesis(weights)

    print(f"\n[LAB] Generation {generation}")
    print("[LAB] Hypothesis:", hypothesis)
    print("[LAB] Observed dominant signal:", summary.get("dominant_signal"))

    # knowledge update via research agent
    if generation % 5 == 0:
        weights = agent_step(scores, weights)

    return weights


# ----------------------------------------------------------------------
# FULL LAB STEP
# ----------------------------------------------------------------------
def lab_cycle(scores: List[Dict], weights: Dict, generation: int) -> Dict:
    """
    Core loop called inside NSGA-II run
    """

    # interpret + adapt
    weights = update_strategy(scores, weights, generation)

    return weights