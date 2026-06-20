"""
research_agent_loop.py

Multi-agent research controller:
- monitors optimisation
- dynamically adjusts knowledge
- updates constraints mid-run
"""

from typing import Dict, List
import logging

from VLAB2.research.research_agent_adaptive import (
    initialise_system)

log = logging.getLogger("virtual_lab.agent")


# ----------------------------------------------------------------------
# ANALYSE POPULATION BEHAVIOUR
# ----------------------------------------------------------------------
def analyse_population(scores: List[Dict]) -> Dict:
    if not scores:
        return {}

    avg = {k: sum(s[k] for s in scores) / len(scores) for k in scores[0]}

    return avg


# ----------------------------------------------------------------------
# DECIDE NEXT RESEARCH DIRECTION
# ----------------------------------------------------------------------
def propose_new_query(avg_scores: Dict) -> str:
    """
    Turn optimisation behaviour into a new literature query
    """

    if avg_scores["motif"] < 0.05:
        return "RNA motifs regulatory elements RGAG RAAG binding"

    if avg_scores["structure"] > 0.4 and avg_scores["kinetic"] < 0.1:
        return "RNA folding kinetics vs thermodynamic stability"

    if avg_scores["binding"] < 0.05:
        return "RNA protein binding affinity determinants"

    return "RNA secondary structure stability mechanisms"


# ----------------------------------------------------------------------
# EXPAND KNOWLEDGE (LIGHTWEIGHT)
# ----------------------------------------------------------------------
def expand_agent_knowledge(query: str) -> Dict:
    """
    Run a lightweight literature refresh
    """
    log.info(f"[Agent] Searching new topic: {query}")

    new_system = initialise_system(query)

    return new_system["weights"]


# ----------------------------------------------------------------------
# MERGE WEIGHT MODELS
# ----------------------------------------------------------------------
def merge_weights(old: Dict, new: Dict, alpha: float = 0.3) -> Dict:
    """
    Blend previous knowledge with new discoveries
    """

    merged = {}

    for k in old:
        merged[k] = (1 - alpha) * old[k] + alpha * new.get(k, 0)

    return merged


# ----------------------------------------------------------------------
# MAIN AGENT STEP
# ----------------------------------------------------------------------
def agent_step(
    scores: List[Dict],
    current_weights: Dict,
) -> Dict:
    """
    Single agent reasoning step
    """

    avg = analyse_population(scores)

    if not avg:
        return current_weights

    query = propose_new_query(avg)

    new_weights = expand_agent_knowledge(query)

    updated = merge_weights(current_weights, new_weights)

    log.info(f"[Agent] Updated weights: {updated}")

    return updated
