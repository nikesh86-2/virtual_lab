from __future__ import annotations
"""
NSGA-II + lightweight Bayesian optimisation for RNA design.

ELI5:
This file is the "smart search engine" for RNA candidates.

It does this:
1. Takes a population of RNA sequences
2. Evaluates each one with:
   - protein binding
   - RNA MD stability
   - ViennaRNA folding
3. Keeps the best trade-offs using NSGA-II
4. Learns simple patterns from what worked
5. Proposes better candidates for the next generation
"""

import logging
import random
from typing import Dict, List, Optional

import numpy as np


try:
    from .ngsa2 import pareto_front, select_nsga2
except ImportError:
    from ngsa2 import pareto_front, select_nsga2


log = logging.getLogger("optimiser")


# ----------------------------------------------------------------------
# Sequence encoding
# ----------------------------------------------------------------------
def encode_sequence(seq: str, max_len: int = 50) -> np.ndarray:
    """
    Turn RNA letters into numbers so the lightweight surrogate model
    can compare sequences.

    A -> 0
    C -> 1
    G -> 2
    U -> 3
    """
    mapping = {"A": 0, "C": 1, "G": 2, "U": 3}

    vec = [mapping.get(c, 0) for c in seq]
    vec = vec[:max_len] + [0] * max(0, max_len - len(vec))

    return np.array(vec, dtype=float)


# ----------------------------------------------------------------------
# Lightweight surrogate model
# ----------------------------------------------------------------------
class SimpleGP:
    """
    Very lightweight surrogate model.

    ELI5:
    This is not a full Gaussian Process library.
    It just says:
    "If this sequence looks like earlier good sequences, it might be good too."
    """

    def __init__(self):
        self.X = np.empty((0, 50), dtype=float)
        self.y = np.array([], dtype=float)

    def fit(self, X: List[np.ndarray], y: List[float]) -> None:
        if not X:
            self.X = np.empty((0, 50), dtype=float)
            self.y = np.array([], dtype=float)
            return

        self.X = np.vstack(X)
        self.y = np.array(y, dtype=float)

    def predict(self, x: np.ndarray) -> tuple[float, float]:
        """
        Return:
            mean score estimate,
            uncertainty estimate
        """
        if len(self.X) == 0:
            return 0.0, 1.0

        dists = np.linalg.norm(self.X - x, axis=1)
        weights = np.exp(-dists)

        weight_sum = float(np.sum(weights)) + 1e-8
        mean = float(np.sum(weights * self.y) / weight_sum)
        uncertainty = float(1.0 / weight_sum)

        return mean, uncertainty


# ----------------------------------------------------------------------
# Objective computation
# ----------------------------------------------------------------------
def compute_objectives(result: dict) -> Optional[Dict[str, float]]:
    """
    Build a multi-objective score vector.

    Higher values are always better here.

    Objectives:
    - binding: lower dg is better -> use -dg
    - stability: lower fluctuation is better -> use -energy_fluctuation
    - folding: lower MFE is better -> use -mfe
    """
    binding = result.get("binding") or {}
    md = result.get("md") or {}
    vienna = result.get("vienna") or {}

    if not binding.get("valid"):
        return None

    dg = binding.get("dg")
    fluct = md.get("energy_fluctuation")
    mfe = vienna.get("mfe")

    if dg is None or fluct is None or mfe is None:
        return None

    return {
        "binding": -float(dg),
        "stability": -float(fluct),
        "folding": -float(mfe),
    }


# ----------------------------------------------------------------------
# Acquisition function
# ----------------------------------------------------------------------
def acquisition(mean: float, uncertainty: float, kappa: float = 2.0) -> float:
    """
    Simple upper-confidence style acquisition.

    ELI5:
    Good candidates are ones that:
    - look promising
    - or are uncertain / unexplored
    """
    return mean + kappa * uncertainty


# ----------------------------------------------------------------------
# Candidate generation
# ----------------------------------------------------------------------
def mutate_random(seq: str, rate: float = 0.15) -> str:
    seq_list = list(seq)

    for i in range(len(seq_list)):
        if random.random() < rate:
            seq_list[i] = random.choice("ACGU")

    return "".join(seq_list)


def mutate_guided(seq: str) -> str:
    """
    Slightly bias AU -> GC for stability,
    but still allow some flexibility.
    """
    seq_list = list(seq)

    for i in range(len(seq_list)):
        if seq_list[i] in "AU" and random.random() < 0.30:
            seq_list[i] = random.choice("GC")
        elif seq_list[i] in "GC" and random.random() < 0.08:
            seq_list[i] = random.choice("AU")

    return "".join(seq_list)


def insert_motif(seq: str, motifs: List[str]) -> str:
    if not motifs:
        return seq

    motif = random.choice(motifs)

    if len(motif) >= len(seq):
        return seq

    pos = random.randint(0, len(seq) - len(motif))
    return seq[:pos] + motif + seq[pos + len(motif):]


def shuffle_block(seq: str, block_size: int = 5) -> str:
    if len(seq) < block_size + 1:
        return seq

    start = random.randint(0, len(seq) - block_size)
    block = list(seq[start:start + block_size])
    random.shuffle(block)

    return seq[:start] + "".join(block) + seq[start + block_size:]


def random_sequence(length: int) -> str:
    return "".join(random.choice("ACGU") for _ in range(length))


def filter_diversity(seqs: List[str], similarity_threshold: float = 0.85) -> List[str]:
    """
    Remove near-duplicate sequences to stop the population collapsing.
    """
    unique: List[str] = []

    for s in seqs:
        keep = True
        for u in unique:
            denom = min(len(s), len(u))
            if denom == 0:
                continue

            match = sum(1 for a, b in zip(s, u) if a == b) / denom
            if match > similarity_threshold:
                keep = False
                break

        if keep:
            unique.append(s)

    return unique


# ----------------------------------------------------------------------
# Full pipeline evaluation
# ----------------------------------------------------------------------
def evaluate_full_pipeline(sequence: str, wrappers: dict, pdb_id: str) -> Optional[dict]:
    try:
        protein = wrappers["protein"]
        md = wrappers["md"]
        vienna = wrappers["vienna"]

        binding_results = protein.evaluate_sequences(pdb_id, [sequence])
        if not binding_results:
            log.warning("No binding results for %s against %s", sequence[:25], pdb_id)
            return None

        binding = binding_results[0]
        if not binding.get("valid"):
            log.warning(
                "Invalid binding for %s against %s: %s",
                sequence[:25], pdb_id, binding.get("error", "unknown")
            )
            return None

        md_res = md.run_md(sequence)
        if not md_res.get("valid"):
            log.warning(
                "Invalid MD for %s: %s",
                sequence[:25], md_res.get("error", "unknown")
            )
            return None

        vienna_res = vienna.run_rnafold(sequence)
        if not vienna_res.get("valid"):
            log.warning(
                "Invalid ViennaRNA for %s: %s",
                sequence[:25], vienna_res.get("error", "unknown")
            )
            return None

        return {
            "sequence": sequence,
            "binding": binding,
            "md": md_res,
            "vienna": vienna_res,
        }

    except Exception as e:
        log.warning("Pipeline failed for sequence %s: %s", sequence[:25], e)
        return None

# ----------------------------------------------------------------------
# Main optimiser
# ----------------------------------------------------------------------
def optimise_sequences(
    initial_sequences: List[str],
    wrapper_bundle: dict,
    pdb_id: str,
    generations: int = 3,
    population_size: int = 20,
    exploration_random: int = 5,
) -> List[dict]:
    """
    Main optimisation loop.

    Parameters:
        initial_sequences : starting RNA candidates
        wrapper_bundle    : wrappers dict from orchestrator state
        pdb_id            : target protein structure to optimise against
        generations       : number of optimisation rounds
        population_size   : max candidate count per generation
        exploration_random: random extra sequences per generation

    Returns:
        final Pareto front from the last generation
    """
    population = list(dict.fromkeys(initial_sequences))[:population_size]

    if not population:
        return []

    surrogate = SimpleGP()

    memory_X: List[np.ndarray] = []
    memory_y: List[float] = []
    motif_memory: List[str] = []

    final_evaluated: List[dict] = []

    for gen in range(generations):
        evaluated: List[dict] = []

        # --------------------------------------------------
        # 1. Evaluate the current population
        # --------------------------------------------------
        for i, seq in enumerate(population):
            result = evaluate_full_pipeline(seq, wrapper_bundle, pdb_id)
            if not result:
                continue

            objectives = compute_objectives(result)
            if not objectives:
                continue

            result["objectives"] = objectives
            result["id"] = f"{gen}_{i}_{seq[:12]}"

            evaluated.append(result)

            # Use summed objectives to train the simple surrogate
            memory_X.append(encode_sequence(seq))
            memory_y.append(sum(objectives.values()))

        if not evaluated:
            log.warning("No valid evaluated sequences in generation %d", gen)
            return final_evaluated or []

        final_evaluated = evaluated

        # --------------------------------------------------
        # 2. Fit surrogate
        # --------------------------------------------------
        surrogate.fit(memory_X, memory_y)

        # --------------------------------------------------
        # 3. NSGA-II selection
        # --------------------------------------------------
        selected = select_nsga2(
            evaluated,
            population_size=min(population_size, len(evaluated)),
        )
        front0 = pareto_front(evaluated)

        log.info(
            "[Gen %d] evaluated=%d selected=%d pareto=%d target=%s",
            gen,
            len(evaluated),
            len(selected),
            len(front0),
            pdb_id,
        )

        # Build motif memory from best front
        for item in front0:
            seq = item["sequence"]
            for j in range(max(0, len(seq) - 4)):
                motif_memory.append(seq[j:j + 5])

        motif_memory = list(dict.fromkeys(motif_memory))[:100]

        # --------------------------------------------------
        # 4. Generate next candidate pool
        # --------------------------------------------------
        candidates: List[str] = []

        for item in selected[: min(10, len(selected))]:
            seq = item["sequence"]

            # keep elite
            candidates.append(seq)

            # exploitation
            candidates.append(mutate_guided(seq))
            candidates.append(mutate_guided(seq))
            candidates.append(insert_motif(seq, motif_memory))
            candidates.append(shuffle_block(seq))

            # exploration around elite
            candidates.append(mutate_random(seq, rate=0.10))
            candidates.append(mutate_random(seq, rate=0.20))

        # Add some random explorers
        base_len = len(population[0]) if population else 30
        for _ in range(exploration_random):
            candidates.append(random_sequence(base_len))

        candidates = filter_diversity(candidates)
        if not candidates:
            candidates = population[:]

        # --------------------------------------------------
        # 5. Use surrogate to rank candidates
        # --------------------------------------------------
        scored_candidates = []

        for seq in candidates:
            x = encode_sequence(seq)
            mean, unc = surrogate.predict(x)
            score = acquisition(mean, unc)
            scored_candidates.append((score, seq))

        scored_candidates.sort(reverse=True, key=lambda x: x[0])

        population = [seq for _, seq in scored_candidates[:population_size]]

        if not population:
            break

    return pareto_front(final_evaluated)

