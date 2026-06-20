from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional

import numpy as np
from VLAB2.optimisation.ngsa2 import pareto_front, select_nsga2
log = logging.getLogger("optimiser")


# ---------------------------------------------------------------------------
# Sequence encoding
# ---------------------------------------------------------------------------
def encode_sequence(seq: str, max_len: int = 50) -> np.ndarray:
    """
    Encode RNA into numeric vector for the surrogate model.

    A -> 0, C -> 1, G -> 2, U -> 3
    """
    mapping = {"A": 0, "C": 1, "G": 2, "U": 3}
    vec = [mapping.get(c, 0) for c in seq]
    vec = vec[:max_len] + [0] * max(0, max_len - len(vec))
    return np.array(vec, dtype=float)


# ---------------------------------------------------------------------------
# Lightweight surrogate model
# ---------------------------------------------------------------------------
class SimpleGP:
    """
    A very lightweight surrogate model.

    ELI5:
    This is not a full Gaussian Process library.
    It's a simple "look at nearby examples and guess" model,
    which is good enough for rebuilding the system cleanly.
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
        Return (mean, uncertainty).

        ELI5:
        - mean = best guess of candidate quality
        - uncertainty = how unsure the model is
        """
        if len(self.X) == 0:
            return 0.0, 1.0

        dists = np.linalg.norm(self.X - x, axis=1)

        # similarity weights
        weights = np.exp(-dists)

        weight_sum = float(np.sum(weights)) + 1e-8
        mean = float(np.sum(weights * self.y) / weight_sum)
        uncertainty = float(1.0 / weight_sum)

        return mean, uncertainty


# ---------------------------------------------------------------------------
# Objective computation
# ---------------------------------------------------------------------------
def compute_objectives(result: dict) -> Optional[Dict[str, float]]:
    """
    Convert raw wrapper outputs into multi-objective values.

    Higher = better for every objective in this function.

    Objectives:
    - binding: stronger binding means lower dg, so use -dg
    - stability: lower fluctuation is better, so use -fluctuation
    - folding: lower MFE is better, so use -mfe
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


# ---------------------------------------------------------------------------
# Acquisition function
# ---------------------------------------------------------------------------
def acquisition(mean: float, uncertainty: float, kappa: float = 2.0) -> float:
    """
    Upper-confidence style acquisition.

    ELI5:
    Good candidate if:
    - predicted good
    - or uncertain / unexplored
    """
    return mean + kappa * uncertainty


# ---------------------------------------------------------------------------
# Mutations / candidate generation
# ---------------------------------------------------------------------------
def mutate_random(seq: str, rate: float = 0.15) -> str:
    seq_list = list(seq)
    for i in range(len(seq_list)):
        if random.random() < rate:
            seq_list[i] = random.choice("ACGU")
    return "".join(seq_list)


def mutate_guided(seq: str) -> str:
    """
    Bias AU -> GC slightly to promote stability,
    but keep some randomness so exploration remains possible.
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

    i = random.randint(0, len(seq) - block_size)
    block = list(seq[i:i + block_size])
    random.shuffle(block)
    return seq[:i] + "".join(block) + seq[i + block_size:]


def random_sequence(length: int) -> str:
    return "".join(random.choice("ACGU") for _ in range(length))


def filter_diversity(seqs: List[str], similarity_threshold: float = 0.85) -> List[str]:
    """
    Remove near-duplicates so the population does not collapse.
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


# ---------------------------------------------------------------------------
# Full pipeline evaluation
# ---------------------------------------------------------------------------
def evaluate_full_pipeline(sequence: str, wrappers: dict, pdb_id: str) -> Optional[dict]:
    """
    Run the scientific wrappers for one sequence.

    Expected wrappers:
        wrappers["protein"]
        wrappers["md"]
        wrappers["vienna"]
    """
    try:
        protein = wrappers["protein"]
        md = wrappers["md"]
        vienna = wrappers["vienna"]

        binding_results = protein.evaluate_sequences(pdb_id, [sequence])
        if not binding_results:
            return None

        binding = binding_results[0]
        if not binding.get("valid"):
            return None

        md_res = md.run_md(sequence)
        vienna_res = vienna.run_rnafold(sequence)

        return {
            "sequence": sequence,
            "binding": binding,
            "md": md_res,
            "vienna": vienna_res,
        }

    except Exception as e:
        log.warning("Pipeline failed for sequence %s: %s", sequence[:25], e)
        return None


# ---------------------------------------------------------------------------
# Main optimiser
# ---------------------------------------------------------------------------
def optimise_sequences(
    sequences: List[str],
    wrappers: dict,
    pdb_id: str,
    generations: int = 3,
    population_size: int = 20,
    exploration_random: int = 5,
) -> List[dict]:
    """
    Main optimisation loop.

    Returns:
        Pareto front (best trade-off solutions) from the final generation.

    Result items contain:
        - sequence
        - binding / md / vienna raw output
        - objectives
        - rank / distance
    """
    population = list(dict.fromkeys(sequences))[:population_size]
    if not population:
        return []

    surrogate = SimpleGP()

    memory_X: List[np.ndarray] = []
    memory_y: List[float] = []
    motif_memory: List[str] = []

    final_population_evaluated: List[dict] = []

    for gen in range(generations):
        evaluated: List[dict] = []

        # ---------------------------------------------------
        # 1. Evaluate current population
        # ---------------------------------------------------
        for i, seq in enumerate(population):
            result = evaluate_full_pipeline(seq, wrappers, pdb_id)
            if not result:
                continue

            objectives = compute_objectives(result)
            if not objectives:
                continue

            result["objectives"] = objectives
            result["id"] = f"{gen}_{i}_{seq[:12]}"
            evaluated.append(result)

            # surrogate training target = simple sum of objectives
            # (keeps the surrogate lightweight while NSGA-II keeps true multi-objective behaviour)
            memory_X.append(encode_sequence(seq))
            memory_y.append(sum(objectives.values()))

        if not evaluated:
            log.warning("No valid evaluated sequences in generation %d", gen)
            return final_population_evaluated or []

        final_population_evaluated = evaluated

        # ---------------------------------------------------
        # 2. Train surrogate
        # ---------------------------------------------------
        surrogate.fit(memory_X, memory_y)

        # ---------------------------------------------------
        # 3. NSGA-II selection on evaluated population
        # ---------------------------------------------------
        selected = select_nsga2(evaluated, population_size=min(population_size, len(evaluated)))
        front0 = pareto_front(evaluated)

        log.info(
            "[Gen %d] evaluated=%d selected=%d pareto=%d",
            gen,
            len(evaluated),
            len(selected),
            len(front0),
        )

        # motif memory from the best trade-offs
        for item in front0:
            seq = item["sequence"]
            for j in range(max(0, len(seq) - 4)):
                motif_memory.append(seq[j:j + 5])

        motif_memory = list(dict.fromkeys(motif_memory))[:100]

        # ---------------------------------------------------
        # 4. Generate candidate pool
        # ---------------------------------------------------
        candidates: List[str] = []

        for item in selected[: min(10, len(selected))]:
            seq = item["sequence"]

            # keep original elite
            candidates.append(seq)

            # guided exploitation
            candidates.append(mutate_guided(seq))
            candidates.append(mutate_guided(seq))
            candidates.append(insert_motif(seq, motif_memory))
            candidates.append(shuffle_block(seq))

            # random exploration around elite
            candidates.append(mutate_random(seq, rate=0.10))
            candidates.append(mutate_random(seq, rate=0.20))

        # add random sequences so we don't get trapped
        base_len = len(population[0]) if population else 30
        for _ in range(exploration_random):
            candidates.append(random_sequence(base_len))

        candidates = filter_diversity(candidates)
        if not candidates:
            candidates = population

        # ---------------------------------------------------
        # 5. Surrogate ranking of candidates
        # ---------------------------------------------------
        scored_candidates = []

        for seq in candidates:
            x = encode_sequence(seq)
