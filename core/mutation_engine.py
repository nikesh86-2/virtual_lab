"""
mutation_engine.py

Smart mutation + crossover for RNA evolution

Features:
- motif-aware mutation
- GC-content bias (thermodynamic proxy)
- structure-conscious mutation
- diversity preservation
"""

import random
from typing import List

NUCLEOTIDES = ["A", "C", "G", "U"]


# ----------------------------------------------------------------------
# BASIC UTILITIES
# ----------------------------------------------------------------------
def random_base(exclude=None):
    choices = [b for b in NUCLEOTIDES if b != exclude]
    return random.choice(choices)


def hamming_distance(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


# ----------------------------------------------------------------------
# MOTIF INSERTION
# ----------------------------------------------------------------------
def insert_motif(seq: str) -> str:
    """
    Insert RGAG / RAAG motif at random position
    """
    motifs = ["AGAG", "GAAG"]  # R = A/G resolved
    motif = random.choice(motifs)

    pos = random.randint(0, len(seq) - len(motif))
    return seq[:pos] + motif + seq[pos + len(motif):]


# ----------------------------------------------------------------------
# GC-BIAS MUTATION
# ----------------------------------------------------------------------
def gc_bias_mutation(seq: str, strength: float = 0.5) -> str:
    """
    Bias toward G/C if thermodynamic weight is high
    """

    seq = list(seq)

    for i in range(len(seq)):
        if random.random() < strength:
            seq[i] = random.choice(["G", "C"])

    return "".join(seq)


# ----------------------------------------------------------------------
# RANDOM POINT MUTATION
# ----------------------------------------------------------------------
def point_mutation(seq: str, rate: float = 0.1) -> str:
    seq = list(seq)

    for i in range(len(seq)):
        if random.random() < rate:
            seq[i] = random_base(seq[i])

    return "".join(seq)


# ----------------------------------------------------------------------
# STRUCTURE-AWARE MUTATION (LIGHTWEIGHT)
# ----------------------------------------------------------------------
def structure_mutation(seq: str, bias: float = 0.3) -> str:
    """
    Bias mutations toward complementary pairs

    (Very approximate, but enough to guide folding)
    """

    comp = {"A": "U", "U": "A", "G": "C", "C": "G"}
    seq = list(seq)

    for i in range(len(seq)):
        if random.random() < bias:
            partner = len(seq) - 1 - i
            seq[i] = comp.get(seq[partner], seq[i])

    return "".join(seq)


# ----------------------------------------------------------------------
# SMART MUTATION PIPELINE
# ----------------------------------------------------------------------
def smart_mutate(seq: str, weights: dict) -> str:
    """
    Main mutation operator informed by literature weights.
    """

    # Derive mutation bias inline from weights (mutation_bias() never existed
    # in research_agent_adaptive \u2014 it was a phantom import that crashed at runtime).
    thermo_w = weights.get("thermo", 1.0)
    structure_w = weights.get("structure", 1.0)
    motif_w = weights.get("motif", 1.0)

    total = max(thermo_w + structure_w + motif_w, 1e-6)

    bias = {
        "gc_bias": min(0.8, thermo_w / total),
        "structure_bias": min(0.8, structure_w / total),
        "motif_insertion_prob": min(0.5, motif_w / total),
    }

    new_seq = seq

    # 1. motif injection
    if random.random() < bias["motif_insertion_prob"]:
        new_seq = insert_motif(new_seq)

    # 2. GC bias (thermodynamics)
    new_seq = gc_bias_mutation(new_seq, strength=bias["gc_bias"])

    # 3. structure bias
    new_seq = structure_mutation(new_seq, bias=bias["structure_bias"])

    # 4. random mutation (exploration)
    new_seq = point_mutation(new_seq, rate=0.1)

    return new_seq


# ----------------------------------------------------------------------
# CROSSOVER (DOMAIN-AWARE)
# ----------------------------------------------------------------------
def crossover(seq1: str, seq2: str) -> str:
    """
    RNA-specific crossover
    """

    point = random.randint(1, len(seq1) - 2)

    child = seq1[:point] + seq2[point:]

    return child


# ----------------------------------------------------------------------
# DIVERSITY FILTER
# ----------------------------------------------------------------------
def enforce_diversity(population: List[str], min_dist: int = 2) -> List[str]:
    """
    Remove near-duplicate sequences
    """

    filtered = []

    for seq in population:
        if all(hamming_distance(seq, s) >= min_dist for s in filtered):
            filtered.append(seq)

    return filtered


# ----------------------------------------------------------------------
# FULL GENERATION STEP
# ----------------------------------------------------------------------
def evolve_population(population: List[str], weights: dict) -> List[str]:
    """
    One mutation + crossover generation
    """

    new_pop = []

    while len(new_pop) < len(population):
        p1, p2 = random.sample(population, 2)

        child = crossover(p1, p2)
        child = smart_mutate(child, weights)

        new_pop.append(child)

    # enforce diversity
    new_pop = enforce_diversity(new_pop)

    # refill if reduced
    while len(new_pop) < len(population):
        new_pop.append(random.choice(population))

    return new_pop
