"""
adaptive_mutation.py

Extracts mutation bias from:
- Pareto front
- conservation data

Returns mutation guidance
"""

from collections import Counter


# ----------------------------------------------------------------------
# FIND IMPORTANT POSITIONS
# ----------------------------------------------------------------------
def extract_position_bias(population, decode_fn):
    sequences = [decode_fn(ind.X) for ind in population]

    if not sequences:
        return {}

    length = len(sequences[0])

    position_data = []

    for i in range(length):
        col = [seq[i] for seq in sequences]
        counts = Counter(col)
        position_data.append(counts)

    return position_data


# ----------------------------------------------------------------------
# CONSERVED MOTIF BIAS
# ----------------------------------------------------------------------
def extract_conservation_bias(state):
    cons = state.get("conservation_signal", {})
    return cons.get("motif_regions", [])


# ----------------------------------------------------------------------
# COMBINE FEEDBACK
# ----------------------------------------------------------------------
def build_mutation_bias(population, decode_fn, state):
    position_bias = extract_position_bias(population, decode_fn)
    conserved_regions = extract_conservation_bias(state)

    return {
        "position_bias": position_bias,
        "conserved_regions": conserved_regions,
    }


import random


NUCLEOTIDES = ["A", "C", "G", "U"]


def guided_mutation(seq, bias, mutation_rate=0.1):
    seq = list(seq)

    position_bias = bias.get("position_bias", [])
    conserved_regions = bias.get("conserved_regions", [])

    protected_positions = set()

    for start, end in conserved_regions:
        for i in range(start, end):
            protected_positions.add(i)

    for i in range(len(seq)):
        if i in protected_positions:
            continue  # ✅ preserve conserved regions

        if random.random() < mutation_rate:
            if i < len(position_bias) and position_bias[i]:
                # ✅ bias toward successful nucleotides
                weights = position_bias[i]
                seq[i] = random.choices(
                    list(weights.keys()),
                    weights=list(weights.values())
                )[0]
            else:
                seq[i] = random.choice(NUCLEOTIDES)

    return "".join(seq)