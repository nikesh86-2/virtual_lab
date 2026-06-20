"""
optimisation_backend.py (FINAL UNIFIED VERSION)

- Uses structure_core (SFold + Vienna fusion)
- Protein fast scoring (with confidence)
- MD stability
- evolutionary optimisation loop
"""

from typing import Dict, List
import random
from optimisation.adaptive_mutation import guided_mutation
from core.structure_core import compute_structural_score

NUCLEOTIDES = ["A", "C", "G", "U"]


# ----------------------------------------------------------------------
# SEQUENCE UTILS
# ----------------------------------------------------------------------
def random_sequence(length=30):
    return "".join(random.choice(NUCLEOTIDES) for _ in range(length))


# ----------------------------------------------------------------------
# FITNESS FUNCTION
# ----------------------------------------------------------------------
def evaluate_sequence(seq: str, state: Dict) -> Dict:
    wrappers = state["wrappers"]

    protein = wrappers["protein"]
    md = wrappers["md"]

    target_pdb = state.get("target_pdb")

    # ------------------------------------------------------------------
    # STRUCTURAL SCORE (SFold + ViennaCore)
    # ------------------------------------------------------------------
    struct = compute_structural_score(seq, wrappers)
    structure_score = struct["structure_score"]

    # ------------------------------------------------------------------
    # PROTEIN BINDING
    # ------------------------------------------------------------------
    binding_score = 0.0
    if target_pdb:
        try:
            br = protein.evaluate_sequences(target_pdb, [seq], fast=True)[0]

            if br.get("valid"):
                dg = float(br.get("dg", 0.0))
                confidence = float(br.get("confidence", 1.0))

                binding_score = -dg * confidence
        except Exception:
            pass

    # ------------------------------------------------------------------
    # MD STABILITY
    # ------------------------------------------------------------------
    md_score = 0.0
    try:
        md_res = md.run_md(seq)

        if md_res.get("valid"):
            md_score = -float(md_res.get("energy_fluctuation", 0.0))
    except Exception:
        pass

    # ------------------------------------------------------------------
    # FINAL SCORE (balanced)
    # ------------------------------------------------------------------
    score = (
        structure_score * 1.0 +
        binding_score * 2.0 +
        md_score * 1.0
    )

    return {
        "sequence": seq,
        "score": score,
        "structure": struct,
        "binding": binding_score,
        "md": md_score,
    }


# ----------------------------------------------------------------------
# EVOLUTION LOOP
# ----------------------------------------------------------------------
def mutate_sequence(seq, state):
    bias = state.get("mutation_bias", {})
    return guided_mutation(seq, bias)

def optimise_sequences_with_nsga(state: Dict, n_iter=3, pop_size=20) -> List[Dict]:
    sequences = state.get("designed_sequences") or []

    # bootstrap
    if not sequences:
        sequences = [random_sequence(30) for _ in range(pop_size)]

    population = sequences

    for _ in range(n_iter):
        scored = [evaluate_sequence(seq, state) for seq in population]
        scored = sorted(scored, key=lambda x: x["score"], reverse=True)

        survivors = scored[: pop_size // 2]

        new_population = [s["sequence"] for s in survivors]

        while len(new_population) < pop_size:
            parent = random.choice(survivors)["sequence"]
            new_population.append(mutate_sequence(parent, state))

        population = new_population

    final = [evaluate_sequence(seq, state) for seq in population]
    final = sorted(final, key=lambda x: x["score"], reverse=True)

    return final
