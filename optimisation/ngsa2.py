"""
NSGA-II with grammar RNA + surrogate + SFold + Vienna integration
"""

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from optimisation.neural_surrogate import NeuralSurrogate
from optimisation.acquisition import acquisition_score
from optimisation.rna_grammar_generator import (
    generate_population,
    structure_preserving_mutation
)

from VLAB2.core.sfold_wrapper import SFoldWrapper
from VLAB2.core.viennarna_wrapper import ViennaRNAWrapper
from VLAB2.research.research_agent_adaptive import score_sequence


NUCLEOTIDES = ["A", "C", "G", "U"]


def decode_sequence(x):
    return "".join(NUCLEOTIDES[i % 4] for i in x)


def encode_sequence(seq):
    return np.array([NUCLEOTIDES.index(b) for b in seq])


# ----------------------------------------------------------
# PROBLEM
# ----------------------------------------------------------
class RNAProblem(Problem):
    def __init__(self, target_pdb, state, surrogate):

        super().__init__(
            n_var=40,
            n_obj=5,
            n_constr=0,
            xl=0,
            xu=3,
            type_var=int,
        )

        self.target_pdb = target_pdb
        self.weights = state.get("research_weights", {})
        self.surrogate = surrogate

        self.sfold = SFoldWrapper()
        self.vienna = ViennaRNAWrapper()

        self.cache = {}
        self.sf_cache = {}
        self.vienna_cache = {}

    # ----------------------------------------------------------
    def get_sfold(self, seq):
        if seq in self.sf_cache:
            return self.sf_cache[seq]

        sf = self.sfold.run_sfold(seq)
        sf["pair_density"] = sf.get("pair_density") or 0.0

        self.sf_cache[seq] = sf
        return sf

    # ----------------------------------------------------------
    def get_vienna(self, seq):
        if seq in self.vienna_cache:
            return self.vienna_cache[seq]

        vr = self.vienna.run_rnafold(seq)

        vr["pair_density"] = vr.get("pair_density") or 0.0

        self.vienna_cache[seq] = vr
        return vr

    # ----------------------------------------------------------
    def evaluate_individual(self, x):

        seq = decode_sequence(x)

        if seq in self.cache:
            return self.cache[seq]

        # ----------------------------------------------------------
        # ✅ SFOLD (ensemble)
        # ----------------------------------------------------------
        sf = self.get_sfold(seq)

        if sf["pair_density"] < 0.05:
            penalty = [10, 10, 10, 10, 10]
            self.cache[seq] = penalty
            return penalty

        # ----------------------------------------------------------
        # ✅ VIENNA (thermodynamics)
        # ----------------------------------------------------------
        vr = self.get_vienna(seq)

        mfe = vr.get("mfe")
        v_pd = vr.get("pair_density", 0.0)

        # normalised MFE score
        if mfe is not None:
            mfe_score = min(abs(mfe) / len(seq), 2.0) / 2.0
        else:
            mfe_score = 0.0

        # ----------------------------------------------------------
        # ✅ SURROGATE (MD proxy)
        # ----------------------------------------------------------
        pred, uncertainty = self.surrogate.predict_with_uncertainty(seq, sf)

        if pred is None or uncertainty is None:
            pred, uncertainty = 0.0, 1.0

        md_estimate = acquisition_score(pred, uncertainty, beta=0.7)

        # ----------------------------------------------------------
        # ✅ BIO/PHYSICS SCORING
        # ----------------------------------------------------------
        scores = score_sequence(seq, sf, self.weights)

        # ----------------------------------------------------------
        # ✅ VIENNA INTEGRATION (CRITICAL)
        # ----------------------------------------------------------

        # reward stable folding
        scores["structure"] *= (1.0 + mfe_score)

        # reward base pairing
        scores["structure"] *= (1.0 + v_pd)

        # boost thermodynamics using MFE
        scores["thermo"] += mfe_score

        # ----------------------------------------------------------
        # ✅ SURROGATE BLENDING
        # ----------------------------------------------------------
        scores["structure"] *= (0.5 + md_estimate)
        scores["kinetic"] *= (0.5 + md_estimate)

        # ----------------------------------------------------------
        # ✅ EXTRA PHYSICS CONSTRAINTS
        # ----------------------------------------------------------
        if sf["pair_density"] > 0.25:
            scores["structure"] *= 1.3

        gc = (seq.count("G") + seq.count("C")) / len(seq)
        if gc < 0.35:
            scores["structure"] *= 0.8

        result = [
            -scores["thermo"],
            -scores["structure"],
            -scores["motif"],
            -scores["binding"],
            -scores["kinetic"],
        ]

        self.cache[seq] = result
        return result

    # ----------------------------------------------------------
    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.array([self.evaluate_individual(x) for x in X])


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def run_multiobjective_nsga2(target_pdb, state):

    print("🚀 NSGA-II (SFold + Vienna + Surrogate + Grammar RNA)")

    surrogate = NeuralSurrogate()

    problem = RNAProblem(target_pdb, state, surrogate)

    # ✅ grammar-based initial population
    init_sequences = generate_population(size=32, length=40)
    init_X = np.array([encode_sequence(seq) for seq in init_sequences])

    algorithm = NSGA2(
        pop_size=32,
        sampling=init_X,
        crossover=SBX(prob=0.9, eta=20),
        mutation=PM(prob=0.15),
        eliminate_duplicates=True,
    )

    result = minimize(
        problem,
        algorithm,
        ("n_gen", 8),
        verbose=False,
    )

    population = result.pop

    # ----------------------------------------------------------
    # ✅ ACTIVE LEARNING
    # ----------------------------------------------------------
    try:
        md = state["wrappers"]["md"]

        candidates = []

        for ind in population:
            seq = decode_sequence(ind.X)
            sf = problem.get_sfold(seq)

            pred, unc = surrogate.predict_with_uncertainty(seq, sf)
            score = acquisition_score(pred, unc, beta=1.0)

            candidates.append((seq, sf, score, unc))

        candidates.sort(key=lambda x: x[2], reverse=True)

        training_data = []

        print(f"🔬 Evaluating {min(8, len(candidates))} candidates via MD")

        for seq, sf, score, unc in candidates[:8]:
            seq_mut = structure_preserving_mutation(seq)

            md_res = md.run_md(seq_mut)

            if md_res.get("valid"):
                # ✅ Issue 11 fix: MDWrapper returns stability_index/compactness,
                # not 'score'.  Use stability_index (abs(min_energy)/(1+fluct)),
                # normalised to [0,1] with a reasonable cap.
                raw_score = (
                    md_res.get("stability_index")
                    or md_res.get("compactness")
                    or 0.0
                )
                md_score = max(0.0, min(1.0, raw_score / 10.0))

                training_data.append((seq_mut, sf, md_score))

        if training_data:
            print(f"✅ Updating surrogate ({len(training_data)} samples)")
            surrogate.update(training_data)

    except Exception as e:
        print("❌ Active learning failed:", e)

    return population, state.get("mutation_bias", {})
