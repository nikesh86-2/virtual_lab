"""
final_rna_design_system.py

HDOCKLITE-INTEGRATED + CONSERVATION/STABILITY/MOTIF-SELECTION UPGRADED VERSION

Features:
- NSGA-II RNA sequence optimisation
- Grammar-driven seeded population
- Exact and near-duplicate sequence deduplication
- SFold/Vienna/MD scoring
- Minimum fold threshold gating
- Conservation-aware scoring
- Motif-selection objective with IUPAC motif support
- RNA design stability objective repair
- Failure-memory score adjustment
- Selective HDOCKlite docking only for fold-valid candidates
- Legacy Vina-compatible binding fields
"""

from __future__ import annotations

import os
import math
import random
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

try:
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
except Exception:
    SBX = None
    PM = None

try:
    from pymoo.operators.repair.rounding import RoundingRepair
except Exception:
    RoundingRepair = None

from VLAB2.core.protein_prep import ensure_protein_pdb, prepare_protein
from VLAB2.core.hdock_wrapper import HDockDocking
from VLAB2.core.sfold_wrapper import SFoldWrapper
from VLAB2.core.rna_prep import prepare_rna_pdb_for_hdock
from VLAB2.research.research_agent_adaptive import (
    score_sequence,
    update_weights,
    build_initial_weights,
)
from VLAB2.orchestration.failure_memory import FailureMemory
from VLAB2.optimisation.rna_grammar_generator import generate_structured_rna

try:
    from VLAB2.core.gpu_manager import clear_gpu
except Exception:
    def clear_gpu():
        return None


log = logging.getLogger("virtual_lab.optimisation")


NUCLEOTIDES = ["A", "C", "G", "U"]

OBJECTIVES = [
    "thermo",
    "structure",
    "motif",
    "binding",
    "kinetic",
    "conservation",
    "diversity",
]

DEFAULT_SEQ_LEN = int(os.getenv("VLAB_RNA_SEQ_LEN", "40"))
POP_SIZE = int(os.getenv("VLAB_NSga_POP_SIZE", os.getenv("VLAB_NSGA_POP_SIZE", "40")))
GENERATIONS = int(os.getenv("VLAB_NSGA_GENERATIONS", "8"))
SELECTIVE_DOCK_TOP_K = int(os.getenv("VLAB_NSGA_HDOCK_TOP_K", "5"))

TARGET_PAIR_DENSITY_MIN = float(os.getenv("VLAB_RNA_PAIR_DENSITY_MIN", "0.30"))
TARGET_PAIR_DENSITY_MAX = float(os.getenv("VLAB_RNA_PAIR_DENSITY_MAX", "0.62"))
TARGET_MFE_PER_NT_MIN = float(os.getenv("VLAB_RNA_MFE_PER_NT_MIN", "-0.75"))
TARGET_MFE_PER_NT_MAX = float(os.getenv("VLAB_RNA_MFE_PER_NT_MAX", "-0.12"))
MAX_HOMOPOLYMER = int(os.getenv("VLAB_RNA_MAX_HOMOPOLYMER", "4"))

# Hard-ish fold acceptance thresholds.
# These are deliberately slightly below the target optimum window so that
# borderline-but-improvable candidates are not immediately killed.
MIN_ACCEPT_PAIR_DENSITY = float(os.getenv("VLAB_RNA_MIN_ACCEPT_PAIR_DENSITY", "0.24"))
MIN_ACCEPT_MFE_PER_NT = float(os.getenv("VLAB_RNA_MIN_ACCEPT_MFE_PER_NT", "-0.08"))
ENFORCE_MIN_FOLD = os.getenv("VLAB_RNA_ENFORCE_MIN_FOLD", "1").strip() == "1"
FOLD_THRESHOLD_PENALTY = float(os.getenv("VLAB_RNA_FOLD_THRESHOLD_PENALTY", "2.5"))

# Deduplication controls.
NEAR_DUP_MAX_HAMMING = int(os.getenv("VLAB_RNA_NEAR_DUP_MAX_HAMMING", "2"))
DEDUP_NEAR_DUPLICATES = os.getenv("VLAB_RNA_DEDUP_NEAR_DUPLICATES", "1").strip() == "1"

# Motif objective configuration.
# Format:
#   motif:weight,motif:weight
# Supports IUPAC R/Y/S/W/K/M/B/D/H/V/N.
TARGET_MOTIFS_ENV = os.getenv(
    "VLAB_RNA_TARGET_MOTIFS",
    "RGAG:1.0,RAAG:1.0,GAG:0.45,AAG:0.45,CAG:0.25,GUC:0.25,AGU:0.25",
)
MOTIF_CONTEXT_WINDOW = int(os.getenv("VLAB_RNA_MOTIF_CONTEXT_WINDOW", "6"))

_EMPTY_MD = {
    "min_energy": None,
    "mean_energy": None,
    "base_pairs": 0,
    "energy_fluctuation": 0.0,
    "stability_index": None,
    "compactness": None,
}

_hdock = None


def get_hdock() -> HDockDocking:
    global _hdock
    if _hdock is None:
        _hdock = HDockDocking()
    return _hdock


def _clean_rna(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    return "".join(c for c in seq.strip().upper().replace("T", "U") if c in "ACGU")


def infer_sequence_length(state: dict, default: int = DEFAULT_SEQ_LEN) -> int:
    if state:
        try:
            if state.get("seq_len"):
                return max(12, int(state["seq_len"]))
        except Exception:
            pass

        target = _clean_rna(state.get("target_sequence", ""))
        if target:
            return len(target)

        lengths = []
        for s in state.get("designed_sequences", []) or []:
            cs = _clean_rna(s)
            if cs:
                lengths.append(len(cs))

        if lengths:
            lengths = sorted(lengths)
            return max(12, int(lengths[len(lengths) // 2]))

    return max(12, int(default))


def quick_sequence_filter(seq: str) -> bool:
    seq = _clean_rna(seq)

    if not seq:
        return False

    gc_frac = (seq.count("G") + seq.count("C")) / max(1, len(seq))

    if gc_frac < 0.30 or gc_frac > 0.80:
        return False

    if any(base * (MAX_HOMOPOLYMER + 1) in seq for base in "ACGU"):
        return False

    if len(set(seq)) < 3:
        return False

    return True


def normalise_sequence(seq: str, seq_len: int) -> Optional[str]:
    seq = _clean_rna(seq)

    if not seq:
        return None

    if len(seq) > seq_len:
        seq = seq[:seq_len]
    elif len(seq) < seq_len:
        seq += "".join(random.choice(NUCLEOTIDES) for _ in range(seq_len - len(seq)))

    if not quick_sequence_filter(seq):
        return None

    return seq


def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(x != y for x, y in zip(a, b))


def deduplicate_sequences(
    sequences: List[str],
    seq_len: int,
    near_duplicate_max_hamming: int = NEAR_DUP_MAX_HAMMING,
    near_duplicates: bool = DEDUP_NEAR_DUPLICATES,
) -> List[str]:
    """
    Remove exact and optionally near-duplicate sequences after normalisation.

    For NSGA initialisation this avoids the optimisation repeatedly exploring
    the same local region, especially when grammar generation or conservation
    consensus seeding produces repeated variants.
    """
    out: List[str] = []
    seen = set()

    for seq in sequences or []:
        s = normalise_sequence(seq, seq_len)
        if not s:
            continue

        if s in seen:
            continue

        if near_duplicates:
            too_close = False
            for existing in out:
                if len(existing) == len(s) and hamming_distance(existing, s) <= near_duplicate_max_hamming:
                    too_close = True
                    break
            if too_close:
                continue

        seen.add(s)
        out.append(s)

    return out


def decode_sequence(x: Any) -> str:
    return "".join(NUCLEOTIDES[int(round(i)) % 4] for i in x)


def encode_sequence(seq: str, seq_len: int) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "U": 3}
    seq = normalise_sequence(seq, seq_len)
    if not seq:
        seq = "".join(random.choice(NUCLEOTIDES) for _ in range(seq_len))
    return np.array([mapping[c] for c in seq], dtype=int)


def fitness_vector(score: Dict[str, Any]) -> List[float]:
    return [-float(score.get(k, 0.0)) for k in OBJECTIVES]


def _norm_lookup_key(seq: str) -> Optional[str]:
    s = _clean_rna(seq)
    return s or None


def _sequence_match_lookup(seq: str, lookup: dict) -> Any:
    key = _norm_lookup_key(seq)

    if not key:
        return None

    if key in lookup:
        return lookup[key]

    for k, v in lookup.items():
        if key.startswith(k) or k.startswith(key):
            return v

    return None


def build_md_lookup(state: dict) -> dict:
    lookup = {}

    for m in state.get("md_results", []) or []:
        seq = m.get("sequence")
        res = m.get("result") or {}
        key = _norm_lookup_key(seq)

        if key:
            lookup[key] = res

    return lookup


def build_binding_lookup(state: dict) -> dict:
    lookup = {}

    def store(seq: str, value: Any) -> None:
        key = _norm_lookup_key(seq)
        if not key:
            return

        try:
            lookup[key] = float(value)
        except Exception:
            return

    for r in state.get("binding_results", []) or []:
        seq = r.get("sequence")
        if not seq:
            continue

        if r.get("dock_score") is not None:
            store(seq, max(-float(r["dock_score"]), 0.0))
        elif r.get("hdock_score") is not None:
            store(seq, max(-float(r["hdock_score"]), 0.0))
        elif r.get("binding_rank_score") is not None:
            store(seq, max(-float(r["binding_rank_score"]), 0.0))
        elif r.get("vina_energy") is not None:
            store(seq, max(-float(r["vina_energy"]), 0.0))
        elif r.get("dg") is not None:
            store(seq, max(-float(r["dg"]), 0.0))
        elif r.get("binding_score") is not None:
            store(seq, float(r["binding_score"]))

    for m in state.get("md_results", []) or []:
        seq = m.get("sequence")
        res = m.get("result") or {}

        if not seq:
            continue

        if res.get("dock_score") is not None:
            store(seq, max(-float(res["dock_score"]), 0.0))
        elif res.get("vina_energy") is not None:
            store(seq, max(-float(res["vina_energy"]), 0.0))

    return lookup


def _normalise_conservation_signal(raw: Any) -> dict:
    if not raw:
        return {"valid": False}

    if isinstance(raw, str):
        return {"valid": False, "summary": raw}

    if not isinstance(raw, dict):
        return {"valid": False}

    if raw.get("valid") is False:
        return raw

    signal = raw.get("conservation_signal") if isinstance(raw.get("conservation_signal"), dict) else raw

    out = {
        "valid": bool(signal.get("valid", raw.get("valid", False))),
        "consensus": _clean_rna(signal.get("consensus", raw.get("consensus", ""))),
        "position_scores": signal.get("position_scores", raw.get("position_scores", [])) or [],
        "conserved_regions": signal.get("conserved_regions", raw.get("conserved_regions", [])) or [],
        "motif_scores": signal.get("motif_scores", raw.get("motif_scores", {})) or {},
        "selected_motifs": signal.get("selected_motifs", raw.get("selected_motifs", [])) or [],
        "conservation_pct": float(signal.get("conservation_pct", raw.get("conservation_pct", 0.0)) or 0.0),
        "summary": signal.get("summary", raw.get("summary", "")),
    }

    if out["position_scores"]:
        out["valid"] = True

    return out


def conservation_fitness(seq: str, conservation_signal: dict) -> float:
    seq = _clean_rna(seq)
    sig = _normalise_conservation_signal(conservation_signal)

    if not seq or not sig.get("valid"):
        return 0.0

    consensus = _clean_rna(sig.get("consensus", ""))
    pos_scores = sig.get("position_scores", []) or []

    if not consensus and not pos_scores:
        return 0.0

    total = 0.0
    weight_sum = 0.0

    for i, base in enumerate(seq):
        if i < len(pos_scores):
            try:
                w = float(pos_scores[i])
            except Exception:
                w = 0.0
        else:
            w = 0.25

        expected = consensus[i] if i < len(consensus) else None

        if expected in "ACGU":
            total += w if base == expected else -0.35 * w
            weight_sum += max(w, 0.05)

    base_score = max(0.0, min(1.0, total / max(weight_sum, 1e-6)))

    region_bonus = 0.0
    for region in sig.get("conserved_regions", []) or []:
        try:
            start, end = int(region[0]), int(region[1])
        except Exception:
            continue

        region_seq = seq[start:min(end, len(seq))]
        region_cons = consensus[start:min(end, len(seq))] if consensus else ""

        if region_seq and region_cons and len(region_seq) == len(region_cons):
            matches = sum(1 for a, b in zip(region_seq, region_cons) if a == b)
            region_bonus += 0.05 * matches / max(1, len(region_seq))

    motif_bonus = 0.0
    for motif, val in (sig.get("motif_scores", {}) or {}).items():
        motif_clean = _clean_rna(motif)
        if motif_clean and motif_clean in seq:
            try:
                motif_bonus += 0.08 * float(val)
            except Exception:
                motif_bonus += 0.04

    return max(0.0, min(1.5, base_score + region_bonus + motif_bonus))


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default

        f = float(x)

        if math.isnan(f) or math.isinf(f):
            return default

        return f

    except Exception:
        return default


def _dotbracket_pair_density(structure: str, seq_len: int) -> Optional[float]:
    if not structure or not isinstance(structure, str):
        return None

    paired = sum(1 for c in structure if c in "()[]{}<>")

    if seq_len <= 0:
        return None

    return paired / (2.0 * seq_len)


def _candidate_structures_from_sf(sf: dict) -> List[str]:
    structures: List[str] = []
    if not isinstance(sf, dict):
        return structures

    for key in ("structure",):
        s = sf.get(key)
        if isinstance(s, str) and s:
            structures.append(s)

    for key in ("structures", "cluster_structures"):
        vals = sf.get(key) or []
        if isinstance(vals, list):
            for s in vals:
                if isinstance(s, str) and s:
                    structures.append(s)

    return list(dict.fromkeys(structures))


def extract_fold_metrics(seq: str, sf: dict) -> dict:
    """
    Extract robust fold metrics.

    Important: some wrappers may report pair_density=0.0 for the primary
    structure while cluster_structures contain a plausible folded stem-loop.
    Therefore we compute both reported and best-observed pair density.
    """
    seq = _clean_rna(seq)
    n = max(1, len(seq))
    sf = sf or {}

    reported_pd = _safe_float(sf.get("pair_density"), None)
    mfe = _safe_float(sf.get("mfe"), None)

    structure_pds = []
    for structure in _candidate_structures_from_sf(sf):
        pd = _dotbracket_pair_density(structure, n)
        if pd is not None:
            structure_pds.append(pd)

    best_structure_pd = max(structure_pds) if structure_pds else None

    if reported_pd is None:
        effective_pd = best_structure_pd
    elif best_structure_pd is None:
        effective_pd = reported_pd
    else:
        effective_pd = max(reported_pd, best_structure_pd)

    mfe_per_nt = mfe / n if mfe is not None else None

    return {
        "mfe": mfe,
        "mfe_per_nt": mfe_per_nt,
        "reported_pair_density": reported_pd,
        "best_structure_pair_density": best_structure_pd,
        "effective_pair_density": effective_pd,
    }


def fold_threshold_status(seq: str, sf: dict) -> dict:
    metrics = extract_fold_metrics(seq, sf)
    reasons = []

    pd = metrics.get("effective_pair_density")
    mfe_per_nt = metrics.get("mfe_per_nt")

    if pd is None:
        reasons.append("missing_pair_density")
    elif pd < MIN_ACCEPT_PAIR_DENSITY:
        reasons.append(f"pair_density_below_min:{pd:.3f}<{MIN_ACCEPT_PAIR_DENSITY:.3f}")

    if mfe_per_nt is None:
        reasons.append("missing_mfe")
    elif mfe_per_nt > MIN_ACCEPT_MFE_PER_NT:
        reasons.append(f"mfe_per_nt_not_negative_enough:{mfe_per_nt:.3f}>{MIN_ACCEPT_MFE_PER_NT:.3f}")

    passed = len(reasons) == 0

    return {
        "passed": passed,
        "reasons": reasons,
        **metrics,
    }


def rna_design_stability_score(seq: str, sf: dict, md: Optional[dict] = None) -> Tuple[float, dict]:
    seq = _clean_rna(seq)
    n = max(1, len(seq))
    sf = sf or {}
    md = md or {}

    fold_metrics = extract_fold_metrics(seq, sf)

    mfe = fold_metrics.get("mfe")
    pair_density = fold_metrics.get("effective_pair_density")
    mfe_per_nt = fold_metrics.get("mfe_per_nt")

    if pair_density is None:
        pd_score = 0.25
    elif TARGET_PAIR_DENSITY_MIN <= pair_density <= TARGET_PAIR_DENSITY_MAX:
        pd_score = 1.0
    elif pair_density < TARGET_PAIR_DENSITY_MIN:
        pd_score = max(0.0, pair_density / max(TARGET_PAIR_DENSITY_MIN, 1e-6))
    else:
        pd_score = max(0.0, 1.0 - (pair_density - TARGET_PAIR_DENSITY_MAX))

    if mfe is None:
        mfe_score = 0.25
    else:
        if TARGET_MFE_PER_NT_MIN <= mfe_per_nt <= TARGET_MFE_PER_NT_MAX:
            mfe_score = 1.0
        elif mfe_per_nt > TARGET_MFE_PER_NT_MAX:
            mfe_score = max(0.0, 1.0 - abs(mfe_per_nt - TARGET_MFE_PER_NT_MAX) * 3.0)
        else:
            mfe_score = max(0.2, 1.0 - abs(mfe_per_nt - TARGET_MFE_PER_NT_MIN) * 1.2)

    entropy = _safe_float(sf.get("ensemble_entropy"), None)
    if entropy is None:
        ensemble_score = 0.5
    else:
        ensemble_score = max(0.0, min(1.0, 1.0 - min(abs(entropy), 80.0) / 100.0))

    md_stability = _safe_float(md.get("stability_index"), None)
    md_fluct = _safe_float(md.get("energy_fluctuation"), None)

    md_score = 0.5 if md_stability is None else max(0.0, min(1.0, md_stability))

    if md_fluct is None:
        fluct_penalty = 0.0
    else:
        fluct_penalty = max(0.0, min(0.35, (md_fluct - 180.0) / 400.0))

    gc = (seq.count("G") + seq.count("C")) / n
    gc_score = 1.0 - min(abs(gc - 0.50) / 0.35, 1.0)

    homopoly_penalty = 0.0
    for b in "ACGU":
        if b * (MAX_HOMOPOLYMER + 1) in seq:
            homopoly_penalty += 0.25

    threshold = fold_threshold_status(seq, sf)
    threshold_penalty = 0.0
    if ENFORCE_MIN_FOLD and not threshold["passed"]:
        threshold_penalty = min(0.65, 0.15 * len(threshold["reasons"]))

    score = (
        0.34 * pd_score
        + 0.28 * mfe_score
        + 0.12 * ensemble_score
        + 0.14 * md_score
        + 0.12 * gc_score
        - fluct_penalty
        - homopoly_penalty
        - threshold_penalty
    )

    details = {
        "mfe": mfe,
        "mfe_per_nt": mfe_per_nt,
        "pair_density": pair_density,
        "reported_pair_density": fold_metrics.get("reported_pair_density"),
        "best_structure_pair_density": fold_metrics.get("best_structure_pair_density"),
        "pd_score": pd_score,
        "mfe_score": mfe_score,
        "ensemble_score": ensemble_score,
        "md_score": md_score,
        "gc_score": gc_score,
        "fluct_penalty": fluct_penalty,
        "homopoly_penalty": homopoly_penalty,
        "threshold_penalty": threshold_penalty,
        "fold_threshold": threshold,
    }

    return max(0.0, min(1.5, score)), details


IUPAC = {
    "A": "A",
    "C": "C",
    "G": "G",
    "U": "U",
    "T": "U",
    "R": "AG",
    "Y": "CU",
    "S": "GC",
    "W": "AU",
    "K": "GU",
    "M": "AC",
    "B": "CGU",
    "D": "AGU",
    "H": "ACU",
    "V": "ACG",
    "N": "ACGU",
}


def parse_target_motifs() -> List[Tuple[str, float]]:
    motifs: List[Tuple[str, float]] = []

    for item in TARGET_MOTIFS_ENV.split(","):
        item = item.strip()
        if not item:
            continue

        if ":" in item:
            motif, weight = item.split(":", 1)
            try:
                w = float(weight)
            except Exception:
                w = 1.0
        else:
            motif, w = item, 1.0

        motif = motif.strip().upper().replace("T", "U")
        motif = "".join(c for c in motif if c in IUPAC)

        if motif:
            motifs.append((motif, max(0.0, w)))

    return motifs or [("RGAG", 1.0), ("RAAG", 1.0)]


def iupac_to_regex(motif: str) -> str:
    parts = []
    for c in motif.upper().replace("T", "U"):
        chars = IUPAC.get(c)
        if not chars:
            return ""
        if len(chars) == 1:
            parts.append(chars)
        else:
            parts.append(f"[{chars}]")
    return "".join(parts)


def find_motif_hits(seq: str, motif: str) -> List[Tuple[int, int, str]]:
    seq = _clean_rna(seq)
    regex = iupac_to_regex(motif)
    if not seq or not regex:
        return []

    hits: List[Tuple[int, int, str]] = []
    pattern = re.compile(f"(?=({regex}))")

    for m in pattern.finditer(seq):
        start = m.start()
        matched = m.group(1)
        hits.append((start, start + len(matched), matched))

    return hits


def _structure_context_score(structure: str, start: int, end: int) -> float:
    """
    Reward motifs exposed in or near a loop but embedded in a folded context.

    Useful for capsid/coat-protein RNA recognition where the recognized motif
    is commonly presented by a stem-loop rather than buried in a long duplex.
    """
    if not structure or start >= len(structure):
        return 0.25

    region = structure[start:min(end, len(structure))]
    if not region:
        return 0.25

    unpaired = region.count(".") / max(1, len(region))

    flank_start = max(0, start - MOTIF_CONTEXT_WINDOW)
    flank_end = min(len(structure), end + MOTIF_CONTEXT_WINDOW)
    flank = structure[flank_start:flank_end]
    paired_flank = sum(1 for c in flank if c in "()[]{}<>") / max(1, len(flank))

    # Best when motif is partly/exposed but has nearby paired stem context.
    exposure_score = 1.0 - abs(unpaired - 0.70)
    context_score = min(1.0, paired_flank / 0.45)

    return max(0.0, min(1.0, 0.6 * exposure_score + 0.4 * context_score))


def motif_selection_fitness(seq: str, sf: dict, conservation_signal: dict) -> Tuple[float, dict]:
    seq = _clean_rna(seq)
    if not seq:
        return 0.0, {"selected_motif": None, "motif_hits": []}

    sig = _normalise_conservation_signal(conservation_signal)
    pos_scores = sig.get("position_scores", []) or []
    cons_motif_scores = sig.get("motif_scores", {}) or {}

    structures = _candidate_structures_from_sf(sf)
    motifs = parse_target_motifs()

    best = {
        "score": 0.0,
        "motif": None,
        "start": None,
        "end": None,
        "matched": None,
        "weight": 0.0,
        "conservation": 0.0,
        "context": 0.0,
    }

    all_hits = []

    for motif, weight in motifs:
        hits = find_motif_hits(seq, motif)

        for start, end, matched in hits:
            if pos_scores and start < len(pos_scores):
                window = pos_scores[start:min(end, len(pos_scores))]
                pos_cons = sum(float(x) for x in window) / max(1, len(window))
            else:
                pos_cons = 0.35

            motif_cons = 0.0
            for key, val in cons_motif_scores.items():
                if key.upper().replace("T", "U") == motif:
                    try:
                        motif_cons = max(motif_cons, float(val))
                    except Exception:
                        pass

            if structures:
                context = max(_structure_context_score(s, start, end) for s in structures)
            else:
                context = 0.25

            # Core motif score:
            # - presence/weight
            # - conservation at positions and motif-level conservation
            # - structural presentation bonus
            raw = (
                0.45 * min(1.0, weight)
                + 0.25 * max(pos_cons, motif_cons)
                + 0.30 * context
            )

            weighted = raw * max(0.1, weight)

            hit_info = {
                "motif": motif,
                "start": start,
                "end": end,
                "matched": matched,
                "weight": weight,
                "position_conservation": round(pos_cons, 4),
                "motif_conservation": round(motif_cons, 4),
                "context": round(context, 4),
                "score": round(weighted, 4),
            }
            all_hits.append(hit_info)

            if weighted > best["score"]:
                best = {
                    "score": weighted,
                    "motif": motif,
                    "start": start,
                    "end": end,
                    "matched": matched,
                    "weight": weight,
                    "conservation": max(pos_cons, motif_cons),
                    "context": context,
                }

    if not all_hits:
        return 0.0, {"selected_motif": None, "motif_hits": []}

    score = max(0.0, min(1.5, best["score"]))
    details = {
        "selected_motif": best["motif"],
        "selected_motif_start": best["start"],
        "selected_motif_end": best["end"],
        "selected_motif_match": best["matched"],
        "selected_motif_weight": best["weight"],
        "selected_motif_conservation": round(best["conservation"], 4),
        "selected_motif_context": round(best["context"], 4),
        "motif_hits": sorted(all_hits, key=lambda x: x["score"], reverse=True)[:10],
    }

    return score, details


def run_hdock_once(seq: str, rna_pdb: str, target_pdb: str) -> Optional[float]:
    try:
        if not target_pdb:
            return None

        if not rna_pdb or not os.path.exists(rna_pdb):
            log.warning("[NSGA HDOCK] RNA PDB missing for %s", seq[:12])
            return None

        receptor_pdb = ensure_protein_pdb(target_pdb)

        if not receptor_pdb or not os.path.exists(receptor_pdb):
            try:
                prepare_protein(target_pdb)
            except Exception:
                pass

            receptor_pdb = ensure_protein_pdb(target_pdb)

        if not receptor_pdb or not os.path.exists(receptor_pdb):
            log.warning("[NSGA HDOCK] receptor PDB missing for target %s", target_pdb)
            return None

        ligand_pdb = prepare_rna_pdb_for_hdock(rna_pdb)

        if not ligand_pdb or not os.path.exists(ligand_pdb):
            log.warning("[NSGA HDOCK] failed to prepare RNA PDB for HDOCK: %s", rna_pdb)
            return None

        result = get_hdock().dock(
            receptor_pdb=receptor_pdb,
            ligand_pdb=ligand_pdb,
        )

        if result and result.get("valid") and result.get("dock_score") is not None:
            return float(result["dock_score"])

        log.warning("[NSGA HDOCK] no valid docking score for %s", seq[:12])
        return None

    except Exception as e:
        log.warning("[NSGA HDOCK] docking failed for %s: %s", seq[:12], e)
        return None


def run_system(
    topic: str,
    target_pdb: Optional[str] = None,
    state: Optional[dict] = None,
    extra_objectives: Optional[dict] = None,
):
    state = state or {}
    wrappers = state.get("wrappers", {})

    seq_len = infer_sequence_length(state)
    state["seq_len"] = seq_len

    sfold = wrappers.get("sfold") or SFoldWrapper()
    vienna = wrappers.get("vienna")

    raw_conservation = (
        state.get("conservation_signal")
        or state.get("conservation")
        or state.get("bioinfo_conservation")
        or {}
    )
    conservation_signal = _normalise_conservation_signal(raw_conservation)

    mutation_bias = state.get("mutation_bias") or build_initial_weights()
    dock_cache = state.setdefault("_dock_cache", {})

    binding_lookup = build_binding_lookup(state)
    md_lookup = build_md_lookup(state)

    log.info(
        "[NSGA] seq_len=%s conservation_valid=%s target_pdb=%s min_pd=%.3f min_mfe_nt=%.3f motifs=%s",
        seq_len,
        conservation_signal.get("valid"),
        target_pdb,
        MIN_ACCEPT_PAIR_DENSITY,
        MIN_ACCEPT_MFE_PER_NT,
        parse_target_motifs(),
    )

    def random_seq() -> str:
        for _ in range(100):
            try:
                seq = generate_structured_rna(length=seq_len)
            except Exception:
                seq = "".join(random.choice(NUCLEOTIDES) for _ in range(seq_len))

            seq = normalise_sequence(seq, seq_len)

            if seq and quick_sequence_filter(seq):
                return seq

        return "".join(random.choice(NUCLEOTIDES) for _ in range(seq_len))

    initial_sequences = []

    if state.get("target_sequence"):
        s_norm = normalise_sequence(state["target_sequence"], seq_len)
        if s_norm:
            initial_sequences.append(s_norm)

    for s in state.get("designed_sequences", []) or []:
        s_norm = normalise_sequence(s, seq_len)
        if s_norm:
            initial_sequences.append(s_norm)

    consensus = conservation_signal.get("consensus")
    if consensus:
        s_norm = normalise_sequence(consensus, seq_len)
        if s_norm:
            initial_sequences.append(s_norm)

    # Motif-seeded population members.
    for motif, _weight in parse_target_motifs():
        regexless = motif.replace("R", random.choice(["A", "G"])).replace("N", random.choice(NUCLEOTIDES))
        regexless = "".join(c if c in "ACGU" else random.choice(NUCLEOTIDES) for c in regexless)
        for _ in range(3):
            backbone = random_seq()
            if len(regexless) < seq_len:
                pos = random.randint(0, seq_len - len(regexless))
                seeded = backbone[:pos] + regexless + backbone[pos + len(regexless):]
                seeded = normalise_sequence(seeded, seq_len)
                if seeded:
                    initial_sequences.append(seeded)

    initial_sequences = deduplicate_sequences(initial_sequences, seq_len)

    while len(initial_sequences) < POP_SIZE:
        initial_sequences.append(random_seq())
        initial_sequences = deduplicate_sequences(initial_sequences, seq_len)

    class SeededSampling(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            unique = deduplicate_sequences(initial_sequences, seq_len)

            while len(unique) < n_samples:
                unique.append(random_seq())
                unique = deduplicate_sequences(unique, seq_len)

            encoded = [encode_sequence(s, seq_len) for s in unique[:n_samples]]
            return np.stack(encoded)

    class RNAProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=seq_len,
                n_obj=len(OBJECTIVES),
                xl=0,
                xu=3,
                type_var=int,
            )
            self.last_scores = {}
            self.fm = FailureMemory()

        def _evaluate(self, X, out, *args, **kwargs):
            penalty = 1e3

            for ind in X:
                seq = decode_sequence(ind)

                if not quick_sequence_filter(seq):
                    self.last_scores[seq] = {k: -penalty for k in OBJECTIVES}
                    self.last_scores[seq]["invalid_reason"] = "quick_sequence_filter_failed"
                    continue

                try:
                    sf = sfold.run_sfold(seq) if sfold else {}
                    if not isinstance(sf, dict):
                        sf = {}

                    if vienna:
                        try:
                            v = vienna.run_rnafold(seq)

                            if isinstance(v, dict) and v.get("valid"):
                                if v.get("mfe") is not None:
                                    sf["mfe"] = v.get("mfe")
                                if v.get("pair_density") is not None:
                                    sf["pair_density"] = v.get("pair_density")
                                if v.get("structure"):
                                    sf["structure"] = v.get("structure")

                        except Exception as ve:
                            log.debug("[NSGA] ViennaRNA failed for %s: %s", seq[:12], ve)

                    md_res = _sequence_match_lookup(seq, md_lookup) or _EMPTY_MD

                    score = score_sequence(
                        seq=seq,
                        sf=sf,
                        md=md_res,
                        weights=mutation_bias,
                        dock=None,
                        conservation_signal=conservation_signal,
                    )

                    stability_signal, stability_details = rna_design_stability_score(seq, sf, md_res)
                    cons_score = conservation_fitness(seq, conservation_signal)
                    motif_signal, motif_details = motif_selection_fitness(seq, sf, conservation_signal)

                    fold_status = stability_details.get("fold_threshold", fold_threshold_status(seq, sf))

                    score["thermo"] = float(score.get("thermo", 0.0)) + (
                        stability_signal * mutation_bias.get("thermo", 1.0)
                    )
                    score["structure"] = float(score.get("structure", 0.0)) + (
                        stability_signal * mutation_bias.get("structure", 1.0)
                    )
                    score["kinetic"] = float(score.get("kinetic", 0.0)) + (
                        0.5 * stability_signal * mutation_bias.get("kinetic", 1.0)
                    )
                    score["conservation"] = float(score.get("conservation", 0.0)) + (
                        cons_score * mutation_bias.get("conservation", 1.0)
                    )

                    score["motif"] = float(score.get("motif", 0.0)) + (
                        motif_signal * mutation_bias.get("motif", 1.0)
                    )

                    # Apply fold threshold penalties to fold-dependent objectives.
                    if ENFORCE_MIN_FOLD and not fold_status.get("passed", False):
                        score["thermo"] = float(score.get("thermo", 0.0)) - FOLD_THRESHOLD_PENALTY
                        score["structure"] = float(score.get("structure", 0.0)) - FOLD_THRESHOLD_PENALTY
                        score["kinetic"] = float(score.get("kinetic", 0.0)) - 0.5 * FOLD_THRESHOLD_PENALTY
                        score["binding"] = float(score.get("binding", 0.0)) - 0.25 * FOLD_THRESHOLD_PENALTY

                    binding_signal = _sequence_match_lookup(seq, binding_lookup)
                    if binding_signal is not None:
                        score["binding"] = (
                            np.tanh(float(binding_signal) / 10.0)
                            * mutation_bias.get("binding", 1.0)
                        )

                    score["diversity"] = (
                        len(set(seq)) / max(1, len(seq))
                    ) * mutation_bias.get("diversity", 1.0)

                    score["rna_stability"] = stability_signal
                    score["rna_stability_details"] = stability_details
                    score["fold_thresholds_passed"] = bool(fold_status.get("passed", False))
                    score["fold_threshold_reasons"] = fold_status.get("reasons", [])
                    score["conservation_raw"] = cons_score
                    score["motif_selection"] = motif_signal
                    score["motif_selection_details"] = motif_details

                    score = self.fm.adjust_score(seq, score)

                    if extra_objectives:
                        scale = 0.1 + 0.1 * state.get("iterations", 0)

                        for k, v in extra_objectives.items():
                            if k in OBJECTIVES:
                                score[k] = score.get(k, 0.0) + scale * float(v)

                    self.last_scores[seq] = score

                except Exception as e:
                    log.debug("[NSGA] scoring failed for %s: %s", seq[:12], e)
                    self.last_scores[seq] = {k: -penalty for k in OBJECTIVES}
                    self.last_scores[seq]["invalid_reason"] = str(e)

            current_seqs = [decode_sequence(ind) for ind in X]
            current_scores = {seq: self.last_scores.get(seq, {}) for seq in current_seqs}

            top_sequences = sorted(
                current_scores.items(),
                key=lambda item: (
                    item[1].get("thermo", 0.0)
                    + item[1].get("structure", 0.0)
                    + item[1].get("motif", 0.0)
                    + item[1].get("conservation", 0.0)
                    + item[1].get("rna_stability", 0.0)
                ),
                reverse=True,
            )[:SELECTIVE_DOCK_TOP_K]

            for seq, score in top_sequences:
                if not target_pdb:
                    continue

                if ENFORCE_MIN_FOLD and not score.get("fold_thresholds_passed", False):
                    log.debug("[NSGA HDOCK] skipping fold-invalid candidate %s: %s", seq[:12], score.get("fold_threshold_reasons"))
                    continue

                if _sequence_match_lookup(seq, binding_lookup) is not None:
                    continue

                md_res = _sequence_match_lookup(seq, md_lookup)

                if not md_res or not md_res.get("rna_pdb"):
                    continue

                cache_key = f"hdock:{target_pdb}:{seq}"

                if cache_key in dock_cache:
                    dock_score = dock_cache[cache_key]
                else:
                    dock_score = run_hdock_once(
                        seq=seq,
                        rna_pdb=md_res["rna_pdb"],
                        target_pdb=target_pdb,
                    )
                    dock_cache[cache_key] = dock_score

                if dock_score is not None:
                    binding_signal = max(-float(dock_score), 0.0)
                    binding_lookup[_norm_lookup_key(seq)] = binding_signal

                    self.last_scores[seq]["binding"] = (
                        np.tanh(binding_signal / 10.0)
                        * mutation_bias.get("binding", 1.0)
                    )
                    self.last_scores[seq]["dock_score"] = dock_score
                    self.last_scores[seq]["binding_units"] = "hdock_relative_score"

            out["F"] = np.array(
                [
                    [
                        -float(self.last_scores.get(decode_sequence(ind), {}).get(k, -penalty))
                        for k in OBJECTIVES
                    ]
                    for ind in X
                ]
            )

    problem = RNAProblem()

    algorithm_kwargs = {
        "pop_size": POP_SIZE,
        "sampling": SeededSampling(),
        "eliminate_duplicates": True,
    }

    if SBX is not None and PM is not None:
        repair = RoundingRepair() if RoundingRepair is not None else None
        algorithm_kwargs["crossover"] = SBX(prob=0.9, eta=15, repair=repair)
        algorithm_kwargs["mutation"] = PM(eta=20, repair=repair)

    algorithm = NSGA2(**algorithm_kwargs)

    try:
        res = minimize(
            problem,
            algorithm,
            ("n_gen", GENERATIONS),
            seed=int(os.getenv("VLAB_NSGA_SEED", "1")),
            verbose=False,
        )
    finally:
        clear_gpu()

    population = list(res.pop) if hasattr(res, "pop") else []

    # Final exact/near deduplication while preserving first encountered Pareto individuals.
    final_population = []
    final_seen = []

    for ind in population:
        seq = decode_sequence(ind.X)
        if not quick_sequence_filter(seq):
            continue

        duplicate = False
        for existing in final_seen:
            if seq == existing:
                duplicate = True
                break
            if DEDUP_NEAR_DUPLICATES and len(seq) == len(existing) and hamming_distance(seq, existing) <= NEAR_DUP_MAX_HAMMING:
                duplicate = True
                break

        if duplicate:
            continue

        final_seen.append(seq)
        final_population.append(ind)

    population_scores = []
    selected_motifs = []

    for ind in final_population:
        seq = decode_sequence(ind.X)
        score = problem.last_scores.get(seq)
        if score:
            population_scores.append(score)
            details = score.get("motif_selection_details") or {}
            if details.get("selected_motif"):
                selected_motifs.append(details)

    if population_scores:
        new_weights = update_weights(population_scores, mutation_bias)
    else:
        new_weights = mutation_bias

    state["_run_system_final_weights"] = new_weights
    state["_run_system_seq_len"] = seq_len
    state["_run_system_conservation_valid"] = conservation_signal.get("valid", False)
    state["_run_system_selected_motifs"] = selected_motifs[:20]
    state["_run_system_min_fold_thresholds"] = {
        "min_accept_pair_density": MIN_ACCEPT_PAIR_DENSITY,
        "min_accept_mfe_per_nt": MIN_ACCEPT_MFE_PER_NT,
        "enforce_min_fold": ENFORCE_MIN_FOLD,
    }

    return final_population


if __name__ == "__main__":
    run_system("RNA secondary structure stability")