"""
Virtual Lab Orchestrator
========================

A LangGraph-based multi-agent workflow for biophysics hypothesis generation,
evidence gathering, structural validation, and peer review.

Agents:
    PI          — proposes/refines hypotheses and sequence optimisation
    Researcher  — retrieves literature evidence (local FAISS → Semantic Scholar fallback)
    Bioinfo     — checks conservation across related genomes
    Structural  — runs SFold + ViennaRNA thermodynamic analysis
    MD          — runs RNA-only MD/coarse-grained dynamics
    Protein     — evaluates RNA-protein interaction / binding ranking
    Skeptic     — provides critical peer review

Usage:
    python3 virtual_lab_orchestrator.py --topic-file research_topics.yaml --topic-index 0 --max-iterations 3
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import random
import re
import sys
import time
from typing import List, Optional, Any

import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

os.environ["HF_HOME"] = "/mnt/scratch/fbsnpat/bot/VLAB2/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

vlab2_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if vlab2_root not in sys.path:
    sys.path.insert(0, vlab2_root)

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# ---------------------------------------------------------------------------
# VLAB imports
# ---------------------------------------------------------------------------
try:
    from VLAB2.core.rcsb_target_selector import select_pdb_targets
except Exception:
    select_pdb_targets = None

from VLAB2.core.rna_prep import prepare_rna_pdb_for_hdock
from VLAB2.optimisation.final_rna_design_system import run_system
from VLAB2.core.gpu_manager import clear_gpu
from VLAB2.core.sfold_wrapper import SFoldWrapper
from VLAB2.core.hdock_wrapper import HDockDocking
from VLAB2.core.protein_prep import prepare_protein
from VLAB2.core.protein_prep import CACHE_DIR as PROTEIN_CACHE_DIR
from VLAB2.optimisation.rna_grammar_generator import generate_structured_rna
from VLAB2.core.viennarna_wrapper import ViennaRNAWrapper
from VLAB2.core.md_wrapper import MDWrapper
from VLAB2.core.protein_wrapper import ProteinWrapper
from VLAB2.core.bioinfo_wrapper import BioinfoWrapper
from VLAB2.utils.pareto_analysis import analyse_pareto
from VLAB2.orchestration.failure_memory import FailureMemory
from VLAB2.research.research_agent_adaptive import (
    search_local_db,
    search_semantic_scholar,
    expand_knowledge,
)
from VLAB2.core.training_data_collector import TrainingDataCollector

from VLAB2.orchestration.state_schema import (
    LabState,
    safe_jsonable,
    add_conversation_entry,
    record_stage_output,
)

try:
    from optimisation_engine import optimise_sequences
except Exception:
    try:
        from optimisation.optimisation_engine import optimise_sequences
    except Exception:
        try:
            from optimisation.optimiser_engine import optimise_sequences
        except Exception:
            optimise_sequences = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

log = logging.getLogger("virtual_lab")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1")
MODEL_NAME = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-32B-Instruct")

data_collector = TrainingDataCollector()
hdock = HDockDocking()

_llm_instances: dict[float, ChatOpenAI] = {}


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """Return cached LLM client for the given temperature."""
    global _llm_instances

    if temperature not in _llm_instances:
        for attempt in range(3):
            try:
                model_kwargs = {
                    "frequency_penalty": float(os.getenv("VLLM_FREQ_PENALTY", "0.3")),
                    "presence_penalty": float(os.getenv("VLLM_PRES_PENALTY", "0.1")),
                    "top_p": float(os.getenv("VLLM_TOP_P", "0.9")),
                    "max_tokens": int(os.getenv("VLLM_MAX_NEW_TOKENS", "512")),
                }

                _llm_instances[temperature] = ChatOpenAI(
                    base_url=VLLM_URL,
                    api_key="empty",
                    model=MODEL_NAME,
                    temperature=temperature,
                    request_timeout=600,
                    model_kwargs=model_kwargs,
                )

                log.info(
                    "LLM client (temp=%.1f) initialised on attempt %d",
                    temperature,
                    attempt + 1,
                )
                break

            except Exception as e:
                log.warning("LLM init attempt %d/3 failed: %s", attempt + 1, e)
                time.sleep(5)

        if temperature not in _llm_instances:
            raise RuntimeError(
                f"Could not connect to vLLM server at {VLLM_URL} after 3 attempts."
            )

    return _llm_instances[temperature]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(state: LabState, path: str = "lab_checkpoint.json"):
    """Persist current state so a killed job can be resumed."""
    try:
        with open(path, "w") as f:
            json.dump(safe_jsonable(dict(state)), f, indent=2)
        log.debug("Checkpoint saved → %s", path)
    except Exception as e:
        log.warning("Failed to save checkpoint: %s", e)


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def critique_reports_high_spread(critique: str) -> bool:
    if not critique:
        return False

    max_spread = getenv_float("VLAB_MAX_ACCEPT_SCORE_SPREAD", 10.0)

    matches = re.findall(
        r"spread\s*=?\s*([-+]?\d+(?:\.\d+)?)",
        critique,
        flags=re.IGNORECASE,
    )

    for m in matches:
        try:
            if float(m) > max_spread:
                return True
        except Exception:
            pass

    if "high" in critique.lower() and "spread" in critique.lower():
        return True

    return False

def strip_metric_literals(text: str) -> str:
    """
    Remove overly specific numeric metric literals from critique text before
    sending it to the PI hypothesis-refinement LLM.

    This prevents hypotheses like:
    'score spread will be less than 68.99677938164322...'
    """
    if not text:
        return ""

    # Remove long decimals and inequality statements.
    text = re.sub(r"[-+]?\d+\.\d{2,}", "<metric_value>", text)
    text = re.sub(
        r"(less than|greater than|below|above|under|over)\s+[-+]?\d+(?:\.\d+)?",
        r"\1 <metric_value>",
        text,
        flags=re.IGNORECASE,
    )

    return text

def critique_signal_summary_for_pi(critique: str) -> str:
    """
    Convert detailed Skeptic critique into qualitative PI guidance.

    The goal is to preserve scientific direction without copying exact
    previous numerical thresholds into the next hypothesis.
    """
    if not critique:
        return "No prior critique available."

    c = critique.lower()

    signals = []

    if "spread" in c:
        signals.append(
            "Docking-score variability is too high; favour designs with more consistent HDOCK-relative ranks."
        )

    if "weak" in c or "support: weak" in c:
        signals.append(
            "Current evidence provides weak support; require stronger structural and docking consistency."
        )

    if "fold" in c or "pair_density" in c or "mfe" in c:
        signals.append(
            "Fold support is marginal; favour higher pair density and more favourable mfe_per_nt."
        )

    if "conservation" in c:
        signals.append(
            "Conservation support is limited; favour motifs and regions with stronger conservation signal."
        )

    if "md" in c or "trajectory" in c or "fluctuation" in c:
        signals.append(
            "MD stability is uncertain; favour candidates with lower fluctuation and stable RNA conformations."
        )

    if "redesign_sequences" in c:
        signals.append(
            "Skeptic recommends sequence redesign rather than accepting the current hypothesis."
        )

    if not signals:
        signals.append(strip_metric_literals(critique[:800]))

    return "\n".join(f"- {s}" for s in signals)


def hypothesis_contains_bad_metric_threshold(text: str) -> bool:
    """
    Detect hypotheses that copied exact numeric thresholds from previous results.
    """
    if not text:
        return False

    patterns = [
        r"(less than|greater than|below|above|under|over)\s*-?\d+(?:\.\d+)?",
        r"[<>]=?\s*-?\d+(?:\.\d+)?",
        r"\b\d+\.\d{3,}\b",
        r"score spread will be less than",
        r"best_binding_score will be lower than",
    ]

    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def fallback_qualitative_hypothesis(state: LabState) -> str:
    """
    Safe fallback hypothesis when the LLM copies exact thresholds.
    """
    topic = (
        state.get("topic_description")
        or state.get("research_topic")
        or "viral RNA motif docking"
    )

    return (
        "Redesigned conserved viral RNA stem-loop candidates with exposed packaging-like "
        "motifs will show stronger and less variable HDOCK-relative docking ranks against "
        "relevant viral capsid RNA-binding targets than prior candidates."
    )

def getenv_int(name: str, default: int, min_value: int | None = None) -> int:
    """
    Safely parse an integer environment variable.
    """
    raw = os.getenv(name, str(default))

    try:
        value = int(raw)
    except Exception:
        log.warning("Invalid integer for %s=%r; using default %d", name, raw, default)
        value = default

    if min_value is not None:
        value = max(min_value, value)

    return value


def getenv_float(name: str, default: float) -> float:
    """
    Safely parse a float environment variable.
    """
    raw = os.getenv(name, str(default))

    try:
        return float(raw)
    except Exception:
        log.warning("Invalid float for %s=%r; using default %.3f", name, raw, default)
        return default
    

def export_docking_outputs(state: LabState, path_prefix: str = "docking_summary") -> dict:
    """
    Export docking results to JSON, CSV, and Markdown for inspection.
    """
    import csv

    timestamp = int(time.time())

    json_path = f"{path_prefix}_{timestamp}.json"
    csv_path = f"{path_prefix}_{timestamp}.csv"
    md_path = f"{path_prefix}_{timestamp}.md"

    rows = []

    md_lookup = {}

    for m in state.get("md_results", []) or []:
        if isinstance(m, dict) and m.get("sequence"):
            md_lookup[m["sequence"]] = m.get("result", {}) or {}

    for r in state.get("binding_results", []) or []:
        if not isinstance(r, dict):
            continue

        seq = r.get("sequence", "")
        md_res = md_lookup.get(seq, {})

        rows.append(
            {
                "iteration": state.get("iterations"),
                "target_pdb": r.get("target_pdb") or state.get("target_pdb"),
                "sequence": seq,
                "rank": r.get("rank"),
                "dock_score": r.get("dock_score"),
                "binding_rank_score": r.get("binding_rank_score", r.get("dg")),
                "proxy_dg": r.get("proxy_dg"),
                "dock_valid": r.get("dock_valid"),
                "dock_method": r.get("dock_method"),
                "dock_error": r.get("dock_error"),
                "dock_output_file": r.get("dock_output_file"),
                "dock_complex_file": r.get("dock_complex_file"),
                "md_mean_energy": md_res.get("mean_energy"),
                "md_min_energy": md_res.get("min_energy"),
                "md_energy_fluctuation": md_res.get("energy_fluctuation"),
                "rna_pdb": md_res.get("rna_pdb"),
                "binding_units": r.get("binding_units", state.get("binding_units")),
            }
        )

    with open(json_path, "w") as f:
        json.dump(safe_jsonable(rows), f, indent=2)

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    with open(md_path, "w") as f:
        f.write("# Docking Summary\n\n")
        f.write(f"Target PDB: {state.get('target_pdb')}\n\n")
        f.write(f"Binding units: {state.get('binding_units', 'hdock_relative_score')}\n\n")

        for row in rows:
            f.write(
                f"- Rank {row.get('rank')} | "
                f"PDB={row.get('target_pdb')} | "
                f"Score={row.get('dock_score')} | "
                f"Seq={row.get('sequence', '')[:30]}... | "
                f"Complex={row.get('dock_complex_file')}\n"
            )

    return {
        "docking_summary_json": json_path,
        "docking_summary_csv": csv_path,
        "docking_summary_md": md_path,
    }

def clean_rna(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    return "".join(c for c in seq.strip().upper().replace("T", "U") if c in "ACGU")


def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(x != y for x, y in zip(a, b))


def dedupe_rna_sequences(
    sequences: list[str],
    max_hamming: Optional[int] = None,
) -> list[str]:
    """
    Exact and optional near-duplicate RNA sequence deduplication.
    """
    max_hamming = (
        int(os.getenv("VLAB_RNA_NEAR_DUP_MAX_HAMMING", "2"))
        if max_hamming is None
        else max_hamming
    )

    use_near = os.getenv("VLAB_RNA_DEDUP_NEAR_DUPLICATES", "1").strip() == "1"

    out = []
    seen = set()

    for seq in sequences or []:
        s = clean_rna(seq)

        if not s:
            continue

        if s in seen:
            continue

        if use_near:
            too_close = False
            for existing in out:
                if len(existing) == len(s) and hamming_distance(existing, s) <= max_hamming:
                    too_close = True
                    break

            if too_close:
                continue

        seen.add(s)
        out.append(s)

    return out


def extract_fold_quality_from_outputs(
    sfold_output: dict | None = None,
    vienna_output: dict | None = None,
    alifold_output: dict | None = None,
) -> dict:
    """
    Normalise folding quality fields from SFold/Vienna/RNAalifold outputs.
    """
    sfold_output = sfold_output or {}
    vienna_output = vienna_output or {}
    alifold_output = alifold_output or {}

    pair_density_values = []
    mfe_values = []
    mfe_per_nt_values = []
    reasons = []

    for label, obj in [
        ("sfold", sfold_output),
        ("vienna", vienna_output),
        ("alifold", alifold_output),
    ]:
        if not isinstance(obj, dict):
            continue

        pd = obj.get("pair_density")
        mfe = obj.get("mfe")
        mfe_per_nt = obj.get("mfe_per_nt")

        try:
            if pd is not None:
                pair_density_values.append(float(pd))
        except Exception:
            pass

        try:
            if mfe is not None:
                mfe_values.append(float(mfe))
        except Exception:
            pass

        try:
            if mfe_per_nt is not None:
                mfe_per_nt_values.append(float(mfe_per_nt))
        except Exception:
            pass

        if obj.get("threshold_reasons"):
            reasons.extend(
                [f"{label}:{x}" for x in obj.get("threshold_reasons", [])]
            )

        if obj.get("passes_min_fold_thresholds") is False:
            reasons.append(f"{label}:failed_min_fold_thresholds")

    best_pair_density = max(pair_density_values) if pair_density_values else None
    best_mfe = min(mfe_values) if mfe_values else None
    best_mfe_per_nt = min(mfe_per_nt_values) if mfe_per_nt_values else None

    min_pd = float(os.getenv("VLAB_RNA_MIN_ACCEPT_PAIR_DENSITY", "0.24"))
    min_mfe_per_nt = float(os.getenv("VLAB_RNA_MIN_ACCEPT_MFE_PER_NT", "-0.08"))

    passed = True

    if best_pair_density is None:
        passed = False
        reasons.append("missing_pair_density")
    elif best_pair_density < min_pd:
        passed = False
        reasons.append(
            f"pair_density_below_min:{best_pair_density:.3f}<{min_pd:.3f}"
        )

    if best_mfe_per_nt is None:
        passed = False
        reasons.append("missing_mfe_per_nt")
    elif best_mfe_per_nt > min_mfe_per_nt:
        passed = False
        reasons.append(
            f"mfe_per_nt_not_negative_enough:{best_mfe_per_nt:.3f}>{min_mfe_per_nt:.3f}"
        )

    return {
        "passed": passed,
        "best_pair_density": best_pair_density,
        "best_mfe": best_mfe,
        "best_mfe_per_nt": best_mfe_per_nt,
        "min_pair_density": min_pd,
        "min_mfe_per_nt": min_mfe_per_nt,
        "reasons": list(dict.fromkeys(reasons)),
    }

def motif_summary_from_state(state: LabState) -> str:
    motifs = []

    conservation_signal = state.get("conservation_signal", {}) or {}

    for item in conservation_signal.get("selected_motifs", []) or []:
        if isinstance(item, dict) and item.get("motif"):
            motifs.append(
                f"{item.get('motif')}@{item.get('start')}-{item.get('end')}"
            )

    for item in state.get("_run_system_selected_motifs", []) or []:
        if isinstance(item, dict) and item.get("selected_motif"):
            motifs.append(
                f"{item.get('selected_motif')}@"
                f"{item.get('selected_motif_start')}-"
                f"{item.get('selected_motif_end')}"
            )

    motifs = list(dict.fromkeys(motifs))
    return ", ".join(motifs[:8]) if motifs else "None"

def safe_binding_rank(r: dict):
    value = r.get("binding_rank_score", r.get("dg"))
    if value is None:
        return float("inf")
    return value


def safe_dg(r: dict):
    """
    Legacy compatibility sorter.

    NOTE:
    In HDOCK mode, r['dg'] is a compatibility alias for binding_rank_score,
    not a physical kcal/mol free energy.
    """
    value = r.get("dg")
    if value is None:
        return float("inf")
    return value


def parse_pdb_candidates(text: str) -> list[str]:
    candidates = []
    blacklist = {"NONE", "NULL", "N/A"}

    for line in text.splitlines():
        line = line.strip().upper()
        matches = re.findall(r"\b[0-9][A-Z0-9]{3}\b", line)
        for m in matches:
            if m not in blacklist:
                candidates.append(m)

    return list(dict.fromkeys(candidates))


def truncate_str(text: str, max_chars: int = 1000) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [TRUNCATED — conclusions must not rely on omitted content]"


def clean_llm_output(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\\\\", "\\")

    for token in [
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|begin_of_text|>",
    ]:
        text = text.replace(token, "")

    return text.strip()


def clean_structural_output(text: str) -> str:
    text = text.replace("\\\\", "\\")
    text = " ".join(text.split())
    return text


def adjust_weights(parsed, weights):
    """
    Adaptive objective-weight update from Skeptic output.

    HDOCK mode:
      parsed['dg_best'] is treated as a relative docking/rank score,
      not kcal/mol.
    """
    new = dict(weights)

    for k in ["binding", "kinetic", "thermo", "structure"]:
        new[k] = new.get(k, 1.0)

    dg_best = parsed.get("dg_best")

    if dg_best is not None:
        try:
            dg_best = float(dg_best)

            # HDOCK/rank-score heuristic:
            # less negative / weakly favourable scores increase binding pressure.
            if dg_best > -50:
                new["binding"] *= 1.3

        except Exception:
            pass

    missing_controls = parsed.get("missing_controls", [])

    if isinstance(missing_controls, str):
        missing_controls_text = missing_controls.lower()
    else:
        missing_controls_text = " ".join(map(str, missing_controls)).lower()

    if "md trajectory" in missing_controls_text or "md" in missing_controls_text:
        new["kinetic"] *= 1.2

    total = sum(new.values()) or 1.0
    return {k: v / total for k, v in new.items()}


def normalise_lit_query(q: str) -> str:
    if not q:
        return ""

    q = q.strip().strip('"').strip("'")
    for ch in [":", ";", ",", "(", ")"]:
        q = q.replace(ch, " ")

    q = " ".join(q.split())
    words = q.split()

    if len(words) > 10:
        q = " ".join(words[:10])

    return q


def keep_research_query(q: str) -> bool:
    if not q:
        return False

    t = q.lower()

    required_any_rna = [
        "rna",
        "stem-loop",
        "stem loop",
        "ires",
        "utr",
    ]

    domain_anchor = [
        "virus",
        "viral",
        "capsid",
        "coat protein",
        "packaging",
        "assembly",
        "rna binding",
        "rna-protein",
        "rna protein",
        "docking",
    ]

    forbidden = [
        "multi-agent",
        "language model",
        "llm",
        "framework",
        "manufacturing",
        "pose graph",
        "robot",
        "slam",
        "education",
        "clinical trial",
        "jailbreak",
        "glass",
        "phosphate glass",
        "plantaricin",
        "cancer drug",
        "anti-cancer",
        "anticancer",
        "dna secondary structure",
        "thermal melting curves",
        "algorithm for dna",
    ]

    return (
        any(x in t for x in required_any_rna)
        and any(x in t for x in domain_anchor)
        and not any(x in t for x in forbidden)
    )


def keep_literature_text(title: str, abstract: str = "") -> bool:
    text = f"{title or ''} {abstract or ''}".lower()

    if not text.strip():
        return False

    required_any_rna = [
        "rna",
        "stem-loop",
        "stem loop",
        "ires",
        "utr",
        "ribonucleic",
    ]

    domain_anchor = [
        "virus",
        "viral",
        "capsid",
        "coat protein",
        "packaging",
        "virion",
        "assembly",
        "rna-binding",
        "rna binding",
        "rna-protein",
        "rna protein",
    ]

    forbidden = [
        "multi-agent",
        "large language model",
        "llm",
        "manufacturing",
        "pose graph",
        "robot",
        "slam",
        "jailbreak",
        "clinical trial multi-agent",
        "phosphate glass",
        "plantaricin",
        "anti-cancer drug",
        "anticancer drug",
    ]

    return (
        any(x in text for x in required_any_rna)
        and any(x in text for x in domain_anchor)
        and not any(x in text for x in forbidden)
    )


def fallback_literature_query(hypothesis: str = "") -> str:
    h = (hypothesis or "").lower()

    if "hpev" in h or "parechovirus" in h:
        return "HPeV1 RNA stem-loop viral capsid binding"

    if "picornaviridae" in h:
        return "Picornaviridae RNA stem-loop capsid packaging"

    return "viral RNA stem-loop capsid binding"


def validate_sequence(seq: str) -> tuple[bool, str]:
    seq = clean_rna(seq)

    if not seq or len(seq) < 12:
        return False, "Sequence too short"

    max_repeat = 1
    current_repeat = 1

    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current_repeat += 1
            max_repeat = max(max_repeat, current_repeat)
        else:
            current_repeat = 1

    max_homopolymer = int(os.getenv("VLAB_RNA_MAX_HOMOPOLYMER", "4"))

    if max_repeat > max_homopolymer:
        return False, f"Too many repeated nucleotides ({max_repeat})"

    if any(c not in "ACGU" for c in seq):
        return False, "Sequence contains invalid characters"

    if len(set(seq)) < 3:
        return False, "Sequence uses fewer than three nucleotide types"

    gc = (seq.count("G") + seq.count("C")) / len(seq)

    if gc < 0.30 or gc > 0.80:
        return False, f"GC content out of range: {gc:.0%} (require 30–80%)"

    return True, "OK"

def validate_text_block(text: str) -> tuple[bool, str]:
    if not text or len(text.strip()) < 10:
        return False, "Text too short"
    return True, "OK"


def mutate_sequence(
    seq: str,
    num_mutations: int = 2,
    locked_positions: Optional[set] = None,
) -> str:
    seq = list(seq)
    locked_positions = locked_positions or set()

    mutable_positions = [i for i in range(len(seq)) if i not in locked_positions]

    if not mutable_positions:
        return "".join(seq)

    for _ in range(num_mutations):
        i = random.choice(mutable_positions)
        choices = [n for n in "ACGU" if n != seq[i]]
        seq[i] = random.choice(choices)

    return "".join(seq)


# ---------------------------------------------------------------------------
# Wrapper construction
# ---------------------------------------------------------------------------

def build_wrapper_bundle() -> dict:
    """Build wrappers once at startup."""
    md = MDWrapper()

    try:
        protein = ProteinWrapper(simrna=md)
    except TypeError:
        protein = ProteinWrapper()

    return {
        "protein": protein,
        "md": md,
        "sfold": SFoldWrapper(),
        "vienna": ViennaRNAWrapper(),
        "bioinfo": BioinfoWrapper(),
        "hdock": hdock,
    }


# ---------------------------------------------------------------------------
# PI Agent
# ---------------------------------------------------------------------------

def pi_agent(state: LabState) -> dict:
    log.info("--- PI AGENT (ADAPTIVE MULTI-OBJECTIVE) ---")

    wrappers = state.get("wrappers", {})
    target_pdb = state.get("target_pdb")
    if (target_pdb and os.getenv("VLAB_TARGET_RESET_ON_HIGH_SPREAD", "1").strip() == "1"
    and critique_reports_high_spread(state.get("critique", "") or "")):
        log.info(
            "Resetting target_pdb=%s because Skeptic reported high docking-score spread.",
            target_pdb,
        )

    failed_targets = list(state.get("failed_target_pdbs", []))
    failed_targets.append(str(target_pdb).upper())

    state["failed_target_pdbs"] = list(dict.fromkeys(failed_targets))
    state["target_pdb"] = None
    target_pdb = None
    
    sequences = dedupe_rna_sequences(state.get("designed_sequences", []))
    critique = state.get("critique", "") or ""

    if not wrappers:
        return {
            "pi_summary": "No wrappers available.",
            "optimisation_status": "no_wrappers",
            "iterations": state.get("iterations", 0) + 1,
        }

    if not target_pdb and sequences:
        log.info(
            "No protein target yet — running sequence-only optimisation without docking."
        )

    if not sequences:
        return {
            "pi_summary": "Bootstrap: waiting for structural design.",
            "optimisation_status": "bootstrap",
            "iterations": state.get("iterations", 0),
        }

    try:
        log.info("Running NSGA-II optimisation (final_rna_design_system)...")

        memory_model = FailureMemory()

        for past in state.get("failure_memory", []):
            memory_model.update(past)

        failure_weights = memory_model.compute_failure_weights()
        log.info("Failure memory weights: %s", failure_weights)

        refined_hypothesis = None

        parsed = {
            "dg_best": None,
            "missing_controls": [],
            "concerns": [],
        }

        extra_objectives = {}

        # ------------------------------------------------------------
        # Parse critique and optionally refine hypothesis
        # ------------------------------------------------------------
        if critique:
            from VLAB2.orchestration.objective_injector import generate_objectives
            from VLAB2.orchestration.skeptic_parser import parse_skeptic_output

            parsed = parse_skeptic_output(critique)
            extra_objectives = generate_objectives(parsed)

            try:
                llm = get_llm(temperature=0.0)

                energy_line = ""
                recommendation_line = ""

                for line in critique.splitlines():
                    if line.startswith("ENERGY:"):
                        energy_line = line.replace("ENERGY:", "").strip()
                    if line.startswith("RECOMMENDATION:"):
                        recommendation_line = line.replace("RECOMMENDATION:", "").strip()
                
                qualitative_critique = critique_signal_summary_for_pi(critique)
                
                prompt = [
                    SystemMessage(
                        content=(
                            "You are a computational biophysics Principal Investigator.\n\n"
                            "Rewrite the hypothesis as a single falsifiable qualitative prediction "
                            "grounded in the evidence below.\n\n"
                            "Output exactly one sentence.\n"
                            "Use HDOCK-relative score terminology when binding data are from HDOCK.\n"
                            "Do not call HDOCK scores kcal/mol.\n"
                            "Do not include exact numeric cutoffs or previous metric values.\n"
                            "Do not write inequalities such as '<', '>', 'less than 9.793', "
                            "or 'lower than -71.865'.\n"
                            "Prefer comparative language such as stronger, weaker, less variable, "
                            "more conserved, more stable, or improved.\n"
                            "Do not use may, could, potentially, or might."
                        )
                    ),

                    HumanMessage(
                        content=(
                            f"Previous hypothesis:\n"
                            f"{truncate_str(state.get('hypothesis', ''), 600)}\n\n"
                            f"Skeptic recommendation: {recommendation_line or 'N/A'}\n\n"
                            f"Qualitative critique signals, with exact metric thresholds removed:\n"
                            f"{qualitative_critique}\n\n"
                            f"Write a qualitative comparative hypothesis. "
                            f"Do not include exact numeric cutoffs, exact score-spread values, "
                            f"or exact previous best_binding_score values."
                        )
                    ),
                ]

                resp = llm.invoke(prompt)
                candidate = clean_llm_output(resp.content)

                if candidate:
                    if hypothesis_contains_bad_metric_threshold(candidate):
                        log.warning(
                            "PI hypothesis copied exact metric thresholds; replacing with qualitative fallback."
                        )
                        candidate = fallback_qualitative_hypothesis(state)

                    refined_hypothesis = candidate
                    log.info("Refined hypothesis: %s", truncate_str(candidate, 200))

            except Exception as e:
                log.warning("Hypothesis refinement failed: %s", e)

        topic = (
            refined_hypothesis
            or state.get("hypothesis")
            or state.get("topic_description")
            or state.get("research_topic", "RNA secondary structure stability")
        )

        conservation = state.get("conservation_signal", {}) or {}
        conservation_fitness = conservation.get("conservation_fitness", 0)

        # ------------------------------------------------------------
        # Adaptive mutation/objective pressure
        # ------------------------------------------------------------
        state["mutation_bias"] = adjust_weights(parsed, state.get("mutation_bias", {}))
        state["conservation_fitness"] = conservation_fitness
        state["conserved_regions"] = conservation.get("conserved_regions", [])

        combined_objectives = {
            "conservation": conservation_fitness,
        }

        if critique:
            combined_objectives.update(extra_objectives)

        # Fold/structure pressure if current fold failed.
        if (
            state.get("fold_thresholds_passed") is False
            and os.getenv("VLAB_RNA_ENFORCE_MIN_FOLD", "1").strip() == "1"
        ):
            combined_objectives["structure"] = combined_objectives.get("structure", 0.0) + 1.0
            combined_objectives["thermo"] = combined_objectives.get("thermo", 0.0) + 0.6
            combined_objectives["diversity"] = combined_objectives.get("diversity", 0.0) + 0.3

        # Docking-spread feedback from Skeptic critique.
        # Logs show high HDOCK score spread is the dominant rejection reason.
        score_spread = None

        try:
            m = re.search(
                r"spread\s*=\s*([-+]?\d+(?:\.\d+)?)",
                critique,
                flags=re.IGNORECASE,
            )
            if m:
                score_spread = float(m.group(1))
        except Exception:
            score_spread = None

        max_accept_spread = getenv_float("VLAB_MAX_ACCEPT_SCORE_SPREAD", 10.0)

        if score_spread is not None and score_spread > max_accept_spread:
            log.info(
                "High binding score spread %.3f detected; increasing diversity/structure pressure.",
                score_spread,
            )
            combined_objectives["diversity"] = combined_objectives.get("diversity", 0.0) + 0.8
            combined_objectives["structure"] = combined_objectives.get("structure", 0.0) + 0.5
            combined_objectives["binding"] = combined_objectives.get("binding", 0.0) + 0.2

        for k, v in failure_weights.items():
            if k == "binding_pressure":
                combined_objectives["binding"] = combined_objectives.get("binding", 0.0) + v
            elif k == "structure_pressure":
                combined_objectives["structure"] = combined_objectives.get("structure", 0.0) + v
            elif k == "diversity_pressure":
                combined_objectives["diversity"] = combined_objectives.get("diversity", 0.0) + v
            elif k == "convergence_pressure":
                combined_objectives["diversity"] = combined_objectives.get("diversity", 0.0) + v
            elif k.startswith("concern"):
                if "entropy" in k or "structure" in k:
                    combined_objectives["structure"] = combined_objectives.get("structure", 0.0) + v

        total = sum(v for v in combined_objectives.values() if v is not None)

        if total > 0:
            combined_objectives = {
                k: (v / total if v is not None else 0.0)
                for k, v in combined_objectives.items()
            }

        combined_objectives["diversity"] = max(
            combined_objectives.get("diversity", 0.0),
            0.05,
        )

        # ------------------------------------------------------------
        # Run optimiser
        # ------------------------------------------------------------
        population = run_system(
            topic=topic,
            target_pdb=target_pdb,
            state=state,
            extra_objectives=combined_objectives,
        )

        if population is None or not hasattr(population, "__len__") or len(population) == 0:
            log.warning("Empty population returned - using fallback")

            fallback_seq = sequences[0] if sequences else ""

            return {
                "pi_summary": "No viable NSGA-II population produced; using fallback sequence.",
                "optimisation_status": "empty_population",
                "designed_sequences": [fallback_seq] if fallback_seq else [],
                "target_sequence": state.get("target_sequence") or fallback_seq or None,
                "structural_candidates": state.get("structural_candidates", []),
                "mutation_bias": state.get("mutation_bias", {}),
                "hypothesis": topic,
                "iterations": state.get("iterations", 0) + 1,
            }

        from VLAB2.optimisation.final_rna_design_system import decode_sequence as _decode

        analysis = analyse_pareto(population, _decode)

        best_idx = analysis["best_idx"]
        extreme_idxs = analysis["extremes"]
        selected_indices = set([best_idx] + extreme_idxs)

        top_sequences = []
        seen = set()

        for i, ind in enumerate(population):
            if i in selected_indices:
                seq = clean_rna(_decode(ind.X))

                if seq and seq not in seen:
                    seen.add(seq)
                    top_sequences.append(seq)

        if not top_sequences:
            if population:
                fallback = clean_rna(_decode(population[0].X))
                top_sequences = [fallback] if fallback else []
            else:
                top_sequences = []

        if not top_sequences:
            fallback_seq = sequences[0] if sequences else ""

            return {
                "pi_summary": "No viable sequences produced.",
                "optimisation_status": "empty_population",
                "designed_sequences": [fallback_seq] if fallback_seq else [],
                "target_sequence": state.get("target_sequence") or fallback_seq or None,
                "structural_candidates": state.get("structural_candidates", []),
                "mutation_bias": state.get("mutation_bias", {}),
                "hypothesis": topic,
                "iterations": state.get("iterations", 0) + 1,
            }

        # Conservation-dependent exploration/exploitation.
        if conservation_fitness > 0.7:
            top_sequences = top_sequences[:3]
        elif conservation_fitness < 0.2:
            log.warning("Low conservation — forcing exploitation phase")
            top_sequences = top_sequences[:2]

        # Preserve old candidates too, but dedupe.
        top_sequences = dedupe_rna_sequences(top_sequences)

        # ------------------------------------------------------------
        # Adaptive bias cleanup
        # ------------------------------------------------------------
        new_bias = dict(state.get("_run_system_final_weights", state.get("mutation_bias", {})) or {})
        selected_motifs = state.get("_run_system_selected_motifs", [])
        min_fold_thresholds = state.get("_run_system_min_fold_thresholds", {})

        new_bias["conservation"] = max(new_bias.get("conservation", 0.0), 0.1)

        if "thermo" in new_bias and new_bias["thermo"] > 0.8:
            new_bias["thermo"] *= 0.9

        if new_bias:
            required = ["thermo", "binding", "conservation", "structure", "diversity"]

            for k in required:
                new_bias[k] = new_bias.get(k, 0.0)

            for k in list(new_bias.keys()):
                new_bias[k] = max(float(new_bias[k]), 0.05)

            for k in list(new_bias.keys()):
                if new_bias[k] > 0.75:
                    new_bias[k] = 0.75

            temp = 0.7 + 0.3 * (
                state.get("iterations", 0) / max(1, state.get("max_iterations", 3))
            )

            new_bias = {k: v ** temp for k, v in new_bias.items()}

            total = sum(new_bias.values())

            if total > 0:
                new_bias = {k: v / total for k, v in new_bias.items()}

        # ------------------------------------------------------------
        # Training data capture
        # ------------------------------------------------------------
        try:
            summary_pop = data_collector.summarise_population(population, _decode)
            data_collector.capture_optimisation_step(
                population_summary=summary_pop,
                selected_sequences=top_sequences,
                mutation_bias=new_bias,
                metadata={
                    "target_pdb": target_pdb,
                    "iteration": state.get("iterations", 0),
                    "binding_units": state.get("binding_units", "hdock_relative_score"),
                    "score_spread": score_spread,
                    "combined_objectives": combined_objectives,
                },
            )
        except Exception as log_err:
            log.warning("Failed to log optimisation step: %s", log_err)

        summary = (
            f"NSGA-II optimisation complete.\n"
            f"Target PDB: {target_pdb}\n"
            f"Selected {len(top_sequences)} sequences.\n"
            f"Conservation fitness: {float(conservation_fitness):.3f}\n"
            f"Selected motifs: {motif_summary_from_state(state)}\n"
            f"Min fold thresholds: {min_fold_thresholds or 'default'}\n"
            f"Binding units: {state.get('binding_units', 'hdock_relative_score')}\n"
            f"Adaptive mutation: {'enabled' if new_bias else 'none'}\n"
            f"Score spread feedback: {score_spread if score_spread is not None else 'N/A'}"
        )

        # Keep previous target_sequence unless absent.
        target_sequence = state.get("target_sequence")

        if not target_sequence and top_sequences:
            target_sequence = top_sequences[0]

        return {
            "pi_summary": summary,
            "optimisation_status": "adaptive_pareto_optimised",
            "designed_sequences": top_sequences,
            "target_sequence": target_sequence,
            "structural_candidates": state.get("structural_candidates", []),
            "mutation_bias": new_bias,
            "hypothesis": topic,
            "iterations": state.get("iterations", 0) + 1,
            "_run_system_selected_motifs": selected_motifs,
            "_run_system_min_fold_thresholds": min_fold_thresholds,
        }

    except Exception as e:
        log.exception("PI optimisation failed")

        fallback_seq = sequences[0] if sequences else ""

        return {
            "pi_summary": f"Fallback due to: {e}",
            "optimisation_status": "fallback",
            "designed_sequences": [fallback_seq] if fallback_seq else [],
            "target_sequence": state.get("target_sequence") or fallback_seq or None,
            "structural_candidates": state.get("structural_candidates", []),
            "mutation_bias": state.get("mutation_bias", {}),
            "hypothesis": state.get("hypothesis", ""),
            "iterations": state.get("iterations", 0) + 1,
        }
    
# ---------------------------------------------------------------------------
# Researcher Agent
# ---------------------------------------------------------------------------

def researcher_agent(state: LabState) -> dict:
    try:
        from VLAB2.orchestration.skeptic_parser import parse_skeptic_output
        from VLAB2.research.literature_refiner import generate_refined_queries
        from VLAB2.research.streaming_literature_agent import start_streaming, rerank

        log.info("--- RESEARCHER AGENT: Querying Knowledge Base ---")

        streamed = set(state.get("streamed_queries", []))
        streaming_started = bool(state.get("streaming_started", False))

        seed_questions = state.get("seed_questions", [])
        idx = int(state.get("current_question_idx", 0) or 0)

        if seed_questions:
            idx = max(0, min(idx, len(seed_questions) - 1))
            hypothesis = seed_questions[idx]
        else:
            hypothesis = (
                state.get("hypothesis")
                or state.get("topic_description")
                or state.get("research_topic", "")
            )

        query_msg = get_llm(temperature=0.0).invoke(
            [
                SystemMessage(
                    content=(
                        "You are a biomedical literature specialist generating "
                        "PubMed-style keyword search queries.\n\n"
                        "Rules:\n"
                        "- 5–8 words, no punctuation, no quotes, no operators\n"
                        "- Must contain RNA\n"
                        "- Must contain at least one viral/capsid/packaging concept\n"
                        "- Must include organism or viral family if named\n"
                        "- Prefer: RNA stem-loop, viral capsid, RNA packaging, "
                        "coat protein, RNA binding, Picornaviridae, HPeV1\n"
                        "- Avoid: framework, multi-agent, LLM, manufacturing, clinical trial\n\n"
                        "Return ONLY the query string."
                    )
                ),
                HumanMessage(content=hypothesis),
            ]
        )

        query = normalise_lit_query(query_msg.content)

        if not keep_research_query(query):
            log.warning("Initial query rejected by filter: %s", query)
            query = fallback_literature_query(hypothesis)

        log.info("    Search query: %s", query)

        started_any_stream = False

        if keep_research_query(query) and query not in streamed:
            start_streaming(query)
            streamed.add(query)
            streaming_started = True
            started_any_stream = True
        elif query in streamed:
            log.info("Streaming already started for query: %s", query)

        parsed = parse_skeptic_output(state.get("critique", ""))

        try:
            refined_queries = generate_refined_queries(parsed, hypothesis=hypothesis)
        except TypeError:
            refined_queries = generate_refined_queries(parsed)

        refined_queries = [normalise_lit_query(q) for q in refined_queries if q]
        refined_queries = [q for q in refined_queries if q]

        accepted_refined = 0

        for q in refined_queries:
            if accepted_refined >= 3:
                break

            if not keep_research_query(q):
                log.info("Skipping broad/refined query: %s", q)
                continue

            if q in streamed:
                log.info("Refined query already streamed: %s", q)
                continue

            start_streaming(q)
            streamed.add(q)
            streaming_started = True
            started_any_stream = True
            accepted_refined += 1

        
        if started_any_stream:
            time.sleep(float(os.getenv("VLAB_STREAM_WARMUP_SECONDS", "6.0")))


        try:
            local_results = search_local_db(query)
        except Exception as e:
            log.warning("Local DB search failed for query '%s': %s", query, e)
            local_results = []

        if local_results:
            local_results = rerank(query, local_results, top_k=7)
        else:
            try:
                fresh_results = search_local_db(query)
            except Exception as e:
                log.warning("Fresh local DB search failed for query '%s': %s", query, e)
                fresh_results = []

            local_results = rerank(query, fresh_results, top_k=7) if fresh_results else []

        filtered_local_results = []

        for doc in local_results:
            title = doc.metadata.get("title", "Unknown")
            abstract = doc.page_content or ""

            if keep_literature_text(title, abstract):
                filtered_local_results.append(doc)
            else:
                log.info("Filtered local DB result as off-topic: %s", str(title)[:120])

        local_results = filtered_local_results[:7]
        evidence: List[str] = []

        if local_results:
            log.info("Found %d relevant papers in local DB.", len(local_results))

            for doc in local_results:
                abstract = doc.page_content or ""
                title = doc.metadata.get("title", "Unknown")

                evidence.append(
                    f"Title: {title} | Abstract: {abstract[:300]}..."
                )

        else:
            log.info("Nothing found in local DB. Falling back to Semantic Scholar...")

            try:
                ss_results = search_semantic_scholar(query, limit=10)
            except Exception as e:
                log.warning("Semantic Scholar fallback failed for query '%s': %s", query, e)
                ss_results = []

            for paper in ss_results:
                title = paper.get("title", "Unknown")
                abstract = paper.get("abstract") or ""

                if not keep_literature_text(title, abstract):
                    log.info(
                        "Filtered Semantic Scholar result as off-topic: %s",
                        str(title)[:120],
                    )
                    continue

                evidence.append(
                    f"Title: {title} | Abstract: {abstract[:300]}..."
                )

        existing_evidence = state.get("evidence", []) or []

        seen = set()
        deduped = []

        def evidence_key(item: str) -> str:
            title = item.split(" | ")[0].replace("Title: ", "").strip().lower()
            title = re.sub(r"[^a-z0-9]+", " ", title)
            return " ".join(title.split())

        for item in existing_evidence:
            key = evidence_key(item)
            if key:
                seen.add(key)

        for item in evidence:
            key = evidence_key(item)

            if not key or key in seen:
                continue

            seen.add(key)
            deduped.append(item)

        # Keep older evidence plus new unique evidence, capped.
        evidence = (existing_evidence + deduped)[-20:]
        result = {
            "evidence": evidence,
            "streaming_started": streaming_started,
            "streamed_queries": list(streamed),
            "stage_outputs": [
                record_stage_output(
                    state,
                    "researcher",
                    "\n\n".join(evidence),
                    summary=f"Collected {len(evidence)} evidence items",
                )
            ],
            "conversation_history": [
                add_conversation_entry(
                    state,
                    "assistant",
                    "Evidence collected.",
                    "researcher",
                )
            ],
        }

        data_collector.capture_agent_interaction(
            "Researcher",
            "Gather literature evidence",
            "\n\n".join(evidence),
        )

        save_checkpoint({**state, **result})
        return result

    except Exception as e:
        log.exception("RESEARCHER AGENT ERROR")

        return {
            "evidence": [f"Researcher failed: {e}"],
            "streaming_started": state.get("streaming_started", False),
            "streamed_queries": list(state.get("streamed_queries", [])),
        }


# ---------------------------------------------------------------------------
# Structural Agent
# ---------------------------------------------------------------------------
def structural_agent(state: LabState) -> dict:
    log.info("--- STRUCTURAL AGENT: Running Simulations ---")

    try:
        prior_seqs = dedupe_rna_sequences(state.get("designed_sequences", []))
        critique = state.get("critique", "") or ""

        exclusion_block = ""

        if prior_seqs:
            exclusion_block = (
                "\nPreviously designed sequences (must not duplicate):\n"
                + "\n".join(f"- {s}" for s in prior_seqs[-10:])
            )

        redesign_note = ""

        if critique:
            for line in critique.splitlines():
                if line.startswith("RECOMMENDATION:") or line.startswith("CONCERNS:"):
                    redesign_note += line + "\n"

            if redesign_note:
                redesign_note = f"\nSkeptic guidance:\n{redesign_note.strip()}"

        design_msg = get_llm(temperature=0.35).invoke(
            [
                SystemMessage(
                    content=(
                        "You are an RNA structural biologist designing candidate "
                        "functional RNA sequences.\n\n"

                        "Design 3–5 DIVERSE RNA sequences, each 25–40 nt, satisfying:\n\n"

                        "Core requirements:\n"
                        "1. Each sequence forms a stem-loop with an exposed recognition motif\n"
                        "2. Each sequence must include ONE motif from: RGAG, RAAG, GAG, AAG, CAG, GUC, AGU\n"
                        "3. GC content must be 40–60%\n"
                        "4. No homopolymer runs of 4 or more\n"
                        "5. Predicted MFE < -5 kcal/mol and pair_density >= 0.24\n"
                        "6. Distinct from previous sequences\n\n"

                        "Diversity constraints, mandatory:\n"
                        "- Sequences must differ from each other by at least 5 positions where possible\n"
                        "- Use at least two different motif types across the set\n"
                        "- Vary motif position, loop length, or stem length across candidates\n"
                        "- Avoid trivial single-mutation variants of the same design\n\n"

                        "Design strategy guidance:\n"
                        "- Vary stem length between 6–10 paired bases\n"
                        "- Vary loop size between 3–8 nt\n"
                        "- Place motifs either in the loop centre or at the stem-loop junction\n"
                        "- Include at least one stability-prioritised candidate and one motif-exposure-prioritised candidate\n\n"

                        "Output EXACTLY this repeated format, with no extra text:\n"
                        "MOTIF: <motif>@<start-end>\n"
                        "SEQ: <uppercase RNA sequence>\n\n"
                        "MOTIF: <motif>@<start-end>\n"
                        "SEQ: <uppercase RNA sequence>"
                    )
                ),
                HumanMessage(
                    content=(
                        (state.get("hypothesis") or state.get("research_topic", ""))
                        + exclusion_block
                        + redesign_note
                    )
                ),
            ]
        )

        # ------------------------------------------------------------
        # Parse 3–5 candidate sequences from LLM output
        # ------------------------------------------------------------

        raw_sequences = []

        for line in design_msg.content.strip().splitlines():
            if line.upper().startswith("SEQ:"):
                seq = line.split(":", 1)[1].strip().upper().replace("T", "U")
                seq = "".join(c for c in seq if c in "ACGU")

                if seq:
                    raw_sequences.append(seq)

        # Fallback if model did not follow format.
        if not raw_sequences:
            tokens = re.findall(r"[ACGUTacgut]{25,40}", design_msg.content)
            for token in tokens:
                seq = token.upper().replace("T", "U")
                seq = "".join(c for c in seq if c in "ACGU")
                if seq:
                    raw_sequences.append(seq)

        raw_sequences = dedupe_rna_sequences(raw_sequences)

        conserved_regions = state.get("conserved_regions", [])
        locked_positions = set()

        for start, end in conserved_regions:
            locked_positions.update(range(start, end))

        final_sequences = []

        for seq in raw_sequences:
            is_valid, msg = validate_sequence(seq)

            if not is_valid:
                log.warning(
                    "Invalid designed sequence (%s) — replacing with grammar-generated candidate",
                    msg,
                )

                try:
                    seq = generate_structured_rna(length=30)
                except Exception:
                    seq = "".join(random.choice("ACGU") for _ in range(30))

                seq = mutate_sequence(seq, 2, locked_positions)

            seq = clean_rna(seq)

            if not seq:
                continue

            if seq in prior_seqs:
                seq = mutate_sequence(seq, 2, locked_positions)

            if seq:
                final_sequences.append(seq)

        final_sequences = dedupe_rna_sequences(final_sequences)

        # Ensure at least 3 candidates if possible.
        attempts = 0
        while len(final_sequences) < 3 and attempts < 25:
            attempts += 1

            try:
                seq = generate_structured_rna(length=30)
            except Exception:
                seq = "".join(random.choice("ACGU") for _ in range(30))

            seq = clean_rna(seq)

            is_valid, _ = validate_sequence(seq)

            if is_valid:
                final_sequences.append(seq)
                final_sequences = dedupe_rna_sequences(final_sequences)

        final_sequences = final_sequences[:5]

        if not final_sequences:
            raise RuntimeError("No valid RNA sequences generated by structural agent.")

        log.info(
            "    Generated %d candidate RNA sequences: %s",
            len(final_sequences),
            ", ".join(final_sequences),
        )

        sfold = state["wrappers"]["sfold"]
        vienna = state["wrappers"]["vienna"]

        msa = state.get("msa_data", "")
        alifold_output = ""

        if msa and ">" in msa:
            try:
                if hasattr(vienna, "run_rnaalifold"):
                    alifold_output = vienna.run_rnaalifold(msa)
                elif (
                    "bioinfo" in state.get("wrappers", {})
                    and hasattr(state["wrappers"]["bioinfo"], "run_rnaalifold")
                ):
                    alifold_output = state["wrappers"]["bioinfo"].run_rnaalifold(msa)
                else:
                    alifold_output = ""
            except Exception as e:
                log.warning("RNAalifold failed/non-fatal: %s", e)
                alifold_output = ""

        # ------------------------------------------------------------
        # Fold all candidates and select best candidate for target_sequence
        # ------------------------------------------------------------

        candidate_records = []

        def _add_mfe_per_nt_if_missing(obj: dict, seq: str) -> dict:
            if not isinstance(obj, dict):
                return {}

            out = dict(obj)

            if out.get("mfe_per_nt") is None and out.get("mfe") is not None:
                try:
                    out["mfe_per_nt"] = float(out["mfe"]) / max(1, len(seq))
                except Exception:
                    pass

            return out

        def _rank_fold_quality(fq: dict) -> float:
            pd = fq.get("best_pair_density")
            mfe_nt = fq.get("best_mfe_per_nt")

            pd_score = 0.0 if pd is None else min(1.0, max(0.0, float(pd) / 0.35))

            if mfe_nt is None:
                mfe_score = 0.0
            else:
                # More negative is better up to a useful cap.
                mfe_score = min(1.0, max(0.0, abs(float(mfe_nt)) / 0.30))

            pass_bonus = 1.0 if fq.get("passed") else 0.0

            return 0.45 * pd_score + 0.35 * mfe_score + 0.20 * pass_bonus

        for seq in final_sequences:
            try:
                sfold_output = sfold.run_sfold(seq)
            except Exception as e:
                log.warning("SFold failed for %s: %s", seq[:15], e)
                sfold_output = {}

            try:
                vienna_output = vienna.run_rnafold(seq)
            except Exception as e:
                log.warning("ViennaRNA failed for %s: %s", seq[:15], e)
                vienna_output = {}

            sfold_output = _add_mfe_per_nt_if_missing(
                sfold_output if isinstance(sfold_output, dict) else {},
                seq,
            )
            vienna_output = _add_mfe_per_nt_if_missing(
                vienna_output if isinstance(vienna_output, dict) else {},
                seq,
            )

            conservation_signal = state.get("conservation_signal", {}) or {}
            conservation_fitness = float(conservation_signal.get("conservation_fitness", 0.0) or 0.0)

            alifold_min_conservation = getenv_float(
                "VLAB_ALIFOLD_MIN_CONSERVATION_FITNESS",
                0.5,
            )

            alifold_advisory_when_low_conservation = (
                os.getenv("VLAB_ALIFOLD_ADVISORY_WHEN_LOW_CONSERVATION", "1").strip() == "1"
            )

            use_alifold_for_hard_gate = (
                isinstance(alifold_output, dict)
                and (
                    conservation_fitness >= alifold_min_conservation
                    or not alifold_advisory_when_low_conservation
                )
            )

            fold_quality = extract_fold_quality_from_outputs(
                sfold_output=sfold_output,
                vienna_output=vienna_output,
                alifold_output=alifold_output if use_alifold_for_hard_gate else {},
            )

            if isinstance(alifold_output, dict) and not use_alifold_for_hard_gate:
                fold_quality.setdefault("advisory_warnings", [])
                fold_quality["advisory_warnings"].append(
                    {
                        "source": "alifold",
                        "reason": (
                            "RNAalifold excluded from hard fold gate because conservation "
                            f"fitness={conservation_fitness:.3f} < {alifold_min_conservation:.3f}"
                        ),
                        "alifold_pair_density": alifold_output.get("pair_density"),
                        "alifold_mfe_per_nt": alifold_output.get("mfe_per_nt"),
                        "alifold_threshold_reasons": alifold_output.get("threshold_reasons", []),
                    }
                )

            candidate_records.append(
                {
                    "sequence": seq,
                    "sfold_output": sfold_output,
                    "vienna_output": vienna_output,
                    "fold_quality": fold_quality,
                    "rank_score": _rank_fold_quality(fold_quality),
                }
            )

        candidate_records.sort(
            key=lambda r: (
                bool(r["fold_quality"].get("passed")),
                float(r.get("rank_score", 0.0)),
            ),
            reverse=True,
        )

        selected = candidate_records[0]
        sequence = selected["sequence"]
        sfold_output = selected["sfold_output"]
        vienna_output = selected["vienna_output"]
        fold_quality = selected["fold_quality"]

        log.info(
            "    Selected target candidate (%d nt): %s | fold_pass=%s | pd=%s | mfe_nt=%s",
            len(sequence),
            sequence,
            fold_quality.get("passed"),
            fold_quality.get("best_pair_density"),
            fold_quality.get("best_mfe_per_nt"),
        )

        clean_sfold = clean_structural_output(str(sfold_output))
        clean_vienna = clean_structural_output(str(vienna_output))
        clean_alifold = clean_structural_output(str(alifold_output))

        candidate_summary_lines = []

        for i, rec in enumerate(candidate_records, start=1):
            fq = rec["fold_quality"]
            candidate_summary_lines.append(
                (
                    f"{i}. {rec['sequence']} | "
                    f"passed={fq.get('passed')} | "
                    f"pair_density={fq.get('best_pair_density')} | "
                    f"mfe_per_nt={fq.get('best_mfe_per_nt')} | "
                    f"reasons={fq.get('reasons', [])}"
                )
            )

        candidate_summary = "\n".join(candidate_summary_lines)

        interpret_msg = get_llm(temperature=0.0).invoke(
            [
                SystemMessage(
                    content=(
                        "You are reviewing RNA folding simulation outputs.\n\n"
                        "Respond in EXACTLY this format:\n"
                        "STABILITY: [STABLE/UNSTABLE/MARGINAL] — MFE=<value or N/A> kcal·mol⁻¹, pair_density=<value or N/A>, ensemble_diversity=<value or N/A>\n"
                        "MOTIF: [PRESENT/ABSENT/AMBIGUOUS] — <motif and nt positions or N/A>\n"
                        "CONSERVATION: [LIKELY/UNLIKELY/UNKNOWN] — <reason or N/A>\n"
                        "ALIFOLD: [SUPPORTED/UNSUPPORTED/NO_DATA] — <evidence or N/A>\n"
                        "VERDICT: [PROCEED/REDESIGN] — <one sentence citing a number>"
                    )
                ),
                HumanMessage(
                    content=(
                        f"Selected sequence ({len(sequence)} nt): {sequence}\n\n"
                        f"All candidate fold summary:\n{truncate_str(candidate_summary, 1500)}\n\n"
                        f"Selected SFold output:\n{truncate_str(clean_sfold, 1500)}\n\n"
                        f"Selected ViennaRNA output:\n{truncate_str(clean_vienna, 1000)}\n\n"
                        f"RNAalifold output:\n{truncate_str(clean_alifold, 1000)}"
                    )
                ),
            ]
        )

        interpreted = clean_llm_output(interpret_msg.content)
        ok, _ = validate_text_block(interpreted)

        if not ok:
            interpreted = (
                "STABILITY: N/A\n"
                "MOTIF: N/A\n"
                "CONSERVATION: UNKNOWN — insufficient data\n"
                "ALIFOLD: NO_DATA\n"
                "VERDICT: REDESIGN — structural outputs could not be parsed"
            )

        analysis = (
            f"Structural Analysis for selected sequence {sequence}\n\n"
            f"{interpreted}\n\n"
            f"All generated candidates:\n"
            f"{candidate_summary}\n\n"
            f"Selected fold threshold status:\n"
            f"{json.dumps(safe_jsonable(fold_quality), indent=2)}\n\n"
            f"Raw selected SFold:\n{truncate_str(clean_sfold, 1200)}\n\n"
            f"Raw selected ViennaRNA:\n{truncate_str(clean_vienna, 800)}\n\n"
            f"Raw RNAalifold:\n{truncate_str(clean_alifold, 800)}"
        )

        result = {
            "structural_analysis": analysis,

            # Return 3–5 candidates to seed PI/NSGA-II.
            "designed_sequences": final_sequences,

            # Selected best-folding candidate for MD/protein path.
            "target_sequence": sequence,

            "fold_quality": fold_quality,
            "fold_thresholds_passed": fold_quality.get("passed", False),
            "fold_threshold_reasons": fold_quality.get("reasons", []),

            # Optional but useful for debugging/training.
            "structural_candidates": [
                {
                    "sequence": rec["sequence"],
                    "fold_quality": rec["fold_quality"],
                    "rank_score": rec["rank_score"],
                }
                for rec in candidate_records
            ],

            "stage_outputs": [
                record_stage_output(
                    state,
                    "structural",
                    analysis,
                    summary=(
                        f"Generated {len(final_sequences)} RNA candidates; "
                        f"selected best fold candidate with "
                        f"pair_density={fold_quality.get('best_pair_density')}, "
                        f"mfe_per_nt={fold_quality.get('best_mfe_per_nt')}"
                    ),
                    metadata={
                        "selected_sequence": sequence,
                        "candidate_count": len(final_sequences),
                        "designed_sequences": final_sequences,
                        "fold_thresholds_passed": fold_quality.get("passed", False),
                        "pair_density": fold_quality.get("best_pair_density"),
                        "mfe_per_nt": fold_quality.get("best_mfe_per_nt"),
                    },
                )
            ],
            "conversation_history": [
                add_conversation_entry(state, "assistant", analysis, "structural")
            ],
        }

        data_collector.capture_agent_interaction(
            "Structural",
            "Run multi-candidate structural analysis",
            analysis,
        )

        save_checkpoint({**state, **result})
        return result

    except Exception as e:
        log.exception("STRUCTURAL AGENT ERROR")
        return {"structural_analysis": f"Structural analysis failed: {e}"}
    
# ---------------------------------------------------------------------------
# MD Agent
# ---------------------------------------------------------------------------
def md_agent(state: LabState) -> dict:
    log.info("--- MD AGENT: Evaluating Structural Stability ---")

    try:
        # ------------------------------------------------------------
        # Candidate selection
        # ------------------------------------------------------------
        eval_top_n = getenv_int("VLAB_AGENT_EVAL_TOP_N", 3, min_value=1)

        sequences = []

        # Preserve original priority: target_sequence first.
        if state.get("target_sequence"):
            sequences.append(state["target_sequence"])

        # Add all designed sequences from structural/PI stages.
        sequences.extend(state.get("designed_sequences", []) or [])

        # Add structural candidates if present from multi-candidate structural agent.
        for candidate in state.get("structural_candidates", []) or []:
            if isinstance(candidate, dict) and candidate.get("sequence"):
                sequences.append(candidate["sequence"])

        sequences = dedupe_rna_sequences(sequences)[:eval_top_n]

        # ------------------------------------------------------------
        # Fold-aware gate
        # ------------------------------------------------------------
        if (
            state.get("fold_thresholds_passed") is False
            and os.getenv("VLAB_RNA_ENFORCE_MIN_FOLD", "1").strip() == "1"
        ):
            structural_candidates = state.get("structural_candidates", []) or []

            any_candidate_passed = any(
                isinstance(candidate, dict)
                and isinstance(candidate.get("fold_quality"), dict)
                and candidate["fold_quality"].get("passed")
                for candidate in structural_candidates
            )

            # If no structural_candidates are available, preserve strict original gate.
            # If structural_candidates exist, skip only when all failed.
            if not any_candidate_passed:
                reasons = state.get("fold_threshold_reasons", []) or []

                summary = (
                    "MD skipped because RNA failed minimum fold thresholds: "
                    + "; ".join(map(str, reasons))
                )

                result = {
                    "md_analysis": summary,
                    "md_results": [],
                    "stage_outputs": [
                        record_stage_output(
                            state,
                            "md",
                            summary,
                            summary="MD skipped: fold thresholds failed",
                            metadata={
                                "fold_thresholds_passed": False,
                                "fold_threshold_reasons": reasons,
                                "candidate_count": len(sequences),
                            },
                        )
                    ],
                    "conversation_history": [
                        add_conversation_entry(state, "assistant", summary, "md")
                    ],
                }

                save_checkpoint({**state, **result})
                return result

            log.info(
                "Selected RNA failed fold thresholds, but at least one structural "
                "candidate passed; continuing MD on candidate set."
            )

        # ------------------------------------------------------------
        # No sequence guard
        # ------------------------------------------------------------
        if not sequences:
            summary = "No sequences available for MD simulation."

            result = {
                "md_analysis": summary,
                "md_results": [],
                "stage_outputs": [
                    record_stage_output(
                        state,
                        "md",
                        summary,
                        summary="MD skipped: no sequences",
                        metadata={"evaluated_sequences": 0},
                    )
                ],
                "conversation_history": [
                    add_conversation_entry(state, "assistant", summary, "md")
                ],
            }

            save_checkpoint({**state, **result})
            return result

        log.info(
            "MD evaluating %d RNA sequence(s): %s",
            len(sequences),
            ", ".join(seq[:20] for seq in sequences),
        )

        md = state["wrappers"]["md"]

        results = []
        valid_results = []

        # ------------------------------------------------------------
        # Run MD
        # ------------------------------------------------------------
        for seq in sequences:
            try:
                output = md.run_md(seq)
                entry = {"sequence": seq, "result": output}
                results.append(entry)

                if output and output.get("valid"):
                    valid_results.append(entry)
                else:
                    err = output.get("error", "unknown error") if output else "no output"
                    log.warning(
                        "MD invalid result for sequence %s: %s",
                        seq[:20],
                        err,
                    )

            except Exception as e:
                log.exception("MD failed for sequence %s", seq[:20])
                results.append(
                    {
                        "sequence": seq,
                        "result": {
                            "valid": False,
                            "error": str(e),
                        },
                    }
                )

            finally:
                clear_gpu()

        # ------------------------------------------------------------
        # Handle all-failed case
        # ------------------------------------------------------------
        if not valid_results:
            failure_lines = []

            for r in results[:5]:
                err = (r.get("result") or {}).get("error", "unknown error")
                failure_lines.append(f"{r['sequence'][:20]}... → {err}")

            summary = "All MD simulations failed."

            if failure_lines:
                summary += "\n" + "\n".join(failure_lines)

            result = {
                "md_analysis": summary,
                "md_results": results,
                "stage_outputs": [
                    record_stage_output(
                        state,
                        "md",
                        summary,
                        summary="MD failed for all evaluated sequences",
                        metadata={
                            "evaluated_sequences": len(results),
                            "valid_results": 0,
                        },
                    )
                ],
                "conversation_history": [
                    add_conversation_entry(state, "assistant", summary, "md")
                ],
            }

            save_checkpoint({**state, **result})
            return result

        # ------------------------------------------------------------
        # Summarise valid MD results
        # ------------------------------------------------------------
        lines = []

        for r in valid_results[:5]:
            lines.append(
                f"{r['sequence'][:20]}... → "
                f"{truncate_str(str(r['result']), 160)}"
            )

        summary_text = "\n".join(lines)

        result = {
            "md_analysis": summary_text,
            "md_results": valid_results,
            "stage_outputs": [
                record_stage_output(
                    state,
                    "md",
                    summary_text,
                    summary="RNA-only MD stability evaluation",
                    metadata={
                        "evaluated_sequences": len(results),
                        "valid_results": len(valid_results),
                    },
                )
            ],
            "conversation_history": [
                add_conversation_entry(state, "assistant", summary_text, "md")
            ],
        }

        save_checkpoint({**state, **result})
        return result

    except Exception as e:
        log.exception("MD AGENT ERROR")

        return {
            "md_analysis": f"MD analysis failed: {e}",
            "md_results": [],
        }

    finally:
        clear_gpu()

def protein_agent(state: LabState) -> dict:
    log.info("--- PROTEIN AGENT: Binding Evaluation ---")

    try:
        require_docking = os.getenv("VLAB_REQUIRE_DOCKING", "1").strip() == "1"
        docking_backend = os.getenv("VLAB_DOCKING_BACKEND", "hdock").strip().lower()
        eval_top_n = getenv_int("VLAB_AGENT_EVAL_TOP_N", 3, min_value=1)

        if docking_backend != "hdock":
            log.warning(
                "Unsupported VLAB_DOCKING_BACKEND=%s; falling back to hdock.",
                docking_backend,
            )
            docking_backend = "hdock"

        log.info("[DOCKING CONFIG] VLAB_REQUIRE_DOCKING=%s", require_docking)
        log.info("[DOCKING CONFIG] VLAB_DOCKING_BACKEND=%s", docking_backend)

        sequences = []

        if state.get("target_sequence"):
            sequences.append(state["target_sequence"])

        sequences.extend(state.get("designed_sequences", []) or [])

        for c in state.get("structural_candidates", []) or []:
            if isinstance(c, dict) and c.get("sequence"):
                sequences.append(c["sequence"])

        sequences = dedupe_rna_sequences(sequences)[:eval_top_n]
        sequences = [s for s in sequences if s]

        log.info("[DOCKING DEBUG] Sequences for evaluation: %d", len(sequences))

        # ------------------------------------------------------------
        # Fold-aware gate
        # ------------------------------------------------------------
        if (
            state.get("fold_thresholds_passed") is False
            and os.getenv("VLAB_RNA_ENFORCE_MIN_FOLD", "1").strip() == "1"
        ):
            structural_candidates = state.get("structural_candidates", []) or []

            any_passed = any(
                isinstance(c, dict)
                and isinstance(c.get("fold_quality"), dict)
                and c["fold_quality"].get("passed")
                for c in structural_candidates
            )

            if not any_passed:
                reasons = state.get("fold_threshold_reasons", [])

                summary = (
                    "Protein docking skipped because all RNA candidates failed minimum "
                    "fold thresholds: "
                    + "; ".join(map(str, reasons))
                )

                return {
                    "protein_analysis": summary,
                    "binding_results": [],
                    "target_pdb": state.get("target_pdb"),
                    "target_pdb_candidates": state.get("target_pdb_candidates", []),
                    "failed_target_pdbs": state.get("failed_target_pdbs", []),
                    "stage_outputs": [
                        record_stage_output(
                            state,
                            "protein",
                            summary,
                            summary="Protein docking skipped: fold thresholds failed",
                            metadata={
                                "fold_thresholds_passed": False,
                                "fold_threshold_reasons": reasons,
                                "candidate_count": len(sequences),
                                "require_docking": require_docking,
                                "docking_backend": docking_backend,
                            },
                        )
                    ],
                    "conversation_history": [
                        add_conversation_entry(state, "assistant", summary, "protein")
                    ],
                }

            log.info(
                "Global selected fold failed, but at least one structural candidate passed; "
                "continuing protein evaluation on candidate set."
            )

        if not sequences:
            summary = "No sequences available for protein analysis."

            return {
                "protein_analysis": summary,
                "binding_results": [],
                "stage_outputs": [
                    record_stage_output(
                        state,
                        "protein",
                        summary,
                        summary="Protein analysis skipped: no sequences",
                        metadata={"sequence_count": 0},
                    )
                ],
                "conversation_history": [
                    add_conversation_entry(state, "assistant", summary, "protein")
                ],
            }

        pw = state["wrappers"]["protein"]

        candidate_pdbs: List[str] = []
        selection_reason = ""

        if state.get("target_pdb"):
            candidate_pdbs.append(str(state["target_pdb"]).upper())
            selection_reason = "reused_existing_target"

        elif state.get("target_pdb_candidates"):
            candidate_pdbs.extend(
                [str(p).upper() for p in state["target_pdb_candidates"]]
            )
            selection_reason = "reused_candidate_list"

        else:
            target_selection_mode = os.getenv(
                "VLAB_TARGET_SELECTION_MODE",
                "llm_fallback",
            ).strip().lower()

            topic_text = (
                state.get("hypothesis")
                or state.get("topic_description")
                or state.get("research_topic")
                or ""
            )

            if target_selection_mode == "rcsb_dynamic" and select_pdb_targets is not None:
                try:
                    ranked_targets = select_pdb_targets(
                        topic=state.get("research_topic", ""),
                        hypothesis=topic_text,
                        virus_name=state.get("virus_name", ""),
                        virus_family=state.get("virus_family", ""),
                        exclude_pdbs=state.get("failed_target_pdbs", []),
                    )

                    state["target_pdb_rankings"] = ranked_targets

                    candidate_pdbs.extend(
                        [x["pdb_id"] for x in ranked_targets if x.get("pdb_id")]
                    )

                    selection_reason = "rcsb_dynamic_ranked"

                    log.info(
                        "RCSB dynamic target candidates: %s",
                        ", ".join(candidate_pdbs),
                    )

                except Exception as e:
                    log.warning("RCSB dynamic target selection failed: %s", e)

            if not candidate_pdbs:
                lookup_msg = get_llm(temperature=0.0).invoke(
                    [
                        SystemMessage(
                            content=(
                                "You are a structural biologist selecting protein targets "
                                "from the RCSB PDB for RNA docking.\n\n"
                                "Return exactly 5 PDB IDs, one per line, no other text.\n"
                                "Prefer experimentally resolved viral capsid or coat-protein "
                                "RNA complexes, RNA packaging, capsid assembly, or RNA stem-loop "
                                "binding systems.\n"
                                "Fallback: 2HW8."
                            )
                        ),
                        HumanMessage(content=topic_text),
                    ]
                )

                candidate_pdbs.extend(parse_pdb_candidates(lookup_msg.content))

                failed_set = {str(x).upper() for x in state.get("failed_target_pdbs", [])}

                candidate_pdbs = [
                    p for p in candidate_pdbs
                    if p.upper() not in failed_set
                ]

                selection_reason = "llm_generated_candidates"

        extra = os.getenv("VLAB_FALLBACK_PDBS", "")

        if extra.strip():
            env_candidates = [x.strip().upper() for x in extra.split(",") if x.strip()]
            log.info("[DOCKING CONFIG] Adding env fallback PDBs: %s", env_candidates)
            candidate_pdbs.extend(env_candidates)

        candidate_pdbs = list(dict.fromkeys(candidate_pdbs))

        invalid_tokens = {"NONE", "NULL", "N/A", ""}

        candidate_pdbs = [
            p for p in candidate_pdbs
            if p not in invalid_tokens and len(p) == 4 and p[0].isdigit()
        ]

        if not candidate_pdbs:
            fallback_env = os.getenv("VLAB_FALLBACK_PDBS", "")
            fallback_list = [
                x.strip().upper()
                for x in fallback_env.split(",")
                if x.strip()
            ]

            if fallback_list:
                selection_reason = "env_fallback_candidates"
                candidate_pdbs = [
                    p for p in fallback_list
                    if len(p) == 4 and p[0].isdigit()
                ]

        candidate_pdbs = list(dict.fromkeys(candidate_pdbs))

        failed_targets = list(state.get("failed_target_pdbs", []))

        if not candidate_pdbs:
            summary = "No valid protein targets available after filtering."

            return {
                "protein_analysis": summary,
                "binding_results": [],
                "target_pdb": None,
                "target_pdb_candidates": [],
                "failed_target_pdbs": failed_targets,
                "target_pdb_selection_reason": "no_valid_candidates",
                "stage_outputs": [
                    record_stage_output(
                        state,
                        "protein",
                        summary,
                        summary="Protein target selection failed (no valid candidates)",
                        metadata={"candidate_count": 0},
                    )
                ],
                "conversation_history": [
                    add_conversation_entry(state, "assistant", summary, "protein")
                ],
            }

        log.info("Protein target candidates: %s", ", ".join(candidate_pdbs))

        chosen_pdb = None
        valid_binding_results: List[dict] = []
        newly_failed: List[str] = []

        for pdb_id in candidate_pdbs:
            log.info("Trying protein target PDB: %s", pdb_id)

            try:
                binding_results = pw.evaluate_sequences(pdb_id, sequences) or []
            except Exception as e:
                log.warning("Protein wrapper failed for target %s: %s", pdb_id, e)
                newly_failed.append(pdb_id)
                continue

            proxy_valid_results = [r for r in binding_results if r.get("valid")]

            log.info(
                "[DOCKING DEBUG] Valid proxy/pre-docking results: %d",
                len(proxy_valid_results),
            )

            if not proxy_valid_results:
                newly_failed.append(pdb_id)
                continue

            enhanced_results = []

            for r in proxy_valid_results:
                seq = r.get("sequence")

                r["dock_score"] = None
                r["dock_valid"] = False
                r["dock_method"] = "not_attempted"
                r["dock_error"] = None
                r["dock_output_file"] = None
                r["dock_complex_file"] = None

                # Legacy compatibility fields.
                r["vina_energy"] = None
                r["vina_valid"] = False
                r["vina_method"] = "not_attempted"
                r["vina_error"] = None

                if not seq:
                    r["dock_method"] = "missing_sequence"
                    r["dock_error"] = "binding result missing sequence"
                    r["vina_method"] = r["dock_method"]
                    r["vina_error"] = r["dock_error"]
                    enhanced_results.append(r)
                    continue

                log.info("[DOCKING DEBUG] Processing sequence: %s...", seq[:20])

                md_res = None

                for m in state.get("md_results", []):
                    if m.get("sequence") == seq:
                        md_res = m.get("result")
                        break

                if not md_res:
                    r["dock_method"] = "skipped_no_md_result"
                    r["dock_error"] = "No MD result for sequence"
                    r["vina_method"] = r["dock_method"]
                    r["vina_error"] = r["dock_error"]
                    enhanced_results.append(r)
                    continue

                if md_res.get("min_energy") is None:
                    r["dock_method"] = "skipped_no_md_energy"
                    r["dock_error"] = "MD result lacks min_energy"
                    r["vina_method"] = r["dock_method"]
                    r["vina_error"] = r["dock_error"]
                    enhanced_results.append(r)
                    continue

                rna_pdb = md_res.get("rna_pdb")

                if not rna_pdb:
                    r["dock_method"] = "skipped_no_rna_pdb"
                    r["dock_error"] = "MD result lacks RNA PDB path"
                    r["vina_method"] = r["dock_method"]
                    r["vina_error"] = r["dock_error"]
                    enhanced_results.append(r)
                    continue

                if not os.path.exists(rna_pdb):
                    r["dock_method"] = "skipped_missing_rna_pdb"
                    r["dock_error"] = f"RNA PDB does not exist: {rna_pdb}"
                    r["vina_method"] = r["dock_method"]
                    r["vina_error"] = r["dock_error"]
                    enhanced_results.append(r)
                    continue

                try:
                    try:
                        _ = prepare_protein(pdb_id)
                    except Exception as prep_err:
                        log.warning(
                            "[HDOCK DEBUG] prepare_protein non-fatal error for %s: %s",
                            pdb_id,
                            prep_err,
                        )

                    receptor_pdb = os.path.join(
                        PROTEIN_CACHE_DIR,
                        f"{pdb_id.lower()}.pdb",
                    )

                    if not os.path.exists(receptor_pdb):
                        r["dock_method"] = "hdock_receptor_pdb_missing"
                        r["dock_error"] = f"Cached receptor PDB missing: {receptor_pdb}"
                        r["vina_method"] = r["dock_method"]
                        r["vina_error"] = r["dock_error"]
                        enhanced_results.append(r)
                        continue

                    ligand_pdb = prepare_rna_pdb_for_hdock(rna_pdb)

                    if not ligand_pdb or not os.path.exists(ligand_pdb):
                        r["dock_method"] = "skipped_rna_pdb_hdock_prep_failed"
                        r["dock_error"] = f"RNA PDB preparation for HDOCK failed: {rna_pdb}"
                        r["vina_method"] = r["dock_method"]
                        r["vina_error"] = r["dock_error"]
                        enhanced_results.append(r)
                        continue

                    dock = state["wrappers"]["hdock"].dock(
                        receptor_pdb=receptor_pdb,
                        ligand_pdb=ligand_pdb,
                    )

                    log.info("[HDOCK RESULT RAW] %s", dock)

                    if dock and dock.get("valid") and dock.get("dock_score") is not None:
                        dock_score = float(dock["dock_score"])

                        r["dock_score"] = dock_score
                        r["dock_valid"] = True
                        r["dock_method"] = dock.get("method", "hdock")
                        r["dock_error"] = None
                        r["dock_output_file"] = dock.get("output_file")
                        r["dock_complex_file"] = dock.get("complex_file")

                        # Legacy compatibility shim.
                        r["vina_energy"] = dock_score
                        r["vina_valid"] = True
                        r["vina_method"] = "hdock"
                        r["vina_error"] = None

                        for m in state.get("md_results", []):
                            if (
                                m.get("sequence") == seq
                                and isinstance(m.get("result"), dict)
                            ):
                                m["result"]["dock_score"] = dock_score
                                m["result"]["dock_valid"] = True
                                m["result"]["dock_target_pdb"] = pdb_id
                                m["result"]["dock_method"] = "hdock"

                                # Legacy compatibility.
                                m["result"]["vina_energy"] = dock_score
                                m["result"]["vina_valid"] = True
                                m["result"]["vina_target_pdb"] = pdb_id

                        log.info(
                            "[HDOCK SUCCESS] pdb=%s seq=%s score=%s",
                            pdb_id,
                            seq[:15],
                            dock_score,
                        )

                    else:
                        r["dock_valid"] = False
                        r["dock_score"] = None
                        r["dock_method"] = (
                            dock.get("method", "hdock_failed")
                            if isinstance(dock, dict)
                            else "hdock_failed"
                        )
                        r["dock_error"] = (
                            dock.get("error")
                            if isinstance(dock, dict) and dock.get("error")
                            else "HDOCK returned no valid docking score"
                        )

                        r["vina_valid"] = False
                        r["vina_energy"] = None
                        r["vina_method"] = r["dock_method"]
                        r["vina_error"] = r["dock_error"]

                        log.warning("[HDOCK FAIL] %s", r["dock_error"])

                except Exception as e:
                    r["dock_valid"] = False
                    r["dock_score"] = None
                    r["dock_method"] = "hdock_exception"
                    r["dock_error"] = str(e)

                    r["vina_valid"] = False
                    r["vina_energy"] = None
                    r["vina_method"] = "hdock_exception"
                    r["vina_error"] = str(e)

                    log.exception(
                        "HDOCK docking failed for %s against %s",
                        seq[:15],
                        pdb_id,
                    )

                enhanced_results.append(r)

            dock_valid = [
                r for r in enhanced_results
                if r.get("dock_valid") or r.get("vina_valid")
            ]

            log.info(
                "[DOCKING SUMMARY] pdb=%s dock_valid=%d",
                pdb_id,
                len(dock_valid),
            )

            if dock_valid:
                chosen_pdb = pdb_id
                valid_binding_results = [
                    {**r, "target_pdb": pdb_id}
                    for r in dock_valid
                ]

                log.info(
                    "Protein target %s accepted with %d docking-valid results.",
                    pdb_id,
                    len(dock_valid),
                )

            elif enhanced_results and not require_docking:
                chosen_pdb = pdb_id
                valid_binding_results = [
                    {**r, "target_pdb": pdb_id}
                    for r in enhanced_results
                ]

                log.info(
                    "Protein target %s accepted with %d proxy results because "
                    "VLAB_REQUIRE_DOCKING=0.",
                    pdb_id,
                    len(enhanced_results),
                )

            else:
                log.warning(
                    "Protein target %s produced no docking-valid results; trying next target.",
                    pdb_id,
                )
                newly_failed.append(pdb_id)
                continue

            for r in valid_binding_results:
                proxy_dg = r.get("dg", 0.0)

                try:
                    proxy_dg = float(proxy_dg)
                except Exception:
                    proxy_dg = 0.0

                if r.get("dock_valid") and r.get("dock_score") is not None:
                    dock_score = float(r["dock_score"])

                    r["proxy_dg"] = proxy_dg
                    r["hdock_score"] = dock_score

                    # Canonical HDOCK-aware ranking score.
                    r["binding_rank_score"] = 0.7 * dock_score + 0.3 * proxy_dg

                    # Legacy compatibility.
                    r["dg"] = r["binding_rank_score"]
                    r["binding_units"] = "hdock_relative_score"

                elif r.get("vina_valid") and r.get("vina_energy") is not None:
                    dock_score = float(r["vina_energy"])

                    r["proxy_dg"] = proxy_dg
                    r["hdock_score"] = dock_score
                    r["binding_rank_score"] = 0.7 * dock_score + 0.3 * proxy_dg
                    r["dg"] = r["binding_rank_score"]
                    r["binding_units"] = "hdock_relative_score"

                else:
                    r["binding_rank_score"] = proxy_dg
                    r["dg"] = proxy_dg
                    r["binding_units"] = "proxy_score"

            valid_binding_results.sort(key=safe_binding_rank)

            rank = 1
            last_score = None

            for r in valid_binding_results:
                score = r.get("binding_rank_score", r.get("dg"))

                if score is None:
                    r["rank"] = rank
                    continue

                if (
                    last_score is not None
                    and abs(float(score) - float(last_score)) > 1e-6
                ):
                    rank += 1

                r["rank"] = rank
                last_score = score

            break

        failed_targets = list(dict.fromkeys(failed_targets + newly_failed))

        if not chosen_pdb or not valid_binding_results:
            tried = ", ".join(candidate_pdbs)

            if require_docking:
                summary = (
                    "No docking-valid protein binding results for any candidate target. "
                    f"Tried: {tried}. Proxy-only results were rejected because "
                    "VLAB_REQUIRE_DOCKING=1."
                )
            else:
                summary = (
                    "No valid protein binding results for any candidate target. "
                    f"Tried: {tried}."
                )

            return {
                "protein_analysis": summary,
                "binding_results": [],
                "target_pdb": None,
                "target_pdb_candidates": candidate_pdbs,
                "failed_target_pdbs": failed_targets,
                "target_pdb_selection_reason": selection_reason,
                "stage_outputs": [
                    record_stage_output(
                        state,
                        "protein",
                        summary,
                        summary="Protein target selection failed",
                        metadata={
                            "candidate_count": len(candidate_pdbs),
                            "sequence_count": len(sequences),
                            "require_docking": require_docking,
                            "docking_backend": docking_backend,
                        },
                    )
                ],
                "conversation_history": [
                    add_conversation_entry(state, "assistant", summary, "protein")
                ],
            }

        lines = [
            f"Binding results for target {chosen_pdb} "
            f"(lower relative HDOCK/rank score = stronger predicted binding):"
        ]

        for r in valid_binding_results[:5]:
            if r.get("dock_valid"):
                display = r.get("dock_score")
                source = "HDOCK score"
            elif r.get("vina_valid"):
                display = r.get("vina_energy")
                source = "legacy docking compatibility"
            else:
                display = r.get("binding_rank_score", r.get("dg"))
                source = "proxy"

            lines.append(
                f"  - Seq: {r.get('sequence', '')[:20]}... | "
                f"Score: {display} ({source}) | "
                f"Rank: {r.get('rank')} | "
                f"Valid: {r.get('valid')} | "
                f"Method: {r.get('dock_method', r.get('vina_method', r.get('method', 'unknown')))}"
            )

            if r.get("dock_error"):
                lines.append(f"    Docking note: {r.get('dock_error')}")

        summary = "\n".join(lines)

        result = {
            "protein_analysis": summary,
            "binding_results": valid_binding_results,
            "target_pdb": chosen_pdb,
            "target_pdb_candidates": candidate_pdbs,
            "failed_target_pdbs": failed_targets,
            "target_pdb_selection_reason": selection_reason,
            "binding_units": "hdock_relative_score",
            "stage_outputs": [
                record_stage_output(
                    state,
                    "protein",
                    summary,
                    summary="Protein binding evaluation summary",
                    metadata={
                        "target_pdb": chosen_pdb,
                        "candidate_count": len(candidate_pdbs),
                        "sequence_count": len(sequences),
                        "require_docking": require_docking,
                        "docking_backend": docking_backend,
                        "dock_valid_count": sum(
                            1 for r in valid_binding_results
                            if r.get("dock_valid") or r.get("vina_valid")
                        ),
                        "binding_units": "hdock_relative_score",
                    },
                )
            ],
            "conversation_history": [
                add_conversation_entry(state, "assistant", summary, "protein")
            ],
        }


        try:
            export_paths = export_docking_outputs({**state, **result})
            result.update(export_paths)
        except Exception as e:
            log.warning("Failed to export docking summary files: %s", e)


        save_checkpoint({**state, **result})
        clear_gpu()
        return result

    except Exception as e:
        log.exception("PROTEIN AGENT ERROR")
        clear_gpu()

        return {
            "protein_analysis": f"Protein analysis failed: {e}",
            "binding_results": [],
        }

# ---------------------------------------------------------------------------
# Bioinfo Agent
# ---------------------------------------------------------------------------

def bioinfo_agent(state: LabState) -> dict:
    log.info("--- BIOINFO AGENT: Checking Conservation ---")

    try:
        bw = state["wrappers"]["bioinfo"]

        topic_text = (
            state.get("hypothesis")
            or state.get("topic_description")
            or state.get("research_topic")
            or ""
        )

        target_sequence = state.get("target_sequence")
        
        designed_sequences = dedupe_rna_sequences(
            state.get("designed_sequences", [])
        )


        # Prefer upgraded BioinfoWrapper.run_pipeline if available.
        if hasattr(bw, "run_pipeline"):
            analysis_dict = bw.run_pipeline(
                topic=topic_text,
                target_sequence=target_sequence,
                designed_sequences=designed_sequences,
                limit=int(os.getenv("VLAB_BIOINFO_FETCH_LIMIT", "20")),
            )
        else:
            # Legacy fallback.
            lookup = get_llm(temperature=0.0).invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a bioinformatician with expert knowledge of NCBI taxonomy.\n\n"
                            "Extract the NCBI Taxon ID for the primary virus or organism named.\n"
                            "Return ONLY a single integer. If ambiguous, return 0.\n\n"
                            "Examples:\n"
                            "HIV-1 → 11676\n"
                            "SARS-CoV-2 → 2697049\n"
                            "HCV → 11103\n"
                            "HPeV1 → 12110\n"
                            "Influenza A → 11520\n"
                            "Poliovirus → 12081\n"
                            "E. coli → 562\n"
                            "Homo sapiens → 9606"
                        )
                    ),
                    HumanMessage(content=topic_text),
                ]
            )

            txid = "".join(c for c in lookup.content if c.isdigit())

            if not txid or txid == "0":
                return {
                    "bioinfo_analysis": (
                        "Skipped: No reliable NCBI Taxon ID identified. "
                        "Conservation analysis requires an unambiguous organism name."
                    ),
                    "conservation_signal": {"valid": False},
                }

            fasta = bw.fetch_related_genomes(txid, limit=3)

            if ">" not in fasta:
                return {
                    "bioinfo_analysis": "No genomes found for conservation analysis.",
                    "conservation_signal": {"valid": False},
                }

            msa = bw.run_msa(fasta)
            analysis_dict = bw.calculate_conservation(msa)
            analysis_dict["msa"] = msa
            analysis_dict["fasta"] = fasta

        msa = analysis_dict.get("msa", "")
        conservation_signal = analysis_dict.get("conservation_signal", {})

        if not conservation_signal:
            conservation_signal = {
                "valid": bool(analysis_dict.get("valid")),
                "consensus": analysis_dict.get("consensus", ""),
                "position_scores": analysis_dict.get("position_scores", []),
                "entropy_scores": analysis_dict.get("entropy_scores", []),
                "conserved_regions": analysis_dict.get("conserved_regions", []),
                "motif_scores": analysis_dict.get("motif_scores", {}),
                "selected_motifs": analysis_dict.get("selected_motifs", []),
                "conservation_pct": analysis_dict.get("conservation_pct", 0.0),
            }

        # Add scalar conservation_fitness expected by PI agent.
        position_scores = conservation_signal.get("position_scores", []) or []
        if position_scores:
            conservation_fitness = sum(float(x) for x in position_scores) / max(1, len(position_scores))
        else:
            conservation_fitness = float(analysis_dict.get("conservation_pct", 0.0) or 0.0) / 100.0

        conservation_signal["conservation_fitness"] = max(0.0, min(1.0, conservation_fitness))

        analysis_str = json.dumps(safe_jsonable(analysis_dict), indent=2)

        result = {
            "bioinfo_analysis": analysis_str,
            "msa_data": msa,
            "conservation_signal": conservation_signal,
            "conserved_regions": conservation_signal.get("conserved_regions", []),
            "stage_outputs": [
                record_stage_output(
                    state,
                    "bioinfo",
                    analysis_str,
                    summary=(
                        f"MSA conservation: "
                        f"{analysis_dict.get('conservation_pct', 0):.1f}% "
                        f"over {analysis_dict.get('num_sequences', '?')} seqs"
                    ),
                    metadata={
                        "conservation_valid": conservation_signal.get("valid", False),
                        "msa_size": len(msa or ""),
                        "conservation_fitness": conservation_signal.get("conservation_fitness", 0.0),
                        "selected_motifs": conservation_signal.get("selected_motifs", [])[:10],
                    },
                )
            ],
            "conversation_history": [
                add_conversation_entry(state, "assistant", analysis_str, "bioinfo")
            ],
        }

        data_collector.capture_agent_interaction(
            "Bioinfo",
            "Compute MSA and conservation",
            analysis_str,
        )

        save_checkpoint({**state, **result})
        return result

    except Exception as e:
        log.exception("BIOINFO AGENT ERROR")
        return {
            "bioinfo_analysis": f"Conservation analysis failed: {e}",
            "conservation_signal": {"valid": False},
        }

# ---------------------------------------------------------------------------
# Skeptic Agent — HDOCK-aware
# ---------------------------------------------------------------------------

def skeptic_agent(state: LabState) -> dict:
    log.info("--- SKEPTIC AGENT: Physics Validation ---")

    try:
        binding_results = state.get("binding_results", [])
        current_target = state.get("target_pdb")

        valid = [
            r for r in binding_results
            if r.get("valid") and (
                not current_target or r.get("target_pdb") == current_target
            )
        ]

        if not valid:
            critique = (
                "SUPPORT: NONE\n"
                "ENERGY: N/A\n"
                "CONCERNS:\n"
                "  1. No valid binding results are available for physical evaluation\n"
                "  2. Hypothesis cannot be assessed without docking/ranking data\n"
                "  3. None\n"
                "MISSING_CONTROLS: docking score, structural validation, MD trajectory\n"
                "RECOMMENDATION: REDESIGN_SEQUENCES\n"
                "REASON: Without binding data no physical claim can be evaluated."
            )

            return {
                "critique": critique,
                "stage_outputs": [
                    record_stage_output(
                        state,
                        "skeptic",
                        critique,
                        summary="Skeptic: insufficient data",
                        metadata={"valid_binding_count": 0},
                    )
                ],
            }

        scores = [
            r.get("binding_rank_score", r.get("dg"))
            for r in valid
            if r.get("binding_rank_score", r.get("dg")) is not None
        ]

        score_range = (max(scores) - min(scores)) if scores else None
        best_score = min(scores) if scores else None
        n_valid = len(valid)

        dock_count = sum(
            1 for r in valid
            if r.get("dock_valid") or r.get("vina_valid")
        )

        hdock_count = sum(
            1 for r in valid
            if r.get("dock_method") == "hdock" or r.get("vina_method") == "hdock"
        )

        score_source = (
            f"HDOCK={hdock_count}/{n_valid}, "
            f"docking={dock_count}/{n_valid}, "
            f"proxy={n_valid - dock_count}/{n_valid}"
        )

        binding_summary = "\n".join(
            f"  {r.get('sequence', '')[:15]}... → "
            f"binding_rank_score={r.get('binding_rank_score', r.get('dg'))} "
            f"({'HDOCK' if r.get('dock_method') == 'hdock' or r.get('vina_method') == 'hdock' else 'proxy'}) "
            f"rank={r.get('rank')}"
            for r in sorted(valid, key=safe_binding_rank)[:5]
        )
        
        fold_quality = state.get("fold_quality", {}) or {}
        fold_passed = state.get("fold_thresholds_passed")
        fold_reasons = state.get("fold_threshold_reasons", [])
        motifs = motif_summary_from_state(state)


        prompt = (
            f"Hypothesis:\n{truncate_str(state.get('hypothesis', ''), 600)}\n\n"
            f"Target PDB: {state.get('target_pdb')}\n"
            f"Binding results (n={n_valid}, source: {score_source}):\n"
            f"{binding_summary}\n"
            f"Best binding rank score: {best_score} "
            f"(HDOCK-relative; not kcal/mol) | "
            f"Score spread: {score_range}\n\n"
            f"Fold threshold status:\n"
            f"passed={fold_passed}, quality={safe_jsonable(fold_quality)}, reasons={fold_reasons}\n\n"
            f"Selected/conserved motifs:\n"
            f"{motifs}\n\n"
            f"Structural analysis:\n"
            f"{truncate_str(state.get('structural_analysis', ''), 500)}\n\n"
            f"MD analysis:\n"
            f"{truncate_str(state.get('md_analysis', ''), 500)}\n\n"
            f"PI optimisation summary:\n"
            f"{truncate_str(state.get('pi_summary', ''), 400)}"
        )

        msg = get_llm(temperature=0.0).invoke(
            [
                SystemMessage(
                    content=(
                        "You are a peer reviewer at a computational biophysics journal.\n\n"
                        "Evaluate whether the evidence supports the hypothesis.\n\n"
                        "IMPORTANT:\n"
                        "- HDOCK scores are relative docking/ranking scores, NOT kcal/mol.\n"
                        "- Do not describe HDOCK scores as binding free energies.\n\n"
                        "Respond in EXACTLY this structured format:\n\n"
                        "SUPPORT: [STRONG/MODERATE/WEAK/NONE]\n"
                        "ENERGY: best_binding_score=<value>, spread=<value>, n=<count>, source=<HDOCK/proxy/mixed>\n"
                        "CONCERNS:\n"
                        "  1. <specific physical/statistical concern with cited value>\n"
                        "  2. <specific physical/statistical concern with cited value>\n"
                        "  3. <third concern, or 'None'>\n"
                        "MISSING_CONTROLS: <comma-separated list or 'None'>\n"
                        "RECOMMENDATION: [ACCEPT/REVISE_HYPOTHESIS/REDESIGN_SEQUENCES/MORE_MD]\n"
                        "REASON: <one sentence citing the most important quantitative value>\n\n"
                        "Scoring standards:\n"
                        "- STRONG: docking-validated, favourable relative score, small spread, confirmed motif\n"
                        "- MODERATE: docking-validated but incomplete controls or moderate spread\n"
                        "- WEAK: weak relative score, high spread, proxy-only, or missing structure support\n"
                        "- NONE: no valid binding data\n\n"
                        "Rules:\n"        
                        "- Every concern must cite a number from the data.\n"
                        "- If fold_threshold_status is failed, recommend REDESIGN_SEQUENCES unless docking and MD are both strong.\n"
                        "- Treat selected/conserved motif presence as supporting evidence only when fold thresholds pass.\n"
                        "- Never write may, could, potentially, or might in REASON."

                    )
                ),
                HumanMessage(content=prompt),
            ]
        )

        critique = clean_llm_output(msg.content)

        from VLAB2.orchestration.skeptic_parser import parse_skeptic_output

        parsed = parse_skeptic_output(critique)

        fm = FailureMemory()
        fm.update(parsed)
        fm.save()

        memory = state.get("failure_memory", [])
        memory.append(parsed)
        state["failure_memory"] = memory[-10:]

        ok, _ = validate_text_block(critique)

        if not ok:
            critique = (
                "SUPPORT: WEAK\n"
                "ENERGY: N/A\n"
                "CONCERNS:\n"
                "  1. Critique generation failed — raw outputs may be malformed\n"
                "  2. Further controlled simulations are required\n"
                "  3. None\n"
                "MISSING_CONTROLS: valid binding data, structural confirmation\n"
                "RECOMMENDATION: REDESIGN_SEQUENCES\n"
                "REASON: Critique could not be generated; sequence redesign is safest."
            )

        if score_range is not None and score_range < 5.0:
            critique += (
                f"\n\n[CONVERGENCE NOTE: Binding score spread is "
                f"{score_range:.2f} (<5.0 threshold) — population may have converged prematurely.]"
            )

        iteration_record = {
            "iteration": state.get("iterations", 0),
            "hypothesis": state.get("hypothesis", ""),
            "target_pdb": state.get("target_pdb"),
            "energy_range": score_range,      # legacy key
            "score_range": score_range,
            "best_dg": best_score,            # legacy key
            "best_binding_score": best_score,
            "n_valid": n_valid,
            "critique": critique,
            "binding_units": "hdock_relative_score",
        }

        result = {
            "critique": critique,
            "results_log": [iteration_record],
            "stage_outputs": [
                record_stage_output(
                    state,
                    "skeptic",
                    critique,
                    summary="Physics-based structured critique",
                    metadata={
                        "score_range": score_range,
                        "best_binding_score": best_score,
                        "n_valid": n_valid,
                        "binding_units": "hdock_relative_score",
                    },
                )
            ],
            "conversation_history": [
                add_conversation_entry(state, "assistant", critique, "skeptic")
            ],
        }

        save_checkpoint({**state, **result})
        return result

    except Exception as e:
        log.exception("SKEPTIC AGENT ERROR")
        return {"critique": f"Skeptic failed: {e}"}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def should_continue(state: LabState) -> str:
    critique = state.get("critique", "") or ""
    max_iter = state.get("max_iterations", 3)
    current_iter = state.get("iterations", 0)    

    if (
        state.get("fold_thresholds_passed") is False
        and os.getenv("VLAB_RNA_ENFORCE_MIN_FOLD", "1").strip() == "1"
    ):
        if current_iter < max_iter:
            log.info("Continuing: RNA failed fold thresholds.")
            return "continue"



    if current_iter >= max_iter:
        return "end"

    results = state.get("binding_results", [])
    current_target = state.get("target_pdb")

    valid = [
        r for r in results
        if r.get("valid") and (
            not current_target or r.get("target_pdb") == current_target
        )
    ]

    if len(valid) >= 3:
        scores = [
            r.get("binding_rank_score", r.get("dg"))
            for r in valid
            if r.get("binding_rank_score", r.get("dg")) is not None
        ]

        if scores:
            spread = max(scores) - min(scores)
            best_score = min(scores)

            if (
                spread < 2.0
                and best_score < float(os.getenv("VLAB_ACCEPT_BINDING_SCORE", "-50"))
                and state.get("fold_thresholds_passed") is not False
            ):
                log.info(
                    "Converged: binding score spread %.3f and best score %.3f pass thresholds.",
                    spread,
                    best_score,
                )
                return "end"

            if spread < 2.0:
                log.info(
                    "Not ending despite small spread %.3f because binding/fold thresholds are not sufficient.",
                    spread,
                )
                critique = state.get("critique", "")

    for line in critique.splitlines():
        if line.startswith("RECOMMENDATION:") and "ACCEPT" in line:
            log.info("Skeptic accepted hypothesis — terminating loop.")
            return "end"

    return "continue"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

workflow = StateGraph(LabState)

workflow.add_node("pi", pi_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("bioinfo", bioinfo_agent)
workflow.add_node("structural", structural_agent)
workflow.add_node("md", md_agent)
workflow.add_node("protein", protein_agent)
workflow.add_node("skeptic", skeptic_agent)

workflow.set_entry_point("pi")

workflow.add_edge("pi", "researcher")
workflow.add_edge("researcher", "bioinfo")
workflow.add_edge("bioinfo", "structural")
workflow.add_edge("structural", "md")
workflow.add_edge("md", "protein")
workflow.add_edge("protein", "skeptic")

workflow.add_conditional_edges(
    "skeptic",
    should_continue,
    {
        "continue": "pi",
        "end": END,
    },
)

virtual_lab = workflow.compile()


# ---------------------------------------------------------------------------
# Topic loading
# ---------------------------------------------------------------------------

def load_topics(path: str = "research_topics.yaml") -> list:
    if not os.path.exists(path):
        log.warning("Topic file %s not found. Using default topic.", path)
        return []

    with open(path) as f:
        data = yaml.safe_load(f)

    return data.get("topics", [])


def select_topic(topics: list, index: Optional[int] = None) -> dict:
    if not topics:
        return {
            "name": "viral packaging motifs",
            "description": "Identify cis-acting RNA packaging signals in viral systems",
            "seed_questions": [],
        }

    if index is not None:
        if 0 <= index < len(topics):
            return topics[index]
        return topics[0]

    print("\n=== Available Research Topics ===")

    for i, t in enumerate(topics):
        print(f"[{i}] {t['name']}: {t['description']}")

    print("[-1] Custom topic")

    try:
        choice = int(input("\nSelect topic number: "))
    except Exception:
        choice = 0

    if choice == -1:
        return {
            "name": input("Topic name: "),
            "description": input("Description: "),
            "seed_questions": [input("Initial question: ")],
        }

    return topics[max(0, min(choice, len(topics) - 1))]


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def generate_final_report(state: LabState) -> str:
    report = f"# Research Report: {state.get('research_topic', 'Unknown')}\n\n"
    report += f"## Scientific Hypothesis\n{truncate_str(state.get('hypothesis', ''), 600)}\n\n"
    report += f"## PI Optimisation Summary\n{truncate_str(state.get('pi_summary', ''), 600)}\n\n"
    report += f"## Target Protein\n{state.get('target_pdb', 'Not selected')}\n\n"
    report += f"## Binding Units\n{state.get('binding_units', 'hdock_relative_score')}\n\n"

    report += "## Iteration History\n"
    report += "| Iter | Target | Best Binding Score | Score Range | n_valid | Critique |\n"
    report += "| :--- | :----- | :----------------- | :---------- | :------ | :------- |\n"

    for item in state.get("results_log", []):
        report += (
            f"| {item.get('iteration')} "
            f"| {item.get('target_pdb')} "
            f"| {item.get('best_binding_score', item.get('best_dg', 'N/A'))} "
            f"| {item.get('score_range', item.get('energy_range', 'N/A'))} "
            f"| {item.get('n_valid', 'N/A')} "
            f"| {truncate_str(item.get('critique', ''), 80)} |\n"
        )

    report += "\n## Agent Outputs\n"

    for entry in state.get("stage_outputs", []):
        report += f"### {entry.get('agent', 'unknown').upper()}\n"
        report += f"{entry.get('summary', '')}\n\n"

    return report


def print_user_friendly_summary(result: dict):
    print("\n" + "=" * 70)
    print(" VIRTUAL LAB: RUN COMPLETE")
    print("=" * 70)

    print("\n[SCIENTIFIC HYPOTHESIS]")
    print(f"  {truncate_str(result.get('hypothesis', ''), 500)}")

    print("\n[PI OPTIMISATION SUMMARY]")
    print(f"  {truncate_str(result.get('pi_summary', ''), 500)}")

    print("\n[RESEARCH TOPIC]")
    print(f"  {result.get('research_topic')}")
    print(f"  {result.get('topic_description')}")

    print("\n[TARGET PROTEIN]")
    print(f"  {result.get('target_pdb', 'Not selected')}")

    print("\n[BINDING UNITS]")
    print(f"  {result.get('binding_units', 'hdock_relative_score')}")

    print("\n[AGENT TAKEAWAYS]")

    for entry in result.get("stage_outputs", []):
        print(f"  {entry.get('agent', 'unknown').upper():<12} : {entry.get('summary', '')}")

    print("\n" + "=" * 70)

    report = generate_final_report(result)
    report_file = f"summary_{int(time.time())}.md"

    try:
        with open(report_file, "w") as f:
            f.write(report)

        print(f"Final report saved to: {report_file}")

    except Exception as e:
        print(f"Failed to save report file: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual Lab Orchestrator")
    parser.add_argument("--topic-file", default="research_topics.yaml", help="Path to topics YAML")
    parser.add_argument("--topic-index", type=int, default=None, help="Index of topic to run")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum iterations")

    args = parser.parse_args()

    topics = load_topics(args.topic_file)
    topic = select_topic(topics, index=args.topic_index)

    log.info("Starting Virtual Lab for topic: %s", topic["name"])

    log.info("Bootstrapping knowledge base with prior training data...")


    try:
        raw_bootstrap_query = (
            topic.get("seed_questions", [topic["description"]])[0]
            if topic.get("seed_questions")
            else topic.get("description") or topic.get("name")
        )

        bootstrap_queries = []

        q1 = normalise_lit_query(raw_bootstrap_query)

        if keep_research_query(q1):
            bootstrap_queries.append(q1)

        fallback_q = fallback_literature_query(
            topic.get("description") or topic.get("name") or raw_bootstrap_query
        )

        if fallback_q and fallback_q not in bootstrap_queries:
            bootstrap_queries.append(fallback_q)

        generic_q = "viral RNA stem-loop capsid binding"

        if generic_q not in bootstrap_queries:
            bootstrap_queries.append(generic_q)

        all_papers = []

        for bq in bootstrap_queries:
            try:
                log.info("Bootstrapping knowledge with query: %s", bq)
                papers = expand_knowledge(bq, build_db=True) or []
                

                for p in papers:
                    if isinstance(p, dict):
                        title = p.get("title", "")
                        abstract = p.get("abstract", "")
                    else:
                        title = getattr(p, "metadata", {}).get("title", "")
                        abstract = getattr(p, "page_content", "")

                    if keep_literature_text(title, abstract):
                        all_papers.append(p)


                if len(all_papers) >= 5:
                    break

            except Exception as qerr:
                log.warning("Bootstrap query failed '%s': %s", bq, qerr)

        # Deduplicate by title if possible.
        seen_titles = set()
        deduped = []

        for p in all_papers:
            title = ""

            if isinstance(p, dict):
                title = str(p.get("title", "")).strip().lower()
            else:
                title = str(getattr(p, "metadata", {}).get("title", "")).strip().lower()

            key = title or str(p)[:200]

            if key in seen_titles:
                continue

            seen_titles.add(key)
            deduped.append(p)

        log.info("Knowledge base rebuilt with %d documents.", len(deduped))

    except Exception as e:
        log.warning("Knowledge bootstrap failed (non-fatal): %s", e)


    wrappers = build_wrapper_bundle()

    initial_state: LabState = {
        "research_topic": topic["name"],
        "topic_description": topic["description"],
        "seed_questions": topic.get("seed_questions", []),
        "failure_memory": [],

        "wrappers": wrappers,
        "mutation_bias": {},

        "current_question_idx": 0,
        "iterations": 0,
        "max_iterations": args.max_iterations,

        "hypothesis": topic.get("description") or topic.get("name") or "",
        "pi_summary": "",
        "optimisation_status": "",

        "evidence": [],

        "structural_analysis": "",
        "md_analysis": "",
        "protein_analysis": "",
        "bioinfo_analysis": "",
        "msa_data": "",
        "critique": "",

        "binding_units": "hdock_relative_score",

        "designed_sequences": [],
        "structural_candidates": [],
        "binding_results": [],
        "md_results": [],
        "structural_candidates": [],

        "target_pdb": None,
        "target_pdb_candidates": [],
        "failed_target_pdbs": [],
        "target_pdb_selection_reason": "",
        "target_sequence": None,

        "results_log": [],
        "stage_outputs": [],
        "conversation_history": [],

        "previous_hypotheses": [],
        "virus_name": "",
        "virus_family": "",
        "virus_genus": "",
        "target_pdb_metadata": {},
        "target_pdb_rankings": [],
        "target_selection_mode": os.getenv("VLAB_TARGET_SELECTION_MODE", "llm_fallback"),
        "docking_summary_json": "",
        "docking_summary_csv": "",
        "docking_summary_md": "",
        
    }

    try:
        final_result = virtual_lab.invoke(initial_state)

        print_user_friendly_summary(final_result)

        output_file = f"lab_results_{topic['name'].replace(' ', '_').lower()}.json"

        with open(output_file, "w") as f:
            json.dump(safe_jsonable(dict(final_result)), f, indent=2)

        log.info("Final results saved to %s", output_file)

        log.info("Running post-run training data pipeline...")

        try:
            from scripts.extract_training_examples import extract_from_checkpoint, load_json
            from scripts.convert_to_supervised import convert

            cp = load_json("lab_checkpoint.json")
            examples = extract_from_checkpoint(cp)

            with open("examples.jsonl", "w") as ef:
                for ex in examples:
                    ef.write(json.dumps(ex) + "\n")

            log.info("Extracted %d training examples.", len(examples))

            convert("examples.jsonl", "supervised.jsonl")
            log.info("Converted to supervised format.")

            postrun_query = (
                topic.get("seed_questions", [topic["name"]])[0]
                if topic.get("seed_questions")
                else topic["name"]
            )

            rebuild_papers = expand_knowledge(postrun_query, build_db=True)

            log.info(
                "Post-run knowledge base rebuilt with %d documents.",
                len(rebuild_papers),
            )

        except Exception as pipe_err:
            log.warning("Post-run training pipeline failed (non-fatal): %s", pipe_err)

    except Exception as e:
        log.error("Virtual Lab failed: %s", e)
        raise
