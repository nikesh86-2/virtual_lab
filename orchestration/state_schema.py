"""
state_schema.py
===============

Shared state schema + helper utilities for the Virtual Lab orchestrator.

This is the single canonical definition of LabState. The orchestrator
imports from here — do NOT define a separate LabState in the orchestrator.
"""

from __future__ import annotations

import operator
import re
import time
from typing import Annotated, Any, List, Optional, TypedDict


# ---------------------------------------------------------------------------
# Reducer helpers
# ---------------------------------------------------------------------------

def _clean_rna_local(seq: str) -> str:
    """
    Local RNA normaliser to avoid importing sequence_utils here and risking
    circular imports.
    """
    if not isinstance(seq, str):
        return ""

    return "".join(
        c for c in seq.strip().upper().replace("T", "U")
        if c in "ACGU"
    )


def _evidence_key(item: str) -> str:
    """
    Stable evidence dedupe key based on title.

    Evidence strings usually look like:
        Title: ... | Abstract: ...
    """
    if not item:
        return ""

    title = str(item).split(" | ")[0].replace("Title: ", "").strip().lower()
    title = re.sub(r"[^a-z0-9]+", " ", title)

    return " ".join(title.split())


def dedupe_evidence_reducer(
    existing: list[str] | None,
    new: list[str] | None,
) -> list[str]:
    """
    Merge evidence while deduplicating by title.

    Keeps insertion order and caps to latest 20 unique evidence items.
    """
    combined = list(existing or []) + list(new or [])

    seen = set()
    out: list[str] = []

    for item in combined:
        key = _evidence_key(item)

        if not key or key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out[-20:]


def dedupe_rna_sequence_reducer(
    existing: list[str] | None,
    new: list[str] | None,
) -> list[str]:
    """
    Merge RNA sequence lists while deduplicating exact cleaned RNA strings.

    Keeps insertion order. Does not do near-duplicate filtering here because
    reducers should remain lightweight and deterministic.
    """
    combined = list(existing or []) + list(new or [])

    seen = set()
    out: list[str] = []

    for seq in combined:
        s = _clean_rna_local(seq)

        if not s or s in seen:
            continue

        seen.add(s)
        out.append(s)

    return out


def _binding_result_key(item: dict) -> tuple:
    """
    Stable-ish binding result key.

    We keep target+sequence+docking output/score. This avoids duplicating the
    exact same docking result while still allowing the same sequence to be
    docked against a new target.
    """
    if not isinstance(item, dict):
        return ("invalid", repr(item))

    seq = _clean_rna_local(item.get("sequence", ""))
    target = str(item.get("target_pdb", "") or "").upper()
    dock_file = str(item.get("dock_output_file", "") or "")
    score = item.get("binding_rank_score", item.get("dg", item.get("dock_score")))

    return (target, seq, dock_file, str(score))


def dedupe_binding_results_reducer(
    existing: list[dict] | None,
    new: list[dict] | None,
) -> list[dict]:
    """
    Merge binding results while avoiding exact duplicate docking records.

    Keeps recent history but prevents unbounded duplicate accumulation.
    """
    combined = list(existing or []) + list(new or [])

    seen = set()
    out: list[dict] = []

    for item in combined:
        key = _binding_result_key(item)

        if key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out[-50:]


def _md_result_key(item: dict) -> tuple:
    """
    Dedupe MD results by sequence and RNA PDB path if present.
    """
    if not isinstance(item, dict):
        return ("invalid", repr(item))

    seq = _clean_rna_local(item.get("sequence", ""))

    result = item.get("result", {})
    if isinstance(result, dict):
        rna_pdb = str(result.get("rna_pdb", "") or "")
        min_energy = str(result.get("min_energy", "") or "")
    else:
        rna_pdb = ""
        min_energy = ""

    return (seq, rna_pdb, min_energy)


def dedupe_md_results_reducer(
    existing: list[dict] | None,
    new: list[dict] | None,
) -> list[dict]:
    """
    Merge MD results while avoiding exact duplicates.
    """
    combined = list(existing or []) + list(new or [])

    seen = set()
    out: list[dict] = []

    for item in combined:
        key = _md_result_key(item)

        if key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out[-50:]


def append_unique_dicts_reducer(
    existing: list[dict] | None,
    new: list[dict] | None,
) -> list[dict]:
    """
    Generic list-of-dicts reducer.

    Keeps all entries that are not exact duplicates by repr().
    Used for results_log/stage_outputs/conversation_history where we usually
    want append-like behaviour but not accidental duplicate records.
    """
    combined = list(existing or []) + list(new or [])

    seen = set()
    out: list[dict] = []

    for item in combined:
        key = repr(item)

        if key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out


# ---------------------------------------------------------------------------
# Canonical LabState
# ---------------------------------------------------------------------------

class LabState(TypedDict, total=False):
    # ----------------------------------------------------------------
    # Topic / setup
    # ----------------------------------------------------------------
    research_topic: str
    topic_description: str
    seed_questions: List[str]

    virus_name: str
    virus_family: str
    virus_genus: str

    # ----------------------------------------------------------------
    # Wrappers bundle (non-serialisable — stripped in safe_jsonable)
    # ----------------------------------------------------------------
    wrappers: dict

    # ----------------------------------------------------------------
    # Iteration control
    # ----------------------------------------------------------------
    current_question_idx: int
    iterations: int
    max_iterations: int

    # ----------------------------------------------------------------
    # PI / optimisation
    # ----------------------------------------------------------------
    hypothesis: str
    pi_summary: str
    optimisation_status: str
    mutation_bias: dict
    joint_physics_feedback: dict

    # ----------------------------------------------------------------
    # Agent outputs
    # ----------------------------------------------------------------
    evidence: Annotated[List[str], dedupe_evidence_reducer]

    structural_analysis: str
    md_analysis: str
    protein_analysis: str
    bioinfo_analysis: str
    msa_data: str
    critique: str

    # ----------------------------------------------------------------
    # Conservation signal
    # ----------------------------------------------------------------
    conservation_signal: dict
    conserved_regions: List
    conservation_fitness: float

    # ----------------------------------------------------------------
    # Designed candidates / structured outputs
    # ----------------------------------------------------------------
    designed_sequences: Annotated[List[str], dedupe_rna_sequence_reducer]
    structural_candidates: Annotated[List[dict], append_unique_dicts_reducer]
    binding_results: Annotated[List[dict], dedupe_binding_results_reducer]
    md_results: Annotated[List[dict], dedupe_md_results_reducer]

    # ----------------------------------------------------------------
    # Target protein selection
    # ----------------------------------------------------------------
    target_pdb: Optional[str]
    target_pdb_candidates: List[str]
    target_pdb_rankings: List[dict]
    failed_target_pdbs: List[str]
    target_pdb_selection_reason: str
    target_pdb_metadata: dict
    target_selection_mode: str
    target_sequence: Optional[str]

    # ----------------------------------------------------------------
    # Docking exports
    # ----------------------------------------------------------------
    binding_units: str
    docking_summary_json: str
    docking_summary_csv: str
    docking_summary_md: str

    # ----------------------------------------------------------------
    # Logs / conversation
    # ----------------------------------------------------------------
    results_log: Annotated[List[dict], append_unique_dicts_reducer]
    stage_outputs: Annotated[List[dict], append_unique_dicts_reducer]
    conversation_history: Annotated[List[dict], append_unique_dicts_reducer]

    # ----------------------------------------------------------------
    # Report / memory
    # ----------------------------------------------------------------
    final_report: str
    previous_hypotheses: List[str]
    failure_memory: List[dict]

    # ----------------------------------------------------------------
    # Optional runtime / training helpers
    # ----------------------------------------------------------------
    data_collector: Any

    # ----------------------------------------------------------------
    # Internal optimiser/runtime metadata
    # ----------------------------------------------------------------
    _run_system_selected_motifs: List[dict]
    _run_system_min_fold_thresholds: dict
    _run_system_final_weights: dict
    _md_cache: dict


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def timestamp() -> str:
    """Return current UTC timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def record_stage_output(
    state: LabState,
    agent_name: str,
    output: str,
    summary: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Create a standard stage output record.

    Every agent writes its result into the same simple shape so later analysis
    does not become messy.
    """
    if output is None:
        output = ""

    output = str(output)

    if not summary:
        summary = output[:300] + ("..." if len(output) > 300 else "")

    return {
        "agent": agent_name,
        "summary": summary,
        "output": output,
        "metadata": metadata or {},
        "timestamp": timestamp(),
    }


def add_conversation_entry(
    state: LabState,
    role: str,
    content: str,
    agent: Optional[str] = None,
) -> dict:
    """
    Create a standard conversation-history entry.
    """
    return {
        "role": role,
        "agent": agent,
        "content": content,
        "timestamp": timestamp(),
    }


def safe_jsonable(obj: Any) -> Any:
    """
    Convert state into something safe for json.dump().

    Python objects like wrapper classes cannot be saved directly to JSON.
    This converts them into readable placeholders.
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, list):
        return [safe_jsonable(x) for x in obj]

    if isinstance(obj, tuple):
        return [safe_jsonable(x) for x in obj]

    if isinstance(obj, set):
        return [safe_jsonable(x) for x in sorted(obj, key=str)]

