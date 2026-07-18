"""
state_schema.py
===============

Shared state schema + helper utilities for the Virtual Lab orchestrator.

This is the single canonical definition of LabState. The orchestrator
imports from here — do NOT define a separate LabState in the orchestrator.
"""

from __future__ import annotations

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


def _normalise_text_key(text: Any) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _evidence_key(item: Any) -> str:
    """
    Stable evidence dedupe key.

    Supports both legacy strings:
        Title: ... | Abstract: ...

    and v2-style dict records:
        {"title": ..., "abstract": ..., "doi": ..., "pmid": ...}
    """
    if not item:
        return ""

    if isinstance(item, dict):
        doi = _normalise_text_key(item.get("doi") or item.get("DOI"))
        pmid = _normalise_text_key(item.get("pmid") or item.get("PMID"))
        title = _normalise_text_key(item.get("title"))
        year = str(item.get("year") or "").strip()
        abstract = _normalise_text_key(
            item.get("abstract")
            or item.get("text")
            or item.get("content")
            or item.get("page_content")
        )[:220]

        if doi:
            return f"doi::{doi}"

        if pmid:
            return f"pmid::{pmid}"

        if title:
            return f"title::{title}|year::{year}|abs::{abstract[:120]}"

        if abstract:
            return f"abstract::{abstract}"

        return ""

    title = str(item).split(" | ")[0].replace("Title: ", "").strip().lower()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return " ".join(title.split())


def dedupe_evidence_reducer(
    existing: list[Any] | None,
    new: list[Any] | None,
) -> list:
    """
    Merge evidence while deduplicating by DOI/PMID/title.

    Keeps insertion order and caps to latest 50 unique evidence items.
    """
    combined = list(existing or []) + list(new or [])

    seen: set[str] = set()
    out: list[Any] = []

    for item in combined:
        key = _evidence_key(item)

        if not key or key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out[-50:]


def dedupe_rna_sequence_reducer(
    existing: list[str] | None,
    new: list[str] | None,
) -> list:
    """
    Merge RNA sequence lists while deduplicating exact cleaned RNA strings.
    """
    combined = list(existing or []) + list(new or [])

    seen: set[str] = set()
    out: list[str] = []

    for seq in combined:
        s = _clean_rna_local(seq)

        if not s or s in seen:
            continue

        seen.add(s)
        out.append(s)

    return out


def dedupe_string_list_reducer(
    existing: list[str] | None,
    new: list[str] | None,
) -> list:
    """
    Merge generic string lists while deduplicating and preserving insertion order.
    """
    combined = list(existing or []) + list(new or [])

    seen: set[str] = set()
    out: list[str] = []

    for item in combined:
        s = str(item or "").strip()

        if not s or s in seen:
            continue

        seen.add(s)
        out.append(s)

    return out


def append_unique_dicts_reducer(
    existing: list[dict] | None,
    new: list[dict] | None,
) -> list:
    """
    Generic list-of-dicts reducer.

    Keeps all entries that are not exact duplicates by repr().
    Used for results_log/stage_outputs/conversation_history where we usually
    want append-like behaviour but not accidental duplicate records.
    """
    combined = list(existing or []) + list(new or [])

    seen: set[str] = set()
    out: list[dict] = []

    for item in combined:
        key = repr(item)

        if key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out


def _binding_result_key(item: dict) -> tuple:
    """
    Stable-ish binding result key.

    Keeps target+sequence+docking output/score. This avoids duplicating the
    exact same docking result while still allowing the same sequence to be
    docked against a new target or in a new HDOCK run.
    """
    if not isinstance(item, dict):
        return ("invalid", repr(item))

    seq = _clean_rna_local(item.get("sequence", ""))
    target = str(item.get("target_pdb", "") or item.get("target_pdb_id", "") or "").upper()
    dock_file = str(item.get("dock_output_file", "") or item.get("output_file", "") or "")
    complex_file = str(
        item.get("dock_complex_file")
        or item.get("complex_file")
        or item.get("complex_pdb")
        or ""
    )
    score = item.get("binding_rank_score", item.get("dg", item.get("dock_score")))

    return (target, seq, dock_file, complex_file, str(score))


def dedupe_binding_results_reducer(
    existing: list[dict] | None,
    new: list[dict] | None,
) -> list:
    """
    Merge binding results while avoiding exact duplicate docking records.

    Keeps recent history but prevents unbounded duplicate accumulation.
    """
    combined = list(existing or []) + list(new or [])

    seen: set[tuple] = set()
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
) -> list:
    """
    Merge MD results while avoiding exact duplicates.
    """
    combined = list(existing or []) + list(new or [])

    seen: set[tuple] = set()
    out: list[dict] = []

    for item in combined:
        key = _md_result_key(item)

        if key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out[-50:]


# ---------------------------------------------------------------------------
# Inhibitor reducers
# ---------------------------------------------------------------------------

def _inhibitor_sm_key(item: dict) -> tuple:
    """
    Small-molecule identity key.

    Important: do NOT include output_file or binding_energy in the identity key,
    otherwise repeated iterations accumulate the same ligand as distinct rows.
    """
    if not isinstance(item, dict):
        return ("invalid", repr(item))

    target = str(
        item.get("target_pdb")
        or item.get("target_pdb_id")
        or item.get("target_tag")
        or ""
    ).upper()

    smiles = str(item.get("smiles") or "").strip()
    cid = str(item.get("cid") or item.get("pubchem_cid") or "").strip()
    name = str(
        item.get("ligand_name")
        or item.get("display_name")
        or item.get("name")
        or item.get("compound_name")
        or item.get("_inhibitor_name")
        or ""
    ).strip().upper()

    if smiles:
        identity = ("smiles", smiles)

    elif cid:
        identity = ("cid", cid)

    else:
        identity = ("name", name)

    return (target, identity)


def dedupe_inhibitor_small_molecules_reducer(
    existing: list[dict] | None,
    new: list[dict] | None,
) -> list:
    """
    Merge small-molecule inhibitor rows by compound identity.

    Keeps the latest row for a target+compound identity, so reruns update
    scores/paths rather than inflating state from 10 to 20/30 rows.
    """
    combined = list(existing or []) + list(new or [])

    by_key: dict[tuple, dict] = {}

    for item in combined:
        if not isinstance(item, dict):
            continue

        key = _inhibitor_sm_key(item)

        if key[0] == "invalid":
            continue

        by_key[key] = item

    return list(by_key.values())[-100:]


def _inhibitor_peptide_key(item: dict) -> tuple:
    """
    Peptide identity key.

    Important: do NOT include complex_file or score in identity, otherwise
    repeated HDOCK runs accumulate the same peptide as distinct rows.
    """
    if not isinstance(item, dict):
        return ("invalid", repr(item))

    target = str(
        item.get("target_pdb")
        or item.get("target_pdb_id")
        or item.get("target_tag")
        or ""
    ).upper()

    seq = str(
        item.get("sequence")
        or item.get("peptide_sequence")
        or ""
    ).upper()

    return (target, seq)


def dedupe_inhibitor_peptides_reducer(
    existing: list[dict] | None,
    new: list[dict] | None,
) -> list:
    """
    Merge peptide inhibitor rows by target+sequence.

    Keeps latest score/complex path for each peptide.
    """
    combined = list(existing or []) + list(new or [])

    by_key: dict[tuple, dict] = {}

    for item in combined:
        if not isinstance(item, dict):
            continue

        key = _inhibitor_peptide_key(item)

        if key[0] == "invalid":
            continue

        by_key[key] = item

    return list(by_key.values())[-100:]


# ---------------------------------------------------------------------------
# Canonical LabState
# ---------------------------------------------------------------------------

class LabState(TypedDict, total=False):
    # ----------------------------------------------------------------
    # Topic / setup
    # ----------------------------------------------------------------
    topic_name: str
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
    pi_action_summary: str
    pi_operational_summary: str
    pi_training_metadata: dict
    best_interface_clean_sequence: Optional[str]
    optimisation_status: str
    mutation_bias: dict
    joint_physics_feedback: dict

    # ----------------------------------------------------------------
    # Agent outputs
    # ----------------------------------------------------------------
    evidence: Annotated[List[Any], dedupe_evidence_reducer]

    structural_analysis: str
    md_analysis: str
    protein_analysis: str
    bioinfo_analysis: str
    msa_data: str
    critique: str
    skeptic_interface_metrics: dict
    skeptic_bioinfo_metrics: dict
    skeptic_error: str | None

    # ----------------------------------------------------------------
    # Conservation signal
    # ----------------------------------------------------------------
    conservation_signal: dict
    conserved_regions: List
    conservation_fitness: float
    bioinfo_num_sequences: int
    bioinfo_alignment_length: int
    bioinfo_quality_passed: bool
    bioinfo_quality_reasons: List[str]

    # ----------------------------------------------------------------
    # Designed candidates / structured outputs
    # ----------------------------------------------------------------
    designed_sequences: Annotated[List[str], dedupe_rna_sequence_reducer]
    structural_candidates: Annotated[List[dict], append_unique_dicts_reducer]
    binding_results: Annotated[List[dict], dedupe_binding_results_reducer]
    md_results: Annotated[List[dict], dedupe_md_results_reducer]
    structural_status: str
    structural_error: str | None

    # ----------------------------------------------------------------
    # Target protein selection
    # ----------------------------------------------------------------
    target_pdb: Optional[str]
    target_pdb_id: Optional[str]
    target_pdb_path: Optional[str]
    target_pdb_candidates: List[str]
    target_pdb_rankings: List[dict]
    failed_target_pdbs: List[Any]
    target_failure_records: List[dict]
    partial_success_targets: List[Any]
    partial_success_sequences: Annotated[List[str], dedupe_string_list_reducer]
    target_pdb_selection_reason: str
    target_pdb_metadata: dict
    target_selection_mode: str
    target_sequence: Optional[str]
    target_status: str | None
    target_status_reason: str | None
    resolved_partial_success_targets: Annotated[List[str], dedupe_string_list_reducer]
    docking_preferences: Annotated[List[dict], append_unique_dicts_reducer]
    seq_len: int

    # ----------------------------------------------------------------
    # Literature learning / biological priors
    # ----------------------------------------------------------------
    research_query: str | None
    literature_query_bundle: Annotated[List[str], dedupe_string_list_reducer]
    literature_topic_profile: dict
    streamed_queries: Annotated[List[str], dedupe_string_list_reducer]
    streamed_query_keys: Annotated[List[str], dedupe_string_list_reducer]
    streaming_started: bool

    literature_motif_hints: Annotated[List[str], dedupe_string_list_reducer]
    literature_target_hints: Annotated[List[str], dedupe_string_list_reducer]
    literature_policy_text: str

    # ----------------------------------------------------------------
    # Docking exports
    # ----------------------------------------------------------------
    binding_units: str
    docking_summary_json: str
    docking_summary_csv: str
    docking_summary_md: str
    interface_contacts: Optional[dict]
    interface_contact_files: Annotated[List[str], dedupe_string_list_reducer]

    # ----------------------------------------------------------------
    # Inhibitor screening
    # ----------------------------------------------------------------
    inhibitor_enabled: bool
    inhibitor_small_molecules: Annotated[List[dict], dedupe_inhibitor_small_molecules_reducer]
    inhibitor_peptides: Annotated[List[dict], dedupe_inhibitor_peptides_reducer]
    inhibitor_best_small_molecule: dict | None
    inhibitor_best_peptide: dict | None

    # Legacy-compatible percent field.
    inhibitor_binding_site_overlap: float

    # Structured atom-overlap fields.
    inhibitor_binding_site_overlap_score: float
    inhibitor_pose_comparison: dict
    inhibitor_small_molecule_comparison: dict | None
    inhibitor_peptide_comparison: dict | None

    inhibitor_docking_box: dict
    inhibitor_analysis: str
    inhibitor_summary: str
    inhibitor_snapshot_paths: Annotated[List[str], dedupe_string_list_reducer]

    # Vina reproducibility and persistent result-cache metadata.
    # Per-ligand fields remain nested inside inhibitor_small_molecules rows.
    inhibitor_vina_seed: int | None
    inhibitor_vina_cache_hits: int
    inhibitor_vina_cache_misses: int
    inhibitor_vina_cache_enabled: bool

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
    _run_system_interface_objective_enabled: bool
    _run_system_interface_lookup_count: int
    _run_system_interface_failure_weights: dict
    _run_system_seq_len: int
    _run_system_conservation_valid: bool
    _run_system_sequence_scores: dict
    _run_system_best_interface_sequence: Optional[str]
    _pi_excluded_interface_clashes: List[str]
    _md_cache: dict
    _dock_cache: dict


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def timestamp() -> str:
    """
    Return current UTC timestamp.
    """
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

    Both 'agent' and 'stage' are included so old/new aggregators do not label
    valid outputs as UNKNOWN.
    """
    if output is None:
        output = ""

    output = str(output)

    if not summary:
        summary = output[:300] + ("..." if len(output) > 300 else "")

    return {
        "agent": agent_name,
        "stage": agent_name,
        "summary": summary,
        "output": output,
        "content": output,
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

    Also handles numpy scalars/arrays and pydantic objects without requiring
    hard imports.
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Path-like support
    try:
        from pathlib import Path
        if isinstance(obj, Path):
            return str(obj)
    except Exception:
        pass

    # NumPy scalar support without importing numpy.
    if hasattr(obj, "item") and not isinstance(obj, (dict, list, tuple, set, str, bytes)):
        try:
            return safe_jsonable(obj.item())
        except Exception:
            pass

    # NumPy arrays / pandas-like support.
    if hasattr(obj, "tolist"):
        try:
            return safe_jsonable(obj.tolist())
        except Exception:
            pass

    # Pydantic v2.
    if hasattr(obj, "model_dump"):
        try:
            return safe_jsonable(obj.model_dump())
        except Exception:
            pass

    # Pydantic v1.
    if hasattr(obj, "dict"):
        try:
            return safe_jsonable(obj.dict())
        except Exception:
            pass

    if isinstance(obj, list):
        return [safe_jsonable(x) for x in obj]

    if isinstance(obj, tuple):
        return [safe_jsonable(x) for x in obj]

    if isinstance(obj, set):
        return [safe_jsonable(x) for x in sorted(obj, key=str)]

    if isinstance(obj, dict):
        out = {}

        for k, v in obj.items():
            # Skip runtime-only/non-serialisable objects.
            if k in {"wrappers", "data_collector", "llm", "language_model"}:
                continue

            out[str(k)] = safe_jsonable(v)

        return out

    try:
        return str(obj)
    except Exception:
        return f"<non_jsonable:{type(obj).__name__}>"