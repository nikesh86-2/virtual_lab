from __future__ import annotations

import logging
import os
from typing import Any, List

from langchain_core.messages import HumanMessage, SystemMessage

from VLAB2.core.gpu_manager import clear_gpu
from VLAB2.core.protein_prep import ensure_protein_pdb
from VLAB2.core.rna_prep import prepare_rna_pdb_for_hdock

from VLAB2.orchestration.failure_memory import FailureMemory
from VLAB2.orchestration.llm import get_llm
from VLAB2.orchestration.state_schema import (
    LabState,
    add_conversation_entry,
    record_stage_output,
)

from VLAB2.orchestration.utils.checkpointing import save_checkpoint
from VLAB2.orchestration.utils.docking_visuals import (
    render_docking_snapshots_for_results,
)
from VLAB2.orchestration.utils.interface_contacts import (
    analyse_interface_contacts_for_results,
)
from VLAB2.orchestration.utils.docking_utils import (
    export_docking_outputs,
    parse_pdb_candidates,
    safe_binding_rank,
)
from VLAB2.orchestration.utils.env_utils import getenv_int
from VLAB2.orchestration.utils.sequence_utils import dedupe_rna_sequences
from VLAB2.orchestration.literature_memory import LiteratureMemory

try:
    from VLAB2.core.rcsb_target_selector import select_pdb_targets
except Exception:
    select_pdb_targets = None


log = logging.getLogger("virtual_lab")

__all__ = ["protein_agent"]


# ---------------------------------------------------------------------------
# Target/docking status constants
# ---------------------------------------------------------------------------

ACCEPTED_TARGET_STATUS = "accepted_target"
PARTIAL_SUCCESS_TARGET_STATUS = "partial_success_target"
FAILED_TARGET_STATUS = "failed_target"

PARTIAL_REASON_HDOCK_INTERFACE = "hdock_passed_interface_partially_failed"
FAILED_REASON_INSUFFICIENT_INTERFACE = "insufficient_interface_clean_docking"
FAILED_REASON_NO_CLEAN_INTERFACE = "no_clean_interface"
FAILED_REASON_STERIC_CLASH = "steric_clash"
FAILED_REASON_SCORE_ARTEFACT = "implausible_hdock_score"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)

    if raw is None:
        return default

    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalise_pdb_id(value: Any) -> str:
    return str(value or "").strip().upper()


def _failed_target_record(
    pdb_id: str,
    reason: str,
    metadata: dict | None = None,
) -> dict:
    return {
        "pdb_id": _normalise_pdb_id(pdb_id),
        "reason": reason or "unknown",
        "metadata": metadata or {},
    }


def _dedupe_failed_targets(records: list[Any]) -> list:
    """
    Dedupe failed targets while supporting both legacy list[str] and new list[dict].

    Latest record for each PDB wins.
    """
    seen: dict[str, dict] = {}

    for item in records or []:
        if isinstance(item, dict):
            pdb = _normalise_pdb_id(item.get("pdb_id") or item.get("target_pdb"))

            if not pdb:
                continue

            seen[pdb] = {
                "pdb_id": pdb,
                "reason": item.get("reason", "unknown"),
                "metadata": item.get("metadata", {}),
            }

        elif item:
            pdb = _normalise_pdb_id(item)

            if pdb:
                seen[pdb] = {
                    "pdb_id": pdb,
                    "reason": "unknown",
                    "metadata": {},
                }

    return list(seen.values())


def _split_failed_target_sets(records: list[Any]) -> tuple[set[str], set[str]]:
    """
    Return:
      hard_failed_set, soft_failed_set

    Legacy string failures are treated as hard/unknown exclusions to preserve
    old behaviour. New dict records can be soft and therefore reusable.
    """
    hard: set[str] = set()
    soft: set[str] = set()

    for item in records or []:
        if isinstance(item, dict):
            pdb = _normalise_pdb_id(item.get("pdb_id") or item.get("target_pdb"))
            reason = item.get("reason", "unknown")

            if not pdb:
                continue

            if reason in FailureMemory.HARD_TARGET_FAILURES:
                hard.add(pdb)
            elif reason in FailureMemory.SOFT_TARGET_FAILURES:
                soft.add(pdb)
            else:
                # Unknown dict reason is conservative but not as hard as old strings.
                soft.add(pdb)

        elif item:
            pdb = _normalise_pdb_id(item)

            if pdb:
                hard.add(pdb)

    return hard, soft


def _is_dock_valid_row(row: dict) -> bool:
    return (
        isinstance(row, dict)
        and row.get("binding_mode") != "rejected_docking"
        and (row.get("dock_valid") is True or row.get("vina_valid") is True)
    )


def _has_clean_interface(row: dict) -> bool:
    return (
        isinstance(row, dict)
        and row.get("dock_valid") is True
        and row.get("interface_passed") is True
        and row.get("interface_steric_clash") is not True
    )


def _has_steric_clash(row: dict) -> bool:
    return isinstance(row, dict) and row.get("interface_steric_clash") is True


def _get_hdock_relative_score(row: dict) -> float | None:
    if not isinstance(row, dict):
        return None

    score = row.get("hdock_score")

    if score is None:
        score = row.get("dock_score")

    if score is None:
        score = row.get("vina_energy")

    try:
        return float(score)
    except Exception:
        return None


def _classify_docking_training_label(row: dict) -> str:
    """
    Assign explicit training label for docking/interface rows.

    This prevents dock_valid=True from being treated as a positive example
    unless the interface is also clean.
    """
    if not isinstance(row, dict):
        return "negative_invalid_row"

    dock_valid = _is_dock_valid_row(row)
    interface_passed = row.get("interface_passed") is True
    steric_clash = row.get("interface_steric_clash") is True

    if dock_valid and interface_passed and not steric_clash:
        return "positive_interface_clean_pose"

    if dock_valid and steric_clash:
        return "negative_steric_clash_pose"

    if dock_valid and not interface_passed:
        return "caution_hdock_valid_interface_failed"

    return "negative_docking_failed"


def _extract_pdb_from_partial_item(item: Any) -> str:
    """
    Supports partial_success_targets stored as either:
      - "8K75"
      - {"target_pdb": "8K75", ...}
      - {"pdb_id": "8K75", ...}
    """
    if isinstance(item, dict):
        return _normalise_pdb_id(item.get("target_pdb") or item.get("pdb_id"))

    return _normalise_pdb_id(item)


def _dedupe_partial_success_targets(records: list[Any]) -> list:
    """
    Dedupe partial success targets.

    Keeps structured dict records. Legacy string entries are upgraded to:
      {
        "target_pdb": PDB,
        "pdb_id": PDB,
        "status": "partial_success_target",
        "reason": "legacy_partial_success_target"
      }
    """
    seen: dict[str, dict] = {}

    for item in records or []:
        if isinstance(item, dict):
            pdb = _normalise_pdb_id(item.get("target_pdb") or item.get("pdb_id"))

            if not pdb:
                continue

            rec = dict(item)
            rec["target_pdb"] = pdb
            rec["pdb_id"] = pdb
            rec.setdefault("status", PARTIAL_SUCCESS_TARGET_STATUS)
            rec.setdefault("reason", PARTIAL_REASON_HDOCK_INTERFACE)
            rec.setdefault("binding_units", "hdock_relative_score")
            rec.setdefault("binding_energy_is_physical", False)
            seen[pdb] = rec

        elif item:
            pdb = _normalise_pdb_id(item)

            if not pdb:
                continue

            seen[pdb] = {
                "target_pdb": pdb,
                "pdb_id": pdb,
                "status": PARTIAL_SUCCESS_TARGET_STATUS,
                "reason": "legacy_partial_success_target",
                "binding_units": "hdock_relative_score",
                "binding_energy_is_physical": False,
                "recommendation": "reuse_as_priority_candidate_but_not_final_target",
            }

    return list(seen.values())


def _build_partial_success_record(
    pdb_id: str,
    docking_results: list[dict],
    reason: str = PARTIAL_REASON_HDOCK_INTERFACE,
) -> dict:
    """
    Build structured lower-confidence target record.

    This is for targets such as 8K75:
      - enough HDOCK-valid poses
      - at least one clean interface
      - not enough clean interfaces for full target acceptance
    """
    pdb_id = _normalise_pdb_id(pdb_id)

    dock_valid = [
        r for r in docking_results or []
        if _is_dock_valid_row(r)
    ]

    interface_clean = [
        r for r in dock_valid
        if _has_clean_interface(r)
    ]

    steric_clash = [
        r for r in dock_valid
        if _has_steric_clash(r)
    ]

    scores = [
        _get_hdock_relative_score(r)
        for r in dock_valid
    ]
    scores = [s for s in scores if s is not None]

    return {
        "target_pdb": pdb_id,
        "pdb_id": pdb_id,
        "status": PARTIAL_SUCCESS_TARGET_STATUS,
        "reason": reason,
        "recommendation": "reuse_as_priority_candidate_but_not_final_target",

        "binding_units": "hdock_relative_score",
        "binding_energy_is_physical": False,

        "dock_valid_count": len(dock_valid),
        "interface_clean_count": len(interface_clean),
        "steric_clash_count": len(steric_clash),

        "best_hdock_relative_score": min(scores) if scores else None,
        "score_spread": (
            max(scores) - min(scores)
            if len(scores) >= 2
            else None
        ),

        "clean_sequences": [
            r.get("sequence") for r in interface_clean if r.get("sequence")
        ],
        "clash_sequences": [
            r.get("sequence") for r in steric_clash if r.get("sequence")
        ],

        "training_label": "partial_success_target",
        "target_selection_label": PARTIAL_REASON_HDOCK_INTERFACE,

        "natural_language_summary": (
            f"Target {pdb_id} produced HDOCK-relative docking-valid results and "
            "at least one interface-clean pose, but failed full target acceptance "
            "because too few poses passed interface validation. Reuse as a "
            "priority partial-success target, but do not treat as a final "
            "validated binding system."
        ),
    }


def _build_interface_preferences(
    binding_results: list[dict],
    research_topic: str = "",
) -> list:
    """
    Generate preference examples:

      1. interface-clean pose > steric-clash pose
      2. reasonable HDOCK score + clean interface >
         stronger HDOCK score + clash

    HDOCK scores are relative docking scores, not physical binding energies.
    """
    clean = []
    clashes = []

    for r in binding_results or []:
        if not _is_dock_valid_row(r):
            continue

        if _has_clean_interface(r):
            clean.append(r)

        elif _has_steric_clash(r):
            clashes.append(r)

    preferences: list[dict] = []

    for good in clean:
        for bad in clashes:
            good_score = _get_hdock_relative_score(good)
            bad_score = _get_hdock_relative_score(bad)

            if good_score is None or bad_score is None:
                continue

            if bad_score < good_score:
                preference_type = (
                    "reasonable_hdock_clean_interface_over_stronger_hdock_clash"
                )
                rationale = (
                    "Preferred the interface-clean pose despite a weaker "
                    "HDOCK-relative score because the alternative has a steric clash."
                )
            else:
                preference_type = "interface_clean_over_steric_clash"
                rationale = (
                    "Preferred the interface-clean pose over a steric-clash pose."
                )

            preferences.append(
                {
                    "schema_version": "docking_preference.v1",
                    "preference_type": preference_type,
                    "research_topic": research_topic,
                    "label": "prefer_clean_interface_over_raw_hdock_score",

                    "preferred": {
                        "sequence": good.get("sequence"),
                        "target_pdb": good.get("target_pdb"),
                        "hdock_relative_score": good_score,
                        "interface_passed": good.get("interface_passed"),
                        "interface_steric_clash": good.get("interface_steric_clash"),
                        "interface_quality_score": good.get("interface_quality_score"),
                        "interface_residue_contacts": good.get("interface_residue_contacts"),
                        "interface_basic_residue_contacts": good.get("interface_basic_residue_contacts"),
                        "interface_min_distance_A": good.get("interface_min_distance_A"),
                        "dock_complex_file": good.get("dock_complex_file"),
                        "training_label": good.get("training_label"),
                    },

                    "rejected": {
                        "sequence": bad.get("sequence"),
                        "target_pdb": bad.get("target_pdb"),
                        "hdock_relative_score": bad_score,
                        "interface_passed": bad.get("interface_passed"),
                        "interface_steric_clash": bad.get("interface_steric_clash"),
                        "interface_quality_score": bad.get("interface_quality_score"),
                        "interface_residue_contacts": bad.get("interface_residue_contacts"),
                        "interface_basic_residue_contacts": bad.get("interface_basic_residue_contacts"),
                        "interface_min_distance_A": bad.get("interface_min_distance_A"),
                        "dock_complex_file": bad.get("dock_complex_file"),
                        "training_label": bad.get("training_label"),
                    },

                    "rationale": rationale,
                    "binding_units": "hdock_relative_score",
                    "binding_energy_is_physical": False,
                }
            )

    return preferences


def _dedupe_preferences(preferences: list[dict]) -> list:
    seen: set[tuple] = set()
    out: list[dict] = []

    for pref in preferences or []:
        if not isinstance(pref, dict):
            continue

        preferred = pref.get("preferred", {}) or {}
        rejected = pref.get("rejected", {}) or {}

        key = (
            pref.get("preference_type"),
            preferred.get("target_pdb"),
            preferred.get("sequence"),
            rejected.get("target_pdb"),
            rejected.get("sequence"),
        )

        if key in seen:
            continue

        seen.add(key)
        out.append(pref)

    return out


# ---------------------------------------------------------------------------
# Sequence collection / fold gating
# ---------------------------------------------------------------------------

def _collect_sequences_for_protein(state: LabState, eval_top_n: int) -> list:
    """
    Collect RNA sequences for protein/docking evaluation.

    Priority:
      1. target_sequence
      2. designed_sequences
      3. structural_candidates
    """
    sequences = []

    if state.get("target_sequence"):
        sequences.append(state["target_sequence"])

    sequences.extend(state.get("designed_sequences", []) or [])

    for candidate in state.get("structural_candidates", []) or []:
        if isinstance(candidate, dict) and candidate.get("sequence"):
            sequences.append(candidate["sequence"])

    sequences = dedupe_rna_sequences(sequences)[:eval_top_n]
    sequences = [s for s in sequences if s]

    return sequences


def _fold_gate_blocks_protein(state: LabState) -> bool:
    """
    Return True if protein docking should be skipped due to fold failure.

    Candidate-aware:
      - if global selected fold failed but at least one structural candidate passed,
        do not block downstream docking.
    """
    if not (
        state.get("fold_thresholds_passed") is False
        and os.getenv("VLAB_RNA_ENFORCE_MIN_FOLD", "1").strip() == "1"
    ):
        return False

    structural_candidates = state.get("structural_candidates", []) or []

    any_passed = any(
        isinstance(c, dict)
        and isinstance(c.get("fold_quality"), dict)
        and c["fold_quality"].get("passed")
        for c in structural_candidates
    )

    return not any_passed


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------

def _select_candidate_pdbs(state: LabState) -> tuple[list[str], str, list[dict]]:
    """
    Select candidate protein PDB IDs.

    Order:
      1. existing target_pdb, unless reuse is disabled or target has hard-failed
      2. partial-success target memory
      3. existing target_pdb_candidates
      4. dynamic RCSB ranking if enabled
      5. LLM PDB selection fallback, if enabled
      6. VLAB_FALLBACK_PDBS env fallback, only if explicitly enabled

    Failed and blacklisted target PDBs are excluded after candidate gathering.

    Generic fallback PDBs are disabled by default via:
      VLAB_ALLOW_GENERIC_TARGET_FALLBACK=0
    """
    candidate_pdbs: list[str] = []
    selection_reason = ""
    target_rankings: list[dict] = list(state.get("target_pdb_rankings", []) or [])

    state_failed_records = state.get("failed_target_pdbs", []) or []
    hard_failed_set, _ = _split_failed_target_sets(state_failed_records)

    try:
        fm = FailureMemory()
        memory_partial_targets = fm.get_partial_success_targets()
        memory_hard_failed_targets = fm.get_hard_failed_targets()
    except Exception as e:
        log.warning("Failed to load FailureMemory for target selection: %s", e)
        memory_partial_targets = []
        memory_hard_failed_targets = set()

    hard_failed_set = set(hard_failed_set) | set(memory_hard_failed_targets)

    blacklist = {
        x.strip().upper()
        for x in os.getenv("VLAB_TARGET_BLACKLIST", "").split(",")
        if x.strip()
    }

    invalid_tokens = {"NONE", "NULL", "N/A", ""}

    allow_reuse_existing_target = _env_bool(
        "VLAB_ALLOW_REUSE_EXISTING_TARGET",
        default=True,
    )

    allow_llm_target_fallback = _env_bool(
        "VLAB_ALLOW_LLM_TARGET_FALLBACK",
        default=True,
    )

    allow_generic_fallback = _env_bool(
        "VLAB_ALLOW_GENERIC_TARGET_FALLBACK",
        default=False,
    )

    use_partial_success_memory = _env_bool(
        "VLAB_USE_PARTIAL_SUCCESS_TARGET_MEMORY",
        default=True,
    )

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

    try:
        lm = LiteratureMemory()
        literature_policy_text = lm.build_target_policy_text()
    except Exception:
        literature_policy_text = ""

    def _normalise_pdbs(values: list[Any]) -> list:

        out: list[str] = []

        for item in values or []:
            if isinstance(item, dict):
                p = _normalise_pdb_id(item.get("target_pdb") or item.get("pdb_id"))
            else:
                p = _normalise_pdb_id(item)

            if p in invalid_tokens:
                continue

            if len(p) != 4 or not p[0].isdigit():
                continue

            if p in hard_failed_set:
                continue

            if p in blacklist:
                continue

            out.append(p)

        return list(dict.fromkeys(out))

    # ------------------------------------------------------------------
    # 1. Reuse existing accepted target only if allowed and not hard-failed.
    # ------------------------------------------------------------------
    existing_target = state.get("target_pdb")

    if existing_target and allow_reuse_existing_target:
        existing = _normalise_pdbs([existing_target])

        if existing:
            candidate_pdbs.extend(existing)
            selection_reason = "reused_existing_target"
        else:
            log.info(
                "Not reusing existing target_pdb=%s because it is invalid, hard-failed, "
                "or blacklisted.",
                existing_target,
            )

    # ------------------------------------------------------------------
    # 2. Reuse partial-success target memory.
    # ------------------------------------------------------------------
    if not candidate_pdbs and use_partial_success_memory:
        partial_items: list[Any] = []
        partial_items.extend(state.get("partial_success_targets", []) or [])
        partial_items.extend(memory_partial_targets or [])

        partial_targets = _normalise_pdbs(partial_items)

        if partial_targets:
            candidate_pdbs.extend(partial_targets)
            selection_reason = "partial_success_target_memory"

            log.info(
                "Using partial-success target candidates from memory: %s",
                ", ".join(candidate_pdbs),
            )

    # ------------------------------------------------------------------
    # 3. Reuse existing candidate list.
    # ------------------------------------------------------------------
    if not candidate_pdbs and state.get("target_pdb_candidates"):
        reused_candidates = _normalise_pdbs(
            [str(p) for p in state.get("target_pdb_candidates", []) or []]
        )

        if reused_candidates:
            candidate_pdbs.extend(reused_candidates)
            selection_reason = "reused_candidate_list"

    # ------------------------------------------------------------------
    # 4. Dynamic RCSB ranking.
    # ------------------------------------------------------------------
    if not candidate_pdbs:
        if target_selection_mode == "rcsb_dynamic" and select_pdb_targets is not None:
            try:
                ranked_targets = select_pdb_targets(
                    topic=state.get("research_topic", ""),
                    hypothesis=(
                        f"{topic_text}\n\n"
                        f"{literature_policy_text}"
                    ),
                    virus_name=state.get("virus_name", ""),
                    virus_family=state.get("virus_family", ""),
                    exclude_pdbs=list(hard_failed_set),
                )

                target_rankings = ranked_targets

                ranked_pdbs = _normalise_pdbs(
                    [x.get("pdb_id") for x in ranked_targets if isinstance(x, dict)]
                )

                if ranked_pdbs:
                    candidate_pdbs.extend(ranked_pdbs)
                    selection_reason = "rcsb_dynamic_ranked"

                    log.info(
                        "RCSB dynamic target candidates: %s",
                        ", ".join(candidate_pdbs),
                    )
                else:
                    log.warning(
                        "RCSB dynamic target selection returned no usable candidates."
                    )

            except Exception as e:
                log.warning("RCSB dynamic target selection failed: %s", e)

    # ------------------------------------------------------------------
    # 5. LLM target fallback, optional.
    # ------------------------------------------------------------------
    if not candidate_pdbs and allow_llm_target_fallback:
        try:
            lookup_msg = get_llm(temperature=0.0).invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a structural biologist selecting protein targets "
                            "from the RCSB PDB for RNA docking.\n\n"
                            "Return exactly 5 PDB IDs, one per line, no other text.\n"
                            "Prefer experimentally resolved viral RNA-binding proteins, "
                            "nucleocapsid or capsid proteins, RNA packaging proteins, "
                            "RNA-protein complexes, or RNA stem-loop binding systems.\n"
                            "Avoid antibody-only, spike-only, polymerase, protease, "
                            "membrane fusion-core, and oversized whole-particle targets."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Topic/hypothesis:\n{topic_text}\n\n"
                            f"{literature_policy_text}"
                        )
                    ),
                ]
            )

            llm_candidates = _normalise_pdbs(
                parse_pdb_candidates(lookup_msg.content)
            )

            if llm_candidates:
                candidate_pdbs.extend(llm_candidates)
                selection_reason = "llm_generated_candidates"

        except Exception as e:
            log.warning("LLM target selection failed: %s", e)

    # ------------------------------------------------------------------
    # 6. Generic env fallback, explicitly opt-in only.
    # ------------------------------------------------------------------
    if not candidate_pdbs and allow_generic_fallback:
        fallback_env = os.getenv("VLAB_FALLBACK_PDBS", "")

        fallback_list = _normalise_pdbs(
            [x.strip().upper() for x in fallback_env.split(",") if x.strip()]
        )

        if fallback_list:
            log.warning(
                "[DOCKING CONFIG] Using generic env fallback PDBs because no "
                "dynamic/LLM/partial-memory candidates were usable: %s",
                fallback_list,
            )
            candidate_pdbs.extend(fallback_list)
            selection_reason = "env_fallback_candidates"

    elif not candidate_pdbs and not allow_generic_fallback:
        log.warning(
            "No usable PDB targets found and generic fallback is disabled. "
            "Set VLAB_ALLOW_GENERIC_TARGET_FALLBACK=1 to allow VLAB_FALLBACK_PDBS."
        )

    candidate_pdbs = _normalise_pdbs(candidate_pdbs)

    if not selection_reason and candidate_pdbs:
        selection_reason = "unknown_candidate_source"

    return candidate_pdbs, selection_reason, target_rankings


# ---------------------------------------------------------------------------
# Docking result helpers
# ---------------------------------------------------------------------------

def _find_md_result_for_sequence(state: LabState, seq: str) -> dict | None:
    for m in state.get("md_results", []):
        if isinstance(m, dict) and m.get("sequence") == seq:
            result = m.get("result")

            if isinstance(result, dict):
                return result

    return None


def _initialise_docking_fields(r: dict) -> dict:
    """
    Initialise HDOCK fields and legacy Vina compatibility fields.
    """
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

    r["binding_mode"] = "proxy"
    r["binding_energy_is_physical"] = False

    return r


def _set_docking_skip(r: dict, method: str, error: str) -> dict:
    r["dock_method"] = method
    r["dock_error"] = error
    r["vina_method"] = method
    r["vina_error"] = error
    r["binding_mode"] = "proxy"
    r["binding_energy_is_physical"] = False

    return r


def _score_and_rank_binding_results(valid_binding_results: list[dict]) -> list:
    """
    Convert raw HDOCK/proxy fields into binding_rank_score and ranks.

    Lower score is better.
    """
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

            # HDOCK-aware ranking score, not a physical kcal/mol free energy.
            r["binding_rank_score"] = 0.7 * dock_score + 0.3 * proxy_dg
            r["dg"] = r["binding_rank_score"]
            r["binding_units"] = "hdock_relative_score"
            r["binding_energy_is_physical"] = False
            r["binding_mode"] = "docked"

        elif r.get("vina_valid") and r.get("vina_energy") is not None:
            dock_score = float(r["vina_energy"])

            r["proxy_dg"] = proxy_dg
            r["hdock_score"] = dock_score
            r["binding_rank_score"] = 0.7 * dock_score + 0.3 * proxy_dg
            r["dg"] = r["binding_rank_score"]
            r["binding_units"] = "hdock_relative_score"
            r["binding_energy_is_physical"] = False
            r["binding_mode"] = "docked"

        else:
            r["binding_rank_score"] = proxy_dg
            r["dg"] = proxy_dg
            r["binding_units"] = "proxy_score"
            r["binding_energy_is_physical"] = False
            r["binding_mode"] = "proxy"

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

    return valid_binding_results


def _count_pdb_atoms(path: str) -> int:
    """
    Count ATOM/HETATM records in a PDB file.
    """
    count = 0

    try:
        with open(path, "r", errors="ignore") as handle:
            for line in handle:
                if line.startswith(("ATOM", "HETATM")):
                    count += 1
    except Exception:
        return 0

    return count


def _reject_implausible_hdock_score(result: dict) -> dict:
    """
    Reject pathological HDOCK-relative scores that usually indicate receptor,
    chain, complex-preparation, or parser artefacts.

    HDOCK scores are relative docking scores, not physical binding energies.
    Extremely large absolute values, e.g. ~-1000, are treated as invalid for
    this workflow.
    """
    if not isinstance(result, dict):
        return result

    try:
        max_abs_reasonable_hdock = float(
            os.getenv("VLAB_MAX_ABS_REASONABLE_HDOCK_SCORE", "300")
        )
    except Exception:
        max_abs_reasonable_hdock = 300.0

    score = result.get("dock_score")

    if score is None:
        score = result.get("hdock_score")

    if score is None:
        return result

    try:
        score_f = float(score)
    except Exception:
        return result

    if abs(score_f) > max_abs_reasonable_hdock:
        error_msg = (
            f"HDOCK score {score_f} exceeds plausible absolute bound "
            f"{max_abs_reasonable_hdock}; likely receptor/complex artefact."
        )

        result["dock_valid"] = False
        result["valid"] = False
        result["vina_valid"] = False
        result["binding_mode"] = "rejected_docking"
        result["dock_error"] = error_msg
        result["error"] = error_msg
        result["vina_error"] = error_msg
        result["implausible_hdock_score"] = True
        result["max_abs_reasonable_hdock_score"] = max_abs_reasonable_hdock
        result["binding_energy_is_physical"] = False
        result["binding_units"] = "hdock_relative_score"

        log.warning(
            "[HDOCK REJECTED] implausible score %.3f exceeds abs bound %.3f",
            score_f,
            max_abs_reasonable_hdock,
        )

    return result


def _updated_md_results_with_docking(
    state: LabState,
    seq: str,
    pdb_id: str,
    dock_score: float,
) -> list:
    """
    Preserve old behaviour of exposing docking info inside md_results, but without
    mutating the incoming state in-place.
    """
    updated = []

    for m in state.get("md_results", []) or []:
        if not isinstance(m, dict):
            updated.append(m)
            continue

        copied = dict(m)

        if m.get("sequence") == seq and isinstance(m.get("result"), dict):
            copied_result = dict(m["result"])
            copied_result["dock_score"] = dock_score
            copied_result["dock_valid"] = True
            copied_result["dock_target_pdb"] = pdb_id
            copied_result["dock_method"] = "hdock"
            copied_result["binding_units"] = "hdock_relative_score"
            copied_result["binding_energy_is_physical"] = False

            # Legacy compatibility.
            copied_result["vina_energy"] = dock_score
            copied_result["vina_valid"] = True
            copied_result["vina_target_pdb"] = pdb_id

            copied["result"] = copied_result

        updated.append(copied)

    return updated


def _update_failure_memory_from_result(result: dict) -> None:
    """
    Persist interface/target outcomes after protein agent has produced result.
    This is best-effort and must never break the run.
    """
    try:
        fm = FailureMemory()
        fm.ingest_run_state(result)
        fm.save()
    except Exception as e:
        log.warning("Failed to update FailureMemory from protein result: %s", e)


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

def protein_agent(state: LabState) -> dict:
    """
    Protein/RNA binding evaluation agent.

    Behaviour:
      - evaluates up to VLAB_AGENT_EVAL_TOP_N RNA candidates
      - dynamically selects or reuses target PDBs
      - uses partial-success target memory as priority candidates
      - runs wrapper proxy scoring
      - requires MD-generated RNA PDBs for HDOCK
      - prepares receptor and RNA ligand PDBs
      - runs HDOCK
      - rejects implausible HDOCK artefacts
      - analyses interface contacts
      - supports lower-confidence partial target state:
          partial_success_target / hdock_passed_interface_partially_failed
      - generates interface-aware preference data
      - exports docking summary files
      - returns ranked binding_results
    """
    log.info("--- PROTEIN AGENT: Binding Evaluation ---")

    try:
        require_docking = os.getenv("VLAB_REQUIRE_DOCKING", "1").strip() == "1"
        docking_backend = os.getenv("VLAB_DOCKING_BACKEND", "hdock").strip().lower()
        eval_top_n = getenv_int("VLAB_AGENT_EVAL_TOP_N", 3, min_value=1)

        max_dockings_per_target = getenv_int(
            "VLAB_MAX_DOCKINGS_PER_TARGET",
            eval_top_n,
            min_value=1,
        )

        min_valid_hdock_score = float(
            os.getenv("VLAB_MIN_VALID_HDOCK_SCORE", "-30")
        )

        min_valid_dockings_per_target = getenv_int(
            "VLAB_MIN_VALID_DOCKINGS_PER_TARGET",
            min(eval_top_n, max_dockings_per_target),
            min_value=1,
        )

        reject_target_on_score_spread = (
            os.getenv("VLAB_REJECT_TARGET_ON_SCORE_SPREAD", "1").strip() == "1"
        )

        max_target_score_spread = float(
            os.getenv("VLAB_MAX_TARGET_SCORE_SPREAD", "25")
        )

        abort_target_on_first_timeout = (
            os.getenv("VLAB_ABORT_TARGET_ON_FIRST_TIMEOUT", "1").strip() == "1"
        )

        max_receptor_atoms = getenv_int(
            "VLAB_MAX_RECEPTOR_ATOMS",
            60000,
            min_value=1000,
        )

        allow_partial_interface_target = _env_bool(
            "VLAB_ALLOW_PARTIAL_INTERFACE_TARGET",
            default=True,
        )

        min_partial_interface_clean = getenv_int(
            "VLAB_MIN_PARTIAL_INTERFACE_CLEAN",
            1,
            min_value=1,
        )

        if docking_backend != "hdock":
            log.warning(
                "Unsupported VLAB_DOCKING_BACKEND=%s; falling back to hdock.",
                docking_backend,
            )
            docking_backend = "hdock"

        sequences = _collect_sequences_for_protein(state, eval_top_n)

        log.info("[DOCKING DEBUG] Sequences for evaluation: %d", len(sequences))
        log.info(
            "[DOCKING CONFIG] require=%s backend=%s eval_top_n=%d max_dockings=%d "
            "min_score=%.3f min_valid=%d reject_spread=%s max_spread=%.3f "
            "max_receptor_atoms=%d abort_timeout=%s partial_interface=%s "
            "min_partial_clean=%d",
            require_docking,
            docking_backend,
            eval_top_n,
            max_dockings_per_target,
            min_valid_hdock_score,
            min_valid_dockings_per_target,
            reject_target_on_score_spread,
            max_target_score_spread,
            max_receptor_atoms,
            abort_target_on_first_timeout,
            allow_partial_interface_target,
            min_partial_interface_clean,
        )

        # ------------------------------------------------------------
        # Fold-aware gate
        # ------------------------------------------------------------
        if _fold_gate_blocks_protein(state):
            reasons = state.get("fold_threshold_reasons", [])

            summary = (
                "Protein docking skipped because all RNA candidates failed minimum "
                "fold thresholds: "
                + "; ".join(map(str, reasons))
            )

            result = {
                "protein_analysis": summary,
                "binding_results": [],
                "failed_sequences": [],
                "target_pdb": state.get("target_pdb"),
                "target_pdb_candidates": state.get("target_pdb_candidates", []),
                "failed_target_pdbs": state.get("failed_target_pdbs", []),
                "partial_success_targets": state.get("partial_success_targets", []),
                "partial_success_sequences": state.get("partial_success_sequences", []),
                "target_failure_records": state.get("target_failure_records", []),
                "target_pdb_rankings": state.get("target_pdb_rankings", []),
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

            save_checkpoint({**state, **result})
            return result

        if (
            state.get("fold_thresholds_passed") is False
            and os.getenv("VLAB_RNA_ENFORCE_MIN_FOLD", "1").strip() == "1"
        ):
            log.info(
                "Global selected fold failed, but at least one structural candidate passed; "
                "continuing protein evaluation on candidate set."
            )

        # ------------------------------------------------------------
        # No sequence guard
        # ------------------------------------------------------------
        if not sequences:
            summary = "No sequences available for protein analysis."

            result = {
                "protein_analysis": summary,
                "binding_results": [],
                "failed_sequences": [],
                "partial_success_targets": state.get("partial_success_targets", []),
                "partial_success_sequences": state.get("partial_success_sequences", []),
                "target_failure_records": state.get("target_failure_records", []),
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

            save_checkpoint({**state, **result})
            return result

        pw = state["wrappers"]["protein"]

        candidate_pdbs, selection_reason, target_rankings = _select_candidate_pdbs(
            state
        )

        failed_targets = _dedupe_failed_targets(
            list(state.get("failed_target_pdbs", []) or [])
        )

        partial_success_targets = _dedupe_partial_success_targets(
            list(state.get("partial_success_targets", []) or [])
        )

        partial_success_sequences = list(
            dict.fromkeys(
                str(x)
                for x in (state.get("partial_success_sequences", []) or [])
                if x
            )
        )

        target_failure_records = list(state.get("target_failure_records", []) or [])

        if not candidate_pdbs:
            summary = "No valid protein targets available after filtering."

            result = {
                "protein_analysis": summary,
                "binding_results": [],
                "failed_sequences": [],
                "target_pdb": None,
                "target_pdb_candidates": [],
                "failed_target_pdbs": failed_targets,
                "partial_success_targets": partial_success_targets,
                "partial_success_sequences": partial_success_sequences,
                "target_failure_records": target_failure_records,
                "target_pdb_selection_reason": "no_valid_candidates",
                "target_pdb_rankings": target_rankings,
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

            save_checkpoint({**state, **result})
            return result

        log.info("Protein target candidates: %s", ", ".join(candidate_pdbs))

        chosen_pdb = None
        valid_binding_results: List[dict] = []
        newly_failed: list[dict] = []
        failed_sequences: list[dict] = []
        md_results_updated = list(state.get("md_results", []) or [])

        # ------------------------------------------------------------
        # Try targets until one yields acceptable docking-valid results
        # ------------------------------------------------------------
        for pdb_id in candidate_pdbs:
            log.info("Trying protein target PDB: %s", pdb_id)

            try:
                binding_results = pw.evaluate_sequences(pdb_id, sequences) or []
            except Exception as e:
                log.warning("Protein wrapper failed for target %s: %s", pdb_id, e)
                newly_failed.append(
                    _failed_target_record(
                        pdb_id,
                        "protein_wrapper_failed",
                        {"error": str(e)},
                    )
                )
                clear_gpu()
                continue

            proxy_valid_results = [
                dict(r) for r in binding_results
                if isinstance(r, dict) and r.get("valid")
            ]

            log.info(
                "[DOCKING DEBUG] Valid proxy/pre-docking results: %d",
                len(proxy_valid_results),
            )

            if not proxy_valid_results:
                newly_failed.append(
                    _failed_target_record(
                        pdb_id,
                        "no_proxy_valid_results",
                        {"sequence_count": len(sequences)},
                    )
                )
                clear_gpu()
                continue

            enhanced_results = []

            for docking_attempt_idx, raw_result in enumerate(proxy_valid_results):
                if docking_attempt_idx >= max_dockings_per_target:
                    log.info(
                        "Reached VLAB_MAX_DOCKINGS_PER_TARGET=%d for %s; "
                        "skipping remaining sequences.",
                        max_dockings_per_target,
                        pdb_id,
                    )
                    break

                r = dict(raw_result)
                seq = r.get("sequence")
                r = _initialise_docking_fields(r)
                abort_current_target = False

                if not seq:
                    r = _set_docking_skip(
                        r,
                        "missing_sequence",
                        "binding result missing sequence",
                    )
                    failed_sequences.append(
                        {"sequence": "", "target_pdb": pdb_id, "error": r["dock_error"]}
                    )
                    enhanced_results.append(r)
                    continue

                log.info("[DOCKING DEBUG] Processing sequence: %s...", seq[:20])

                md_res = _find_md_result_for_sequence(state, seq)

                if not md_res:
                    r = _set_docking_skip(
                        r,
                        "skipped_no_md_result",
                        "No MD result for sequence",
                    )
                    failed_sequences.append(
                        {"sequence": seq, "target_pdb": pdb_id, "error": "no_md_result"}
                    )
                    enhanced_results.append(r)
                    continue

                if md_res.get("min_energy") is None:
                    r = _set_docking_skip(
                        r,
                        "skipped_no_md_energy",
                        "MD result lacks min_energy",
                    )
                    failed_sequences.append(
                        {"sequence": seq, "target_pdb": pdb_id, "error": "no_md_energy"}
                    )
                    enhanced_results.append(r)
                    continue

                rna_pdb = md_res.get("rna_pdb")

                if not rna_pdb:
                    r = _set_docking_skip(
                        r,
                        "skipped_no_rna_pdb",
                        "MD result lacks RNA PDB path",
                    )
                    failed_sequences.append(
                        {"sequence": seq, "target_pdb": pdb_id, "error": "no_rna_pdb"}
                    )
                    enhanced_results.append(r)
                    continue

                if not os.path.exists(rna_pdb):
                    r = _set_docking_skip(
                        r,
                        "skipped_missing_rna_pdb",
                        f"RNA PDB does not exist: {rna_pdb}",
                    )
                    failed_sequences.append(
                        {
                            "sequence": seq,
                            "target_pdb": pdb_id,
                            "error": f"missing_rna_pdb:{rna_pdb}",
                        }
                    )
                    enhanced_results.append(r)
                    continue

                try:
                    receptor_pdb = ensure_protein_pdb(pdb_id)

                    if not receptor_pdb or not os.path.exists(receptor_pdb):
                        r = _set_docking_skip(
                            r,
                            "hdock_receptor_pdb_missing",
                            f"Cached receptor PDB missing/unavailable for {pdb_id}",
                        )
                        failed_sequences.append(
                            {
                                "sequence": seq,
                                "target_pdb": pdb_id,
                                "error": f"missing_receptor:{pdb_id}",
                            }
                        )
                        enhanced_results.append(r)

                        newly_failed.append(
                            _failed_target_record(
                                pdb_id,
                                "missing_receptor",
                                {"sequence": seq},
                            )
                        )

                        log.warning(
                            "Aborting target %s because receptor PDB is unavailable.",
                            pdb_id,
                        )
                        break

                    receptor_atoms = _count_pdb_atoms(receptor_pdb)

                    if receptor_atoms > max_receptor_atoms:
                        r = _set_docking_skip(
                            r,
                            "hdock_receptor_too_large",
                            (
                                f"Receptor PDB has {receptor_atoms} atoms, exceeding "
                                f"VLAB_MAX_RECEPTOR_ATOMS={max_receptor_atoms}"
                            ),
                        )
                        failed_sequences.append(
                            {
                                "sequence": seq,
                                "target_pdb": pdb_id,
                                "error": f"receptor_too_large:{receptor_atoms}",
                            }
                        )
                        enhanced_results.append(r)

                        newly_failed.append(
                            _failed_target_record(
                                pdb_id,
                                "too_large",
                                {
                                    "receptor_atoms": receptor_atoms,
                                    "max_receptor_atoms": max_receptor_atoms,
                                },
                            )
                        )

                        log.warning(
                            "Aborting target %s because receptor has %d atoms > max %d.",
                            pdb_id,
                            receptor_atoms,
                            max_receptor_atoms,
                        )
                        break

                    ligand_pdb = prepare_rna_pdb_for_hdock(rna_pdb)

                    if not ligand_pdb or not os.path.exists(ligand_pdb):
                        r = _set_docking_skip(
                            r,
                            "skipped_rna_pdb_hdock_prep_failed",
                            f"RNA PDB preparation for HDOCK failed: {rna_pdb}",
                        )
                        failed_sequences.append(
                            {
                                "sequence": seq,
                                "target_pdb": pdb_id,
                                "error": "rna_hdock_prep_failed",
                            }
                        )
                        enhanced_results.append(r)
                        continue

                    dock = state["wrappers"]["hdock"].dock(
                        receptor_pdb=receptor_pdb,
                        ligand_pdb=ligand_pdb,
                    )

                    log.info("[HDOCK RESULT RAW] %s", dock)

                    dock = _reject_implausible_hdock_score(dock)

                    if isinstance(dock, dict) and dock.get("implausible_hdock_score"):
                        newly_failed.append(
                            _failed_target_record(
                                pdb_id,
                                FAILED_REASON_SCORE_ARTEFACT,
                                {
                                    "score": dock.get("dock_score", dock.get("hdock_score")),
                                    "sequence": seq,
                                },
                            )
                        )

                    if isinstance(dock, dict) and dock.get("valid") and dock.get("dock_score") is not None:
                        log.info(
                            "[HDOCK RESULT ACCEPTABLE_FOR_FILTERING] pdb=%s seq=%s score=%s",
                            pdb_id,
                            seq[:16],
                            dock.get("dock_score"),
                        )
                    else:
                        log.warning(
                            "[HDOCK REJECTED/INVALID] pdb=%s seq=%s error=%s",
                            pdb_id,
                            seq[:16],
                            (
                                dock.get("dock_error")
                                or dock.get("error")
                                or dock.get("vina_error")
                                or "HDOCK returned no valid docking score"
                            )
                            if isinstance(dock, dict)
                            else "HDOCK returned non-dict result",
                        )

                    if dock and dock.get("valid") and dock.get("dock_score") is not None:
                        dock_score = float(dock["dock_score"])

                        if dock_score > min_valid_hdock_score:
                            r["dock_valid"] = False
                            r["dock_score"] = dock_score
                            r["dock_method"] = "hdock_rejected_weak_score"
                            r["dock_error"] = (
                                f"HDOCK score {dock_score:.3f} is weaker than "
                                f"VLAB_MIN_VALID_HDOCK_SCORE={min_valid_hdock_score:.3f}"
                            )

                            r["vina_valid"] = False
                            r["vina_energy"] = dock_score
                            r["vina_method"] = "hdock_rejected_weak_score"
                            r["vina_error"] = r["dock_error"]
                            r["binding_mode"] = "rejected_docking"
                            r["binding_units"] = "hdock_relative_score"
                            r["binding_energy_is_physical"] = False

                            failed_sequences.append(
                                {
                                    "sequence": seq,
                                    "target_pdb": pdb_id,
                                    "error": r["dock_error"],
                                    "dock_score": dock_score,
                                }
                            )

                            log.warning("[HDOCK REJECTED] %s", r["dock_error"])

                        else:
                            r["dock_score"] = dock_score
                            r["dock_valid"] = True
                            r["dock_method"] = dock.get("method", "hdock")
                            r["dock_error"] = None
                            r["dock_output_file"] = dock.get("output_file")
                            r["dock_complex_file"] = dock.get("complex_file")
                            r["binding_mode"] = "docked"
                            r["binding_units"] = "hdock_relative_score"
                            r["binding_energy_is_physical"] = False

                            # Legacy compatibility shim.
                            r["vina_energy"] = dock_score
                            r["vina_valid"] = True
                            r["vina_method"] = "hdock"
                            r["vina_error"] = None

                            r["linked_md"] = {
                                "min_energy": md_res.get("min_energy"),
                                "mean_energy": md_res.get("mean_energy"),
                                "energy_fluctuation": md_res.get("energy_fluctuation"),
                                "rna_pdb": md_res.get("rna_pdb"),
                            }

                            md_results_updated = _updated_md_results_with_docking(
                                {**state, "md_results": md_results_updated},
                                seq=seq,
                                pdb_id=pdb_id,
                                dock_score=dock_score,
                            )

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
                        if isinstance(dock, dict):
                            r["dock_error"] = (
                                dock.get("error")
                                or dock.get("dock_error")
                                or dock.get("vina_error")
                                or "HDOCK returned no valid docking score"
                            )
                        else:
                            r["dock_error"] = "HDOCK returned no valid docking score"

                        r["vina_valid"] = False
                        r["vina_energy"] = None
                        r["vina_method"] = r["dock_method"]
                        r["vina_error"] = r["dock_error"]
                        r["binding_mode"] = "proxy"
                        r["binding_energy_is_physical"] = False

                        failed_sequences.append(
                            {
                                "sequence": seq,
                                "target_pdb": pdb_id,
                                "error": r["dock_error"],
                            }
                        )

                        log.warning("[HDOCK FAIL] %s", r["dock_error"])

                        if (
                            abort_target_on_first_timeout
                            and r.get("dock_error")
                            and "timed out" in str(r.get("dock_error")).lower()
                        ):
                            newly_failed.append(
                                _failed_target_record(
                                    pdb_id,
                                    "hdock_timeout",
                                    {"sequence": seq, "error": r.get("dock_error")},
                                )
                            )

                            log.warning(
                                "Aborting target %s after HDOCK timeout because "
                                "VLAB_ABORT_TARGET_ON_FIRST_TIMEOUT=1.",
                                pdb_id,
                            )
                            abort_current_target = True

                except Exception as e:
                    r["dock_valid"] = False
                    r["dock_score"] = None
                    r["dock_method"] = "hdock_exception"
                    r["dock_error"] = str(e)

                    r["vina_valid"] = False
                    r["vina_energy"] = None
                    r["vina_method"] = "hdock_exception"
                    r["vina_error"] = str(e)
                    r["binding_mode"] = "proxy"
                    r["binding_energy_is_physical"] = False

                    failed_sequences.append(
                        {"sequence": seq, "target_pdb": pdb_id, "error": str(e)}
                    )

                    newly_failed.append(
                        _failed_target_record(
                            pdb_id,
                            "hdock_exception",
                            {"sequence": seq, "error": str(e)},
                        )
                    )

                    log.exception(
                        "HDOCK docking failed for %s against %s",
                        seq[:15],
                        pdb_id,
                    )

                enhanced_results.append(r)

                if abort_current_target:
                    break

            dock_valid = [
                r for r in enhanced_results
                if _is_dock_valid_row(r)
            ]

            dock_scores = [
                float(r["dock_score"])
                for r in dock_valid
                if r.get("dock_score") is not None
            ]

            score_spread = None

            if len(dock_scores) >= 2:
                score_spread = max(dock_scores) - min(dock_scores)

            log.info(
                "[DOCKING FILTER] pdb=%s dock_valid=%d min_required=%d score_spread=%s "
                "min_valid_hdock_score=%.3f",
                pdb_id,
                len(dock_valid),
                min_valid_dockings_per_target,
                f"{score_spread:.3f}" if score_spread is not None else "N/A",
                min_valid_hdock_score,
            )

            if require_docking and len(dock_valid) < min_valid_dockings_per_target:
                log.warning(
                    "Rejecting target %s: only %d docking-valid results after filtering; "
                    "required %d.",
                    pdb_id,
                    len(dock_valid),
                    min_valid_dockings_per_target,
                )

                newly_failed.append(
                    _failed_target_record(
                        pdb_id,
                        "insufficient_dock_valid",
                        {
                            "dock_valid": len(dock_valid),
                            "required": min_valid_dockings_per_target,
                        },
                    )
                )

                clear_gpu()
                continue

            if (
                require_docking
                and reject_target_on_score_spread
                and score_spread is not None
                and score_spread > max_target_score_spread
            ):
                log.warning(
                    "Rejecting target %s: docking score spread %.3f exceeds %.3f.",
                    pdb_id,
                    score_spread,
                    max_target_score_spread,
                )

                newly_failed.append(
                    _failed_target_record(
                        pdb_id,
                        "score_spread",
                        {
                            "score_spread": score_spread,
                            "max_target_score_spread": max_target_score_spread,
                        },
                    )
                )

                clear_gpu()
                continue

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

                newly_failed.append(
                    _failed_target_record(
                        pdb_id,
                        "no_docking_valid_results",
                        {"enhanced_result_count": len(enhanced_results)},
                    )
                )

                clear_gpu()
                continue

            valid_binding_results = _score_and_rank_binding_results(
                valid_binding_results
            )

            clear_gpu()
            break

        failed_targets = _dedupe_failed_targets(failed_targets + newly_failed)

        # ------------------------------------------------------------
        # No target succeeded at HDOCK/proxy level
        # ------------------------------------------------------------
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

            result = {
                "protein_analysis": summary,
                "binding_results": [],
                "failed_sequences": failed_sequences,
                "target_pdb": None,
                "target_pdb_candidates": candidate_pdbs,
                "failed_target_pdbs": failed_targets,
                "partial_success_targets": partial_success_targets,
                "partial_success_sequences": partial_success_sequences,
                "target_failure_records": target_failure_records,
                "target_pdb_selection_reason": selection_reason,
                "target_pdb_rankings": target_rankings,
                "md_results": md_results_updated,
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

            _update_failure_memory_from_result({**state, **result})
            save_checkpoint({**state, **result})
            clear_gpu()
            return result

        # ------------------------------------------------------------
        # Human-readable summary
        # ------------------------------------------------------------
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
                f"Mode: {r.get('binding_mode')} | "
                f"Method: {r.get('dock_method', r.get('vina_method', r.get('method', 'unknown')))}"
            )

            if r.get("dock_error"):
                lines.append(f"    Docking note: {r.get('dock_error')}")

        summary = "\n".join(lines)

        result = {
            "protein_analysis": summary,
            "binding_results": valid_binding_results,
            "failed_sequences": failed_sequences,
            "target_pdb": chosen_pdb,
            "target_pdb_candidates": candidate_pdbs,
            "failed_target_pdbs": failed_targets,
            "partial_success_targets": partial_success_targets,
            "partial_success_sequences": partial_success_sequences,
            "target_failure_records": target_failure_records,
            "target_pdb_selection_reason": selection_reason,
            "target_pdb_rankings": target_rankings,
            "binding_units": "hdock_relative_score",
            "binding_energy_is_physical": False,
            "md_results": md_results_updated,
            "docking_preferences": list(state.get("docking_preferences", []) or []),
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
                            if _is_dock_valid_row(r)
                        ),
                        "binding_units": "hdock_relative_score",
                        "binding_energy_is_physical": False,
                    },
                )
            ],
            "conversation_history": [
                add_conversation_entry(state, "assistant", summary, "protein")
            ],
        }

        # ------------------------------------------------------------
        # Render docking visual snapshots
        # ------------------------------------------------------------
        try:
            visual_paths = render_docking_snapshots_for_results({**state, **result})

            if visual_paths:
                result.update(visual_paths)

        except Exception as e:
            log.warning("Failed to render docking snapshots: %s", e)

        # ------------------------------------------------------------
        # Extract protein-RNA interface contact metrics
        # ------------------------------------------------------------
        try:
            interface_paths = analyse_interface_contacts_for_results({**state, **result})

            if interface_paths:
                result.update(interface_paths)

                require_interface_for_target_acceptance = _env_bool(
                    "VLAB_REQUIRE_INTERFACE_FOR_TARGET_ACCEPTANCE",
                    default=False,
                )

                rows = [
                    r for r in result.get("binding_results", []) or []
                    if isinstance(r, dict)
                ]

                # Add explicit row-level training labels after interface metrics exist.
                for row in rows:
                    row["training_label"] = _classify_docking_training_label(row)
                    row["binding_units"] = "hdock_relative_score"
                    row["binding_energy_is_physical"] = False

                dock_valid_count = sum(1 for r in rows if _is_dock_valid_row(r))
                interface_clean = [r for r in rows if _has_clean_interface(r)]
                steric_clash_count = sum(1 for r in rows if _has_steric_clash(r))

                # Build preference data regardless of final target acceptance.
                new_preferences = _build_interface_preferences(
                    rows,
                    research_topic=state.get("research_topic", ""),
                )
                result["docking_preferences"] = _dedupe_preferences(
                    list(result.get("docking_preferences", []) or []) + new_preferences
                )

                if new_preferences:
                    log.info(
                        "Generated %d interface preference examples for target %s.",
                        len(new_preferences),
                        chosen_pdb,
                    )

                if require_interface_for_target_acceptance:
                    if len(interface_clean) < min_valid_dockings_per_target:
                        # ----------------------------------------------------
                        # New behaviour:
                        # partial_success_target if:
                        #   enough HDOCK-valid poses AND at least one clean interface
                        # ----------------------------------------------------
                        is_partial_success = (
                            allow_partial_interface_target
                            and dock_valid_count >= min_valid_dockings_per_target
                            and len(interface_clean) >= min_partial_interface_clean
                        )

                        if is_partial_success:
                            partial_record = _build_partial_success_record(
                                pdb_id=chosen_pdb,
                                docking_results=rows,
                                reason=PARTIAL_REASON_HDOCK_INTERFACE,
                            )

                            partial_success_targets = _dedupe_partial_success_targets(
                                partial_success_targets + [partial_record]
                            )

                            for clean_row in interface_clean:
                                if clean_row.get("sequence"):
                                    partial_success_sequences.append(clean_row["sequence"])

                            partial_success_sequences = list(
                                dict.fromkeys(
                                    str(x)
                                    for x in partial_success_sequences
                                    if x
                                )
                            )

                            failure_record = _failed_target_record(
                                chosen_pdb,
                                PARTIAL_REASON_HDOCK_INTERFACE,
                                {
                                    "status": PARTIAL_SUCCESS_TARGET_STATUS,
                                    "dock_valid_count": dock_valid_count,
                                    "interface_clean_count": len(interface_clean),
                                    "steric_clash_count": steric_clash_count,
                                    "required_interface_clean": min_valid_dockings_per_target,
                                    "recommendation": (
                                        "reuse_as_priority_candidate_but_not_final_target"
                                    ),
                                },
                            )

                            failed_targets = _dedupe_failed_targets(
                                failed_targets + [failure_record]
                            )

                            target_failure_records = list(target_failure_records) + [
                                partial_record
                            ]

                            summary = (
                                f"Target {chosen_pdb} produced docking-valid "
                                f"HDOCK-relative results and at least one clean interface, "
                                f"but failed full target acceptance: "
                                f"{len(interface_clean)} interface-clean pose(s), "
                                f"required {min_valid_dockings_per_target}. "
                                f"Recorded as {PARTIAL_SUCCESS_TARGET_STATUS} with "
                                f"reason={PARTIAL_REASON_HDOCK_INTERFACE}. "
                                "Reuse as a priority lower-confidence target, but do not "
                                "treat as a final validated binding system."
                            )

                            log.warning(
                                "Target %s recorded as %s: dock_valid=%d "
                                "interface_clean=%d steric_clash=%d reason=%s",
                                chosen_pdb,
                                PARTIAL_SUCCESS_TARGET_STATUS,
                                dock_valid_count,
                                len(interface_clean),
                                steric_clash_count,
                                PARTIAL_REASON_HDOCK_INTERFACE,
                            )

                            result.update(
                                {
                                    "protein_analysis": summary,
                                    # Important: not a final accepted target.
                                    "target_pdb": None,
                                    "failed_target_pdbs": failed_targets,
                                    "partial_success_targets": partial_success_targets,
                                    "partial_success_sequences": partial_success_sequences,
                                    "target_failure_records": target_failure_records,
                                    "target_pdb_selection_reason": (
                                        f"{PARTIAL_SUCCESS_TARGET_STATUS}:"
                                        f"{_normalise_pdb_id(chosen_pdb)}:"
                                        f"{PARTIAL_REASON_HDOCK_INTERFACE}"
                                    ),
                                    "target_status": PARTIAL_SUCCESS_TARGET_STATUS,
                                    "target_status_reason": PARTIAL_REASON_HDOCK_INTERFACE,
                                    "stage_outputs": [
                                        record_stage_output(
                                            state,
                                            "protein",
                                            summary,
                                            summary=(
                                                "Protein target recorded as partial "
                                                "success after interface validation"
                                            ),
                                            metadata={
                                                "target_pdb": chosen_pdb,
                                                "target_status": PARTIAL_SUCCESS_TARGET_STATUS,
                                                "target_status_reason": (
                                                    PARTIAL_REASON_HDOCK_INTERFACE
                                                ),
                                                "dock_valid_count": dock_valid_count,
                                                "interface_clean_count": len(interface_clean),
                                                "steric_clash_count": steric_clash_count,
                                                "required_interface_clean": (
                                                    min_valid_dockings_per_target
                                                ),
                                                "require_interface_for_target_acceptance": True,
                                                "binding_units": "hdock_relative_score",
                                                "binding_energy_is_physical": False,
                                                "preference_count": len(new_preferences),
                                            },
                                        )
                                    ],
                                    "conversation_history": [
                                        add_conversation_entry(
                                            state,
                                            "assistant",
                                            summary,
                                            "protein",
                                        )
                                    ],
                                }
                            )

                        else:
                            log.warning(
                                "Rejecting accepted target %s after interface analysis: "
                                "only %d interface-clean docking results; required %d.",
                                chosen_pdb,
                                len(interface_clean),
                                min_valid_dockings_per_target,
                            )

                            if len(interface_clean) > 0:
                                fail_reason = "only_one_clean_pose"
                            else:
                                fail_reason = FAILED_REASON_NO_CLEAN_INTERFACE

                            failed_targets = _dedupe_failed_targets(
                                failed_targets
                                + [
                                    _failed_target_record(
                                        chosen_pdb,
                                        fail_reason,
                                        {
                                            "dock_valid_count": dock_valid_count,
                                            "interface_clean_count": len(interface_clean),
                                            "steric_clash_count": steric_clash_count,
                                            "required": min_valid_dockings_per_target,
                                        },
                                    )
                                ]
                            )

                            target_failure_records = list(target_failure_records) + [
                                {
                                    "target_pdb": _normalise_pdb_id(chosen_pdb),
                                    "pdb_id": _normalise_pdb_id(chosen_pdb),
                                    "status": FAILED_TARGET_STATUS,
                                    "reason": FAILED_REASON_INSUFFICIENT_INTERFACE,
                                    "dock_valid_count": dock_valid_count,
                                    "interface_clean_count": len(interface_clean),
                                    "steric_clash_count": steric_clash_count,
                                    "required_interface_clean": min_valid_dockings_per_target,
                                    "binding_units": "hdock_relative_score",
                                    "binding_energy_is_physical": False,
                                }
                            ]

                            summary = (
                                f"Target {chosen_pdb} produced docking-valid HDOCK "
                                f"results, but failed interface validation: only "
                                f"{len(interface_clean)} interface-clean pose(s), "
                                f"required {min_valid_dockings_per_target}. "
                                "Treating target as failed for this run."
                            )

                            result.update(
                                {
                                    "protein_analysis": summary,
                                    "target_pdb": None,
                                    "failed_target_pdbs": failed_targets,
                                    "partial_success_targets": partial_success_targets,
                                    "partial_success_sequences": partial_success_sequences,
                                    "target_failure_records": target_failure_records,
                                    "target_pdb_selection_reason": selection_reason,
                                    "target_status": FAILED_TARGET_STATUS,
                                    "target_status_reason": FAILED_REASON_INSUFFICIENT_INTERFACE,
                                    "stage_outputs": [
                                        record_stage_output(
                                            state,
                                            "protein",
                                            summary,
                                            summary=(
                                                "Protein target rejected after interface "
                                                "validation"
                                            ),
                                            metadata={
                                                "target_pdb": chosen_pdb,
                                                "target_status": FAILED_TARGET_STATUS,
                                                "target_status_reason": (
                                                    FAILED_REASON_INSUFFICIENT_INTERFACE
                                                ),
                                                "dock_valid_count": dock_valid_count,
                                                "interface_clean_count": len(interface_clean),
                                                "steric_clash_count": steric_clash_count,
                                                "required_interface_clean": (
                                                    min_valid_dockings_per_target
                                                ),
                                                "require_interface_for_target_acceptance": True,
                                                "binding_units": "hdock_relative_score",
                                                "binding_energy_is_physical": False,
                                                "preference_count": len(new_preferences),
                                            },
                                        )
                                    ],
                                    "conversation_history": [
                                        add_conversation_entry(
                                            state,
                                            "assistant",
                                            summary,
                                            "protein",
                                        )
                                    ],
                                }
                            )

                # Add compact interface notes to the human-readable protein summary.
                interface_lines = ["", "Interface contact summary:"]

                for r in result.get("binding_results", [])[:5]:
                    if not isinstance(r, dict):
                        continue

                    if not r.get("dock_valid"):
                        continue

                    interface_lines.append(
                        "  - Seq: {seq}... | contacts={contacts} | basic={basic} | "
                        "min_dist={dist} Å | interface_score={score} | "
                        "passed={passed} | clash={clash} | label={label}".format(
                            seq=(r.get("sequence") or "")[:20],
                            contacts=r.get("interface_residue_contacts"),
                            basic=r.get("interface_basic_residue_contacts"),
                            dist=r.get("interface_min_distance_A"),
                            score=r.get("interface_quality_score"),
                            passed=r.get("interface_passed"),
                            clash=r.get("interface_steric_clash"),
                            label=r.get("training_label"),
                        )
                    )

                result["protein_analysis"] = (
                    result.get("protein_analysis", "")
                    + "\n"
                    + "\n".join(interface_lines)
                )

        except Exception as e:
            log.warning("Failed to analyse interface contacts: %s", e)

        # ------------------------------------------------------------
        # Export docking summaries after visuals/contact metrics
        # ------------------------------------------------------------
        try:
            export_paths = export_docking_outputs({**state, **result})
            result.update(export_paths)

        except Exception as e:
            log.warning("Failed to export docking summary files: %s", e)

        _update_failure_memory_from_result({**state, **result})
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