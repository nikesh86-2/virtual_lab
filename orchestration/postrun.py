from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from VLAB2.orchestration.utils.literature_utils import (
    fallback_literature_query,
    keep_literature_text,
    keep_research_query,
    normalise_lit_query,
)
from VLAB2.research.research_agent_adaptive import expand_knowledge
from VLAB2.orchestration.failure_memory import FailureMemory
from VLAB2.orchestration.literature_memory import LiteratureMemory


log = logging.getLogger("virtual_lab")

POSTRUN_DIR = Path("postrun_training_data")


# ---------------------------------------------------------------------------
# Docking/interface scoring helpers
# ---------------------------------------------------------------------------

def _training_preference_score(r: dict) -> float:
    """
    Higher = better for preference training.

    Combines:
      - lower binding_rank_score / dg
      - better interface quality
      - broader RNA span coverage
      - better contact entropy
      - fewer excessive clusters
      - fewer steric clashes
      - more basic contacts
      - better MD min energy, lightly
    """
    if not isinstance(r, dict):
        return -1e9

    binding_rank_score = r.get("binding_rank_score", r.get("dg"))

    try:
        binding_component = -float(binding_rank_score)
    except Exception:
        binding_component = 0.0

    try:
        interface_quality = float(r.get("interface_quality_score") or 0.0)
    except Exception:
        interface_quality = 0.0

    try:
        residue_contacts = float(r.get("interface_residue_contacts") or 0.0)
    except Exception:
        residue_contacts = 0.0

    try:
        basic_contacts = float(r.get("interface_basic_residue_contacts") or 0.0)
    except Exception:
        basic_contacts = 0.0

    try:
        entropy = float(r.get("interface_contact_entropy") or 0.0)
    except Exception:
        entropy = 0.0

    try:
        span = float(r.get("interface_rna_span_covered") or 0.0)
    except Exception:
        span = 0.0

    try:
        clusters = int(r.get("interface_cluster_count") or 0)
    except Exception:
        clusters = 0

    steric_clash = bool(r.get("interface_steric_clash"))

    linked_md = r.get("linked_md") or {}

    try:
        md_min = float(linked_md.get("min_energy") or 0.0)
    except Exception:
        md_min = 0.0

    cluster_penalty = 1.5 * max(0, clusters - 2)

    score = (
        binding_component
        + 14.0 * interface_quality
        + 0.12 * residue_contacts
        + 0.75 * basic_contacts
        + 2.0 * entropy
        + 8.0 * span
        - cluster_penalty
        + 0.015 * abs(md_min)
    )

    try:
        min_dist = float(r.get("interface_min_distance_A") or 999.0)
    except Exception:
        min_dist = 999.0

    if min_dist < 1.0:
        score -= 120.0
    elif steric_clash:
        score -= 60.0

    if not r.get("dock_valid"):
        score -= 50.0

    if r.get("binding_mode") == "rejected_docking":
        score -= 50.0

    return round(score, 6)


def _has_clean_interface(r: dict) -> bool:
    """
    True only when the docking pose passes interface validation and has no steric clash.
    """
    if not isinstance(r, dict):
        return False

    return bool(r.get("interface_passed")) and not bool(r.get("interface_steric_clash"))


def _normalise_failed_pdb_ids(records: list[Any]) -> list:
    """
    Extract PDB IDs from failed_target_pdbs supporting legacy list[str]
    and new list[dict].
    """
    out: list[str] = []

    for item in records or []:
        if isinstance(item, dict):
            pdb = str(
                item.get("pdb_id")
                or item.get("target_pdb")
                or ""
            ).strip().upper()
        else:
            pdb = str(item or "").strip().upper()

        if pdb:
            out.append(pdb)

    return sorted(set(out))


def _normalise_partial_success_target_ids(records: list[Any]) -> list:
    """
    Extract PDB IDs from partial_success_targets supporting legacy list[str]
    and structured list[dict].
    """
    out: list[str] = []

    for item in records or []:
        if isinstance(item, dict):
            pdb = str(
                item.get("target_pdb")
                or item.get("pdb_id")
                or ""
            ).strip().upper()
        else:
            pdb = str(item or "").strip().upper()

        if pdb:
            out.append(pdb)

    return sorted(set(out))


def _interface_clash_severity(r: dict) -> str:
    """
    Qualitative clash severity label for filtering/training.
    """
    if not isinstance(r, dict):
        return "unknown"

    try:
        min_dist = float(r.get("interface_min_distance_A") or 999.0)
    except Exception:
        min_dist = 999.0

    if not r.get("interface_steric_clash"):
        return "none"

    if min_dist < 1.0:
        return "severe"

    if min_dist < 1.5:
        return "moderate"

    if min_dist < 1.8:
        return "borderline"

    return "unknown"


def _pose_training_label(r: dict) -> str:
    """
    Coarse label for downstream corpus filtering.
    """
    if not isinstance(r, dict):
        return "invalid"

    if not r.get("dock_valid"):
        return "dock_invalid"

    if r.get("interface_steric_clash"):
        severity = _interface_clash_severity(r)
        return f"reject_interface_clash_{severity}"

    if r.get("interface_passed"):
        return "accept_interface_valid"

    return "weak_interface"


def _sanitize_docking_row_for_training(r: dict) -> dict:
    """
    Remove legacy Vina compatibility fields from HDOCK rows before supervised export.

    This prevents the model from learning that HDOCK-relative RNA docking scores
    are Vina kcal/mol scores.
    """
    if not isinstance(r, dict):
        return {}

    out = dict(r)

    method = str(
        out.get("dock_method")
        or out.get("method")
        or out.get("vina_method")
        or ""
    ).lower()

    binding_units = str(out.get("binding_units") or "").lower()

    if method == "hdock" or binding_units == "hdock_relative_score":
        out.pop("vina_energy", None)
        out.pop("vina_valid", None)
        out.pop("vina_method", None)
        out.pop("vina_error", None)

    out["binding_units"] = out.get("binding_units") or "hdock_relative_score"
    out["binding_energy_is_physical"] = bool(
        out.get("binding_energy_is_physical", False)
    )

    return out


def _select_best_interface_clean_sequence_from_state(state: dict) -> str | None:
    """
    Select best clean-interface sequence from binding results, if available.
    """
    rows = [
        r for r in state.get("binding_results", []) or []
        if isinstance(r, dict)
        and r.get("dock_valid")
        and _has_clean_interface(r)
        and r.get("sequence")
    ]

    if not rows:
        return None

    def _key(r: dict) -> tuple[float, float]:
        try:
            rank_score = float(r.get("binding_rank_score", r.get("dg", 1e9)))
        except Exception:
            rank_score = 1e9

        try:
            hdock_score = float(r.get("dock_score", r.get("hdock_score", 1e9)))
        except Exception:
            hdock_score = 1e9

        return rank_score, hdock_score

    rows.sort(key=_key)
    return rows[0].get("sequence")


def _normalise_pi_training_metadata_for_export(
    state: dict,
    docking_rows: list[dict] | None = None,
) -> dict:
    """
    Normalise PI training metadata against final-state target outcome.

    Prevents contradictory export such as:
      manifest.target_status = partial_success_target
      pi_training_metadata.target_status = failed_target

    Final state is authoritative for target_status/target_status_reason.
    Docking rows are authoritative for clean/clash counts.
    """
    if not isinstance(state, dict):
        return {}

    meta = dict(state.get("pi_training_metadata") or {})

    target_status = state.get("target_status")
    target_status_reason = state.get("target_status_reason")

    target_pdb = (
        state.get("target_pdb")
        or state.get("target_pdb_id")
    )

    binding_units = state.get("binding_units") or "hdock_relative_score"

    if docking_rows is None:
        docking_rows = _extract_docking_rows(state)

    clean_count = sum(
        1 for r in docking_rows or []
        if isinstance(r, dict)
        and r.get("dock_valid")
        and r.get("interface_passed")
        and not r.get("interface_steric_clash")
    )

    clash_count = sum(
        1 for r in docking_rows or []
        if isinstance(r, dict)
        and r.get("dock_valid")
        and r.get("interface_steric_clash")
    )

    dock_valid_count = sum(
        1 for r in docking_rows or []
        if isinstance(r, dict)
        and r.get("dock_valid")
    )

    if target_pdb is not None:
        meta["target_pdb"] = target_pdb

    if target_status:
        meta["target_status"] = target_status

    if target_status_reason:
        meta["target_status_reason"] = target_status_reason

    meta["binding_units"] = binding_units
    meta["binding_energy_is_physical"] = False

    meta["dock_valid_count"] = dock_valid_count
    meta["clean_interface_count"] = clean_count
    meta["interface_clean_count"] = clean_count
    meta["interface_clash_count"] = clash_count
    meta["steric_clash_count"] = clash_count

    meta["best_interface_clean_sequence"] = (
        state.get("best_interface_clean_sequence")
        or _select_best_interface_clean_sequence_from_state(state)
    )

    if not meta.get("training_quality"):
        if dock_valid_count and clean_count:
            meta["training_quality"] = "high"
        elif dock_valid_count:
            meta["training_quality"] = "medium"
        else:
            meta["training_quality"] = "low"

    if target_status == "partial_success_target":
        meta["target_policy"] = (
            "reuse_as_priority_candidate_but_not_final_validated_target"
        )

    return meta


def _sanitize_critique_for_training(critique: Any) -> str:
    """
    Remove explicit internal threshold notes from Skeptic critique before training export.
    Keeps scientific critique while avoiding threshold memorisation.
    """
    text = "" if critique is None else str(critique)

    # Remove bracketed convergence notes that expose internal thresholds.
    text = re.sub(
        r"\n?\[CONVERGENCE NOTE:.*?\]\s*",
        "\n",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove explicit parenthetical threshold fragments if any remain.
    text = re.sub(
        r"\s*\(<\s*[-+]?\d+(?:\.\d+)?\s*threshold\)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def _preference_payload(r: dict) -> dict:
    if not isinstance(r, dict):
        return {}

    return {
        "sequence": r.get("sequence"),
        "target_pdb": r.get("target_pdb"),
        "dock_score": r.get("dock_score"),
        "binding_rank_score": r.get("binding_rank_score"),
        "binding_units": r.get("binding_units") or "hdock_relative_score",
        "binding_energy_is_physical": r.get("binding_energy_is_physical", False),
        "interface_quality_score": r.get("interface_quality_score"),
        "interface_residue_contacts": r.get("interface_residue_contacts"),
        "interface_basic_residue_contacts": r.get("interface_basic_residue_contacts"),
        "interface_min_distance_A": r.get("interface_min_distance_A"),
        "interface_steric_clash": r.get("interface_steric_clash"),
        "interface_passed": r.get("interface_passed"),
        "interface_contact_entropy": r.get("interface_contact_entropy"),
        "interface_contact_entropy_normalized": r.get(
            "interface_contact_entropy_normalized"
        ),
        "interface_rna_span_covered": r.get("interface_rna_span_covered"),
        "interface_cluster_count": r.get("interface_cluster_count"),
        "md_min_energy": (r.get("linked_md") or {}).get("min_energy"),
        "training_preference_score": _training_preference_score(r),
        "interface_clash_severity": _interface_clash_severity(r),
        "pose_training_label": _pose_training_label(r),
        "training_label": r.get("training_label"),
    }


def _inhibitor_name(r: dict, default: str = "unknown_ligand") -> str:
    """
    Robustly recover a small-molecule name from inhibitor result records.
    """
    if not isinstance(r, dict):
        return default

    name = (
        r.get("ligand_name")
        or r.get("display_name")
        or r.get("name")
        or r.get("compound_name")
        or r.get("title")
        or r.get("_inhibitor_name")
    )

    if name:
        return str(name)

    for key in (
        "ligand_file",
        "ligand_path",
        "pdbqt_path",
        "output_file",
        "docked_pdbqt",
    ):
        value = r.get(key)

        if not value:
            continue

        try:
            stem = Path(str(value)).stem
            stem = stem.replace("_docked", "")
            return stem or default
        except Exception:
            continue

    return default


def _dedupe_inhibitor_records(
    records: list[dict],
    ligand_type: str = "small_molecule",
) -> list:
    """
    Dedupe inhibitor records before training export and manifest counts.
    """
    seen: set[tuple] = set()
    out: list[dict] = []

    for r in records or []:
        if not isinstance(r, dict):
            continue

        if ligand_type == "small_molecule":
            key = (
                r.get("target_pdb") or r.get("target_tag"),
                r.get("smiles"),
                _inhibitor_name(r),
                r.get("binding_energy"),
                r.get("output_file") or r.get("docked_pdbqt"),
            )
        else:
            key = (
                r.get("target_pdb") or r.get("target_tag"),
                r.get("sequence") or r.get("peptide_sequence"),
                r.get("score") or r.get("hdock_score"),
                r.get("complex_file")
                or r.get("complex_pdb")
                or r.get("dock_complex_file"),
            )

        if key in seen:
            continue

        seen.add(key)
        out.append(r)

    return out


def _dedupe_preferences(preferences: list[dict]) -> list:
    """
    Dedupe pairwise preference examples.
    """
    seen: set[tuple] = set()
    out: list[dict] = []

    for pref in preferences or []:
        if not isinstance(pref, dict):
            continue

        chosen = pref.get("chosen") or {}
        rejected = pref.get("rejected") or {}

        key = (
            pref.get("type"),
            chosen.get("target_pdb"),
            chosen.get("sequence") or chosen.get("name"),
            chosen.get("binding_energy")
            or chosen.get("score")
            or chosen.get("dock_score"),
            rejected.get("target_pdb"),
            rejected.get("sequence") or rejected.get("name"),
            rejected.get("binding_energy")
            or rejected.get("score")
            or rejected.get("dock_score"),
        )

        if key in seen:
            continue

        seen.add(key)
        out.append(pref)

    return out


# ---------------------------------------------------------------------------
# Preference extractors
# ---------------------------------------------------------------------------

def _extract_explicit_docking_preferences(state: dict) -> list:
    """
    Consume explicit docking preference examples generated by protein_agent.
    """
    if not isinstance(state, dict):
        return []

    out: list[dict] = []

    for pref in state.get("docking_preferences", []) or []:
        if not isinstance(pref, dict):
            continue

        chosen = pref.get("preferred") or pref.get("chosen")
        rejected = pref.get("rejected")

        if not isinstance(chosen, dict) or not isinstance(rejected, dict):
            continue

        pref_type = (
            pref.get("preference_type")
            or pref.get("type")
            or "interface_clean_over_steric_clash"
        )

        out.append(
            {
                "type": pref_type,
                "prompt": (
                    "Choose the better RNA-protein docking pose. Prefer clean "
                    "protein-RNA interface geometry over raw HDOCK-relative score "
                    "when the alternative pose has a steric clash. HDOCK scores "
                    "are relative docking scores only, not physical binding free "
                    "energies."
                ),
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "source": "protein_agent_docking_preferences",
                    "schema_version": pref.get("schema_version"),
                    "label": pref.get("label"),
                    "rationale": pref.get("rationale"),
                    "preference_type": pref_type,
                    "binding_units": "hdock_relative_score",
                    "binding_energy_is_physical": False,
                },
            }
        )

    return out


def _extract_interface_clean_over_clash_preferences(state: dict) -> list:
    """
    Build explicit pairwise preferences:
      clean interface > steric clash
      reasonable HDOCK + clean interface > stronger HDOCK + clash
    """
    if not isinstance(state, dict):
        return []

    binding_results = state.get("binding_results", []) or []

    clean: list[dict] = []
    clash: list[dict] = []

    for r in binding_results:
        if not isinstance(r, dict):
            continue

        if not r.get("dock_valid"):
            continue

        if _has_clean_interface(r):
            clean.append(r)
        elif r.get("interface_steric_clash"):
            clash.append(r)

    preferences: list[dict] = []

    for good in clean:
        for bad in clash:
            good_score = good.get("dock_score")
            bad_score = bad.get("dock_score")

            try:
                good_score_f = float(good_score)
                bad_score_f = float(bad_score)
            except Exception:
                good_score_f = None
                bad_score_f = None

            if (
                good_score_f is not None
                and bad_score_f is not None
                and bad_score_f < good_score_f
            ):
                pref_type = "reasonable_hdock_clean_interface_over_stronger_hdock_clash"
                rationale = (
                    "The rejected pose has a stronger HDOCK-relative score, but "
                    "it has steric clash evidence. The preferred pose is retained "
                    "because it has a clean interface."
                )
            else:
                pref_type = "interface_clean_over_steric_clash"
                rationale = (
                    "The preferred pose has a clean protein-RNA interface, while "
                    "the rejected pose has steric clash evidence."
                )

            preferences.append(
                {
                    "type": pref_type,
                    "prompt": (
                        "Choose the better RNA-protein docking pose. Prefer "
                        "interface-clean poses over steric-clash poses, even if "
                        "the clashing pose has a stronger HDOCK-relative score. "
                        "Do not treat HDOCK as a physical binding free energy."
                    ),
                    "chosen": _preference_payload(good),
                    "rejected": _preference_payload(bad),
                    "metadata": {
                        "source": "postrun_interface_clean_over_clash",
                        "target_pdb": good.get("target_pdb") or bad.get("target_pdb"),
                        "criterion": "clean_interface_over_steric_clash",
                        "rationale": rationale,
                        "binding_units": "hdock_relative_score",
                        "binding_energy_is_physical": False,
                    },
                }
            )

    return preferences


def _extract_partial_success_target_examples(state: dict) -> list:
    """
    Train target-selection behaviour for lower-confidence partial targets.
    """
    if not isinstance(state, dict):
        return []

    examples: list[dict] = []

    partial_targets = state.get("partial_success_targets", []) or []
    target_failure_records = state.get("target_failure_records", []) or []

    combined: list[dict] = []

    for item in partial_targets:
        if isinstance(item, dict):
            combined.append(item)
        elif item:
            pdb = str(item).strip().upper()
            combined.append(
                {
                    "target_pdb": pdb,
                    "pdb_id": pdb,
                    "status": "partial_success_target",
                    "reason": "legacy_partial_success_target",
                }
            )

    for item in target_failure_records:
        if (
            isinstance(item, dict)
            and item.get("status") == "partial_success_target"
        ):
            combined.append(item)

    seen: set[str] = set()

    for rec in combined:
        pdb = str(
            rec.get("target_pdb")
            or rec.get("pdb_id")
            or ""
        ).strip().upper()

        if not pdb or pdb in seen:
            continue

        seen.add(pdb)

        reason = rec.get("reason") or "hdock_passed_interface_partially_failed"

        examples.append(
            {
                "type": "partial_success_target_policy",
                "instruction": (
                    "Interpret this partial-success RNA-protein docking target. "
                    "Decide whether it should be reused, rejected, or treated as "
                    "a final accepted target. HDOCK scores are relative docking "
                    "scores only, not physical binding free energies."
                ),
                "input": json.dumps(
                    {
                        "target_pdb": pdb,
                        "status": rec.get("status"),
                        "reason": reason,
                        "dock_valid_count": rec.get("dock_valid_count"),
                        "interface_clean_count": rec.get("interface_clean_count"),
                        "steric_clash_count": rec.get("steric_clash_count"),
                        "best_hdock_relative_score": rec.get(
                            "best_hdock_relative_score"
                        ),
                        "score_spread": rec.get("score_spread"),
                        "clean_sequences": rec.get("clean_sequences", []),
                        "clash_sequences": rec.get("clash_sequences", []),
                        "target_pdb_final": state.get("target_pdb"),
                        "target_pdb_selection_reason": state.get(
                            "target_pdb_selection_reason"
                        ),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "output": (
                    "TARGET_STATUS: partial_success_target\n"
                    f"TARGET: {pdb}\n"
                    f"REASON: {reason}\n"
                    "ACTION: Reuse this target as a priority lower-confidence "
                    "candidate in the next docking iteration, but do not treat it "
                    "as a final accepted protein target. The target produced "
                    "HDOCK-relative docking-valid poses and at least one clean "
                    "interface, but failed full target acceptance because interface "
                    "validation was not consistent across poses. Optimisation should "
                    "increase interface-clean pose frequency and penalise steric "
                    "clash geometries."
                ),
                "metadata": {
                    "source": "partial_success_targets",
                    "target_pdb": pdb,
                    "status": "partial_success_target",
                    "reason": reason,
                    "accepted_target_present": (
                        state.get("target_pdb") is not None
                        and state.get("target_status") == "accepted_target"
                    ),
                    "partial_success_target_present": True,
                    "binding_units": "hdock_relative_score",
                    "binding_energy_is_physical": False,
                },
            }
        )

    return examples


# ---------------------------------------------------------------------------
# Generic JSON/example helpers
# ---------------------------------------------------------------------------

def _topic_seed_text(topic: dict) -> str:
    """
    Robustly extract a useful query/description from a topic dict.
    """
    if not isinstance(topic, dict):
        return ""

    seed_questions = topic.get("seed_questions") or []

    if seed_questions:
        return str(seed_questions[0] or "").strip()

    return str(
        topic.get("description")
        or topic.get("topic_description")
        or topic.get("name")
        or ""
    ).strip()


def _jsonable(obj: Any) -> Any:
    """
    Conservative JSON converter for training examples.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]

    if hasattr(obj, "model_dump"):
        try:
            return _jsonable(obj.model_dump())
        except Exception:
            pass

    if hasattr(obj, "dict"):
        try:
            return _jsonable(obj.dict())
        except Exception:
            pass

    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass

    return repr(obj)


def _truncate(text: Any, limit: int = 4000) -> str:
    """
    Keep examples compact enough for supervised fine-tuning.
    """
    text = "" if text is None else str(text)

    if len(text) <= limit:
        return text

    return text[:limit] + "... [TRUNCATED]"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if not isinstance(row, dict):
                continue

            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _as_supervised(example: dict) -> dict | None:
    """
    Convert internal example format into chat-style supervised format.
    """
    if not isinstance(example, dict):
        return None

    instruction = example.get("instruction")
    input_text = example.get("input")
    output_text = example.get("output")

    if not instruction or output_text is None:
        return None

    user_content = instruction

    if input_text:
        user_content += "\n\nINPUT:\n" + _truncate(input_text, 5000)

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a computational biophysics assistant helping operate "
                    "a multi-agent RNA design, folding, MD, and HDOCK workflow. "
                    "Use HDOCK scores only as relative docking scores, not physical "
                    "kcal/mol binding free energies."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": _truncate(output_text, 5000),
            },
        ],
        "metadata": example.get("metadata", {}),
    }


# ---------------------------------------------------------------------------
# Knowledge bootstrap
# ---------------------------------------------------------------------------

def bootstrap_knowledge_base(topic: dict) -> None:
    """
    Build or refresh local knowledge base before the run.

    Non-fatal: failures are logged and execution continues.
    """
    log.info("Bootstrapping knowledge base with prior training data...")

    try:
        raw_bootstrap_query = _topic_seed_text(topic)

        if not raw_bootstrap_query:
            raw_bootstrap_query = "viral RNA stem-loop capsid binding"

        bootstrap_queries: list[str] = []

        q1 = normalise_lit_query(raw_bootstrap_query)

        if keep_research_query(q1):
            bootstrap_queries.append(q1)

        fallback_q = fallback_literature_query(
            topic.get("description")
            or topic.get("topic_description")
            or topic.get("name")
            or raw_bootstrap_query
        )

        if fallback_q and fallback_q not in bootstrap_queries:
            bootstrap_queries.append(fallback_q)

        generic_q = "viral RNA stem-loop capsid binding"

        if generic_q not in bootstrap_queries:
            bootstrap_queries.append(generic_q)

        all_papers: list[Any] = []

        for query in bootstrap_queries:
            try:
                log.info("Bootstrapping knowledge with query: %s", query)
                papers = expand_knowledge(query, build_db=True) or []

                for paper in papers:
                    if isinstance(paper, dict):
                        title = paper.get("title", "")
                        abstract = paper.get("abstract", "")
                    else:
                        metadata = getattr(paper, "metadata", {}) or {}
                        title = metadata.get("title", "")
                        abstract = getattr(paper, "page_content", "")

                    if keep_literature_text(title, abstract):
                        all_papers.append(paper)

                if len(all_papers) >= 5:
                    break

            except Exception as query_error:
                log.warning("Bootstrap query failed '%s': %s", query, query_error)

        seen_titles: set[str] = set()
        deduped: list[Any] = []

        for paper in all_papers:
            if isinstance(paper, dict):
                title = str(paper.get("title", "")).strip().lower()
            else:
                metadata = getattr(paper, "metadata", {}) or {}
                title = str(metadata.get("title", "")).strip().lower()

            key = title or str(paper)[:200]

            if key in seen_titles:
                continue

            seen_titles.add(key)
            deduped.append(paper)

        log.info("Knowledge base rebuilt with %d documents.", len(deduped))

    except Exception as e:
        log.warning("Knowledge bootstrap failed/non-fatal: %s", e)


def _load_checkpoint_or_state(final_state: dict | None = None) -> dict | None:
    """
    Prefer final state passed by CLI. If absent, fall back to lab_checkpoint.json.
    """
    if isinstance(final_state, dict) and final_state:
        return final_state

    checkpoint_path = Path("lab_checkpoint.json")

    if not checkpoint_path.exists():
        log.warning(
            "No final_state provided and %s does not exist; skipping extraction.",
            checkpoint_path,
        )
        return None

    try:
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))

        if not isinstance(data, dict):
            log.warning(
                "Checkpoint loaded as %s, expected dict; skipping extraction.",
                type(data).__name__,
            )
            return None

        return data

    except Exception as e:
        log.warning("Failed to load checkpoint %s: %s", checkpoint_path, e)
        return None


# ---------------------------------------------------------------------------
# Example extractors
# ---------------------------------------------------------------------------

def _extract_stage_examples(state: dict) -> list:
    """
    Train agent-style summarisation/decision behaviour from stage_outputs.
    """
    examples: list[dict] = []

    stage_outputs = state.get("stage_outputs", []) or []

    for stage in stage_outputs:
        if not isinstance(stage, dict):
            continue

        agent = stage.get("agent", "unknown")
        summary = stage.get("summary")
        output = stage.get("output")
        metadata = stage.get("metadata", {})

        if output is None:
            if agent == "bioinfo" and isinstance(metadata, dict):
                output = (
                    "BIOINFO SUMMARY\n"
                    f"Conservation valid: {metadata.get('conservation_valid')}\n"
                    f"MSA size: {metadata.get('msa_size')}\n"
                    f"Conservation fitness: {metadata.get('conservation_fitness')}\n"
                    f"Selected motifs: {metadata.get('selected_motifs', [])}"
                )
            else:
                continue

        if str(output).strip().lower() in {"", "none", "null"}:
            if agent == "bioinfo" and isinstance(metadata, dict):
                output = (
                    "BIOINFO SUMMARY\n"
                    f"Conservation valid: {metadata.get('conservation_valid')}\n"
                    f"MSA size: {metadata.get('msa_size')}\n"
                    f"Conservation fitness: {metadata.get('conservation_fitness')}\n"
                    f"Selected motifs: {metadata.get('selected_motifs', [])}"
                )
            else:
                continue

        examples.append(
            {
                "type": "stage_output",
                "instruction": (
                    f"Given the current Virtual Lab context, produce the {agent} "
                    "agent output and concise summary."
                ),
                "input": json.dumps(
                    {
                        "research_topic": state.get("research_topic"),
                        "hypothesis": state.get("hypothesis"),
                        "target_pdb": state.get("target_pdb"),
                        "target_status": state.get("target_status"),
                        "target_status_reason": state.get("target_status_reason"),
                        "agent": agent,
                        "metadata": metadata,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "output": _truncate(output, 5000),
                "metadata": {
                    "agent": agent,
                    "summary": summary,
                    "source": "stage_outputs",
                },
            }
        )

    return examples


def _extract_structural_selection_examples(state: dict) -> list:
    """
    Train sequence/fold selection from structural candidates.
    """
    candidates = state.get("structural_candidates", []) or []
    target_sequence = state.get("target_sequence")

    if not candidates or not target_sequence:
        return []

    return [
        {
            "type": "structural_selection",
            "instruction": (
                "Select the best RNA candidate for downstream MD and docking. "
                "Prioritise candidates passing fold thresholds, with adequate pair "
                "density and favourable MFE per nucleotide."
            ),
            "input": json.dumps(
                {
                    "candidates": candidates,
                    "fold_thresholds_passed": state.get("fold_thresholds_passed"),
                    "fold_threshold_reasons": state.get("fold_threshold_reasons", []),
                },
                indent=2,
                ensure_ascii=False,
            ),
            "output": target_sequence,
            "metadata": {
                "source": "structural_candidates",
                "selected_sequence": target_sequence,
            },
        }
    ]


def _extract_docking_examples(state: dict) -> tuple[list[dict], list[dict]]:
    """
    Build supervised docking interpretation examples and pairwise preferences.
    """
    examples: list[dict] = []
    preferences: list[dict] = []

    binding_results = state.get("binding_results", []) or []

    valid = [
        r for r in binding_results
        if isinstance(r, dict)
        and r.get("dock_valid")
        and r.get("binding_mode") == "docked"
    ]

    if not valid:
        return examples, preferences

    valid = sorted(
        valid,
        key=_training_preference_score,
        reverse=True,
    )

    sanitized_valid = [
        _sanitize_docking_row_for_training(r)
        for r in valid
    ]

    accepted_target = state.get("target_pdb")
    target_status = state.get("target_status")
    accepted_target_is_final = (
        accepted_target is not None
        and target_status == "accepted_target"
    )

    any_interface_passed = any(
        _has_clean_interface(r)
        and accepted_target_is_final
        and r.get("target_pdb") == accepted_target
        for r in valid
    )

    any_partial_interface_passed = any(
        _has_clean_interface(r)
        for r in valid
    )

    examples.append(
        {
            "type": "docking_ranking",
            "instruction": (
                "Rank these RNA candidates by HDOCK-relative docking quality. "
                "Use lower HDOCK-relative scores as better, but do not describe "
                "them as physical kcal/mol binding energies. Consider interface "
                "contact topology and steric clash flags when interpreting pose quality."
            ),
            "input": json.dumps(sanitized_valid, indent=2, ensure_ascii=False),
            "output": json.dumps(
                [
                    {
                        "rank": i + 1,
                        "sequence": r.get("sequence"),
                        "target_pdb": r.get("target_pdb"),
                        "dock_score": r.get("dock_score"),
                        "binding_rank_score": r.get("binding_rank_score"),
                        "binding_units": r.get("binding_units") or "hdock_relative_score",
                        "binding_energy_is_physical": r.get(
                            "binding_energy_is_physical",
                            False,
                        ),
                        "interface_quality_score": r.get("interface_quality_score"),
                        "interface_residue_contacts": r.get("interface_residue_contacts"),
                        "interface_basic_residue_contacts": r.get(
                            "interface_basic_residue_contacts"
                        ),
                        "interface_min_distance_A": r.get("interface_min_distance_A"),
                        "interface_steric_clash": r.get("interface_steric_clash"),
                        "interface_passed": r.get("interface_passed"),
                        "interface_contact_entropy": r.get("interface_contact_entropy"),
                        "interface_contact_entropy_normalized": r.get(
                            "interface_contact_entropy_normalized"
                        ),
                        "interface_rna_span_covered": r.get(
                            "interface_rna_span_covered"
                        ),
                        "interface_cluster_count": r.get("interface_cluster_count"),
                        "interface_clash_severity": _interface_clash_severity(r),
                        "training_label": r.get("training_label"),
                        "training_preference_score": _training_preference_score(r),
                        "interpretation": (
                            "clean_interface"
                            if _has_clean_interface(r)
                            else "docking_valid_but_interface_caution"
                        ),
                        "reason": (
                            "Candidate ranked by combined HDOCK-relative score, "
                            "interface quality, basic residue contacts, RNA span "
                            "coverage, contact entropy, cluster compactness, MD "
                            "support, and steric clash penalty."
                        ),
                    }
                    for i, r in enumerate(valid)
                ],
                indent=2,
                ensure_ascii=False,
            ),
            "metadata": {
                "source": "binding_results",
                "target_pdb": state.get("target_pdb"),
                "target_status": state.get("target_status"),
                "target_status_reason": state.get("target_status_reason"),
                "n_valid": len(valid),
                "any_interface_passed": any_interface_passed,
                "any_partial_interface_passed": any_partial_interface_passed,
                "binding_units": "hdock_relative_score",
                "binding_energy_is_physical": False,
            },
        }
    )

    if len(valid) >= 2:
        best = valid[0]

        if any_interface_passed:
            pref_type = "docking_interface_preference"
        elif any_partial_interface_passed:
            pref_type = "partial_interface_evidence_preference"
        else:
            pref_type = "least_bad_docking_interface_preference"

        prompt = (
            "Choose the better RNA binder using HDOCK-relative docking score, "
            "interface contact quality, basic residue enrichment, MD stability, "
            "and steric clash evidence. HDOCK scores are relative docking scores, "
            "not physical kcal/mol free energies."
            if any_interface_passed or any_partial_interface_passed
            else (
                "Both RNA docking poses have interface problems. Choose the less "
                "problematic candidate using HDOCK-relative docking score, interface "
                "contact topology, basic residue enrichment, MD stability, and steric "
                "clash severity. Do not treat either pose as experimentally validated."
            )
        )

        for other in valid[1:]:
            preferences.append(
                {
                    "type": pref_type,
                    "prompt": prompt,
                    "chosen": _preference_payload(best),
                    "rejected": _preference_payload(other),
                    "metadata": {
                        "source": "binding_results",
                        "target_pdb": state.get("target_pdb"),
                        "target_status": state.get("target_status"),
                        "target_status_reason": state.get("target_status_reason"),
                        "criterion": "combined_docking_interface_md_score",
                        "any_interface_passed": any_interface_passed,
                        "any_partial_interface_passed": any_partial_interface_passed,
                        "binding_units": "hdock_relative_score",
                        "binding_energy_is_physical": False,
                    },
                }
            )

    return examples, preferences


def _extract_target_filter_examples(state: dict) -> list:
    """
    Train target filtering/failure memory from failed, partial, and accepted targets.
    """
    examples: list[dict] = []

    rankings = state.get("target_pdb_rankings", []) or []

    failed = _normalise_failed_pdb_ids(
        state.get("failed_target_pdbs", []) or []
    )

    partial_ids = _normalise_partial_success_target_ids(
        state.get("partial_success_targets", []) or []
    )

    accepted = state.get("target_pdb")

    if not rankings and not failed and not accepted and not partial_ids:
        return examples

    examples.append(
        {
            "type": "target_filtering",
            "instruction": (
                "Given ranked PDB target candidates and observed docking outcomes, "
                "select a suitable target for RNA docking. Avoid oversized receptors, "
                "weak docking targets, and hard-failed PDBs. Reuse partial-success "
                "targets as lower-confidence priority candidates, but do not treat "
                "them as final accepted targets unless interface validation passes."
            ),
            "input": json.dumps(
                {
                    "target_pdb_rankings": rankings,
                    "failed_target_pdbs": failed,
                    "partial_success_targets": partial_ids,
                    "target_blacklist": state.get("target_blacklist", []),
                    "accepted_target": accepted,
                    "target_status": state.get("target_status"),
                    "target_status_reason": state.get("target_status_reason"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            "output": json.dumps(
                {
                    "selected_target_pdb": accepted,
                    "partial_success_targets": partial_ids,
                    "avoid_targets": failed,
                    "reason": (
                        "Accepted targets require docking-valid poses with clean "
                        "interface validation. Partial-success targets should be "
                        "reused as priority candidates when they have HDOCK-valid "
                        "poses and at least one clean interface, but they are not "
                        "confirmed binding systems until target-level interface "
                        "criteria are met."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            ),
            "metadata": {
                "source": "target_selection",
                "accepted_target": accepted,
                "accepted_target_present": (
                    accepted is not None
                    and state.get("target_status") == "accepted_target"
                ),
                "partial_success_targets": partial_ids,
                "partial_success_target_count": len(partial_ids),
                "failed_target_count": len(failed),
            },
        }
    )

    return examples


def _extract_critique_revision_examples(state: dict) -> list:
    """
    Train PI/skeptic feedback interpretation.
    """
    critique = state.get("critique")
    hypothesis = state.get("hypothesis")

    pi_action_summary = (
        state.get("pi_action_summary")
        or state.get("pi_training_summary")
        or state.get("pi_summary")
    )

    joint_feedback = state.get("joint_physics_feedback", {})

    if not critique:
        return []

    normalised_pi_training_metadata = _normalise_pi_training_metadata_for_export(state)
    critique_for_training = _sanitize_critique_for_training(critique)

    return [
        {
            "type": "critique_interpretation",
            "instruction": (
                "Interpret the Skeptic critique and describe the next optimisation "
                "action for the PI agent. Use qualitative language and avoid copying "
                "exact thresholds into the hypothesis."
            ),
            "input": json.dumps(
                {
                    "hypothesis": hypothesis,
                    "critique": critique_for_training,
                    "joint_physics_feedback": joint_feedback,
                    "conservation_signal": state.get("conservation_signal", {}),
                    "binding_units": state.get("binding_units"),
                    "target_status": state.get("target_status"),
                    "target_status_reason": state.get("target_status_reason"),
                    "partial_success_targets": state.get("partial_success_targets", []),
                    "pi_training_metadata": normalised_pi_training_metadata,
                    "best_interface_clean_sequence": state.get(
                        "best_interface_clean_sequence"
                    ),
                },
                indent=2,
                ensure_ascii=False,
            ),
            "output": pi_action_summary or critique_for_training,
            "metadata": {
                "source": "critique",
                "recommendation_present": "RECOMMENDATION:" in str(critique),
                "training_quality": normalised_pi_training_metadata.get(
                    "training_quality"
                ),
                "target_status": normalised_pi_training_metadata.get("target_status"),
                "target_status_reason": normalised_pi_training_metadata.get(
                    "target_status_reason"
                ),
            },
        }
    ]


def _extract_inhibitor_examples(state: dict) -> tuple[list[dict], list[dict]]:
    """
    Extract inhibitor screening training examples and preferences.
    """
    examples: list[dict] = []
    preferences: list[dict] = []

    if not isinstance(state, dict):
        return examples, preferences

    small_mols = _dedupe_inhibitor_records(
        state.get("inhibitor_small_molecules", []) or [],
        ligand_type="small_molecule",
    )
    peptides = _dedupe_inhibitor_records(
        state.get("inhibitor_peptides", []) or [],
        ligand_type="peptide",
    )
    inhibitor_enabled = state.get("inhibitor_enabled", False)

    if not inhibitor_enabled or (not small_mols and not peptides):
        return examples, preferences

    valid_sm = [
        r for r in small_mols
        if isinstance(r, dict)
        and r.get("valid")
        and r.get("binding_energy") is not None
    ]

    if valid_sm:
        valid_sm = sorted(valid_sm, key=lambda x: x.get("binding_energy", 999))

        examples.append(
            {
                "type": "small_molecule_ranking",
                "instruction": (
                    "Rank small-molecule inhibitors by AutoDock Vina binding energy. "
                    "Lower binding energy (more negative) indicates stronger predicted binding. "
                    "Vina binding energies are approximate kcal/mol estimates, not exact "
                    "experimental values."
                ),
                "input": json.dumps(
                    {
                        "target_pdb": state.get("target_pdb"),
                        "n_screened": len(small_mols),
                        "n_valid": len(valid_sm),
                        "compounds": [
                            {
                                "name": _inhibitor_name(r),
                                "target_pdb": r.get("target_pdb") or state.get("target_pdb"),
                                "binding_energy": r.get("binding_energy"),
                                "binding_units": r.get("binding_units") or "vina_kcal_mol",
                                "binding_energy_is_physical": bool(
                                    r.get("binding_energy_is_physical", True)
                                ),
                                "valid": r.get("valid"),
                            }
                            for r in small_mols
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "output": json.dumps(
                    [
                        {
                            "rank": i + 1,
                            "name": _inhibitor_name(r),
                            "target_pdb": r.get("target_pdb") or state.get("target_pdb"),
                            "binding_energy": r.get("binding_energy"),
                            "binding_units": r.get("binding_units") or "vina_kcal_mol",
                            "binding_energy_is_physical": bool(
                                r.get("binding_energy_is_physical", True)
                            ),
                            "interpretation": (
                                "strong_binder"
                                if r.get("binding_energy", 0) < -7
                                else "moderate_binder"
                            ),
                        }
                        for i, r in enumerate(valid_sm)
                    ],
                    indent=2,
                    ensure_ascii=False,
                ),
                "metadata": {
                    "source": "inhibitor_small_molecules",
                    "target_pdb": state.get("target_pdb"),
                    "binding_units": "vina_kcal_mol",
                    "binding_energy_is_physical": True,
                    "n_screened": len(small_mols),
                    "n_valid": len(valid_sm),
                },
            }
        )

        if len(valid_sm) >= 2:
            best = valid_sm[0]

            for other in valid_sm[1:]:
                preferences.append(
                    {
                        "type": "small_molecule_binding_preference",
                        "prompt": (
                            f"Choose the better small-molecule inhibitor for target "
                            f"{state.get('target_pdb')}. Compare only AutoDock Vina "
                            "scores within the small-molecule set. Lower Vina binding "
                            "energy indicates stronger predicted binding."
                        ),
                        "chosen": {
                            "name": _inhibitor_name(best),
                            "target_pdb": best.get("target_pdb") or state.get("target_pdb"),
                            "binding_energy": best.get("binding_energy"),
                            "binding_units": "vina_kcal_mol",
                            "binding_energy_is_physical": True,
                        },
                        "rejected": {
                            "name": _inhibitor_name(other),
                            "target_pdb": other.get("target_pdb") or state.get("target_pdb"),
                            "binding_energy": other.get("binding_energy"),
                            "binding_units": "vina_kcal_mol",
                            "binding_energy_is_physical": True,
                        },
                        "metadata": {
                            "source": "inhibitor_small_molecules",
                            "target_pdb": state.get("target_pdb"),
                            "criterion": "vina_binding_energy",
                            "binding_units": "vina_kcal_mol",
                            "binding_energy_is_physical": True,
                        },
                    }
                )

    valid_pep = [
        r for r in peptides
        if isinstance(r, dict)
        and r.get("valid")
        and (
            r.get("score") is not None
            or r.get("hdock_score") is not None
        )
    ]

    if valid_pep:
        valid_pep = sorted(
            valid_pep,
            key=lambda x: x.get("score", x.get("hdock_score", 999)),
        )

        examples.append(
            {
                "type": "peptide_ranking",
                "instruction": (
                    "Rank peptide inhibitors by HDOCK docking score. Lower HDOCK score "
                    "indicates better predicted binding. HDOCK scores are relative docking "
                    "scores, not physical binding free energies."
                ),
                "input": json.dumps(
                    {
                        "target_pdb": state.get("target_pdb"),
                        "n_screened": len(peptides),
                        "n_valid": len(valid_pep),
                        "peptides": [
                            {
                                "sequence": (
                                    r.get("sequence")
                                    or r.get("peptide_sequence")
                                ),
                                "target_pdb": r.get("target_pdb") or state.get("target_pdb"),
                                "score": r.get("score", r.get("hdock_score")),
                                "binding_units": "hdock_relative_score",
                                "binding_energy_is_physical": False,
                                "valid": r.get("valid"),
                            }
                            for r in peptides
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "output": json.dumps(
                    [
                        {
                            "rank": i + 1,
                            "sequence": (
                                r.get("sequence")
                                or r.get("peptide_sequence")
                            ),
                            "target_pdb": r.get("target_pdb") or state.get("target_pdb"),
                            "score": r.get("score", r.get("hdock_score")),
                            "binding_units": "hdock_relative_score",
                            "binding_energy_is_physical": False,
                            "interpretation": (
                                "strong_binder"
                                if r.get("score", r.get("hdock_score", 0)) < -200
                                else "moderate_binder"
                            ),
                        }
                        for i, r in enumerate(valid_pep)
                    ],
                    indent=2,
                    ensure_ascii=False,
                ),
                "metadata": {
                    "source": "inhibitor_peptides",
                    "target_pdb": state.get("target_pdb"),
                    "binding_units": "hdock_relative_score",
                    "binding_energy_is_physical": False,
                    "n_screened": len(peptides),
                    "n_valid": len(valid_pep),
                },
            }
        )

        if len(valid_pep) >= 2:
            best = valid_pep[0]

            for other in valid_pep[1:]:
                preferences.append(
                    {
                        "type": "peptide_binding_preference",
                        "prompt": (
                            "Choose the better peptide inhibitor. Lower HDOCK score indicates "
                            "better predicted binding affinity. HDOCK scores are relative docking "
                            "scores, not physical kcal/mol binding free energies."
                        ),
                        "chosen": {
                            "sequence": (
                                best.get("sequence")
                                or best.get("peptide_sequence")
                            ),
                            "target_pdb": best.get("target_pdb") or state.get("target_pdb"),
                            "score": best.get("score", best.get("hdock_score")),
                            "binding_units": "hdock_relative_score",
                            "binding_energy_is_physical": False,
                        },
                        "rejected": {
                            "sequence": (
                                other.get("sequence")
                                or other.get("peptide_sequence")
                            ),
                            "target_pdb": other.get("target_pdb") or state.get("target_pdb"),
                            "score": other.get("score", other.get("hdock_score")),
                            "binding_units": "hdock_relative_score",
                            "binding_energy_is_physical": False,
                        },
                        "metadata": {
                            "source": "inhibitor_peptides",
                            "target_pdb": state.get("target_pdb"),
                            "criterion": "hdock_score",
                            "binding_units": "hdock_relative_score",
                            "binding_energy_is_physical": False,
                        },
                    }
                )

    overlap = state.get("inhibitor_binding_site_overlap", 0.0)

    if overlap > 0:
        examples.append(
            {
                "type": "inhibitor_rna_overlap_analysis",
                "instruction": (
                    "Interpret the binding-site overlap between inhibitor poses and RNA "
                    "binding poses. Higher overlap indicates the inhibitor competes with "
                    "RNA for the same pocket, suggesting competitive inhibition potential."
                ),
                "input": json.dumps(
                    {
                        "target_pdb": state.get("target_pdb"),
                        "overlap_percent": overlap,
                        "best_small_molecule": (valid_sm[0] if valid_sm else None),
                        "best_peptide": (valid_pep[0] if valid_pep else None),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "output": (
                    f"Binding-site overlap with RNA interface: {overlap:.1f}%. "
                    f"{'High overlap suggests competitive inhibition potential.' if overlap > 50 else 'Moderate overlap indicates partial pocket competition.' if overlap > 20 else 'Low overlap suggests inhibitor binds a different region.'}"
                ),
                "metadata": {
                    "source": "inhibitor_binding_site_overlap",
                    "target_pdb": state.get("target_pdb"),
                    "overlap_percent": overlap,
                },
            }
        )

    return examples, preferences


def _extract_final_summary_example(state: dict) -> list:
    """
    Train compact final reporting with correct handling of accepted vs partial targets.
    """
    binding_results = state.get("binding_results", []) or []

    if not binding_results:
        return []

    accepted_target = state.get("target_pdb")
    target_sequence = state.get("target_sequence")
    target_status = state.get("target_status")
    target_status_reason = state.get("target_status_reason")
    partial_targets = state.get("partial_success_targets", []) or []
    partial_target_ids = _normalise_partial_success_target_ids(partial_targets)

    best_clean_sequence = (
        state.get("best_interface_clean_sequence")
        or _select_best_interface_clean_sequence_from_state(state)
    )

    dock_valid_count = sum(
        1 for r in binding_results
        if isinstance(r, dict) and r.get("dock_valid")
    )

    interface_pass_count = sum(
        1 for r in binding_results
        if isinstance(r, dict)
        and r.get("dock_valid")
        and _has_clean_interface(r)
    )

    clash_count = sum(
        1 for r in binding_results
        if isinstance(r, dict)
        and r.get("dock_valid")
        and r.get("interface_steric_clash")
    )

    if dock_valid_count and interface_pass_count == 0:
        interface_sentence = (
            " However, none of the docking-valid poses passed interface validation; "
            f"{clash_count} pose(s) showed steric clash evidence. These results should "
            "therefore be treated as docking hits requiring redesign or pose refinement "
            "rather than accepted physical binders."
        )

    elif target_status == "partial_success_target" or partial_target_ids:
        target_list_text = (
            ", ".join(partial_target_ids)
            if partial_target_ids
            else str(accepted_target or "unknown")
        )
        interface_sentence = (
            f" {interface_pass_count} docking-valid pose(s) passed interface "
            "contact validation, but target-level acceptance criteria were not met. "
            f"Target(s) {target_list_text} should be treated as lower-confidence "
            "partial-success targets for reuse and redesign, not confirmed "
            "binding systems."
        )

    elif accepted_target is None and interface_pass_count > 0:
        interface_sentence = (
            f" {interface_pass_count} docking-valid pose(s) passed interface validation, "
            "but the target-level acceptance criteria were not met, so this should be "
            "treated as partial interface evidence rather than a confirmed binding system."
        )

    else:
        interface_sentence = (
            f" {interface_pass_count} docking-valid pose(s) also passed interface "
            "contact validation."
        )

    if accepted_target is not None:
        target_sentence = f"Target {accepted_target} produced"
    else:
        attempted_targets = sorted({
            r.get("target_pdb")
            for r in binding_results
            if isinstance(r, dict) and r.get("target_pdb")
        })

        if attempted_targets:
            target_sentence = (
                "No final protein target was accepted. Attempted targets "
                f"{', '.join(attempted_targets)} produced"
            )
        else:
            target_sentence = "No final protein target was accepted. Docking attempts produced"

    if best_clean_sequence:
        sequence_sentence = (
            f"The strongest interface-clean sequence was {best_clean_sequence}. "
        )
    elif target_sequence:
        sequence_sentence = (
            f"The selected target sequence was {target_sequence}. "
        )
    else:
        sequence_sentence = ""

    summary = {
        "research_topic": state.get("research_topic"),
        "target_pdb": accepted_target,
        "target_status": target_status,
        "target_status_reason": target_status_reason,
        "partial_success_targets": partial_targets,
        "target_sequence": target_sequence,
        "best_interface_clean_sequence": best_clean_sequence,
        "binding_results": [
            _sanitize_docking_row_for_training(r)
            for r in binding_results
            if isinstance(r, dict)
        ],
        "conservation_signal": state.get("conservation_signal", {}),
        "md_analysis": state.get("md_analysis"),
        "protein_analysis": state.get("protein_analysis"),
        "critique": _sanitize_critique_for_training(state.get("critique")),
    }

    output_text = (
        f"{target_sentence} {dock_valid_count} docking-valid RNA binding results. "
        f"{sequence_sentence}"
        "HDOCK-relative scores should be interpreted as relative docking scores, "
        "not physical binding free energies."
        f"{interface_sentence}"
    )

    return [
        {
            "type": "final_report_summary",
            "instruction": (
                "Write a concise final scientific summary of this RNA design and "
                "docking run. Clearly distinguish HDOCK-relative docking scores "
                "from physical binding energies, and report whether interface "
                "validation passed."
            ),
            "input": json.dumps(summary, indent=2, ensure_ascii=False),
            "output": output_text,
            "metadata": {
                "source": "final_state",
                "target_pdb": accepted_target,
                "target_status": target_status,
                "target_status_reason": target_status_reason,
                "partial_success_targets": partial_target_ids,
                "partial_success_target_count": len(partial_target_ids),
                "accepted_target_present": (
                    accepted_target is not None
                    and target_status == "accepted_target"
                ),
                "partial_success_target_present": (
                    target_status == "partial_success_target"
                    or bool(partial_target_ids)
                ),
                "dock_valid_count": dock_valid_count,
                "interface_pass_count": interface_pass_count,
                "interface_steric_clash_count": clash_count,
                "best_interface_clean_sequence": best_clean_sequence,
                "attempted_target_count": len(
                    {
                        r.get("target_pdb")
                        for r in binding_results
                        if isinstance(r, dict) and r.get("target_pdb")
                    }
                ),
            },
        }
    ]


def _extract_literature_target_policy_examples(state: dict) -> list:
    """
    Train the model to convert literature evidence into target-selection policy.
    """
    if not isinstance(state, dict):
        return []

    try:
        lm = LiteratureMemory()
        policy_text = lm.build_target_policy_text()
    except Exception:
        policy_text = ""

    evidence = state.get("evidence", []) or []

    if not evidence:
        return []

    compact_evidence = []

    for item in evidence[:8]:
        if isinstance(item, dict):
            compact_evidence.append(
                {
                    "title": item.get("title"),
                    "abstract": _truncate(
                        item.get("abstract")
                        or item.get("text")
                        or item.get("content"),
                        1200,
                    ),
                    "source": item.get("source"),
                    "query": item.get("query") or item.get("search_query"),
                }
            )
        else:
            compact_evidence.append(_truncate(str(item), 1200))

    output = (
        f"{policy_text}\n\n"
        "Target-selection guidance: prioritise compact experimentally resolved "
        "viral RNA-binding proteins or RNA-binding domains, especially "
        "nucleocapsid/nucleoprotein/capsid-associated systems when supported by "
        "the literature. Avoid antibody-only, spike/fusion-core, polymerase, "
        "protease, RNA-only, and oversized assemblies unless no better "
        "RNA-interacting target is available. Reuse partial-success targets as "
        "lower-confidence priority candidates when they show HDOCK-valid docking "
        "and at least one clean interface, but do not treat them as final accepted "
        "targets until interface validation is consistent."
    )

    return [
        {
            "type": "literature_target_policy",
            "instruction": (
                "Given literature evidence for an RNA docking topic, derive a "
                "target-selection policy for RCSB/PDB protein target selection."
            ),
            "input": json.dumps(
                {
                    "research_topic": state.get("research_topic"),
                    "virus_family": state.get("virus_family"),
                    "evidence": compact_evidence,
                    "partial_success_targets": state.get(
                        "partial_success_targets",
                        [],
                    ),
                },
                indent=2,
                ensure_ascii=False,
            ),
            "output": output,
            "metadata": {
                "source": "literature_memory",
                "target_pdb": state.get("target_pdb"),
                "target_status": state.get("target_status"),
                "target_status_reason": state.get("target_status_reason"),
            },
        }
    ]


def _extract_literature_rows_from_memory_like_parse(state: dict) -> list:
    """
    Extract literature evidence rows without mutating LiteratureMemory.
    """
    if not isinstance(state, dict):
        return []

    rows: list[dict] = []

    research_topic = state.get("research_topic")
    query = (
        state.get("research_query")
        or state.get("literature_query")
        or state.get("topic_description")
        or research_topic
    )

    lm = LiteratureMemory()

    for item in state.get("evidence", []) or []:
        try:
            rec = lm.ingest_evidence_item(
                item,
                research_topic=research_topic,
                query=query,
            )
            if rec:
                rows.append(rec)
        except Exception:
            continue

    return rows


def _extract_interface_critique_examples(state: dict) -> list:
    """
    Train the model to critique docking poses using contact metrics.
    """
    examples: list[dict] = []

    for r in state.get("binding_results", []) or []:
        if not isinstance(r, dict) or not r.get("dock_valid"):
            continue

        steric_clash = bool(r.get("interface_steric_clash"))
        passed = _has_clean_interface(r)

        if steric_clash:
            verdict = "CAUTION"
            reason = (
                "The pose has a severe steric clash or unrealistically short "
                "protein-RNA atom distance despite a docking-valid HDOCK result."
            )
        elif passed:
            verdict = "SUPPORT"
            reason = (
                "The pose has sufficient protein-RNA residue contacts, at least "
                "one basic residue contact, adequate RNA span coverage, and no "
                "severe steric clash."
            )
        else:
            verdict = "WEAK"
            reason = (
                "The docking result is valid but the protein-RNA interface contact "
                "metrics are insufficient for strong support."
            )

        examples.append(
            {
                "type": "interface_critique",
                "instruction": (
                    "Critique this RNA-protein docking pose using HDOCK-relative "
                    "score and interface contact metrics. Do not treat HDOCK as "
                    "a physical binding free energy."
                ),
                "input": json.dumps(
                    {
                        "sequence": r.get("sequence"),
                        "target_pdb": r.get("target_pdb"),
                        "dock_score": r.get("dock_score"),
                        "binding_rank_score": r.get("binding_rank_score"),
                        "binding_units": r.get("binding_units"),
                        "binding_energy_is_physical": r.get(
                            "binding_energy_is_physical",
                            False,
                        ),
                        "interface_residue_contacts": r.get(
                            "interface_residue_contacts"
                        ),
                        "interface_basic_residue_contacts": r.get(
                            "interface_basic_residue_contacts"
                        ),
                        "interface_min_distance_A": r.get("interface_min_distance_A"),
                        "interface_quality_score": r.get("interface_quality_score"),
                        "interface_passed": r.get("interface_passed"),
                        "interface_steric_clash": r.get("interface_steric_clash"),
                        "interface_contact_entropy": r.get("interface_contact_entropy"),
                        "interface_contact_entropy_normalized": r.get(
                            "interface_contact_entropy_normalized"
                        ),
                        "interface_rna_span_covered": r.get(
                            "interface_rna_span_covered"
                        ),
                        "interface_cluster_count": r.get("interface_cluster_count"),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "output": (
                    f"VERDICT: {verdict}\n"
                    f"REASON: {reason}\n"
                    f"INTERFACE_SCORE: {r.get('interface_quality_score')}\n"
                    f"CONTACTS: {r.get('interface_residue_contacts')} residue-pair "
                    f"contacts, {r.get('interface_basic_residue_contacts')} basic "
                    f"residue contacts, minimum distance "
                    f"{r.get('interface_min_distance_A')} Å.\n"
                    f"TOPOLOGY: RNA span={r.get('interface_rna_span_covered')}, "
                    f"entropy={r.get('interface_contact_entropy')}, "
                    f"normalized_entropy={r.get('interface_contact_entropy_normalized')}, "
                    f"clusters={r.get('interface_cluster_count')}."
                ),
                "metadata": {
                    "source": "interface_contacts",
                    "target_pdb": r.get("target_pdb"),
                    "sequence": r.get("sequence"),
                    "verdict": verdict,
                    "interface_passed": passed,
                    "interface_steric_clash": steric_clash,
                    "binding_units": "hdock_relative_score",
                    "binding_energy_is_physical": False,
                },
            }
        )

    return examples


def _extract_examples_from_state(state: dict) -> tuple[list[dict], list[dict]]:
    """
    Extract all post-run supervised and preference examples.
    """
    examples: list[dict] = []
    preferences: list[dict] = []

    examples.extend(_extract_stage_examples(state))
    examples.extend(_extract_structural_selection_examples(state))
    examples.extend(_extract_literature_target_policy_examples(state))

    docking_examples, docking_preferences = _extract_docking_examples(state)
    examples.extend(docking_examples)
    preferences.extend(docking_preferences)

    preferences.extend(_extract_explicit_docking_preferences(state))
    preferences.extend(_extract_interface_clean_over_clash_preferences(state))

    examples.extend(_extract_target_filter_examples(state))
    examples.extend(_extract_partial_success_target_examples(state))

    inhibitor_examples, inhibitor_preferences = _extract_inhibitor_examples(state)
    examples.extend(inhibitor_examples)
    preferences.extend(inhibitor_preferences)

    examples.extend(_extract_interface_critique_examples(state))
    examples.extend(_extract_critique_revision_examples(state))
    examples.extend(_extract_final_summary_example(state))

    examples = [
        ex for ex in examples
        if isinstance(ex, dict)
        and ex.get("instruction")
        and ex.get("output") is not None
    ]

    preferences = [
        pref for pref in preferences
        if isinstance(pref, dict)
        and pref.get("chosen")
        and pref.get("rejected")
    ]

    preferences = _dedupe_preferences(preferences)

    return examples, preferences


def _extract_docking_rows(state: dict) -> list:
    """
    Save clean docking rows for downstream analysis.
    """
    rows: list[dict] = []

    for r in state.get("binding_results", []) or []:
        if not isinstance(r, dict):
            continue

        rows.append(
            {
                "rank": r.get("rank"),
                "sequence": r.get("sequence"),
                "target_pdb": r.get("target_pdb"),
                "dock_score": r.get("dock_score"),
                "binding_rank_score": r.get("binding_rank_score"),
                "dock_valid": r.get("dock_valid"),
                "binding_mode": r.get("binding_mode"),
                "binding_units": r.get("binding_units"),
                "binding_energy_is_physical": r.get(
                    "binding_energy_is_physical",
                    False,
                ),
                "training_label": r.get("training_label"),
                "target_status": state.get("target_status"),
                "target_status_reason": state.get("target_status_reason"),
                "md_min_energy": (r.get("linked_md") or {}).get("min_energy"),
                "md_mean_energy": (r.get("linked_md") or {}).get("mean_energy"),
                "md_energy_fluctuation": (
                    r.get("linked_md") or {}
                ).get("energy_fluctuation"),
                "interface_quality_score": r.get("interface_quality_score"),
                "interface_passed": r.get("interface_passed"),
                "interface_steric_clash": r.get("interface_steric_clash"),
                "interface_residue_contacts": r.get("interface_residue_contacts"),
                "interface_basic_residue_contacts": r.get(
                    "interface_basic_residue_contacts"
                ),
                "interface_min_distance_A": r.get("interface_min_distance_A"),
                "interface_contact_csv": r.get("interface_contact_csv"),
                "interface_contact_json": r.get("interface_contact_json"),
                "docking_snapshot_png": r.get("docking_snapshot_png"),
                "interface_contact_entropy": r.get("interface_contact_entropy"),
                "interface_contact_entropy_normalized": r.get(
                    "interface_contact_entropy_normalized"
                ),
                "interface_rna_span_covered": r.get("interface_rna_span_covered"),
                "interface_cluster_count": r.get("interface_cluster_count"),
                "training_preference_score": _training_preference_score(r),
                "interface_clash_severity": _interface_clash_severity(r),
                "pose_training_label": _pose_training_label(r),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Automatic LoRA training trigger
# ---------------------------------------------------------------------------

def _check_training_quality(manifest: dict) -> bool:
    """
    Check if collected training data meets quality thresholds for auto-training.
    """
    min_rows = int(os.getenv("VLAB_MIN_TRAIN_ROWS", "20"))
    min_interface_valid = int(os.getenv("VLAB_MIN_INTERFACE_VALID", "5"))
    min_literature = int(os.getenv("VLAB_MIN_LITERATURE_EVIDENCE", "2"))

    examples_count = (
        manifest.get("example_count")
        or manifest.get("counts", {}).get("examples", 0)
    )

    supervised_count = (
        manifest.get("supervised_count")
        or manifest.get("counts", {}).get("supervised", 0)
    )

    if examples_count < min_rows and supervised_count < min_rows:
        log.info(
            "Dataset too small for training: %s examples < %s",
            examples_count,
            min_rows,
        )
        return False

    interface_clean_count = manifest.get("interface_clean_count", 0)

    if interface_clean_count < min_interface_valid:
        log.info(
            "Insufficient interface-valid examples: %s < %s",
            interface_clean_count,
            min_interface_valid,
        )
        return False

    literature_count = manifest.get("literature_evidence_count", 0)

    if literature_count < min_literature:
        log.info(
            "Insufficient literature evidence: %s < %s",
            literature_count,
            min_literature,
        )
        return False

    log.info("Dataset quality check passed for auto-training")
    return True


def _trigger_auto_training(manifest: dict) -> None:
    """
    Trigger automatic LoRA training if enabled and data quality is sufficient.
    """
    if os.getenv("VLAB_LORA_TRAIN", "0") != "1":
        log.info("Auto-training disabled (VLAB_LORA_TRAIN != 1)")
        return

    if not _check_training_quality(manifest):
        return

    py = os.getenv("PY", "python")

    log.info("Triggering automatic LoRA training...")

    try:
        subprocess.Popen(
            [py, "training/auto_retrain.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        log.info("Auto-training started in background")
    except Exception as e:
        log.warning("Failed to trigger auto-training: %s", e)


# ---------------------------------------------------------------------------
# Main post-run pipeline
# ---------------------------------------------------------------------------

def run_postrun_pipeline(topic: dict, final_state: dict | None = None) -> None:
    """
    Extract training data and rebuild/refresh knowledge after run.

    Non-fatal. This function must never invalidate an otherwise successful run.
    """
    log.info("Running post-run training data pipeline...")

    try:
        state = _load_checkpoint_or_state(final_state)

        literature_rows: list[dict] = []

        if state is not None:
            try:
                fm = FailureMemory()
                fm.ingest_run_state(state)
                fm.save()
            except Exception as e:
                log.warning("Failed to update persistent FailureMemory: %s", e)

            try:
                lm = LiteratureMemory()
                literature_rows = lm.ingest_state(state)
                lm.save()
                log.info(
                    "Updated LiteratureMemory with %d evidence records.",
                    len(literature_rows),
                )
            except Exception as e:
                log.warning("Failed to update persistent LiteratureMemory: %s", e)
                literature_rows = []

        if state is None:
            log.warning("Skipping training example extraction: no usable state.")
            examples: list[dict] = []
            preferences: list[dict] = []
        else:
            examples, preferences = _extract_examples_from_state(state)

        supervised: list[dict] = []

        target_selection_reason = (state or {}).get("target_pdb_selection_reason")
        target_rankings = (state or {}).get("target_pdb_rankings", []) or []

        target_selection_quality = "unknown"

        if target_selection_reason:
            target_selection_quality = str(target_selection_reason)

        all_fallback_targets = bool(target_rankings) and all(
            isinstance(x, dict) and "env_fallback" in (x.get("reasons") or [])
            for x in target_rankings
        )

        any_fallback_targets = bool(target_rankings) and any(
            isinstance(x, dict) and "env_fallback" in (x.get("reasons") or [])
            for x in target_rankings
        )

        if all_fallback_targets:
            if str(target_selection_reason) == "reused_existing_target":
                target_selection_quality = "reused_env_fallback_target"
            else:
                target_selection_quality = "env_fallback"

        elif any_fallback_targets:
            target_selection_quality = "mixed_with_env_fallback"

        for ex in examples:
            converted = _as_supervised(ex)

            if converted is not None:
                supervised.append(converted)

        docking_rows = _extract_docking_rows(state or {})

        normalised_pi_training_metadata = _normalise_pi_training_metadata_for_export(
            state or {},
            docking_rows=docking_rows,
        )

        dock_valid_count = sum(
            1 for r in docking_rows
            if isinstance(r, dict) and r.get("dock_valid")
        )

        interface_pass_count = sum(
            1 for r in docking_rows
            if isinstance(r, dict)
            and r.get("dock_valid")
            and r.get("interface_passed")
            and not r.get("interface_steric_clash")
        )

        interface_clash_count = sum(
            1 for r in docking_rows
            if isinstance(r, dict)
            and r.get("dock_valid")
            and r.get("interface_steric_clash")
        )

        interface_clean_count = sum(
            1 for r in docking_rows
            if isinstance(r, dict)
            and r.get("dock_valid")
            and r.get("interface_passed")
            and not r.get("interface_steric_clash")
        )

        weak_interface_count = sum(
            1 for r in docking_rows
            if isinstance(r, dict)
            and r.get("dock_valid")
            and not r.get("interface_passed")
            and not r.get("interface_steric_clash")
        )

        clash_severity_counts = {
            "severe": 0,
            "moderate": 0,
            "borderline": 0,
            "none": 0,
            "unknown": 0,
        }

        pose_label_counts: dict[str, int] = {}

        for row in docking_rows:
            if not isinstance(row, dict):
                continue

            severity = row.get("interface_clash_severity") or "unknown"

            if severity not in clash_severity_counts:
                severity = "unknown"

            clash_severity_counts[severity] += 1

            label = row.get("pose_training_label") or "unknown"
            pose_label_counts[label] = pose_label_counts.get(label, 0) + 1

        preference_type_counts: dict[str, int] = {}

        for pref in preferences:
            if not isinstance(pref, dict):
                continue

            pref_type = pref.get("type") or "unknown"
            preference_type_counts[pref_type] = preference_type_counts.get(pref_type, 0) + 1

        true_preference_count = preference_type_counts.get(
            "docking_interface_preference",
            0,
        )
        partial_interface_preference_count = preference_type_counts.get(
            "partial_interface_evidence_preference",
            0,
        )
        least_bad_preference_count = preference_type_counts.get(
            "least_bad_docking_interface_preference",
            0,
        )
        clean_over_clash_preference_count = preference_type_counts.get(
            "interface_clean_over_steric_clash",
            0,
        )
        clean_over_stronger_clash_preference_count = preference_type_counts.get(
            "reasonable_hdock_clean_interface_over_stronger_hdock_clash",
            0,
        )

        small_molecule_preference_count = preference_type_counts.get(
            "small_molecule_binding_preference",
            0,
        )
        peptide_preference_count = preference_type_counts.get(
            "peptide_binding_preference",
            0,
        )

        partial_success_targets = (state or {}).get("partial_success_targets", []) or []
        partial_success_target_ids = _normalise_partial_success_target_ids(
            partial_success_targets
        )
        resolved_partial_success_targets = (
            (state or {}).get("resolved_partial_success_targets", []) or []
        )

        inhibitor_enabled = (state or {}).get("inhibitor_enabled", False)

        small_mols_raw = (state or {}).get("inhibitor_small_molecules", []) or []
        peptides_raw = (state or {}).get("inhibitor_peptides", []) or []

        small_mols = _dedupe_inhibitor_records(
            small_mols_raw,
            ligand_type="small_molecule",
        )
        peptides = _dedupe_inhibitor_records(
            peptides_raw,
            ligand_type="peptide",
        )

        inhibitor_overlap = (state or {}).get("inhibitor_binding_site_overlap", 0.0)

        manifest = {
            "schema_version": "postrun_training_manifest.v5",
            "topic_name": topic.get("name") if isinstance(topic, dict) else None,
            "research_topic": (state or {}).get("research_topic"),
            "virus_family": (state or {}).get("virus_family"),
            "virus_genus": (state or {}).get("virus_genus"),
            "virus_name": (state or {}).get("virus_name"),

            "target_pdb": (state or {}).get("target_pdb"),
            "target_pdb_id": (state or {}).get("target_pdb_id"),
            "target_pdb_path": (state or {}).get("target_pdb_path"),
            "target_status": (state or {}).get("target_status"),
            "target_status_reason": (state or {}).get("target_status_reason"),
            "partial_success_target_count": len(partial_success_target_ids),
            "partial_success_targets": partial_success_target_ids,
            "resolved_partial_success_targets": resolved_partial_success_targets,
            "resolved_partial_success_target_count": len(resolved_partial_success_targets),
            "target_sequence": (state or {}).get("target_sequence"),
            "best_interface_clean_sequence": (state or {}).get(
                "best_interface_clean_sequence"
            )
            or _select_best_interface_clean_sequence_from_state(state or {}),
            "target_selection_mode": (state or {}).get("target_selection_mode"),
            "target_pdb_selection_reason": target_selection_reason,
            "target_selection_quality": target_selection_quality,
            "target_ranked_candidate_count": len(target_rankings),
            "target_failed_count": len((state or {}).get("failed_target_pdbs", []) or []),

            "example_count": len(examples),
            "supervised_count": len(supervised),
            "preference_count": len(preferences),
            "true_preference_count": true_preference_count,
            "partial_interface_preference_count": partial_interface_preference_count,
            "least_bad_preference_count": least_bad_preference_count,
            "clean_over_clash_preference_count": clean_over_clash_preference_count,
            "clean_over_stronger_clash_preference_count": (
                clean_over_stronger_clash_preference_count
            ),
            "preference_type_counts": preference_type_counts,
            "docking_row_count": len(docking_rows),

            "binding_units": (state or {}).get("binding_units"),
            "dock_valid_count": dock_valid_count,
            "interface_pass_count": interface_pass_count,
            "interface_clean_count": interface_clean_count,
            "weak_interface_count": weak_interface_count,
            "interface_steric_clash_count": interface_clash_count,
            "interface_clash_severity_counts": clash_severity_counts,
            "pose_label_counts": pose_label_counts,

            "inhibitor_enabled": inhibitor_enabled,
            "inhibitor_small_molecule_count": len(small_mols),
            "inhibitor_small_molecule_valid_count": sum(
                1 for r in small_mols
                if isinstance(r, dict) and r.get("valid")
            ),
            "inhibitor_peptide_count": len(peptides),
            "inhibitor_peptide_valid_count": sum(
                1 for r in peptides
                if isinstance(r, dict) and r.get("valid")
            ),
            "inhibitor_binding_site_overlap_percent": inhibitor_overlap,
            "small_molecule_preference_count": small_molecule_preference_count,
            "peptide_preference_count": peptide_preference_count,

            "conservation_valid": (
                ((state or {}).get("conservation_signal") or {}).get("valid")
            ),
            "conservation_fitness": (
                ((state or {}).get("conservation_signal") or {}).get(
                    "conservation_fitness"
                )
            ),
            "designed_sequence_count": len(
                (state or {}).get("designed_sequences", []) or []
            ),
            "binding_result_count": len(
                (state or {}).get("binding_results", []) or []
            ),
            "iteration_count": (state or {}).get("iterations"),
            "max_iterations": (state or {}).get("max_iterations"),
            "pi_training_metadata": normalised_pi_training_metadata,
            "literature_evidence_count": len(literature_rows),

            "files": {
                "examples": str(POSTRUN_DIR / "examples.jsonl"),
                "supervised": str(POSTRUN_DIR / "supervised.jsonl"),
                "preferences": str(POSTRUN_DIR / "preferences.jsonl"),
                "docking_rows": str(POSTRUN_DIR / "docking_rows.jsonl"),
                "manifest": str(POSTRUN_DIR / "manifest.json"),
                "literature_evidence": str(POSTRUN_DIR / "literature_evidence.jsonl"),
            },
        }

        if state is not None:
            try:
                fm = FailureMemory()
                fm.ingest_run_state(
                    {
                        **state,
                        "postrun_training_manifest": manifest,
                    }
                )
                fm.save()
            except Exception as e:
                log.warning("Failed to update persistent FailureMemory: %s", e)

        POSTRUN_DIR.mkdir(parents=True, exist_ok=True)

        _write_json(POSTRUN_DIR / "manifest.json", manifest)
        _write_jsonl(POSTRUN_DIR / "examples.jsonl", examples)
        _write_jsonl(POSTRUN_DIR / "supervised.jsonl", supervised)
        _write_jsonl(POSTRUN_DIR / "preferences.jsonl", preferences)
        _write_jsonl(POSTRUN_DIR / "docking_rows.jsonl", docking_rows)
        _write_jsonl(POSTRUN_DIR / "literature_evidence.jsonl", literature_rows)

        _write_jsonl(Path("examples.jsonl"), examples)
        _write_jsonl(Path("supervised.jsonl"), supervised)

        log.info(
            "Post-run training data written: examples=%d supervised=%d "
            "preferences=%d docking_rows=%d",
            len(examples),
            len(supervised),
            len(preferences),
            len(docking_rows),
        )

        try:
            _trigger_auto_training(manifest)
        except Exception as e:
            log.warning("Auto-training trigger failed/non-fatal: %s", e)

        postrun_query = _topic_seed_text(topic)

        if not postrun_query:
            postrun_query = "viral RNA stem-loop capsid binding"

        try:
            rebuild_papers = expand_knowledge(postrun_query, build_db=True) or []

            log.info(
                "Post-run knowledge base rebuilt with %d documents.",
                len(rebuild_papers),
            )

        except Exception as e:
            log.warning("Post-run knowledge rebuild failed/non-fatal: %s", e)

    except Exception as e:
        log.warning("Post-run training pipeline failed/non-fatal: %s", e)