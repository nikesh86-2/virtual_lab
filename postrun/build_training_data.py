"""
build_training_data.py

Post-run builder for VLAB2 training data.

Usage:
  python -m VLAB2.postrun.build_training_data lab_results_*.json

Writes:
  postrun_training_data/
    examples.jsonl
    supervised.jsonl
    preferences.jsonl
    docking_rows.jsonl
    literature_evidence.jsonl
    run_feedback_memory.jsonl
    manifest.json
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


OUT_DIR = Path("postrun_training_data")


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            if not isinstance(row, dict):
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _has_clean_interface(r: dict) -> bool:
    return (
        isinstance(r, dict)
        and r.get("dock_valid") is True
        and r.get("interface_passed") is True
        and r.get("interface_steric_clash") is not True
    )


def _pose_label(row: dict) -> str:
    if not isinstance(row, dict):
        return "reject_invalid_row"

    if row.get("implausible_hdock_score"):
        return "reject_implausible_hdock_score"

    if not row.get("dock_valid"):
        return "reject_docking_invalid"

    if _has_clean_interface(row):
        return "accept_interface_valid"

    if row.get("interface_steric_clash"):
        severity = row.get("interface_clash_severity") or "unknown"
        return f"reject_interface_clash_{severity}"

    if row.get("interface_contacts_valid") is False:
        return "dock_valid_interface_analysis_failed"

    return "dock_valid_interface_uncertain"


def _pose_quality_rank(row: dict) -> int:
    if _has_clean_interface(row):
        return 4

    if row.get("dock_valid") and row.get("interface_clash_severity") == "borderline":
        return 3

    if row.get("dock_valid") and row.get("interface_clash_severity") == "moderate":
        return 2

    if row.get("dock_valid"):
        return 1

    return 0


def _extract_attempted_targets(binding_results: list[dict]) -> list[str]:
    return sorted({
        str(r.get("target_pdb")).upper()
        for r in binding_results
        if isinstance(r, dict) and r.get("target_pdb")
    })


def _canonical_docking_row(state: dict, row: dict) -> dict:
    conservation = state.get("conservation_signal", {}) or {}

    return {
        "schema_version": "docking_row.v2",
        "timestamp": _now_iso(),
        "research_topic": state.get("research_topic"),
        "virus_family": state.get("virus_family"),
        "virus_genus": state.get("virus_genus"),
        "virus_name": state.get("virus_name"),
        "sequence": row.get("sequence"),
        "target_pdb": row.get("target_pdb"),
        "target_accepted": bool(state.get("target_pdb") and row.get("target_pdb") == state.get("target_pdb")),

        "dock_valid": row.get("dock_valid"),
        "dock_score": row.get("dock_score"),
        "binding_rank_score": row.get("binding_rank_score"),
        "binding_units": row.get("binding_units", state.get("binding_units", "hdock_relative_score")),
        "binding_mode": row.get("binding_mode"),

        "interface_contacts_valid": row.get("interface_contacts_valid"),
        "interface_passed": row.get("interface_passed"),
        "interface_steric_clash": row.get("interface_steric_clash"),
        "interface_clash_severity": row.get("interface_clash_severity"),
        "interface_quality_score": row.get("interface_quality_score"),
        "interface_min_distance_A": row.get("interface_min_distance_A"),
        "interface_residue_contacts": row.get("interface_residue_contacts"),
        "interface_basic_residue_contacts": row.get("interface_basic_residue_contacts"),
        "interface_basic_contact_fraction": row.get("interface_basic_contact_fraction"),
        "interface_rna_span_covered": row.get("interface_rna_span_covered"),
        "interface_contact_entropy_normalized": row.get("interface_contact_entropy_normalized"),
        "interface_cluster_count": row.get("interface_cluster_count"),

        "linked_md": row.get("linked_md", {}),
        "conservation_valid": conservation.get("valid"),
        "conservation_fitness": conservation.get("conservation_fitness"),

        "pose_label": _pose_label(row),
        "preference_class": None,
        "target_selection_mode": state.get("target_selection_mode"),
        "target_selection_reason": state.get("target_pdb_selection_reason"),
    }


def _extract_docking_rows(state: dict) -> list[dict]:
    rows = []
    for r in state.get("binding_results", []) or []:
        if isinstance(r, dict):
            rows.append(_canonical_docking_row(state, r))
    return rows


def _extract_preferences(state: dict) -> list[dict]:
    valid = [
        r for r in state.get("binding_results", []) or []
        if isinstance(r, dict) and r.get("dock_valid")
    ]

    if len(valid) < 2:
        return []

    accepted_target = state.get("target_pdb")

    any_interface_passed = any(
        _has_clean_interface(r)
        and accepted_target is not None
        and r.get("target_pdb") == accepted_target
        for r in valid
    )

    any_partial_interface_passed = any(
        _has_clean_interface(r)
        for r in valid
    )

    if any_interface_passed:
        pref_type = "docking_interface_preference"
    elif any_partial_interface_passed:
        pref_type = "partial_interface_evidence_preference"
    else:
        pref_type = "least_bad_docking_interface_preference"

    sorted_rows = sorted(valid, key=_pose_quality_rank, reverse=True)
    chosen = sorted_rows[0]
    rejected = sorted_rows[-1]

    if _pose_quality_rank(chosen) <= _pose_quality_rank(rejected):
        return []

    prompt = {
        "research_topic": state.get("research_topic"),
        "accepted_target": accepted_target,
        "binding_units": state.get("binding_units", "hdock_relative_score"),
        "candidate_a": _canonical_docking_row(state, chosen),
        "candidate_b": _canonical_docking_row(state, rejected),
        "instruction": (
            "Choose the better RNA-protein docking pose. Prefer clean interface "
            "geometry over raw HDOCK-relative score. HDOCK scores are relative "
            "ranking scores, not physical binding free energies."
        ),
    }

    return [
        {
            "schema_version": "preference_pair.v2",
            "timestamp": _now_iso(),
            "preference_type": pref_type,
            "chosen": json.dumps(_canonical_docking_row(state, chosen), ensure_ascii=False),
            "rejected": json.dumps(_canonical_docking_row(state, rejected), ensure_ascii=False),
            "prompt": json.dumps(prompt, ensure_ascii=False),
            "metadata": {
                "source": "binding_results",
                "accepted_target": accepted_target,
                "chosen_target": chosen.get("target_pdb"),
                "rejected_target": rejected.get("target_pdb"),
                "chosen_pose_label": _pose_label(chosen),
                "rejected_pose_label": _pose_label(rejected),
                "uses_interface_metrics": True,
                "uses_literature_context": False,
            },
        }
    ]


def _extract_final_summary_example(state: dict) -> list[dict]:
    binding_results = state.get("binding_results", []) or []
    if not binding_results:
        return []

    accepted_target = state.get("target_pdb")
    target_sequence = state.get("target_sequence")

    dock_valid_count = sum(
        1 for r in binding_results
        if isinstance(r, dict) and r.get("dock_valid")
    )

    interface_pass_count = sum(
        1 for r in binding_results
        if isinstance(r, dict) and _has_clean_interface(r)
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
        attempted_targets = _extract_attempted_targets(binding_results)
        if attempted_targets:
            target_sentence = (
                "No final protein target was accepted. Attempted targets "
                f"{', '.join(attempted_targets)} produced"
            )
        else:
            target_sentence = "No final protein target was accepted. Docking attempts produced"

    summary = {
        "research_topic": state.get("research_topic"),
        "target_pdb": accepted_target,
        "target_sequence": target_sequence,
        "binding_results": binding_results,
        "conservation_signal": state.get("conservation_signal", {}),
        "md_analysis": state.get("md_analysis"),
        "protein_analysis": state.get("protein_analysis"),
        "critique": state.get("critique"),
    }

    output_text = (
        f"{target_sentence} {dock_valid_count} docking-valid RNA binding results. "
        f"The best target sequence was {target_sequence}. "
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
                "accepted_target_present": accepted_target is not None,
                "dock_valid_count": dock_valid_count,
                "interface_pass_count": interface_pass_count,
                "interface_steric_clash_count": clash_count,
                "attempted_target_count": len(_extract_attempted_targets(binding_results)),
            },
        }
    ]


def _extract_literature_evidence(state: dict) -> list[dict]:
    """
    Best-effort extraction from state["evidence"].

    Adjust field names if your evidence objects differ.
    """
    rows = []
    evidence = state.get("evidence", []) or []

    for item in evidence:
        if not isinstance(item, dict):
            continue

        text = item.get("abstract") or item.get("text") or item.get("content") or ""
        title = item.get("title") or ""

        if not text and not title:
            continue

        lower = f"{title} {text}".lower()

        target_hints = []
        for term in ["nucleocapsid", "nucleoprotein", "capsid", "rna-binding", "rna binding", "ribonucleoprotein"]:
            if term in lower:
                target_hints.append(term)

        negative_hints = []
        for term in ["spike", "fusion core", "antibody", "fab", "polymerase", "protease"]:
            if term in lower:
                negative_hints.append(term)

        motif_hints = []
        for term in ["stem-loop", "stem loop", "packaging signal", "g-rich", "hairpin"]:
            if term in lower:
                motif_hints.append(term)

        rows.append(
            {
                "schema_version": "literature_evidence.v1",
                "timestamp": _now_iso(),
                "research_topic": state.get("research_topic"),
                "query": item.get("query") or item.get("search_query"),
                "source": item.get("source", "local_or_streamed_literature"),
                "title": title,
                "abstract": text,
                "year": item.get("year"),
                "pmid": item.get("pmid"),
                "doi": item.get("doi"),
                "retrieval_score": item.get("score") or item.get("retrieval_score"),
                "rerank_score": item.get("rerank_score"),
                "evidence_type": "literature_chunk",
                "target_hints": sorted(set(target_hints)),
                "motif_hints": sorted(set(motif_hints)),
                "negative_hints": sorted(set(negative_hints)),
            }
        )

    return rows


def _extract_literature_to_target_examples(literature_rows: list[dict], state: dict) -> list[dict]:
    if not literature_rows:
        return []

    snippets = []
    for row in literature_rows[:5]:
        title = row.get("title") or ""
        abstract = row.get("abstract") or ""
        snippets.append(f"TITLE: {title}\nABSTRACT: {abstract[:1200]}")

    evidence_text = "\n\n---\n\n".join(snippets)

    output = (
        "Prioritise experimentally resolved viral RNA-binding protein targets, "
        "especially nucleocapsid/nucleoprotein or capsid RNA-binding domains. "
        "Prefer compact protein receptors suitable for docking. Avoid RNA-only "
        "structures, antibody-only complexes, spike/fusion-core proteins, "
        "polymerases, proteases, and oversized assemblies unless no better "
        "RNA-interacting target is available."
    )

    return [
        {
            "type": "literature_to_target_policy",
            "instruction": (
                "Given the topic and literature evidence, suggest target-selection "
                "priorities for RNA-protein docking."
            ),
            "input": json.dumps(
                {
                    "research_topic": state.get("research_topic"),
                    "virus_family": state.get("virus_family"),
                    "evidence": evidence_text,
                },
                indent=2,
                ensure_ascii=False,
            ),
            "output": output,
            "metadata": {
                "source": "literature_stream",
                "example_type": "literature_to_target_policy",
            },
        }
    ]


def _build_run_feedback_memory(state: dict) -> list[dict]:
    binding_results = state.get("binding_results", []) or []

    clean_rows = [
        r for r in binding_results
        if isinstance(r, dict) and _has_clean_interface(r)
    ]

    failed_targets = state.get("failed_target_pdbs", []) or []

    best_clean = None
    if clean_rows:
        best_clean = sorted(
            clean_rows,
            key=lambda r: float(r.get("interface_quality_score") or 0.0),
            reverse=True,
        )[0]

    run_label = "successful_target_selected" if state.get("target_pdb") else (
        "partial_success_clean_pose_no_target" if clean_rows else "failed_no_valid_target"
    )

    row = {
        "schema_version": "run_feedback_memory.v1",
        "timestamp": _now_iso(),
        "research_topic": state.get("research_topic"),
        "run_label": run_label,
        "target_pdb": state.get("target_pdb"),
        "best_partial_target": best_clean.get("target_pdb") if best_clean else None,
        "best_clean_sequence": best_clean.get("sequence") if best_clean else None,
        "failed_targets": failed_targets,
        "successful_features": {
            "interface_quality_score": best_clean.get("interface_quality_score") if best_clean else None,
            "interface_min_distance_A": best_clean.get("interface_min_distance_A") if best_clean else None,
            "interface_basic_residue_contacts": best_clean.get("interface_basic_residue_contacts") if best_clean else None,
            "interface_rna_span_covered": best_clean.get("interface_rna_span_covered") if best_clean else None,
        },
        "next_pi_bias": {
            "increase_interface_geometry_weight": bool(clean_rows),
            "promote_targets": sorted({r.get("target_pdb") for r in clean_rows if r.get("target_pdb")}),
            "promote_sequences": sorted({r.get("sequence") for r in clean_rows if r.get("sequence")}),
        },
    }

    return [row]


def build_training_data(state: dict) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    docking_rows = _extract_docking_rows(state)
    preferences = _extract_preferences(state)
    final_summary_examples = _extract_final_summary_example(state)
    literature_rows = _extract_literature_evidence(state)
    literature_examples = _extract_literature_to_target_examples(literature_rows, state)
    run_feedback_rows = _build_run_feedback_memory(state)

    supervised = final_summary_examples + literature_examples
    examples = supervised

    counts = {
        "examples": _append_jsonl(OUT_DIR / "examples.jsonl", examples),
        "supervised": _append_jsonl(OUT_DIR / "supervised.jsonl", supervised),
        "preferences": _append_jsonl(OUT_DIR / "preferences.jsonl", preferences),
        "docking_rows": _append_jsonl(OUT_DIR / "docking_rows.jsonl", docking_rows),
        "literature_evidence": _append_jsonl(OUT_DIR / "literature_evidence.jsonl", literature_rows),
        "run_feedback_memory": _append_jsonl(OUT_DIR / "run_feedback_memory.jsonl", run_feedback_rows),
    }

    manifest = {
        "schema_version": "postrun_training_manifest.v3",
        "timestamp": _now_iso(),
        "research_topic": state.get("research_topic"),
        "virus_family": state.get("virus_family"),
        "target_pdb": state.get("target_pdb"),
        "target_sequence": state.get("target_sequence"),
        "target_selection_mode": state.get("target_selection_mode"),
        "target_pdb_selection_reason": state.get("target_pdb_selection_reason"),
        "counts": counts,
        "dock_valid_count": sum(1 for r in docking_rows if r.get("dock_valid")),
        "interface_clean_count": sum(1 for r in docking_rows if r.get("pose_label") == "accept_interface_valid"),
        "partial_success_targets": sorted({
            r.get("target_pdb")
            for r in docking_rows
            if r.get("pose_label") == "accept_interface_valid"
            and not state.get("target_pdb")
            and r.get("target_pdb")
        }),
        "files": {
            "examples": str(OUT_DIR / "examples.jsonl"),
            "supervised": str(OUT_DIR / "supervised.jsonl"),
            "preferences": str(OUT_DIR / "preferences.jsonl"),
            "docking_rows": str(OUT_DIR / "docking_rows.jsonl"),
            "literature_evidence": str(OUT_DIR / "literature_evidence.jsonl"),
            "run_feedback_memory": str(OUT_DIR / "run_feedback_memory.jsonl"),
            "manifest": str(OUT_DIR / "manifest.json"),
        },
    }

    _write_json(OUT_DIR / "manifest.json", manifest)
    return manifest


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m VLAB2.postrun.build_training_data lab_results.json")
        raise SystemExit(2)

    path = Path(sys.argv[1])
    state = json.loads(path.read_text(encoding="utf-8"))
    manifest = build_training_data(state)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()