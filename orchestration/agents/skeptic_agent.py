from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from VLAB2.orchestration.failure_memory import FailureMemory
from VLAB2.orchestration.llm import get_llm
from VLAB2.orchestration.state_schema import (
    LabState,
    safe_jsonable,
    add_conversation_entry,
    record_stage_output,
)
from VLAB2.orchestration.utils.checkpointing import save_checkpoint
from VLAB2.orchestration.utils.docking_utils import safe_binding_rank
from VLAB2.orchestration.utils.fold_utils import motif_summary_from_state
from VLAB2.orchestration.utils.sequence_utils import clean_rna, dedupe_rna_sequences
from VLAB2.orchestration.utils.text_utils import (
    clean_llm_output,
    truncate_str,
    validate_text_block,
)


log = logging.getLogger("virtual_lab")

__all__ = ["skeptic_agent"]


def _normalise_target_id(value: Any) -> str:
    """Normalise a PDB ID or PDB path to an uppercase target key."""
    text = str(value or "").strip()

    if not text:
        return ""

    try:
        text = Path(text).stem
    except Exception:
        pass

    return text.upper()


def _is_usable_binding_row(row: dict) -> bool:
    """Return True for usable measured or legacy-valid binding rows."""
    return (
        isinstance(row, dict)
        and row.get("binding_mode") != "rejected_docking"
        and (
            row.get("dock_valid") is True
            or row.get("vina_valid") is True
            or row.get("valid") is True
        )
    )


def _safe_numeric_score(row: dict) -> float | None:
    """Extract a finite binding rank score without conflating score methods."""
    if not isinstance(row, dict):
        return None

    value = row.get("binding_rank_score", row.get("dg"))

    if value is None:
        value = row.get("dock_score", row.get("hdock_score"))

    try:
        value = float(value)
    except Exception:
        return None

    if not math.isfinite(value):
        return None

    return value


def _target_status_summary(state: LabState) -> str:
    """
    Describe target status without treating partial success as acceptance.
    Uses best_validated_target_evaluation if available.
    """
    # Prioritize best validated target for summary
    best_target = state.get("best_validated_target_evaluation")
    latest_target = state.get("latest_target_evaluation")
    target_record = best_target or latest_target
    
    if target_record and isinstance(target_record, dict):
        target_id = _normalise_target_id(target_record.get("target_pdb"))
        status = target_record.get("status") or "unknown"
        reason = target_record.get("status_reason") or "unknown"
    else:
        # Fall back to legacy fields
        target_id = _normalise_target_id(
            state.get("target_pdb_id")
            or state.get("target_pdb")
        )
        status = state.get("target_status") or "unknown"
        reason = state.get("target_status_reason") or "unknown"

    if status == "accepted_target":
        return (
            f"Target PDB: {target_id or 'None'}; "
            f"status=accepted_target; reason={reason}."
        )

    if status == "partial_success_target":
        return (
            f"Target PDB: {target_id or 'None'}; "
            f"status=partial_success_target; reason={reason}. "
            "This is reusable lower-confidence evidence, not a fully "
            "validated target."
        )

    if status == "failed_target":
        return (
            f"Target PDB: {target_id or 'None'}; "
            f"status=failed_target; reason={reason}."
        )

    return (
        f"Target PDB: {target_id or 'None'}; "
        f"status={status}; reason={reason}."
    )


def _interface_summary(rows: list[dict]) -> tuple[str, dict]:
    """Summarise clean, clashing, and interface-unanalysed docking rows."""
    dock_valid = [
        row
        for row in rows or []
        if isinstance(row, dict)
        and (
            row.get("dock_valid") is True
            or row.get("vina_valid") is True
        )
    ]

    analysed = [
        row
        for row in dock_valid
        if any(
            row.get(key) is not None
            for key in (
                "interface_passed",
                "interface_steric_clash",
                "interface_quality_score",
                "interface_signed_score",
                "interface_evidence_known",
            )
        )
    ]

    clean = [
        row
        for row in analysed
        if row.get("interface_passed") is True
        and row.get("interface_steric_clash") is not True
    ]

    clashes = [
        row
        for row in analysed
        if row.get("interface_steric_clash") is True
    ]

    signed_scores: list[float] = []

    for row in analysed:
        try:
            value = float(row.get("interface_signed_score"))
        except Exception:
            continue

        if math.isfinite(value):
            signed_scores.append(value)

    best_clean_sequence = None

    if clean:
        best_clean_sequence = min(
            clean,
            key=safe_binding_rank,
        ).get("sequence")

    metrics = {
        "dock_valid_count": len(dock_valid),
        "interface_analysed_count": len(analysed),
        "clean_interface_count": len(clean),
        "steric_clash_count": len(clashes),
        "interface_unknown_count": max(0, len(dock_valid) - len(analysed)),
        "best_interface_score": max(signed_scores) if signed_scores else None,
        "worst_interface_score": min(signed_scores) if signed_scores else None,
        "best_interface_clean_sequence": best_clean_sequence,
    }

    summary = (
        f"dock_valid={metrics['dock_valid_count']}, "
        f"analysed={metrics['interface_analysed_count']}, "
        f"clean={metrics['clean_interface_count']}, "
        f"clash={metrics['steric_clash_count']}, "
        f"unknown={metrics['interface_unknown_count']}, "
        f"best_signed_score={metrics['best_interface_score']}, "
        f"worst_signed_score={metrics['worst_interface_score']}, "
        f"best_clean_sequence={best_clean_sequence or 'N/A'}"
    )

    return summary, metrics


def _bioinfo_summary(state: LabState) -> tuple[str, dict]:
    """Summarise MSA quality separately from motif/conservation values."""
    signal = state.get("conservation_signal", {}) or {}

    quality_passed = state.get("bioinfo_quality_passed")

    if quality_passed is None:
        quality_passed = signal.get("quality_passed")

    num_sequences = (
        state.get("bioinfo_num_sequences")
        or signal.get("num_sequences")
        or 0
    )
    alignment_length = (
        state.get("bioinfo_alignment_length")
        or signal.get("alignment_length")
        or 0
    )
    quality_reasons = (
        state.get("bioinfo_quality_reasons")
        or signal.get("quality_reasons")
        or []
    )
    conservation_fitness = float(
        signal.get("conservation_fitness", 0.0) or 0.0
    )

    metrics = {
        "quality_passed": bool(quality_passed),
        "num_sequences": int(num_sequences or 0),
        "alignment_length": int(alignment_length or 0),
        "quality_reasons": quality_reasons,
        "conservation_fitness": conservation_fitness,
    }

    summary = (
        f"quality_passed={metrics['quality_passed']}, "
        f"num_sequences={metrics['num_sequences']}, "
        f"alignment_length={metrics['alignment_length']}, "
        f"quality_reasons={metrics['quality_reasons']}, "
        f"conservation_fitness={metrics['conservation_fitness']:.3f}"
    )

    return summary, metrics


def _inhibitor_summary(state: LabState) -> str:
    """Generate a method-separated inhibitor summary for the skeptic."""
    if not state.get("inhibitor_enabled"):
        return "Inhibitor screening disabled."

    small_mols = state.get("inhibitor_small_molecules", []) or []
    peptides = state.get("inhibitor_peptides", []) or []
    overlap = state.get("inhibitor_binding_site_overlap", 0.0)
    analysis = state.get("inhibitor_analysis", "")

    n_valid_sm = sum(
        1 for row in small_mols
        if isinstance(row, dict) and row.get("valid")
    )
    n_valid_pep = sum(
        1 for row in peptides
        if isinstance(row, dict) and row.get("valid")
    )

    lines: list[str] = []

    if small_mols:
        lines.append(
            f"Small molecules: {n_valid_sm}/{len(small_mols)} valid by AutoDock Vina"
        )
        valid_sm = [
            row for row in small_mols
            if isinstance(row, dict)
            and row.get("valid")
            and row.get("binding_energy") is not None
        ]

        if valid_sm:
            best_sm = min(
                valid_sm,
                key=lambda row: float(row.get("binding_energy", 999.0)),
            )
            lines.append(
                f"  Best within Vina: {best_sm.get('name', 'unknown')} "
                f"(energy={best_sm.get('binding_energy', 'N/A')} kcal/mol)"
            )

    if peptides:
        lines.append(
            f"Peptides: {n_valid_pep}/{len(peptides)} valid by HDOCK"
        )
        valid_pep = [
            row for row in peptides
            if isinstance(row, dict)
            and row.get("valid")
            and row.get("score") is not None
        ]

        if valid_pep:
            best_pep = min(
                valid_pep,
                key=lambda row: float(row.get("score", 999.0)),
            )
            lines.append(
                f"  Best within HDOCK: "
                f"{best_pep.get('sequence', 'unknown')[:12]}... "
                f"(HDOCK-relative score={best_pep.get('score', 'N/A')})"
            )

    try:
        overlap_f = float(overlap)
    except Exception:
        overlap_f = 0.0

    if overlap_f > 0:
        lines.append(
            f"Binding-site overlap with RNA interface: {overlap_f:.1f}%"
        )

    if analysis:
        lines.append(f"Analysis: {str(analysis)[:200]}")

    return "\n".join(lines) if lines else "No inhibitor results available."


def _insufficient_data_result(
    state: LabState,
    target_summary: str,
) -> dict:
    critique = (
        "SUPPORT: NONE\n"
        "ENERGY: best_binding_score=N/A, spread=N/A, n=0, source=none\n"
        "CONCERNS:\n"
        "  1. No usable docking-valid binding results are available for physical evaluation\n"
        "  2. The hypothesis cannot be assessed without measured docking or ranking data\n"
        "  3. Target status requires re-evaluation before acceptance\n"
        "MISSING_CONTROLS: docking score, interface validation, structural validation, MD trajectory\n"
        "RECOMMENDATION: REDESIGN_SEQUENCES\n"
        "REASON: No docking-valid binding rows are available for quantitative assessment."
    )

    result = {
        "critique": critique,
        "results_log": [
            {
                "iteration": state.get("iterations", 0),
                "hypothesis": state.get("hypothesis", ""),
                "target_pdb": state.get("target_pdb"),
                "target_status": state.get("target_status"),
                "energy_range": None,
                "score_range": None,
                "best_dg": None,
                "best_binding_score": None,
                "n_valid": 0,
                "critique": critique,
                "binding_units": "hdock_relative_score",
            }
        ],
        "stage_outputs": [
            record_stage_output(
                state,
                "skeptic",
                critique,
                summary="Skeptic: insufficient docking-valid data",
                metadata={
                    "valid_binding_count": 0,
                    "target_summary": target_summary,
                },
            )
        ],
        "conversation_history": [
            add_conversation_entry(state, "assistant", critique, "skeptic")
        ],
    }

    save_checkpoint({**state, **result})
    return result


def skeptic_agent(state: LabState) -> dict:
    """Physics-, interface-, conservation-, and method-aware peer reviewer."""
    log.info("--- SKEPTIC AGENT: Physics Validation ---")

    try:
        binding_results = state.get("binding_results", []) or []
        target_status_summary = _target_status_summary(state)
        current_target = _normalise_target_id(
            state.get("target_pdb_id")
            or state.get("target_pdb")
        )

        all_valid = [
            row for row in binding_results
            if _is_usable_binding_row(row)
        ]

        if current_target:
            target_valid = []

            for row in all_valid:
                row_target = _normalise_target_id(
                    row.get("target_pdb_id")
                    or row.get("target_pdb")
                )

                if not row_target or row_target == current_target:
                    target_valid.append(row)

            valid = target_valid or all_valid
        else:
            valid = all_valid

        if not valid:
            return _insufficient_data_result(
                state,
                target_status_summary,
            )

        score_pairs = [
            (row, _safe_numeric_score(row))
            for row in valid
        ]
        score_pairs = [
            (row, score)
            for row, score in score_pairs
            if score is not None
        ]

        scores = [score for _, score in score_pairs]
        score_range = (
            max(scores) - min(scores)
            if len(scores) >= 2
            else None
        )
        best_score = min(scores) if scores else None
        n_valid = len(valid)

        dock_count = sum(
            1 for row in valid
            if row.get("dock_valid") is True
            or row.get("vina_valid") is True
        )
        hdock_count = sum(
            1 for row in valid
            if row.get("dock_method") == "hdock"
            or row.get("vina_method") == "hdock"
        )
        proxy_count = max(0, n_valid - dock_count)

        if hdock_count == n_valid and n_valid > 0:
            compact_source = "HDOCK"
        elif dock_count == 0:
            compact_source = "proxy"
        else:
            compact_source = "mixed"

        score_source = (
            f"HDOCK={hdock_count}/{n_valid}, "
            f"docking={dock_count}/{n_valid}, "
            f"proxy={proxy_count}/{n_valid}"
        )

        binding_summary_lines = []

        for row in sorted(valid, key=safe_binding_rank)[:5]:
            row_score = _safe_numeric_score(row)
            method = (
                "HDOCK"
                if row.get("dock_method") == "hdock"
                or row.get("vina_method") == "hdock"
                else "proxy/other"
            )
            binding_summary_lines.append(
                f"  {row.get('sequence', '')[:15]}... -> "
                f"binding_rank_score={row_score}, method={method}, "
                f"rank={row.get('rank')}, "
                f"interface_passed={row.get('interface_passed')}, "
                f"steric_clash={row.get('interface_steric_clash')}"
            )

        binding_summary = "\n".join(binding_summary_lines)
        interface_summary, interface_metrics = _interface_summary(valid)
        bioinfo_summary, bioinfo_metrics = _bioinfo_summary(state)

        fold_quality = state.get("fold_quality", {}) or {}
        fold_passed = state.get("fold_thresholds_passed")
        fold_reasons = state.get("fold_threshold_reasons", []) or []
        motifs = motif_summary_from_state(state)

        prompt = (
            f"Hypothesis:\n{truncate_str(state.get('hypothesis', ''), 600)}\n\n"
            f"{target_status_summary}\n\n"
            f"Binding results (n={n_valid}, source: {score_source}):\n"
            f"{binding_summary}\n"
            f"Best binding rank score: {best_score} "
            f"(HDOCK-relative when method is HDOCK; not kcal/mol) | "
            f"Score spread: {score_range}\n\n"
            f"Interface validation:\n{interface_summary}\n\n"
            f"Fold threshold status:\n"
            f"passed={fold_passed}, quality={safe_jsonable(fold_quality)}, "
            f"reasons={fold_reasons}\n\n"
            f"Bioinformatics evidence quality:\n{bioinfo_summary}\n\n"
            f"Selected/conserved motifs:\n{motifs}\n\n"
            f"Structural analysis:\n"
            f"{truncate_str(state.get('structural_analysis', ''), 500)}\n\n"
            f"MD analysis:\n"
            f"{truncate_str(state.get('md_analysis', ''), 500)}\n\n"
            f"Inhibitor screening results:\n{_inhibitor_summary(state)}\n\n"
            f"PI optimisation summary:\n"
            f"{truncate_str(state.get('pi_summary', ''), 400)}"
        )

        msg = get_llm(temperature=0.0).invoke(
            [
                SystemMessage(
                    content=(
                        "You are a peer reviewer at a computational biophysics journal.\n\n"
                        "Evaluate whether the evidence supports the hypothesis.\n\n"
                        "Mandatory method rules:\n"
                        "- HDOCK scores are relative docking/ranking scores, not kcal/mol.\n"
                        "- Do not describe HDOCK scores as binding free energies.\n"
                        "- AutoDock Vina small-molecule scores are reported in kcal/mol.\n"
                        "- Peptide HDOCK values are relative ranking scores.\n"
                        "- Never compare the numerical magnitude of Vina and HDOCK scores directly.\n"
                        "- Evaluate small molecules and peptides within their own docking method.\n\n"
                        "Interface-aware rules:\n"
                        "- Prefer clean-interface evidence over a more favourable raw HDOCK-relative score with a steric clash.\n"
                        "- A known steric-clash sequence must not receive STRONG support.\n"
                        "- Do not treat interface-unanalysed docking rows as clean poses.\n"
                        "- A target with at least one clean pose but insufficient clean-pose replication is partial success, not complete failure.\n"
                        "- ACCEPT requires target_status=accepted_target and sufficient clean-interface replication.\n"
                        "- A partial_success_target should normally receive REVISE_HYPOTHESIS, REDESIGN_SEQUENCES, or MORE_MD rather than ACCEPT.\n\n"
                        "Fold and conservation rules:\n"
                        "- If fold_quality advisory_warnings show RNAalifold was excluded from hard gating, mention it only as an advisory conservation caveat.\n"
                        "- Do not recommend redesign solely because advisory-only RNAalifold pair_density is 0.0.\n"
                        "- Do not treat motifs or conservation as strong evidence when bioinfo quality_passed is false.\n"
                        "- Treat selected/conserved motif presence as support only when hard fold thresholds pass.\n\n"
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
                        "- STRONG requires docking-valid evidence, at least two clean interface-valid poses, passed fold thresholds, and no unresolved target-status conflict.\n"
                        "- MODERATE is appropriate for a partial-success target with at least one clean pose, or an accepted target with incomplete MD or conservation controls.\n"
                        "- WEAK is appropriate when clashes dominate, only one docking-valid pose exists, hard fold thresholds fail, or conservation evidence fails quality gating.\n"
                        "- NONE is appropriate only when no usable docking-valid evidence exists.\n\n"
                        "Additional rules:\n"
                        "- Every concern must cite a number from the data.\n"
                        "- If hard fold thresholds fail, recommend REDESIGN_SEQUENCES unless docking and MD are both strong.\n"
                        "- Never write may, could, potentially, or might in REASON."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )

        critique = clean_llm_output(msg.content)
        ok, _ = validate_text_block(critique)

        if not ok:
            critique = (
                "SUPPORT: WEAK\n"
                f"ENERGY: best_binding_score={best_score}, spread={score_range}, "
                f"n={n_valid}, source={compact_source}\n"
                "CONCERNS:\n"
                "  1. Structured critique generation failed validation despite available docking data\n"
                f"  2. Interface evidence includes {interface_metrics['clean_interface_count']} clean and {interface_metrics['steric_clash_count']} clashing pose(s)\n"
                f"  3. Bioinformatics quality_passed={bioinfo_metrics['quality_passed']} with {bioinfo_metrics['num_sequences']} sequence(s)\n"
                "MISSING_CONTROLS: independently replicated docking, extended MD validation\n"
                "RECOMMENDATION: REVISE_HYPOTHESIS\n"
                f"REASON: The fallback review retained {n_valid} usable binding rows but the generated critique was malformed."
            )

        designed_sequences = dedupe_rna_sequences(
            state.get("designed_sequences", []) or []
        )
        unique_sequence_count = len(designed_sequences)

        if (
            score_range is not None
            and score_range < 5.0
            and unique_sequence_count < 3
        ):
            critique += (
                "\n\n[CONVERGENCE NOTE: "
                f"Binding-score spread is {score_range:.2f} and only "
                f"{unique_sequence_count} unique designed sequences remain; "
                "the population may have converged prematurely.]"
            )

        elif score_range is not None and score_range < 5.0:
            critique += (
                "\n\n[CONSISTENCY NOTE: "
                f"Binding-score spread is {score_range:.2f} across "
                f"{unique_sequence_count} unique designed sequences; "
                "the relative rankings are comparatively consistent.]"
            )

        from VLAB2.orchestration.skeptic_parser import parse_skeptic_output

        parsed = parse_skeptic_output(critique)
        fm = FailureMemory()
        fm.update(parsed)

        try:
            fm.ingest_run_state(state)
        except Exception as mem_err:
            log.warning("FailureMemory ingest_run_state failed: %s", mem_err)

        fm.save()

        memory = list(state.get("failure_memory", []) or [])
        memory.append(parsed)
        state["failure_memory"] = memory[-10:]

        iteration_record = {
            "iteration": state.get("iterations", 0),
            "hypothesis": state.get("hypothesis", ""),
            "target_pdb": state.get("target_pdb"),
            "target_pdb_id": current_target or None,
            "target_status": state.get("target_status"),
            "target_status_reason": state.get("target_status_reason"),
            "energy_range": score_range,
            "score_range": score_range,
            "best_dg": best_score,
            "best_binding_score": best_score,
            "n_valid": n_valid,
            "critique": critique,
            "binding_units": "hdock_relative_score",
            "interface_metrics": interface_metrics,
            "bioinfo_metrics": bioinfo_metrics,
        }

        result = {
            "critique": critique,
            "results_log": [iteration_record],
            "skeptic_interface_metrics": interface_metrics,
            "skeptic_bioinfo_metrics": bioinfo_metrics,
            "stage_outputs": [
                record_stage_output(
                    state,
                    "skeptic",
                    critique,
                    summary="Physics-, interface-, and conservation-aware critique",
                    metadata={
                        "score_range": score_range,
                        "best_binding_score": best_score,
                        "n_valid": n_valid,
                        "binding_units": "hdock_relative_score",
                        "target_status": state.get("target_status"),
                        "interface_metrics": interface_metrics,
                        "bioinfo_metrics": bioinfo_metrics,
                    },
                )
            ],
            "conversation_history": [
                add_conversation_entry(state, "assistant", critique, "skeptic")
            ],
        }

        save_checkpoint({**state, **result})
        return result

    except Exception as exc:
        log.exception("SKEPTIC AGENT ERROR")
        return {
            "critique": f"Skeptic failed: {exc}",
            "skeptic_error": str(exc),
        }
