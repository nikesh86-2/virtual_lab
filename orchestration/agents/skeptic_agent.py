from __future__ import annotations

import logging

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
from VLAB2.orchestration.utils.text_utils import (
    clean_llm_output,
    truncate_str,
    validate_text_block,
)


log = logging.getLogger("virtual_lab")


__all__ = ["skeptic_agent"]


def _inhibitor_summary(state: LabState) -> str:
    """
    Generate a concise summary of inhibitor screening results for the skeptic.
    """
    if not state.get("inhibitor_enabled"):
        return "Inhibitor screening disabled."

    small_mols = state.get("inhibitor_small_molecules", []) or []
    peptides = state.get("inhibitor_peptides", []) or []
    overlap = state.get("inhibitor_binding_site_overlap", 0.0)
    analysis = state.get("inhibitor_analysis", "")

    n_valid_sm = sum(1 for r in small_mols if r.get("valid"))
    n_valid_pep = sum(1 for r in peptides if r.get("valid"))

    lines = []
    if small_mols:
        lines.append(f"Small molecules: {n_valid_sm}/{len(small_mols)} valid")
        # Show best small molecule if available
        valid_sm = [r for r in small_mols if r.get("valid")]
        if valid_sm:
            best_sm = min(valid_sm, key=lambda x: x.get("binding_energy", 999))
            lines.append(
                f"  Best: {best_sm.get('name', 'unknown')} "
                f"(energy={best_sm.get('binding_energy', 'N/A')} kcal/mol)"
            )

    if peptides:
        lines.append(f"Peptides: {n_valid_pep}/{len(peptides)} valid")
        # Show best peptide if available
        valid_pep = [r for r in peptides if r.get("valid")]
        if valid_pep:
            best_pep = min(valid_pep, key=lambda x: x.get("score", 999))
            lines.append(
                f"  Best: {best_pep.get('sequence', 'unknown')[:12]}... "
                f"(score={best_pep.get('score', 'N/A')})"
            )

    if overlap > 0:
        lines.append(f"Binding-site overlap with RNA interface: {overlap:.1f}%")

    if analysis:
        lines.append(f"Analysis: {analysis[:200]}")

    return "\n".join(lines) if lines else "No inhibitor results available."


def skeptic_agent(state: LabState) -> dict:
    """
    Physics-aware peer reviewer.

    Evaluates:
      - HDOCK-relative binding ranks
      - score spread
      - fold threshold status
      - motif/conservation support
      - structural/MD summaries

    Produces structured critique used by PI/optimisation.
    """
    log.info("--- SKEPTIC AGENT: Physics Validation ---")

    try:
        binding_results = state.get("binding_results", [])
        current_target = state.get("target_pdb")

        all_valid = [
            r for r in binding_results
            if isinstance(r, dict) and r.get("valid")
        ]

        if current_target:
            valid = [
                r for r in all_valid
                if r.get("target_pdb") == current_target
            ]
        else:
            valid = all_valid

        # ------------------------------------------------------------
        # No valid binding data
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # Score summary
        # ------------------------------------------------------------
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
            if r.get("dock_method") == "hdock"
            or r.get("vina_method") == "hdock"
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
        accepted_target_status = (
                        f"Accepted target PDB: {current_target}"
                        if current_target
                        else "Accepted target PDB: None. Binding rows represent attempted-target evidence only."
                    )

        prompt = (

            f"Hypothesis:\n{truncate_str(state.get('hypothesis', ''), 600)}\n\n"
            f"{accepted_target_status}\n"
            f"Binding results (n={n_valid}, source: {score_source}):\n"
            f"{binding_summary}\n"
            f"Best binding rank score: {best_score} "
            f"(HDOCK-relative; not kcal/mol) | "
            f"Score spread: {score_range}\n\n"
            f"Fold threshold status:\n"
            f"passed={fold_passed}, quality={safe_jsonable(fold_quality)}, "
            f"reasons={fold_reasons}\n\n"
            f"Selected/conserved motifs:\n"
            f"{motifs}\n\n"
            f"Structural analysis:\n"
            f"{truncate_str(state.get('structural_analysis', ''), 500)}\n\n"
            f"MD analysis:\n"
            f"{truncate_str(state.get('md_analysis', ''), 500)}\n\n"
            f"Inhibitor screening results:\n"
            f"{_inhibitor_summary(state)}\n\n"
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
                        "- Do not describe HDOCK scores as binding free energies.\n"
                        "- If fold_quality contains advisory_warnings showing RNAalifold was excluded "
                        "from hard gating, do not treat RNAalifold failure as a hard structural failure. "
                        "Mention it only as an advisory conservation caveat.\n\n"
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
                        "- STRONG: docking-validated, favourable relative score, small spread, confirmed motif, fold thresholds passed\n"
                        "- MODERATE: docking-validated but incomplete controls, advisory conservation caveats, or moderate spread\n"
                        "- WEAK: weak relative score, high spread, proxy-only, failed hard fold thresholds, or missing MD support\n"
                        "- NONE: no valid binding data\n\n"
                        "Rules:\n"
                        "- Every concern must cite a number from the data.\n"
                        "- If hard fold_threshold_status is failed, recommend REDESIGN_SEQUENCES unless docking and MD are both strong.\n"
                        "- If fold thresholds passed but RNAalifold is advisory-only, do not recommend redesign solely because RNAalifold pair_density is 0.0.\n"
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

        # New: ingest state-level docking/interface information.
        try:
            fm.ingest_run_state(state)
        except Exception as mem_err:
            log.warning("FailureMemory ingest_run_state failed: %s", mem_err)

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
            "energy_range": score_range,
            "score_range": score_range,
            "best_dg": best_score,
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