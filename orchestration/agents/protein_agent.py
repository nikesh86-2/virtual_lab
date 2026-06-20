from __future__ import annotations

import logging
import os
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from VLAB2.core.gpu_manager import clear_gpu
from VLAB2.core.protein_prep import ensure_protein_pdb
from VLAB2.core.rna_prep import prepare_rna_pdb_for_hdock

from VLAB2.orchestration.llm import get_llm
from VLAB2.orchestration.state_schema import (
    LabState,
    add_conversation_entry,
    record_stage_output,
)
from VLAB2.orchestration.utils.docking_visuals import (
    render_docking_snapshots_for_results,
)
from VLAB2.orchestration.utils.interface_contacts import (
    analyse_interface_contacts_for_results,
)

from VLAB2.orchestration.utils.checkpointing import save_checkpoint

from VLAB2.orchestration.utils.docking_visuals import (
    render_docking_snapshots_for_results,
)

from VLAB2.orchestration.utils.docking_utils import (
    export_docking_outputs,
    parse_pdb_candidates,
    safe_binding_rank,
)
from VLAB2.orchestration.utils.env_utils import getenv_int
from VLAB2.orchestration.utils.sequence_utils import dedupe_rna_sequences


try:
    from VLAB2.core.rcsb_target_selector import select_pdb_targets
except Exception:
    select_pdb_targets = None


log = logging.getLogger("virtual_lab")

__all__ = ["protein_agent"]


def _collect_sequences_for_protein(state: LabState, eval_top_n: int) -> list[str]:
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


def _select_candidate_pdbs(state: LabState) -> tuple[list[str], str, list[dict]]:
    """
    Select candidate protein PDB IDs.

    Order:
      1. existing target_pdb
      2. existing target_pdb_candidates
      3. dynamic RCSB ranking if enabled
      4. LLM PDB selection fallback
      5. VLAB_FALLBACK_PDBS env fallback

    Failed and blacklisted target PDBs are excluded after candidate gathering.
    """
    candidate_pdbs: list[str] = []
    selection_reason = ""
    target_rankings: list[dict] = list(state.get("target_pdb_rankings", []) or [])

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

                target_rankings = ranked_targets

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

    failed_set = {str(x).upper() for x in state.get("failed_target_pdbs", [])}

    blacklist = {
        x.strip().upper()
        for x in os.getenv("VLAB_TARGET_BLACKLIST", "").split(",")
        if x.strip()
    }

    candidate_pdbs = [
        p for p in candidate_pdbs
        if p.upper() not in failed_set and p.upper() not in blacklist
    ]

    if not candidate_pdbs:
        fallback_env = os.getenv("VLAB_FALLBACK_PDBS", "")
        fallback_list = [
            x.strip().upper()
            for x in fallback_env.split(",")
            if x.strip()
        ]

        fallback_list = [
            p for p in fallback_list
            if (
                len(p) == 4
                and p[0].isdigit()
                and p.upper() not in failed_set
                and p.upper() not in blacklist
            )
        ]

        if fallback_list:
            selection_reason = "env_fallback_candidates"
            candidate_pdbs = fallback_list

    candidate_pdbs = list(dict.fromkeys(candidate_pdbs))

    return candidate_pdbs, selection_reason, target_rankings


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

    return r


def _set_docking_skip(r: dict, method: str, error: str) -> dict:
    r["dock_method"] = method
    r["dock_error"] = error
    r["vina_method"] = method
    r["vina_error"] = error
    r["binding_mode"] = "proxy"

    return r


def _score_and_rank_binding_results(valid_binding_results: list[dict]) -> list[dict]:
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
            r["binding_mode"] = "docked"

        elif r.get("vina_valid") and r.get("vina_energy") is not None:
            dock_score = float(r["vina_energy"])

            r["proxy_dg"] = proxy_dg
            r["hdock_score"] = dock_score
            r["binding_rank_score"] = 0.7 * dock_score + 0.3 * proxy_dg
            r["dg"] = r["binding_rank_score"]
            r["binding_units"] = "hdock_relative_score"
            r["binding_mode"] = "docked"

        else:
            r["binding_rank_score"] = proxy_dg
            r["dg"] = proxy_dg
            r["binding_units"] = "proxy_score"
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


def _updated_md_results_with_docking(
    state: LabState,
    seq: str,
    pdb_id: str,
    dock_score: float,
) -> list[dict]:
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

            # Legacy compatibility.
            copied_result["vina_energy"] = dock_score
            copied_result["vina_valid"] = True
            copied_result["vina_target_pdb"] = pdb_id

            copied["result"] = copied_result

        updated.append(copied)

    return updated


def protein_agent(state: LabState) -> dict:
    """
    Protein/RNA binding evaluation agent.

    Behaviour:
      - evaluates up to VLAB_AGENT_EVAL_TOP_N RNA candidates
      - dynamically selects or reuses target PDBs
      - runs wrapper proxy scoring
      - requires MD-generated RNA PDBs for HDOCK
      - prepares receptor and RNA ligand PDBs
      - runs HDOCK
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
            "max_receptor_atoms=%d abort_timeout=%s",
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

        failed_targets = list(state.get("failed_target_pdbs", []))

        if not candidate_pdbs:
            summary = "No valid protein targets available after filtering."

            result = {
                "protein_analysis": summary,
                "binding_results": [],
                "failed_sequences": [],
                "target_pdb": None,
                "target_pdb_candidates": [],
                "failed_target_pdbs": failed_targets,
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
        newly_failed: List[str] = []
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
                newly_failed.append(pdb_id)
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
                newly_failed.append(pdb_id)
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
                        r["dock_error"] = (
                            dock.get("error")
                            if isinstance(dock, dict) and dock.get("error")
                            else "HDOCK returned no valid docking score"
                        )

                        r["vina_valid"] = False
                        r["vina_energy"] = None
                        r["vina_method"] = r["dock_method"]
                        r["vina_error"] = r["dock_error"]
                        r["binding_mode"] = "proxy"

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

                    failed_sequences.append(
                        {"sequence": seq, "target_pdb": pdb_id, "error": str(e)}
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
                if r.get("dock_valid") or r.get("vina_valid")
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

            # ------------------------------------------------------------
            # Docking-convergence enforcement.
            #
            # Only enforce these hard gates when VLAB_REQUIRE_DOCKING=1.
            # If VLAB_REQUIRE_DOCKING=0, allow the later proxy/enhanced-results
            # fallback branch to preserve legacy/debug behaviour.
            # ------------------------------------------------------------
            if require_docking and len(dock_valid) < min_valid_dockings_per_target:
                log.warning(
                    "Rejecting target %s: only %d docking-valid results after filtering; "
                    "required %d.",
                    pdb_id,
                    len(dock_valid),
                    min_valid_dockings_per_target,
                )
                newly_failed.append(pdb_id)
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
                newly_failed.append(pdb_id)
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
                newly_failed.append(pdb_id)
                clear_gpu()
                continue

            valid_binding_results = _score_and_rank_binding_results(
                valid_binding_results
            )

            clear_gpu()
            break

        failed_targets = list(dict.fromkeys(failed_targets + newly_failed))

        # ------------------------------------------------------------
        # No target succeeded
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
            "target_pdb_selection_reason": selection_reason,
            "target_pdb_rankings": target_rankings,
            "binding_units": "hdock_relative_score",
            "md_results": md_results_updated,
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

                # Add compact interface notes to the human-readable protein summary.
                interface_lines = ["", "Interface contact summary:"]

                for r in result.get("binding_results", [])[:5]:
                    if not isinstance(r, dict):
                        continue

                    if not r.get("dock_valid"):
                        continue

                    interface_lines.append(
                        "  - Seq: {seq}... | contacts={contacts} | basic={basic} | "
                        "min_dist={dist} Å | interface_score={score} | passed={passed}".format(
                            seq=(r.get("sequence") or "")[:20],
                            contacts=r.get("interface_residue_contacts"),
                            basic=r.get("interface_basic_residue_contacts"),
                            dist=r.get("interface_min_distance_A"),
                            score=r.get("interface_quality_score"),
                            passed=r.get("interface_passed"),
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