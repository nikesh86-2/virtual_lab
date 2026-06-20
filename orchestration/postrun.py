from __future__ import annotations

import json
import logging
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


log = logging.getLogger("virtual_lab")


POSTRUN_DIR = Path("postrun_training_data")

def _training_preference_score(r: dict) -> float:
    """
    Higher = better for preference training.

    Combines:
      - lower binding_rank_score / dg
      - better interface quality
      - fewer clashes
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

    steric_clash = bool(r.get("interface_steric_clash"))

    linked_md = r.get("linked_md") or {}

    try:
        md_min = float(linked_md.get("min_energy") or 0.0)
    except Exception:
        md_min = 0.0

    score = (
        binding_component
        + 12.0 * interface_quality
        + 0.15 * residue_contacts
        + 0.75 * basic_contacts
        + 0.015 * abs(md_min)
    )

    if steric_clash:
        score -= 35.0

    if not r.get("dock_valid"):
        score -= 50.0

    if r.get("binding_mode") == "rejected_docking":
        score -= 50.0

    return score

def _preference_payload(r: dict) -> dict:
    return {
        "sequence": r.get("sequence"),
        "target_pdb": r.get("target_pdb"),
        "dock_score": r.get("dock_score"),
        "binding_rank_score": r.get("binding_rank_score"),
        "interface_quality_score": r.get("interface_quality_score"),
        "interface_residue_contacts": r.get("interface_residue_contacts"),
        "interface_basic_residue_contacts": r.get(
            "interface_basic_residue_contacts"
        ),
        "interface_min_distance_A": r.get("interface_min_distance_A"),
        "interface_steric_clash": r.get("interface_steric_clash"),
        "interface_passed": r.get("interface_passed"),
        "md_min_energy": (r.get("linked_md") or {}).get("min_energy"),
        "training_preference_score": _training_preference_score(r),
    }

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

        if not output:
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

    examples.append(
        {
            "type": "docking_ranking",
            "instruction": (
                "Rank these RNA candidates by HDOCK-relative docking quality. "
                "Use lower HDOCK-relative scores as better, but do not describe "
                "them as physical kcal/mol binding energies."
            ),
            "input": json.dumps(valid, indent=2, ensure_ascii=False),
            "output": json.dumps(
                [
                    {
                        "rank": i + 1,
                        "sequence": r.get("sequence"),
                        "target_pdb": r.get("target_pdb"),
                        "dock_score": r.get("dock_score"),
                        "binding_rank_score": r.get("binding_rank_score"),
                        "interface_quality_score": r.get("interface_quality_score"),
                        "interface_residue_contacts": r.get("interface_residue_contacts"),
                        "interface_basic_residue_contacts": r.get(
                            "interface_basic_residue_contacts"
                        ),
                        "interface_min_distance_A": r.get("interface_min_distance_A"),
                        "interface_steric_clash": r.get("interface_steric_clash"),
                        "interface_passed": r.get("interface_passed"),
                        "training_preference_score": _training_preference_score(r),
                        "reason": (
                            "Candidate ranked by combined HDOCK-relative score, "
                            "interface quality, basic residue contacts, MD support, "
                            "and steric clash penalty."
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
                "n_valid": len(valid),
            },
        }
    )

    if len(valid) >= 2:
        best = valid[0]

        for other in valid[1:]:
            preferences.append(
                {
                    "type": "docking_interface_preference",
                    "prompt": (
                        "Choose the better RNA binder using HDOCK-relative docking score, "
                        "interface contact quality, basic residue enrichment, MD stability, "
                        "and steric clash evidence. HDOCK scores are relative docking scores, "
                        "not physical kcal/mol free energies."
                    ),
                    "chosen": _preference_payload(best),
                    "rejected": _preference_payload(other),
                    "metadata": {
                        "source": "binding_results",
                        "target_pdb": state.get("target_pdb"),
                        "criterion": "combined_docking_interface_md_score",
                    },
                }
            )
    return examples, preferences


def _extract_target_filter_examples(state: dict) -> list:
    """
    Train target filtering/failure memory from failed and accepted targets.
    """
    examples: list[dict] = []

    rankings = state.get("target_pdb_rankings", []) or []
    failed = set(str(x).upper() for x in state.get("failed_target_pdbs", []) or [])
    accepted = state.get("target_pdb")

    if not rankings and not failed and not accepted:
        return examples

    examples.append(
        {
            "type": "target_filtering",
            "instruction": (
                "Given ranked PDB target candidates and observed docking failures, "
                "select a suitable target for RNA docking. Avoid oversized receptors, "
                "weak docking targets, and previously failed PDBs."
            ),
            "input": json.dumps(
                {
                    "target_pdb_rankings": rankings,
                    "failed_target_pdbs": sorted(failed),
                    "target_blacklist": state.get("target_blacklist", []),
                    "accepted_target": accepted,
                },
                indent=2,
                ensure_ascii=False,
            ),
            "output": json.dumps(
                {
                    "selected_target_pdb": accepted,
                    "avoid_targets": sorted(failed),
                    "reason": (
                        "Accepted target produced sufficient docking-valid results "
                        "with HDOCK-relative scores passing the configured filters."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            ),
            "metadata": {
                "source": "target_selection",
                "accepted_target": accepted,
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
    pi_summary = state.get("pi_summary")
    joint_feedback = state.get("joint_physics_feedback", {})

    if not critique:
        return []

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
                    "critique": critique,
                    "joint_physics_feedback": joint_feedback,
                    "conservation_signal": state.get("conservation_signal", {}),
                    "binding_units": state.get("binding_units"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            "output": pi_summary or critique,
            "metadata": {
                "source": "critique",
                "recommendation_present": "RECOMMENDATION:" in critique,
            },
        }
    ]


def _extract_final_summary_example(state: dict) -> list:
    """
    Train compact final reporting.
    """
    if not state.get("binding_results"):
        return []

    summary = {
        "research_topic": state.get("research_topic"),
        "target_pdb": state.get("target_pdb"),
        "target_sequence": state.get("target_sequence"),
        "binding_results": state.get("binding_results", []),
        "conservation_signal": state.get("conservation_signal", {}),
        "md_analysis": state.get("md_analysis"),
        "protein_analysis": state.get("protein_analysis"),
        "critique": state.get("critique"),
    }

    return [
        {
            "type": "final_report_summary",
            "instruction": (
                "Write a concise final scientific summary of this RNA design and "
                "docking run. Clearly distinguish HDOCK-relative docking scores "
                "from physical binding energies."
            ),
            "input": json.dumps(summary, indent=2, ensure_ascii=False),
            "output": (
                f"Target {state.get('target_pdb')} produced "
                f"{len(state.get('binding_results', []) or [])} docking-valid RNA "
                f"binding results. The best target sequence was "
                f"{state.get('target_sequence')}. HDOCK-relative scores should be "
                f"interpreted as relative docking scores, not physical binding "
                f"free energies."
            ),
            "metadata": {
                "source": "final_state",
                "target_pdb": state.get("target_pdb"),
            },
        }
    ]


def _extract_examples_from_state(state: dict) -> tuple[list[dict], list[dict]]:
    """
    Extract all post-run supervised and preference examples.
    """
    examples: list[dict] = []
    preferences: list[dict] = []

    examples.extend(_extract_stage_examples(state))
    examples.extend(_extract_structural_selection_examples(state))

    docking_examples, docking_preferences = _extract_docking_examples(state)
    examples.extend(docking_examples)
    preferences.extend(docking_preferences)

    examples.extend(_extract_target_filter_examples(state))
    examples.extend(_extract_interface_critique_examples(state))
    examples.extend(_extract_critique_revision_examples(state))
    examples.extend(_extract_final_summary_example(state))

    # Remove invalid examples.
    examples = [
        ex for ex in examples
        if isinstance(ex, dict) and ex.get("instruction") and ex.get("output") is not None
    ]

    preferences = [
        pref for pref in preferences
        if isinstance(pref, dict) and pref.get("chosen") and pref.get("rejected")
    ]

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
                "md_min_energy": (r.get("linked_md") or {}).get("min_energy"),
                "md_mean_energy": (r.get("linked_md") or {}).get("mean_energy"),
                "md_energy_fluctuation": (
                    r.get("linked_md") or {}
                ).get("energy_fluctuation"),
                "interface_quality_score": r.get("interface_quality_score"),
                "interface_passed": r.get("interface_passed"),
                "interface_steric_clash": r.get("interface_steric_clash"),
                "interface_residue_contacts": r.get("interface_residue_contacts"),
                "interface_basic_residue_contacts": r.get("interface_basic_residue_contacts"),
                "interface_min_distance_A": r.get("interface_min_distance_A"),
                "interface_contact_csv": r.get("interface_contact_csv"),
                "interface_contact_json": r.get("interface_contact_json"),
                "docking_snapshot_png": r.get("docking_snapshot_png"),
                "training_preference_score": _training_preference_score(r),
            }
        )

    return rows

def _extract_interface_critique_examples(state: dict) -> list"""
    Train the model to critique docking poses using contact metrics.
    """
    examples = []

    for r in state.get("binding_results", []) or []:
        if not isinstance(r, dict) or not r.get("dock_valid"):
            continue

        steric_clash = bool(r.get("interface_steric_clash"))
        passed = bool(r.get("interface_passed"))

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
                "one basic residue contact, and no severe steric clash."
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
                    f"{r.get('interface_min_distance_A')} Å."
                ),
                "metadata": {
                    "source": "interface_contacts",
                    "target_pdb": r.get("target_pdb"),
                    "sequence": r.get("sequence"),
                },
            }
        )

    return examples

def run_postrun_pipeline(topic: dict, final_state: dict | None = None) -> None:
    """
    Extract training data and rebuild/refresh knowledge after run.

    Non-fatal. This function must never invalidate an otherwise successful run.
    """
    log.info("Running post-run training data pipeline...")

    try:
        state = _load_checkpoint_or_state(final_state)

        if state is None:
            log.warning("Skipping training example extraction: no usable state.")
            examples: list[dict] = []
            preferences: list[dict] = []
        else:
            examples, preferences = _extract_examples_from_state(state)

        supervised = []

        for ex in examples:
            converted = _as_supervised(ex)

            if converted is not None:
                supervised.append(converted)

        docking_rows = _extract_docking_rows(state or {})

        manifest = {
            "topic_name": topic.get("name") if isinstance(topic, dict) else None,
            "research_topic": (state or {}).get("research_topic"),
            "target_pdb": (state or {}).get("target_pdb"),
            "target_sequence": (state or {}).get("target_sequence"),
            "example_count": len(examples),
            "supervised_count": len(supervised),
            "preference_count": len(preferences),
            "docking_row_count": len(docking_rows),
            "binding_units": (state or {}).get("binding_units"),
        }

        POSTRUN_DIR.mkdir(parents=True, exist_ok=True)

        _write_json(POSTRUN_DIR / "manifest.json", manifest)
        _write_jsonl(POSTRUN_DIR / "examples.jsonl", examples)
        _write_jsonl(POSTRUN_DIR / "supervised.jsonl", supervised)
        _write_jsonl(POSTRUN_DIR / "preferences.jsonl", preferences)
        _write_jsonl(POSTRUN_DIR / "docking_rows.jsonl", docking_rows)

        # Backwards-compatible output names.
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