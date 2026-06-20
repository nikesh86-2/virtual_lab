from __future__ import annotations

import os
from typing import Any

from VLAB2.orchestration.state_schema import LabState


def build_initial_state(
    topic: dict,
    wrappers: dict,
    max_iterations: int = 3,
    data_collector: Any | None = None,
) -> LabState:
    """
    Build initial LabState for a Virtual Lab run.
    """
    state: LabState = {
        "research_topic": topic["name"],
        "topic_description": topic["description"],
        "seed_questions": topic.get("seed_questions", []),
        "failure_memory": [],

        "wrappers": wrappers,
        "mutation_bias": {},

        "current_question_idx": 0,
        "iterations": 0,
        "max_iterations": max_iterations,

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

        "target_pdb": None,
        "target_pdb_candidates": [],
        "failed_target_pdbs": [],
        "target_pdb_selection_reason": "",
        "target_pdb_metadata": {},
        "target_pdb_rankings": [],
        "target_selection_mode": os.getenv(
            "VLAB_TARGET_SELECTION_MODE",
            "llm_fallback",
        ),
        "target_sequence": None,

        "virus_name": topic.get("virus_name", ""),
        "virus_family": topic.get("virus_family", ""),
        "virus_genus": topic.get("virus_genus", ""),

        "docking_summary_json": "",
        "docking_summary_csv": "",
        "docking_summary_md": "",

        "results_log": [],
        "stage_outputs": [],
        "conversation_history": [],

        "previous_hypotheses": [],

        "_md_cache": {},
        "joint_physics_feedback": {},
    }

    if data_collector is not None:
        state["data_collector"] = data_collector

    return state