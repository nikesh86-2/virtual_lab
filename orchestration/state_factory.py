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

    Keep this aligned with VLAB2.orchestration.state_schema.LabState.
    """
    topic_name = topic.get("topic_name") or topic.get("name") or ""
    research_topic = topic.get("research_topic") or topic_name
    topic_description = topic.get("description") or topic.get("topic_description") or ""

    virus_name = topic.get("virus_name", "")
    virus_family = topic.get("virus_family", "")
    virus_genus = topic.get("virus_genus", "")

    state: LabState = {
        # ------------------------------------------------------------------
        # Topic / setup
        # ------------------------------------------------------------------
        "topic_name": topic_name,
        "research_topic": research_topic,
        "topic_description": topic_description,
        "seed_questions": topic.get("seed_questions", []),
        "failure_memory": [],

        "virus_name": virus_name,
        "virus_family": virus_family,
        "virus_genus": virus_genus,

        # ------------------------------------------------------------------
        # Runtime wrappers / PI setup
        # ------------------------------------------------------------------
        "wrappers": wrappers,
        "mutation_bias": {},

        "current_question_idx": 0,
        "iterations": 0,
        "max_iterations": max_iterations,

        "hypothesis": topic_description or research_topic or "",
        "pi_summary": "",
        "optimisation_status": "",
        "pi_action_summary": "",
        "pi_operational_summary": "",
        "pi_training_metadata": {},
        "best_interface_clean_sequence": None,
        # Phase 4.1: NSGA health reporting defaults
        "nsga_valid_candidate_count": 0,
        "nsga_invalid_candidate_count": 0,
        "nsga_penalty_only_count": 0,
        "nsga_failure_reason_counts": {},
        "nsga_best_sequence": None,
        "nsga_used_fallback_population": False,

        # ------------------------------------------------------------------
        # Literature / researcher state
        # ------------------------------------------------------------------
        "evidence": [],
        "research_query": None,
        "literature_query_bundle": [],
        "literature_topic_profile": {},
        "streamed_queries": [],
        "streaming_started": False,

        "literature_motif_hints": [],
        "literature_target_hints": [],
        "literature_policy_text": "",

        # ------------------------------------------------------------------
        # Agent analysis text
        # ------------------------------------------------------------------
        "structural_analysis": "",
        "structural_status": "",
        "structural_error": None,
        "md_analysis": "",
        "protein_analysis": "",
        "bioinfo_analysis": "",
        "msa_data": "",
        "critique": "",
        "skeptic_interface_metrics": {},
        "skeptic_bioinfo_metrics": {},
        "skeptic_error": None,

        # ------------------------------------------------------------------
        # Conservation
        # ------------------------------------------------------------------
        "conservation_signal": {},
        "conserved_regions": [],
        "conservation_fitness": 0.0,
        "bioinfo_num_sequences": 0,
        "bioinfo_alignment_length": 0,
        "bioinfo_quality_passed": False,
        "bioinfo_quality_reasons": [],
        # Current batch conservation (reset each iteration)
        "current_batch_conservation_signal": {},
        "current_batch_conserved_regions": [],
        "current_batch_conservation_fitness": 0.0,
        "current_batch_msa_mapping": None,
        # Historical conservation (accumulated across iterations)
        "historical_conservation_signal": {},
        "historical_conservation_fitness": None,
        "historical_msa_sequence_count": 0,
        "conservation_iteration_history": [],

        # ------------------------------------------------------------------
        # Design / docking state
        # ------------------------------------------------------------------
        "binding_units": "hdock_relative_score",

        "designed_sequences": [],
        "structural_candidates": [],
        "binding_results": [],
        "md_results": [],
        "interface_contacts": None,
        "interface_contact_files": [],
        "seq_len": 0,

        # Current structural batch (non-reducer - reset each iteration)
        "current_structural_sequences": [],
        "current_structural_candidates": [],
        "current_structural_iteration": 0,

        # ------------------------------------------------------------------
        # Target protein state
        # ------------------------------------------------------------------
        "target_pdb": None,
        "target_pdb_id": None,
        "target_pdb_path": None,
        "target_pdb_candidates": [],
        "failed_target_pdbs": [],
        "partial_success_targets": [],
        "partial_success_sequences": [],
        "target_failure_records": [],
        "target_status": None,
        "target_status_reason": None,
        "resolved_partial_success_targets": [],
        "docking_preferences": [],
        "target_pdb_selection_reason": "",
        "target_pdb_metadata": {},
        "target_pdb_rankings": [],
        "target_selection_mode": os.getenv(
            "VLAB_TARGET_SELECTION_MODE",
            "llm_fallback",
        ),
        "target_sequence": None,
        # Priority 6 fix: Best validated target preservation defaults
        "best_validated_target_status": None,
        "best_validated_target_status_reason": None,
        "best_validated_binding_results": [],
        "best_validated_clean_interface_count": 0,

        # ------------------------------------------------------------------
        # Inhibitor screening
        # ------------------------------------------------------------------
        "inhibitor_enabled": False,
        "inhibitor_small_molecules": [],
        "inhibitor_peptides": [],
        "inhibitor_best_small_molecule": None,
        "inhibitor_best_peptide": None,

        # Legacy-compatible percent field.
        "inhibitor_binding_site_overlap": 0.0,

        # New structured overlap fields.
        "inhibitor_binding_site_overlap_score": 0.0,
        "inhibitor_pose_comparison": {},
        "inhibitor_small_molecule_comparison": None,
        "inhibitor_peptide_comparison": None,

        "inhibitor_docking_box": {},
        "inhibitor_analysis": "",
        "inhibitor_summary": "",
        "inhibitor_snapshot_paths": [],
        # Vina reproducibility and persistent result-cache metadata.
        "inhibitor_vina_seed": None,
        "inhibitor_vina_cache_hits": 0,
        "inhibitor_vina_cache_misses": 0,
        "inhibitor_vina_cache_enabled": False,
        # Inhibitor screening signature for skip logic
        "inhibitor_screen_signature": None,
        "inhibitor_screen_last_signature": None,
        "inhibitor_screen_skipped_count": 0,
        # Phase 2.4: PubChem provider-source metrics
        "pubchem_live_requests": 0,
        "pubchem_cache_hits": 0,
        "pubchem_manifest_hits": 0,
        "pubchem_failures": 0,
        "pubchem_circuit_opened": False,
        # Phase 4.2: Method-specific cache metrics
        "rna_hdock_cache_hits": 0,
        "rna_hdock_cache_misses": 0,
        "peptide_hdock_cache_hits": 0,
        "peptide_hdock_cache_misses": 0,

        # ------------------------------------------------------------------
        # Docking exports
        # ------------------------------------------------------------------
        "docking_summary_json": "",
        "docking_summary_csv": "",
        "docking_summary_md": "",

        # ------------------------------------------------------------------
        # Logs / conversation
        # ------------------------------------------------------------------
        "results_log": [],
        "stage_outputs": [],
        "conversation_history": [],

        # ------------------------------------------------------------------
        # Report / memory
        # ------------------------------------------------------------------
        "final_report": "",
        "previous_hypotheses": [],

        # ------------------------------------------------------------------
        # Runtime caches
        # ------------------------------------------------------------------
        "_md_cache": {},
        "_dock_cache": {},
        "joint_physics_feedback": {},

        "_run_system_selected_motifs": [],
        "_run_system_min_fold_thresholds": {},
        "_run_system_final_weights": {},
        "_run_system_interface_objective_enabled": False,
        "_run_system_interface_lookup_count": 0,
        "_run_system_interface_failure_weights": {},
        "_run_system_seq_len": 0,
        "_run_system_conservation_valid": False,
        "_run_system_sequence_scores": {},
        "_run_system_best_interface_sequence": None,
        "_pi_excluded_interface_clashes": [],
        "streamed_query_keys": [],
    }

    if data_collector is not None:
        state["data_collector"] = data_collector

    return state