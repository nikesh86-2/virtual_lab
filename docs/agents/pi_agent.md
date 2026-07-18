# PI Agent Documentation

## Overview
The **pi_agent** (Principal Investigator Agent) implements adaptive multi-objective optimisation using NSGA-II guided by a neural surrogate model. It serves as the evolutionary engine of the VLAB2 pipeline, continuously refining RNA sequences based on feedback from downstream agents (Skeptic, MD, Structural).

## Responsibilities
- Run NSGA-II optimisation with adaptive multi-objective weights
- Refine hypotheses qualitatively based on Skeptic critique
- Integrate feedback from multiple sources (docking, MD, interface, literature)
- Select Pareto-optimal and extreme trade-off sequences
- Apply conservation constraints and motif-based guidance
- Manage target PDB selection and reset logic
- Track and learn from failures via FailureMemory
- Incorporate literature-derived constraints via LiteratureMemory
- Generate training-friendly action summaries for supervised learning

## State Inputs
The agent reads the following fields from `LabState`:

| Field | Type | Description |
|---|---|---|
| `designed_sequences` | `list[str]` | Current population of designed sequences |
| `target_pdb` | `str \| None` | Current protein target PDB ID or path |
| `target_pdb_id` | `str \| None` | PDB ID for the target |
| `target_pdb_path` | `str \| None` | Path to target PDB file |
| `target_status` | `str` | Target status: `accepted_target`, `partial_success_target`, `failed_target`, or `None` |
| `target_status_reason` | `str \| None` | Reason for target status |
| `partial_success_targets` | `list` | Previously identified partial-success targets |
| `resolved_partial_success_targets` | `list` | Resolved partial-success target IDs |
| `failed_target_pdbs` | `list` | List of failed target PDBs |
| `critique` | `str` | Skeptic agent critique from previous iteration |
| `hypothesis` | `str \| None` | Current working hypothesis |
| `topic_description` | `str \| None` | Research topic description |
| `research_topic` | `str \| None` | Alternative topic description |
| `conservation_signal` | `dict` | Conservation analysis results |
| `mutation_bias` | `dict` | Adaptive mutation weights from previous iteration |
| `interface_contacts` | `dict` | RNA-protein interface contacts |
| `binding_results` | `list[dict]` | Docking results from previous iteration |
| `md_results` | `list[dict]` | MD simulation results |
| `fold_thresholds_passed` | `bool` | Whether fold quality thresholds passed |
| `failure_memory` | `list[dict]` | Historical failure data |
| `literature_motif_hints` | `list[str]` | Literature-derived motif hints |
| `literature_target_hints` | `list[str]` | Literature-derived target hints |
| `literature_policy_text` | `str` | Literature-derived policy text |
| `wrappers` | `dict` | Available simulation wrappers (MD, etc.) |
| `data_collector` | `object \| None` | Optional data collection hook |

## State Outputs
The agent adds the following fields to `LabState`:

| Field | Type | Description |
|---|---|---|
| `pi_summary` | `str` | Operational/debug summary (backwards-compatible) |
| `pi_action_summary` | `str` | Qualitative action summary for supervised training (preferred) |
| `pi_operational_summary` | `str` | Runtime/debug summary (alias for pi_summary) |
| `pi_training_metadata` | `dict` | Structured metadata for filtering training examples |
| `optimisation_status` | `str` | Status: `adaptive_pareto_optimised`, `fallback`, `empty_population`, etc. |
| `designed_sequences` | `list[str]` | Selected Pareto-optimal sequences for next iteration |
| `target_sequence` | `str \| None` | Best sequence to use as target |
| `best_interface_clean_sequence` | `str \| None` | Best sequence with clean protein-RNA interface |
| `mutation_bias` | `dict` | Updated adaptive mutation weights |
| `hypothesis` | `str` | Refined hypothesis |
| `iterations` | `int` | Updated iteration count |
| `joint_physics_feedback` | `dict` | Combined docking + MD + interface feedback signals |
| `literature_motif_hints` | `list[str]` | Updated literature motif hints |
| `literature_target_hints` | `list[str]` | Updated literature target hints |
| `literature_policy_text` | `str` | Updated literature policy text |
| `target_pdb` | `str \| None` | Current target PDB (may be reset) |
| `target_status` | `str \| None` | Updated target status |
| `target_status_reason` | `str \| None` | Updated target status reason |
| `failed_target_pdbs` | `list` | Updated list of failed targets |
| `structural_candidates` | `list` | Structural candidates from previous iteration |
| `_run_system_selected_motifs` | `list` | Motifs selected by NSGA-II system |
| `_run_system_min_fold_thresholds` | `dict` | Minimum fold thresholds used |
| `_run_system_sequence_scores` | `dict` | Sequence-level scores from NSGA-II |
| `_run_system_best_interface_sequence` | `str \| None` | Best interface sequence from NSGA-II |

### `pi_training_metadata` Structure
```python
{
    "schema_version": "pi_action_summary.v2",
    "target_pdb": str,  # PDB ID or None
    "target_pdb_id": str,  # PDB ID only
    "target_pdb_path": str,  # File path
    "target_status": str,  # accepted_target, partial_success_target, failed_target
    "target_status_reason": str,  # Reason for status
    "target_policy": str,  # Policy: reuse_as_priority_candidate, accepted_target, avoid_failed_target
    "accepted_target_present": bool,
    "partial_success_target_present": bool,
    "binding_units": str,  # hdock_relative_score
    "binding_energy_is_physical": False,
    "dock_valid_count": int,
    "clean_interface_count": int,
    "interface_clash_count": int,
    "best_interface_clean_sequence": str,  # Best clean interface sequence
    "has_binding": bool,  # Whether binding data available
    "has_md": bool,  # Whether MD data available
    "training_quality": str,  # high, medium, or low
}
```

## Environment Variables
| Variable | Default | Description |
|---|---|---|
| `VLAB_TARGET_RESET_ON_HIGH_SPREAD` | `1` | Enable automatic target reset on high docking score spread |
| `VLAB_TARGET_RESET_REQUIRE_WEAK_BINDING` | `1` | Require weak binding to trigger target reset |
| `VLAB_ACCEPT_BINDING_SCORE` | `-50.0` | Threshold for considering HDOCK relative score favourable |
| `VLAB_MAX_ACCEPT_SCORE_SPREAD` | `10.0` | Maximum acceptable HDOCK relative score spread |
| `VLAB_TARGET_RESET_EXTREME_SPREAD_MULTIPLIER` | `2.0` | Multiplier for extreme spread detection |
| `VLAB_TARGET_MARK_RESET_AS_FAILED` | `0` | Mark reset targets as failed |
| `VLAB_TARGET_RESET_MIN_VALID_N` | `3` | Minimum valid samples before considering target reset |
| `VLAB_RNA_ENFORCE_MIN_FOLD` | `1` | Enforce fold quality thresholds in optimisation |
| `VLAB_MIN_VALID_HDOCK_SCORE` | `-30.0` | Minimum valid HDOCK relative score |

## Core Components

### Target Management
- `_maybe_reset_target_on_high_spread(state)` – Automatically resets target PDB when Skeptic reports problematic docking score spread
- `_target_id_from_state(state)` – Recovers stable target PDB ID from state
- `_target_status_is_final_accepted(state)` – Checks if target is fully accepted
- `_partial_success_target_present(state)` – Checks for partial-success targets
- `_build_target_phrase_for_pi(state, target_pdb, target_status)` – Builds target-aware wording for PI instructions

### Feedback Integration
- `_compute_joint_physics_feedback(state)` – Combines docking, interface, and MD signals into unified feedback:
  - `binding_signal`: Normalised binding strength (0-1)
  - `stability_signal`: Normalised MD stability (0-1)
  - `fluctuation_signal`: Inverse of energy fluctuation (0-1)
  - `interface_signal`: Interface quality signal (0-1)
  - `clean_interface_count`: Number of clean interface sequences
  - `interface_clash_count`: Number of sequences with steric clashes
  - `score_spread`: Variability in docking scores

- `_refine_hypothesis_from_critique(state, critique)` – Uses LLM to rewrite hypothesis qualitatively from Skeptic critique, stripping exact numeric thresholds

- `_build_combined_objectives(state, parsed, extra_objectives, failure_weights, conservation_fitness, critique, score_spread, joint_feedback)` – Builds and normalises multi-objective weights for NSGA-II:
  - `conservation`: Conservation fitness score
  - `binding`: Binding strength pressure
  - `structure`: Fold quality pressure
  - `thermo`: Thermodynamic stability pressure
  - `diversity`: Sequence diversity pressure
  - `interface`: Interface quality pressure

### Optimisation Pipeline
- `run_system(topic, target_pdb, state, extra_objectives)` – Runs NSGA-II optimisation using neural surrogate model
- `analyse_pareto(population, decode_sequence)` – Extracts Pareto front and extreme solutions
- `_select_top_sequences_from_population(population, analysis)` – Decodes Pareto-selected candidates
- `_cleanup_mutation_bias(state)` – Normalises and bounds adaptive mutation weights

### Sequence Selection & Filtering
- `_filter_known_interface_clashes(state, selected_sequences, fallback_sequences, max_count)` – Excludes sequences with known steric clashes, refills from fallback
- `_promote_interface_sequence(state, selected_sequences, max_count)` – Reserves slot for best positive-interface candidate
- `_log_selected_interface_scores(state, selected_sequences)` – Logs interface evidence for selected sequences

### Training Output Generation
- `_build_pi_training_action_summary(state, target_pdb, target_status, target_status_reason, top_sequences, conservation_fitness, score_spread, joint_feedback, selected_motifs)` – Builds qualitative action summary for supervised training
- `_build_pi_operational_summary(target_pdb, top_sequences, conservation_fitness, selected_motifs, min_fold_thresholds, binding_units, new_bias, score_spread, joint_feedback, literature_target_hints, literature_motif_hints, state)` – Runtime/debug summary
- `_capture_optimisation_step_if_available(state, population, top_sequences, new_bias, target_pdb, score_spread, combined_objectives, joint_feedback, training_action_summary)` – Captures optimisation data for data collector

## Agent Workflow

### 1. Target Reset Check
- Parses Skeptic critique for docking score spread
- If spread exceeds `VLAB_MAX_ACCEPT_SCORE_SPREAD` and binding is weak:
  - Resets `target_pdb` to `None` (unless protected by clean interface evidence)
  - Optionally marks reset target as failed

### 2. Sequence Collection
- Deduplicates `designed_sequences`
- If no target PDB and no sequences: returns bootstrap status
- If no target PDB but sequences exist: runs sequence-only optimisation

### 3. Memory Integration
- Loads `FailureMemory` and computes failure weights from historical failures
- Loads `LiteratureMemory` and retrieves motif/target hints
- Stores literature hints back to state for downstream agents

### 4. Critique Parsing & Hypothesis Refinement
- Parses Skeptic critique via `parse_skeptic_output()`
- Generates extra objectives from critique via `generate_objectives()`
- Refines hypothesis qualitatively via LLM (strips exact numeric thresholds)
- Falls back to existing hypothesis if refinement fails

### 5. Feedback Computation
- Computes joint physics feedback from docking, MD, and interface data
- Logs detailed feedback signals (binding, stability, fluctuation, interface)

### 6. Objective Construction
- Builds combined multi-objective weights:
  - Starts with conservation fitness
  - Adds critique-derived objectives
  - Applies fold threshold penalties
  - Adjusts for interface quality (clean interfaces rewarded, clashes penalised)
  - Increases diversity/structure pressure on high score spread
  - Applies failure memory weights
  - Normalises all weights to sum to 1.0

### 7. NSGA-II Optimisation
- Runs `run_system()` with topic, target PDB, and combined objectives
- If population is empty: returns fallback result

### 8. Pareto Selection
- Analyzes Pareto front and extreme solutions
- Selects top sequences based on conservation fitness:
  - High conservation (>0.7): select 3 sequences
  - Low conservation (<0.2): select 2 sequences
  - Medium: select 6 sequences
- Promotes best interface-clean sequence to leading position
- Filters out sequences with known steric clashes
- Refills filtered slots from remaining Pareto candidates

### 9. Mutation Bias Update
- Cleans up and bounds mutation weights
- Applies temperature decay based on iteration count
- Normalises weights

### 10. Training Output Generation
- Builds operational summary for debugging
- Builds qualitative action summary for supervised training
- Builds training metadata for example filtering
- Captures optimisation step if data collector available

### 11. Return State Update
- Returns all updated fields including sequences, hypothesis, mutation bias, feedback, and training outputs

## Error Handling & Edge Cases

The agent handles several failure modes:

- **No wrappers available** – Returns early with `optimisation_status="no_wrappers"`
- **Bootstrap phase** – Returns early waiting for structural design if no sequences
- **Sequence-only optimisation** – Runs without docking when no target PDB
- **Empty population** – Falls back to previous best sequence
- **All sequences have clashes** – Falls back to best safe previously designed sequence
- **Literature memory failure** – Logs warning, continues with empty hints
- **Hypothesis refinement failure** – Falls back to existing hypothesis
- **General exceptions** – Logs stack trace, returns fallback result with previous sequences

## Integration Points

- **Preceded by**: Skeptic Agent (provides critique and feedback)
- **Followed by**: Structural Agent, MD Agent, Inhibitor Agent (receive designed sequences)
- **Memory integration**: Uses `FailureMemory` and `LiteratureMemory` for historical learning
- **Data collection**: Optionally captures optimisation steps via `data_collector`
- **Training data**: Generates `pi_action_summary` and `pi_training_metadata` for supervised fine-tuning

## Key Design Principles

1. **Qualitative reasoning** – Hypotheses and actions use comparative language (stronger, weaker, more stable) rather than exact numeric thresholds
2. **HDOCK-relative scores** – Explicitly treats HDOCK scores as relative ranking, not physical binding free energies
3. **Interface quality priority** – Favors clean-interface variants over raw docking-score improvements
4. **Target status awareness** – Distinguishes between accepted targets, partial-success targets, and failed targets
5. **Adaptive multi-objective** – Dynamically adjusts objective weights based on multidimensional feedback
6. **Training-friendly outputs** – Generates structured, qualitative summaries for supervised learning
7. **Graceful degradation** – Multiple fallback paths ensure pipeline continues even when optimisation fails

---
*Last updated: 2026-07-01*