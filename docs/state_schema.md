# State Schema Documentation

## Overview

The **State Schema** defines the canonical `LabState` TypedDict that flows through the entire VLAB2 pipeline. Every agent reads from and writes to this shared state, enabling seamless handoff between research phases.

Located in: `orchestration/state_schema.py`

## Design Principles

1. **Single source of truth** — Only one `LabState` definition; all agents import from here
2. **Type safety** — Uses `TypedDict` with `total=False` for optional fields
3. **Automatic deduplication** — Reducer functions prevent duplicate accumulation
4. **Serialisability** — `safe_jsonable()` handles non-serialisable objects (wrappers, numpy arrays)
5. **Extensibility** — New fields can be added without breaking existing agents

## LabState Structure

### Topic & Setup Fields

| Field | Type | Description |
|---|---|---|
| `topic_name` | `str` | Name of the research topic |
| `research_topic` | `str` | Full research topic description |
| `topic_description` | `str` | Detailed topic description |
| `seed_questions` | `List[str]` | Seed questions for research |
| `virus_name` | `str` | Virus name (if applicable) |
| `virus_family` | `str` | Virus family classification |
| `virus_genus` | `str` | Virus genus classification |

### Iteration Control

| Field | Type | Description |
|---|---|---|
| `iterations` | `int` | Current iteration count (starts at 0) |
| `max_iterations` | `int` | Maximum allowed iterations |
| `current_question_idx` | `int` | Current question index being addressed |

### PI / Optimisation Fields

| Field | Type | Description |
|---|---|---|
| `hypothesis` | `str` | Current working hypothesis |
| `pi_summary` | `str` | PI operational/debug summary |
| `pi_action_summary` | `str` | Qualitative action summary for training |
| `pi_operational_summary` | `str` | Alias for pi_summary |
| `pi_training_metadata` | `dict` | Structured metadata for training filtering |
| `optimisation_status` | `str` | Status: `adaptive_pareto_optimised`, `fallback`, etc. |
| `mutation_bias` | `dict` | Adaptive mutation weights |
| `joint_physics_feedback` | `dict` | Combined docking + MD + interface feedback |
| `best_interface_clean_sequence` | `str \| None` | Best sequence with clean interface |

### Agent Output Fields

| Field | Type | Description |
|---|---|---|
| `evidence` | `List[Any]` | Literature evidence (deduplicated) |
| `structural_analysis` | `str` | Structural agent analysis text |
| `md_analysis` | `str` | MD agent analysis text |
| `protein_analysis` | `str` | Protein agent analysis text |
| `bioinfo_analysis` | `str` | Bioinformatics agent analysis text |
| `msa_data` | `str` | Multiple sequence alignment data |
| `critique` | `str` | Skeptic agent critique |
| `skeptic_interface_metrics` | `dict` | Skeptic interface quality metrics |
| `skeptic_bioinfo_metrics` | `dict` | Skeptic bioinformatics metrics |
| `skeptic_error` | `str \| None` | Skeptic error message (if any) |

### Conservation Fields

| Field | Type | Description |
|---|---|---|
| `conservation_signal` | `dict` | Conservation analysis results |
| `conserved_regions` | `List` | Identified conserved regions |
| `conservation_fitness` | `float` | Normalised conservation fitness score |
| `bioinfo_num_sequences` | `int` | Number of sequences in MSA |
| `bioinfo_alignment_length` | `int` | Length of multiple sequence alignment |
| `bioinfo_quality_passed` | `bool` | Whether bioinformatics quality checks passed |
| `bioinfo_quality_reasons` | `List[str]` | Reasons for quality pass/fail |

### Design & Docking Fields

| Field | Type | Description |
|---|---|---|
| `designed_sequences` | `List[str]` | Designed RNA sequences (deduplicated, cleaned) |
| `structural_candidates` | `List[dict]` | Structural design candidates |
| `binding_results` | `List[dict]` | Docking results (deduplicated by target+seq+score) |
| `md_results` | `List[dict]` | MD simulation results (deduplicated by seq+structure) |
| `interface_contacts` | `dict \| None` | Interface contact analysis results |
| `interface_contact_files` | `List[str]` | Paths to interface contact files |
| `seq_len` | `int` | RNA sequence length |
| `binding_units` | `str` | Binding score units (e.g., `hdock_relative_score`) |

### Target Protein Fields

| Field | Type | Description |
|---|---|---|
| `target_pdb` | `str \| None` | Current target PDB ID or path |
| `target_pdb_id` | `str \| None` | PDB ID only (e.g., "6M71") |
| `target_pdb_path` | `str \| None` | Full path to PDB file |
| `target_pdb_candidates` | `List[str]` | Candidate PDBs considered |
| `target_pdb_rankings` | `List[dict]` | PDB rankings with scores |
| `target_status` | `str \| None` | Status: `accepted_target`, `partial_success_target`, `failed_target` |
| `target_status_reason` | `str \| None` | Reason for target status |
| `failed_target_pdbs` | `List` | PDBs that failed docking |
| `partial_success_targets` | `List` | Targets with partial success |
| `resolved_partial_success_targets` | `List[str]` | Resolved partial-success target IDs |
| `target_failure_records` | `List[dict]` | Detailed failure records |
| `target_pdb_selection_reason` | `str` | Reason for target selection |
| `target_pdb_metadata` | `dict` | Additional target metadata |
| `target_selection_mode` | `str` | Selection mode: `llm_fallback`, etc. |
| `target_sequence` | `str \| None` | Target protein sequence |
| `docking_preferences` | `List[dict]` | User-defined docking preferences |

### Literature Fields

| Field | Type | Description |
|---|---|---|
| `research_query` | `str \| None` | Primary research query |
| `literature_query_bundle` | `List[str]` | Expanded query variants (deduplicated) |
| `literature_topic_profile` | `dict` | Topic profile from literature |
| `streamed_queries` | `List[str]` | Streaming literature queries (deduplicated) |
| `streamed_query_keys` | `List[str]` | Query keys for deduplication |
| `streaming_started` | `bool` | Whether streaming has begun |
| `literature_motif_hints` | `List[str]` | Motif hints from literature (deduplicated) |
| `literature_target_hints` | `List[str]` | Target hints from literature (deduplicated) |
| `literature_policy_text` | `str` | Policy text derived from literature |

### Inhibitor Screening Fields

| Field | Type | Description |
|---|---|---|
| `inhibitor_enabled` | `bool` | Whether inhibitor screening is active |
| `inhibitor_small_molecules` | `List[dict]` | Small-molecule inhibitor results (deduplicated by identity) |
| `inhibitor_peptides` | `List[dict]` | Peptide inhibitor results (deduplicated by target+seq) |
| `inhibitor_best_small_molecule` | `dict \| None` | Best small-molecule inhibitor |
| `inhibitor_best_peptide` | `dict \| None` | Best peptide inhibitor |
| `inhibitor_binding_site_overlap` | `float` | Legacy overlap percentage |
| `inhibitor_binding_site_overlap_score` | `float` | Structured overlap score |
| `inhibitor_pose_comparison` | `dict` | Pose comparison metrics |
| `inhibitor_small_molecule_comparison` | `dict \| None` | Small-molecule comparison |
| `inhibitor_peptide_comparison` | `dict \| None` | Peptide comparison |
| `inhibitor_docking_box` | `dict` | Docking box parameters |
| `inhibitor_analysis` | `str` | Inhibitor analysis text |
| `inhibitor_summary` | `str` | Inhibitor summary |
| `inhibitor_snapshot_paths` | `List[str]` | PyMOL snapshot paths (deduplicated) |
| `inhibitor_vina_seed` | `int \| None` | Vina reproducibility seed |
| `inhibitor_vina_cache_hits` | `int` | Vina cache hit count |
| `inhibitor_vina_cache_misses` | `int` | Vina cache miss count |
| `inhibitor_vina_cache_enabled` | `bool` | Vina cache status |

### Docking Export Fields

| Field | Type | Description |
|---|---|---|
| `docking_summary_json` | `str` | Path to JSON summary |
| `docking_summary_csv` | `str` | Path to CSV summary |
| `docking_summary_md` | `str` | Path to Markdown summary |

### Logs & Conversation Fields

| Field | Type | Description |
|---|---|---|
| `results_log` | `List[dict]` | Per-iteration results (deduplicated by repr) |
| `stage_outputs` | `List[dict]` | Per-agent outputs (deduplicated by repr) |
| `conversation_history` | `List[dict]` | Agent conversation entries (deduplicated by repr) |

### Report & Memory Fields

| Field | Type | Description |
|---|---|---|
| `final_report` | `str` | Final research report |
| `previous_hypotheses` | `List[str]` | Historical hypotheses |
| `failure_memory` | `List[dict]` | Historical failure data |

### Runtime Fields (Non-Serialisable)

| Field | Type | Description |
|---|---|---|
| `wrappers` | `dict` | Computational tool bundle (MD, protein, ViennaRNA, etc.) |
| `data_collector` | `TrainingDataCollector` | Training data collection hook |

### Internal Optimiser/Runtime Metadata

| Field | Type | Description |
|---|---|---|
| `_run_system_selected_motifs` | `List[dict]` | Motifs selected by NSGA-II system |
| `_run_system_min_fold_thresholds` | `dict` | Minimum fold thresholds used |
| `_run_system_final_weights` | `dict` | Final optimisation weights |
| `_run_system_interface_objective_enabled` | `bool` | Interface objective status |
| `_run_system_interface_lookup_count` | `int` | Interface lookup count |
| `_run_system_interface_failure_weights` | `dict` | Interface failure weights |
| `_run_system_seq_len` | `int` | Sequence length used in optimisation |
| `_run_system_conservation_valid` | `bool` | Conservation signal validity |
| `_run_system_sequence_scores` | `dict` | Per-sequence scores from NSGA-II |
| `_run_system_best_interface_sequence` | `str \| None` | Best interface sequence from NSGA-II |
| `_pi_excluded_interface_clashes` | `List[str]` | Sequences excluded due to clashes |
| `_md_cache` | `dict` | MD simulation cache |
| `_dock_cache` | `dict` | Docking result cache |

## Reducer Functions

Reducer functions automatically merge new state updates with existing values, preventing duplicate accumulation and enforcing limits.

### Evidence Reducers

#### `dedupe_evidence_reducer(existing, new)`
- **Purpose**: Merge literature evidence while deduplicating by DOI/PMID/title
- **Strategy**: Extracts stable keys from dict records (DOI > PMID > title+abstract)
- **Limit**: Keeps latest 50 unique items
- **Input format**: Supports both legacy strings (`"Title: ... | Abstract: ..."`) and v2 dicts (`{"title": ..., "doi": ..., "abstract": ...}`)

#### `dedupe_rna_sequence_reducer(existing, new)`
- **Purpose**: Merge RNA sequence lists while deduplicating exact cleaned strings
- **Strategy**: Normalises sequences (uppercase, T→U, strips non-ACGU chars)
- **Order**: Preserves insertion order

#### `dedupe_string_list_reducer(existing, new)`
- **Purpose**: Merge generic string lists while deduplicating
- **Strategy**: Strips whitespace, exact string match
- **Order**: Preserves insertion order

#### `append_unique_dicts_reducer(existing, new)`
- **Purpose**: Merge list-of-dicts while avoiding exact duplicates
- **Strategy**: Uses `repr(item)` for comparison
- **Use case**: `results_log`, `stage_outputs`, `conversation_history`

### Docking Result Reducers

#### `dedupe_binding_results_reducer(existing, new)`
- **Purpose**: Merge docking results while avoiding exact duplicate records
- **Key**: `(target, sequence, dock_output_file, complex_file, score)`
- **Limit**: Keeps latest 50 unique items

#### `dedupe_md_results_reducer(existing, new)`
- **Purpose**: Merge MD results while avoiding exact duplicates
- **Key**: `(sequence, rna_pdb_path, min_energy)`
- **Limit**: Keeps latest 50 unique items

### Inhibitor Reducers

#### `dedupe_inhibitor_small_molecules_reducer(existing, new)`
- **Purpose**: Merge small-molecule inhibitor rows by compound identity
- **Key**: `(target, (smiles OR cid OR name))`
- **Strategy**: Keeps latest row for each identity (updates scores/paths on reruns)
- **Limit**: Keeps latest 100 unique compounds

#### `dedupe_inhibitor_peptides_reducer(existing, new)`
- **Purpose**: Merge peptide inhibitor rows by target+sequence
- **Key**: `(target, sequence)`
- **Strategy**: Keeps latest score/complex path for each peptide

## Helper Utilities

### `timestamp()`
Returns current UTC timestamp in ISO format (`"2026-07-01T12:00:00Z"`).

### `record_stage_output(state, agent_name, output, summary, metadata)`
Creates a standard stage output record with:
- `agent`, `stage` — Agent name (for backwards compatibility)
- `summary` — First 300 chars of output
- `output`, `content` — Full output text
- `metadata` — Additional context
- `timestamp` — UTC timestamp

### `add_conversation_entry(state, role, content, agent)`
Creates a conversation-history entry with:
- `role` — `"system"`, `"user"`, or `"assistant"`
- `agent` — Agent name (optional)
- `content` — Message content
- `timestamp` — UTC timestamp

### `safe_jsonable(obj)`
Converts state into JSON-safe payload:
- Skips runtime-only keys: `wrappers`, `data_collector`, `llm`, `language_model`
- Handles `Path` objects → converts to string
- Handles NumPy scalars/arrays → converts via `.item()` or `.tolist()`
- Handles Pydantic v1/v2 → converts via `.dict()` or `.model_dump()`
- Handles tuples/sets → converts to lists
- Last resort: `str(obj)` or `"<non_jsonable:ClassName>"`

## State Flow Example

```python
# Initial state (from state_factory.py)
state = {
    "topic_name": "SARS-CoV-2 RNA inhibition",
    "hypothesis": "Design RNA aptamers that bind SARS-CoV-2 spike protein",
    "iterations": 0,
    "designed_sequences": [],
    "evidence": [],
    ...
}

# After PI agent (iteration 0)
state["hypothesis"] = "RNA aptamers targeting the receptor-binding domain of SARS-CoV-2 spike..."
state["designed_sequences"] = ["AAGCUUCCGA", "GGCUAAGCUU", ...]
state["optimisation_status"] = "adaptive_pareto_optimised"

# After Researcher agent
state["evidence"] = [
    {"title": "...", "doi": "...", "abstract": "..."},
    {"title": "...", "pmid": "...", "abstract": "..."},
]

# After Skeptic agent (iteration 0)
state["critique"] = "RECOMMENDATION: REVISE_HYPOTHESIS\n..."
state["iterations"] = 1

# Graph routes to "pi" again (continues loop)
# ...
```

## Best Practices

1. **Always use reducers** — Never directly assign lists/dicts; use the reducer functions to merge
2. **Clean sequences** — Use `_clean_rna()` before adding RNA sequences
3. **Validate before docking** — Check `fold_thresholds_passed` before adding binding results
4. **Handle None gracefully** — All fields are optional (`total=False`); always use `.get()` with defaults
5. **Log state changes** — Use `record_stage_output()` for agent outputs
6. **Strip runtime objects** — Rely on `safe_jsonable()` for JSON serialization

---
*Last updated: 2026-07-01*