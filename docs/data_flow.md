# Data Flow & Integration Documentation

## Overview

The **Data Flow & Integration** layer describes how data moves through VLAB2's multi-agent pipeline, how components interact, and how information is transformed between stages. This document covers the complete lifecycle of research data from initial hypothesis to final output.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Research Topic                            │
│              (research_topics.yaml)                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Initial State                              │
│              (state_factory.py)                              │
│                                                              │
│  topic_name, hypothesis, wrappers, max_iterations,          │
│  evidence=[], designed_sequences=[]                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              LangGraph State Machine                         │
│                   (orchestrator.py)                          │
│                                                              │
│  PI → Researcher → Bioinfo → Structural → MD → Protein     │
│                                       ↓                      │
│                            Inhibitor (conditional)           │
│                                       ↓                      │
│                                    Skeptic                   │
│                                       ↓                      │
│                              (continue → PI)                 │
│                              (or END)                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Post-Run Pipeline                           │
│                   (postrun.py)                               │
│                                                              │
│  1. Build training data                                     │
│  2. Validate dataset                                        │
│  3. Trigger LoRA training (if enabled)                      │
│  4. Update model (if enabled)                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Output Artifacts                            │
│                                                              │
│  lab_results_*.json, training_data/*.jsonl,                 │
│  docking_snapshots/*.png, summary_*.md                      │
└─────────────────────────────────────────────────────────────┘
```

## State Flow

### LabState as Central Data Hub

The `LabState` TypedDict (`orchestration/state_schema.py`) is the single source of truth that flows through all agents. Each agent reads from and writes to this shared state.

```python
# Initial state structure
initial_state = {
    # Topic & setup
    "topic_name": "SARS-CoV-2 RNA Inhibition",
    "research_topic": "...",
    "topic_description": "...",
    "seed_questions": [...],
    
    # Iteration control
    "iterations": 0,
    "max_iterations": 3,
    
    # PI/optimisation
    "hypothesis": "...",
    "designed_sequences": [],
    "optimisation_status": "",
    
    # Agent outputs (initially empty)
    "evidence": [],
    "structural_analysis": "",
    "md_analysis": "",
    "protein_analysis": "",
    "bioinfo_analysis": "",
    "critique": "",
    
    # Target protein
    "target_pdb": None,
    "target_status": None,
    
    # Results
    "binding_results": [],
    "md_results": [],
    "interface_contacts": None,
    
    # Runtime (non-serialisable)
    "wrappers": {...},
    "data_collector": TrainingDataCollector(),
}
```

### State Reducers

Reducers automatically merge new state updates with existing values, preventing duplicates:

```python
# Evidence deduplication
state["evidence"] = dedupe_evidence_reducer(
    state.get("evidence", []),
    new_evidence
)

# Sequence deduplication
state["designed_sequences"] = dedupe_rna_sequence_reducer(
    state.get("designed_sequences", []),
    new_sequences
)

# Binding results deduplication
state["binding_results"] = dedupe_binding_results_reducer(
    state.get("binding_results", []),
    new_binding_results
)
```

## Agent Data Flow

### 1. PI Agent

**Input**:
- Previous iteration's `critique` (from Skeptic)
- `designed_sequences` (from Structural or previous PI iteration)
- `binding_results`, `md_results` (from previous iterations)
- `conservation_signal` (from Bioinformatics)
- `failure_memory`, `literature_motif_hints` (from memory systems)

**Processing**:
1. Parses critique for docking score spread
2. Resets target if needed
3. Builds combined multi-objective weights
4. Runs NSGA-II optimisation
5. Selects Pareto-optimal sequences

**Output**:
- Updated `hypothesis` (refined from critique)
- New `designed_sequences` (Pareto-optimal)
- `mutation_bias` (adaptive weights)
- `optimisation_status`
- `joint_physics_feedback`
- `pi_action_summary`, `pi_training_metadata`

**Data transformations**:
```
critique (text) → parsed objectives (dict)
designed_sequences (list[str]) → optimised sequences (list[str])
binding_results + md_results → joint_physics_feedback (dict)
```

### 2. Researcher Agent

**Input**:
- `topic_description`, `research_topic`
- `seed_questions`

**Processing**:
1. Generates research queries
2. Searches literature databases
3. Retrieves relevant papers
4. Extracts key findings

**Output**:
- `evidence` (list of paper dicts with title, doi, abstract)
- `literature_query_bundle` (expanded queries)
- `literature_motif_hints` (motifs from literature)
- `literature_target_hints` (target hints from literature)
- `literature_policy_text` (policy derived from literature)

**Data transformations**:
```
topic (text) → research queries (list[str])
literature search results → evidence (list[dict])
evidence → motif/target hints (list[str])
```

### 3. Bioinformatics Agent

**Input**:
- `virus_name`, `virus_family`
- `research_topic`

**Processing**:
1. Retrieves viral sequences from NCBI
2. Performs multiple sequence alignment (MAFFT)
3. Calculates conservation scores
4. Identifies conserved regions

**Output**:
- `msa_data` (aligned sequences)
- `conservation_signal` (dict with fitness scores)
- `conserved_regions` (list of position ranges)
- `conservation_fitness` (float 0-1)
- `bioinfo_num_sequences`, `bioinfo_alignment_length`
- `bioinfo_analysis` (text summary)

**Data transformations**:
```
virus_name → retrieved sequences (list[str])
sequences → MSA (str)
MSA → conservation scores (dict)
conservation scores → fitness (float)
```

### 4. Structural Agent

**Input**:
- `designed_sequences` (from PI or initial)
- `conserved_regions` (from Bioinformatics)
- `wrappers` (ViennaRNA, SFold)

**Processing**:
1. Filters sequences by conservation
2. Predicts secondary structure (ViennaRNA)
3. Evaluates folding stability
4. Applies fold quality thresholds

**Output**:
- Filtered `designed_sequences`
- `structural_candidates` (list of dicts with structure info)
- `fold_thresholds_passed` (bool)
- `structural_analysis` (text summary)

**Data transformations**:
```
sequences (list[str]) → filtered sequences (list[str])
sequences → structure predictions (dict)
structure predictions → fold quality (bool)
```

### 5. MD Agent

**Input**:
- `designed_sequences` (from Structural)
- `wrappers` (MDWrapper)

**Processing**:
1. Generates RNA structures (SimRNA)
2. Runs coarse-grained MD simulations
3. Calculates energy fluctuation
4. Assesses stability

**Output**:
- `md_results` (list of dicts with stability metrics)
- `md_analysis` (text summary)
- Updated `designed_sequences` (filtered by stability)

**Data transformations**:
```
sequences (list[str]) → MD results (list[dict])
MD results → stability scores (float)
stability scores → filtered sequences (list[str])
```

### 6. Protein Agent

**Input**:
- `designed_sequences` (from MD)
- `target_pdb` (from PI or literature)
- `wrappers` (ProteinWrapper, HDockDocking)

**Processing**:
1. Prepares target protein PDB
2. Selects target if not set
3. Runs RNA-protein docking (HDOCK)
4. Extracts binding scores
5. Analyzes interface contacts

**Output**:
- `binding_results` (list of docking dicts)
- `interface_contacts` (dict with contact info)
- `target_pdb`, `target_status`
- `protein_analysis` (text summary)

**Data transformations**:
```
sequences (list[str]) + target_pdb → docking results (list[dict])
docking results → interface contacts (dict)
docking results → binding scores (float)
```

### 7. Inhibitor Agent (Conditional)

**Input**:
- `interface_contacts` (from Protein)
- `target_pdb` (from Protein)
- `wrappers` (VinaWrapper)

**Processing**:
1. Checks if inhibitor screening enabled
2. Queries databases for small molecules/peptides
3. Docks inhibitors against binding site
4. Compares inhibitor binding to RNA binding

**Output**:
- `inhibitor_small_molecules` (list of dicts)
- `inhibitor_peptides` (list of dicts)
- `inhibitor_best_small_molecule`, `inhibitor_best_peptide`
- `inhibitor_analysis` (text summary)

**Data transformations**:
```
interface_contacts → docking box (dict)
databases → inhibitor candidates (list[dict])
inhibitors → docking results (list[dict])
RNA binding + inhibitor binding → comparison (dict)
```

### 8. Skeptic Agent

**Input**:
- All agent outputs (`evidence`, `md_analysis`, `protein_analysis`, etc.)
- `binding_results`, `md_results`
- `critique` (from previous iteration)

**Processing**:
1. Evaluates evidence quality
2. Checks convergence criteria
3. Assesses scientific validity
4. Generates critique

**Output**:
- `critique` (text with RECOMMENDATION line)
- `skeptic_interface_metrics` (dict)
- `skeptic_bioinfo_metrics` (dict)

**Data transformations**:
```
all results → quality assessment (dict)
quality assessment → critique (text)
critique → recommendation (ACCEPT/REVISE)
```

## Memory Systems

### FailureMemory (`orchestration/failure_memory.py`)

**Purpose**: Track historical failures to avoid repeating mistakes.

**Data Flow**:
```
Previous runs → failure records (list[dict])
failure records → failure weights (dict)
failure weights → objective penalties in PI agent
```

**Storage**: Saved in `LabState["failure_memory"]`

**Example record**:
```python
{
    "iteration": 2,
    "agent": "protein",
    "failure_type": "docking_failed",
    "target_pdb": "6M71",
    "sequence": "AAGCUUCCGA",
    "error": "No valid docking pose",
    "timestamp": "2026-07-01T12:00:00Z"
}
```

### LiteratureMemory (`orchestration/literature_memory.py`)

**Purpose**: Store and retrieve literature-derived constraints.

**Data Flow**:
```
Researcher agent → literature hints (list[str])
literature hints → stored in state
state → PI agent for motif guidance
```

**Storage**: Saved in `LabState["literature_motif_hints"]`, `["literature_target_hints"]`

## Training Data Flow

### Data Collection (`core/training_data_collector.py`)

**Purpose**: Capture training examples from research cycles.

**Collection Points**:
1. **After each agent execution**: `capture_agent_interaction()`
2. **After PI optimisation**: `capture_optimisation_step()`
3. **After each iteration**: `collect_cycle()`

**Data Format** (JSONL):
```json
{
  "instruction": "Act as PI Agent. Task: Optimise RNA sequences",
  "input": "Context: target=6M71, sequences=[...]",
  "output": "{\"selected_sequences\": [...], \"mutation_bias\": {...}}",
  "metadata": {
    "type": "agent_reasoning",
    "agent": "pi_agent",
    "session_id": "20260701_120000",
    "timestamp": "2026-07-01T12:00:00Z"
  }
}
```

### Post-Run Pipeline (`orchestration/postrun.py`)

**Execution**: After each Virtual Lab run completes

**Steps**:
1. **Bootstrap knowledge base**: Load literature and failure memory
2. **Build training data**: Convert results to instruction-tuning format
3. **Validate dataset**: Check minimum requirements
4. **Trigger LoRA training**: If `VLAB_LORA_TRAIN=1`
5. **Update model**: If `VLAB_AUTO_MODEL_REPLACE=1`

**Data Flow**:
```
LabState → TrainingDataCollector → training_data/*.jsonl
training_data/*.jsonl → HF dataset → LoRA training → output_model/
```

## Output Artifacts

### Primary Outputs

| Artifact | Location | Format | Description |
|---|---|---|---|
| `lab_results_{topic}.json` | Project root | JSON | Full structured results |
| `summary_{timestamp}.md` | Project root | Markdown | Human-readable summary |
| `training_data/*.jsonl` | `training_data/` | JSONL | Instruction-tuning data |

### Secondary Outputs

| Artifact | Location | Format | Description |
|---|---|---|---|
| `docking_snapshots/*.png` | `output_data/docking_snapshots/` | PNG | PyMOL renders |
| `docking_contacts/*.json` | `output_data/docking_contacts/` | JSON | Interface contacts |
| `coverage/*.html` | `output_data/coverage/` | HTML | Coverage reports |
| `output_model/` | `training/output_model/` | Binaries | LoRA adapter weights |

### Result JSON Structure

```json
{
  "topic_name": "SARS-CoV-2 RNA Inhibition",
  "iterations": 3,
  "hypothesis": "...",
  "designed_sequences": ["AAGCUUCCGA", ...],
  "evidence": [...],
  "binding_results": [...],
  "md_results": [...],
  "critique": "...",
  "optimisation_status": "adaptive_pareto_optimised",
  "final_report": "..."
}
```

## Error Handling & Fault Tolerance

### Agent-Level Fault Tolerance

Each agent implements graceful degradation:

```python
def agent_function(state: LabState) -> LabState:
    try:
        # Normal processing
        result = process(state)
        return update_state(state, result)
    except Exception as e:
        log.warning("Agent failed: %s", e)
        return {
            **state,
            "failure_memory": append_unique_dicts_reducer(
                state.get("failure_memory", []),
                [{"agent": "agent_name", "error": str(e)}]
            )
        }
```

### Graph-Level Fault Tolerance

LangGraph handles node failures:
- Failed nodes return fallback state
- Conditional edges handle missing data
- State reducers prevent corruption

### Serialization-Level Fault Tolerance

`safe_jsonable()` handles non-serialisable objects:
- Skips runtime objects (`wrappers`, `data_collector`)
- Converts `Path` → `str`
- Converts numpy → native types
- Last resort: `str(obj)` or `"<non_jsonable:ClassName>"`

### Atomic Writes

`_write_json_atomic()` prevents corrupt output files:
1. Write to temporary file
2. Rename to final filename
3. Atomic on POSIX systems

## Integration Points

### External Services

| Service | Integration | Purpose |
|---|---|---|
| **vLLM** | `VLLM_API_BASE` | LLM inference |
| **NCBI/GenBank** | BioinfoWrapper | Sequence retrieval |
| **HDOCKlite** | HDockDocking | Protein-RNA docking |
| **SimRNA** | MDWrapper, ProteinWrapper | RNA structure/MD |
| **ViennaRNA** | ViennaRNAWrapper | Secondary structure |
| **AutoDock Vina** | VinaWrapper | Small-molecule docking |
| **MAFFT** | BioinfoWrapper | Multiple sequence alignment |

### Internal Modules

| Module | Import From | Purpose |
|---|---|---|
| **Agents** | `orchestration/agents/` | AI agent implementations |
| **Wrappers** | `core/` | Scientific software interfaces |
| **Optimisation** | `optimisation/` | NSGA-II, surrogate models |
| **Training** | `training/` | LoRA fine-tuning pipeline |
| **Memory** | `orchestration/failure_memory.py`, `literature_memory.py` | Historical learning |

## Data Lifecycle

```
1. INITIALIZATION
   ├── Load research topics
   ├── Bootstrap knowledge base (literature, failure memory)
   └── Build initial LabState

2. RESEARCH LOOP (per iteration)
   ├── PI Agent: Generate hypothesis, optimise sequences
   ├── Researcher: Gather literature evidence
   ├── Bioinformatics: MSA, conservation analysis
   ├── Structural: Design RNA sequences, predict folding
   ├── MD Agent: Simulate stability, filter candidates
   ├── Protein Agent: Dock against target, extract scores
   ├── Inhibitor Agent: (Conditional) Screen competing ligands
   └── Skeptic Agent: Review, decide continue/end

3. CONVERGENCE CHECK
   ├── If "continue": Loop back to step 2
   └── If "end": Proceed to post-run pipeline

4. POST-RUN PIPELINE
   ├── Bootstrap knowledge base
   ├── Build training data from LabState
   ├── Validate dataset quality
   ├── Trigger LoRA training (if enabled)
   └── Update model (if enabled)

5. OUTPUT GENERATION
   ├── Save lab_results_{topic}.json
   ├── Generate summary_{timestamp}.md
   ├── Save training_data/*.jsonl
   └── Generate visualisations (PNG, HTML)
```

## Cross-Agent Communication Patterns

### Direct State Passing

Most agents communicate through shared state:

```python
# Agent A writes to state
state["designed_sequences"] = dedupe_rna_sequence_reducer(
    state.get("designed_sequences", []),
    new_sequences
)

# Agent B reads from state
sequences = state.get("designed_sequences", [])
```

### Conditional Routing

Agents route based on state conditions:

```python
# Protein agent → Inhibitor agent (if enabled)
def _inhibitor_should_run(state) -> str:
    if not env_bool("VLAB_INHIBITOR_ENABLED"):
        return "skeptic"
    if not state.get("interface_contacts", {}).get("interface_residues"):
        return "skeptic"
    return "inhibitor"

# Skeptic → PI (continue) or END
def should_continue(state) -> str:
    if reached_max_iterations(state):
        return "end"
    if convergence_criteria_met(state):
        return "end"
    return "continue"
```

### Feedback Loops

```
Skeptic Critique → PI Agent Hypothesis Refinement
PI Optimisation → Structural Design → MD Validation → Protein Docking
Docking Results → Skeptic Review → (loop back if needed)
```

## Data Quality Gates

### Fold Quality Gate

```python
# Structural agent enforces fold thresholds
if not fold_thresholds_passed:
    state["fold_thresholds_passed"] = False
    # Forces redesign in next iteration
```

### Docking Validity Gate

```python
# Protein agent filters valid docking results
if not (binding.get("valid") and docking.get("dock_valid")):
    # Result excluded from convergence criteria
    continue
```

### Interface Quality Gate

```python
# Convergence requires interface-clean poses
if require_interface:
    if (r.get("interface_steric_clash") or 
        not r.get("interface_passed")):
        # Excluded from validated results
        continue
```

### Conservation Gate

```python
# Optional: Require conservation signal for convergence
if require_conservation:
    if state.get("conservation_fitness", 0) < min_conservation:
        # Excluded from convergence
        continue
```

## Summary

This document describes the complete data flow through VLAB2's multi-agent research pipeline. Key takeaways:

- **LabState** is the central data hub, flowing through all agents
- **State reducers** prevent duplicate entries and merge updates cleanly
- **Memory systems** (FailureMemory, LiteratureMemory) enable cross-iteration learning
- **Training data** is collected throughout the pipeline for future model improvement
- **Fault tolerance** is built in at every level — agent, graph, serialization, and output
- **Data quality gates** ensure only valid, high-quality results contribute to convergence

---
*Last updated: 2026-07-15*