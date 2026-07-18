# Orchestrator / Workflow Documentation

## Overview

The **Orchestrator** is the central control plane of VLAB2. It builds a LangGraph state machine that coordinates specialised AI agents through a closed-loop research pipeline. The orchestrator manages:

1. **Workflow compilation** — Constructs the directed graph of agent nodes and routing edges
2. **State management** — Maintains `LabState` across all agent invocations
3. **Conditional routing** — Decides whether to continue the loop, skip nodes, or terminate
4. **Convergence detection** — Evaluates whether the system has reached scientific consensus

## Architecture

### LangGraph State Machine

```
┌─────────────────────────────────────────────────────────────┐
│                  LangGraph State Machine                    │
│                                                             │
│   PI → Researcher → Bioinfo → Structural → MD → Protein    │
│                                              ↓             │
│                              Inhibitor (conditional)        │
│                                              ↓             │
│                                          Skeptic            │
│                                              ↓             │
│                                    (continue → PI)          │
│                                    (or END)                 │
└─────────────────────────────────────────────────────────────┘
```

### Node Definitions

| Node | Agent | Description |
|---|---|---|
| `pi` | PI Agent | Hypothesis generation, NSGA-II optimisation, target management |
| `researcher` | Researcher Agent | Literature retrieval, evidence gathering |
| `bioinfo` | Bioinformatics Agent | MSA, conservation analysis, genome retrieval |
| `structural` | Structural Agent | RNA sequence design, folding prediction |
| `md` | MD Agent | SimRNA molecular dynamics simulations |
| `protein` | Protein Agent | RNA-protein docking via HDOCK |
| `inhibitor` | Inhibitor Agent | Small-molecule/peptide inhibitor screening |
| `skeptic` | Skeptic Agent | Peer review, convergence evaluation |

### Edge Definitions

| From | To | Type | Condition |
|---|---|---|---|
| `pi` | `researcher` | Direct | Always |
| `researcher` | `bioinfo` | Direct | Always |
| `bioinfo` | `structural` | Direct | Always |
| `structural` | `md` | Direct | Always |
| `md` | `protein` | Direct | Always |
| `protein` | `inhibitor` | Conditional | `VLAB_INHIBITOR_ENABLED=1` AND target PDB exists AND interface contacts available |
| `protein` | `skeptic` | Conditional | Inhibitor disabled or no target/interface |
| `inhibitor` | `skeptic` | Direct | Always |
| `skeptic` | `pi` | Conditional | Convergence not reached (`"continue"`) |
| `skeptic` | `END` | Conditional | Convergence reached (`"end"`) |

## Entry Point

The orchestrator is invoked via the CLI:

```bash
python -m orchestration.cli \
    --topic-file research_topics.yaml \
    --topic-index 0 \
    --max-iterations 3
```

Or from the run script:

```bash
"${PY}" orchestration/virtual_lab_orchestrator.py \
    --topic-index "${TOPIC_INDEX}" \
    --max-iterations "${MAX_ITERATIONS}"
```

## Workflow Builder

The `build_virtual_lab()` function in `orchestration/orchestrator.py` constructs the compiled graph:

```python
from langgraph.graph import StateGraph
from VLAB2.orchestration.state_schema import LabState

def build_virtual_lab():
    workflow = StateGraph(LabState)

    # Add all agent nodes
    workflow.add_node("pi", pi_agent)
    workflow.add_node("researcher", researcher_agent)
    # ... (all other agents)

    # Set entry point
    workflow.set_entry_point("pi")

    # Add direct edges
    workflow.add_edge("pi", "researcher")
    workflow.add_edge("researcher", "bioinfo")
    # ... (all direct edges)

    # Add conditional edges
    workflow.add_conditional_edges(
        "protein",
        _inhibitor_should_run,
        {"inhibitor": "inhibitor", "skeptic": "skeptic"},
    )

    workflow.add_conditional_edges(
        "skeptic",
        should_continue,
        {"continue": "pi", "end": END},
    )

    return workflow.compile()
```

The compiled graph is stored as a module-level singleton `virtual_lab` for reuse across invocations.

## State Factory

The `build_initial_state()` function in `orchestration/state_factory.py` creates the initial `LabState` dictionary:

```python
initial_state = build_initial_state(
    topic=topic,                    # Parsed from research_topics.yaml
    wrappers=wrappers,              # MD, protein, ViennaRNA, etc.
    max_iterations=args.max_iterations,
    data_collector=TrainingDataCollector(),
)
```

Key initial state fields:
- `topic_name`, `research_topic`, `topic_description` — Research context
- `hypothesis` — Initial hypothesis (defaults to topic description)
- `iterations` — Starts at 0
- `max_iterations` — User-specified or default (3)
- `designed_sequences` — Empty list (populated by Structural/PI agents)
- `evidence` — Empty list (populated by Researcher agent)
- `wrappers` — Computational tool bundle
- `failure_memory` — Empty list (populated over time)

## Routing Logic

### Inhibitor Routing (`_inhibitor_should_run`)

Determines whether to run inhibitor screening after protein docking:

**Routes to `inhibitor` when ALL conditions are met:**
1. `VLAB_INHIBITOR_ENABLED=1`
2. `target_pdb` exists OR `partial_success_targets` has valid entry
3. `interface_contacts["interface_residues"]` is non-empty

**Routes to `skeptic` when ANY condition fails:**
- Inhibitor screening disabled
- No target PDB available
- No interface contacts available

### Convergence Routing (`should_continue`)

Determines whether to continue the research loop or terminate after Skeptic review:

**Routes to `end` (terminate) when ALL conditions are met:**
1. `len(validated_docking_results) >= VLAB_MIN_VALID_DOCKINGS_FOR_CONVERGENCE` (default: 2)
2. Docking score spread < `VLAB_CONVERGENCE_SPREAD_THRESHOLD` (default: 5.0)
3. Best binding score < `VLAB_ACCEPT_BINDING_SCORE` (default: -50.0)
4. Fold thresholds passed (`fold_thresholds_passed is not False`)
5. Conservation fitness >= `VLAB_MIN_CONSERVATION_FOR_ACCEPT` if `VLAB_REQUIRE_CONSERVATION_FOR_CONVERGENCE=1`
6. Skeptic does NOT recommend revision (unless `VLAB_ALLOW_CONVERGENCE_ON_REVISE=1`)

**Routes to `continue` (loop back to PI) when:**
- Max iterations reached
- Fold thresholds failed (forces redesign)
- Insufficient validated docking results
- Convergence thresholds not met
- Skeptic recommends revision (unless override enabled)

### Validated Docking Results

The `_validated_docking_results()` function filters `binding_results` with strict criteria:

**Required fields:**
- `valid` = True
- `dock_valid` = True (or `vina_valid`)
- `binding_mode` != `"rejected_docking"`
- `dock_score` < `VLAB_MIN_VALID_HDOCK_SCORE` (default: -30.0)
- `target_pdb` matches current target (if `VLAB_REQUIRE_CURRENT_TARGET_FOR_CONVERGENCE=1`)

**Interface requirements (if `VLAB_REQUIRE_INTERFACE_FOR_CONVERGENCE=1`):**
- `interface_contacts_valid` is not False
- `interface_passed` = True
- `interface_steric_clash` is not True

## Execution Flow

### 1. Bootstrap

```python
# Load topics
topics = load_topics("research_topics.yaml")
topic = select_topic(topics, index=0)

# Bootstrap knowledge base (literature/failure memory)
bootstrap_knowledge_base(topic)

# Build computational wrappers
wrappers = build_wrapper_bundle()

# Build data collector for training pipeline
data_collector = TrainingDataCollector()

# Create initial state
initial_state = build_initial_state(
    topic=topic,
    wrappers=wrappers,
    max_iterations=3,
    data_collector=data_collector,
)
```

### 2. Graph Invocation

```python
final_result = virtual_lab.invoke(initial_state)
```

This triggers the full agent pipeline:
1. PI generates hypothesis and runs NSGA-II optimisation
2. Researcher gathers literature evidence
3. Bioinformatics performs MSA and conservation analysis
4. Structural designs RNA sequences
5. MD validates folding stability
6. Protein docks RNA against target
7. (Optional) Inhibitor screens for competing ligands
8. Skeptic reviews and decides: continue or end

### 3. Result Processing

```python
# Log diagnostics
_log_final_result_summary(final_result)

# Convert to JSON-safe payload
payload = _coerce_final_result_for_json(final_result)

# Print terminal summary
print_user_friendly_summary(final_result)

# Save atomic JSON
output_file = _topic_output_name(topic["name"])
_write_json_atomic(payload, output_file)

# Run post-run training pipeline
run_postrun_pipeline(topic, final_result)
```

## Output Artifacts

| Artifact | Location | Description |
|---|---|---|
| `lab_results_{topic_name}.json` | Project root | Full structured results per topic |
| `summary_{timestamp}.md` | Project root | Markdown summary report |
| `training_data/*.jsonl` | `training_data/` | Instruction-tuning datasets |
| `docking_snapshots/*.png` | `output_data/docking_snapshots/` | PyMOL-rendered docking images |
| `docking_contacts/*.json` | `output_data/docking_contacts/` | Interface contact analysis |

## Error Handling

The orchestrator implements multiple layers of fault tolerance:

1. **Agent-level** — Each agent catches exceptions and returns fallback state
2. **Graph-level** — LangGraph handles node failures gracefully
3. **Serialization-level** — `safe_jsonable()` and `_fallback_jsonable()` prevent runtime objects from corrupting JSON output
4. **Atomic writes** — `_write_json_atomic()` uses temp file + rename to prevent corrupt result files
5. **Refusal to save null** — Explicit checks prevent saving empty/null result payloads

## Integration with SLURM

The run script (`run_virtual_lab_conda.sh`) handles:
- Module loading (CUDA, Miniforge)
- Environment setup (cache paths, GPU allocation)
- vLLM server startup and readiness check
- Orchestrator invocation with topic index and max iterations
- Cleanup trap for vLLM process termination

```bash
sbatch --export=ALL,TOPIC_INDEX=0,MAX_ITERATIONS=3 run_virtual_lab_conda.sh
```

---
*Last updated: 2026-07-01*