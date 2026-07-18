# VLAB2 — Virtual Lab Orchestrator

## Overview

**VLAB2** is an autonomous multi-agent biophysics research platform that orchestrates specialised AI agents to conduct end-to-end structural biology research. The system performs closed-loop hypothesis generation, molecular simulation, evolutionary optimisation, and peer review — all driven by a local LLM and backed by real scientific software.

The platform is designed for GPU-accelerated HPC infrastructure (SLURM), serving a local [vLLM](https://github.com/vllm-project/vllm) instance (`Qwen/Qwen2.5-32B-Instruct`) as the reasoning backbone.

## What It Does

1. **Hypothesises** — The PI agent proposes research hypotheses based on a topic.
2. **Gathers Evidence** — The Researcher agent retrieves relevant literature; the Bioinformatics agent retrieves genomes and performs multiple sequence alignment.
3. **Designs** — The Structural agent designs RNA sequences and evaluates folding stability.
4. **Validates** — The MD agent performs coarse-grained molecular dynamics simulations.
5. **Evaluates Binding** — The Protein agent docks RNA candidates against target proteins.
6. **Screens Inhibitors** — The Inhibitor agent searches for small-molecule and peptide inhibitors that compete with RNA binding.
7. **Critiques** — The Skeptic agent performs critical peer review, assessing convergence and validity.
8. **Refines** — If the Skeptic deems the hypothesis worth pursuing, the loop repeats from the PI agent.

## Architecture

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
        │
   ┌────▼────┐
   │  vLLM   │  ← Qwen/Qwen2.5-32B-Instruct
   └─────────┘
```

### Key Components

| Component | Location | Purpose |
|---|---|---|
| **Orchestrator** | `orchestration/orchestrator.py` | Builds and compiles the LangGraph workflow |
| **CLI Entry Point** | `orchestration/cli.py` | Main script invoked by the run script |
| **State Schema** | `orchestration/state_schema.py` | Canonical `LabState` TypedDict with reducers |
| **Agents** | `orchestration/agents/` | Individual agent implementations |
| **Scientific Wrappers** | `core/` | Interfaces to ViennaRNA, SimRNA, HDOCK, etc. |
| **Optimisation** | `optimisation/` | NSGA-II, surrogate models, adaptive mutation |
| **Training** | `training/` | LoRA fine-tuning pipeline |
| **Research** | `research/` | Literature agents, self-driving lab logic |
| **Run Script** | `run_virtual_lab_conda.sh` | SLURM job submission with full environment setup |

## Running the System

### SLURM (HPC)

```bash
# Run with default topic (index 0) and 3 iterations
sbatch --export=ALL,TOPIC_INDEX=0,MAX_ITERATIONS=3 run_virtual_lab_conda.sh

# Run a specific topic
sbatch --export=ALL,TOPIC_INDEX=5,MAX_ITERATIONS=5 run_virtual_lab_conda.sh
```

### Local

```bash
# Ensure vLLM is running on port 8000 first
python -m vllm.entrypoints.openai.api_server \
    --model <path-to-model> --port 8000

# Then run the orchestrator
python orchestration/virtual_lab_orchestrator.py \
    --topic-index 0 --max-iterations 3
```

## Outputs

| Path | Description |
|---|---|
| `lab_results_*.json` | Full structured results per topic |
| `logs/vllm_*.log` | vLLM server log |
| `output_data/hdock_work/` | HDOCK working directory |
| `output_data/docking_snapshots/` | PyMOL-rendered docking images |
| `output_data/docking_contacts/` | Interface contact analysis |
| `output_data/coverage/` | Coverage data and HTML reports |
| `training_data/` | Instruction-tuning JSONL datasets |

## Configuration

- **Topics**: Defined in `research_topics.yaml`
- **Environment**: `.env` file for API keys and paths (see `.env.example`)
- **Convergence controls**: Environment variables like `VLAB_ALLOW_CONVERGENCE_ON_REVISE`, `VLAB_REQUIRE_CONSERVATION_FOR_CONVERGENCE`
- **Inhibitor screening**: Controlled by `VLAB_INHIBITOR_ENABLED`, `VLAB_INHIBITOR_MAX_SMALL_MOLECULES`, etc.

## Documentation

Detailed documentation is in the `docs/` directory:

| Document | Description |
|---|---|
| [`docs/README.md`](../README.md) | This file — project overview |
| [`docs/run_script.md`](run_script.md) | SLURM run script breakdown |
| [`docs/agents/`](agents/) | Per-agent documentation |
| [`docs/orchestrator.md`](orchestrator.md) | Workflow and state machine |
| [`docs/state_schema.md`](state_schema.md) | LabState schema and reducers |
| [`docs/training_pipeline.md`](training_pipeline.md) | LoRA fine-tuning guide |
| [`docs/optimisation.md`](optimisation.md) | NSGA-II and surrogate models |
| [`docs/environment_config.md`](environment_config.md) | Environment setup and variables |
| [`docs/CHECKLIST.md`](CHECKLIST.md) | Documentation progress tracker |

## Project Structure

```
VLAB2/
├── orchestration/          # LangGraph workflow, agents, CLI
│   ├── agents/             # Individual agent implementations
│   ├── orchestrator.py     # Workflow builder
│   ├── state_schema.py     # LabState definition
│   ├── cli.py              # Entry point
│   └── ...                 # Config, routing, reporting, etc.
├── core/                   # Scientific software wrappers
│   ├── viennarna_wrapper.py
│   ├── md_wrapper.py
│   ├── protein_wrapper.py
│   ├── hdock_wrapper.py
│   ├── vina_wrapper.py
│   └── ...                 # PDB prep, RNA prep, etc.
├── optimisation/           # NSGA-II, surrogate models, adaptive mutation
├── training/               # LoRA fine-tuning pipeline
├── research/               # Literature agents, self-driving lab
├── wrappers/               # Additional wrapper implementations
├── docs/                   # This documentation
├── run_virtual_lab_conda.sh # SLURM job submission script
├── research_topics.yaml    # Research topic definitions
└── .env.example            # Environment variable template
```

## Key Design Principles

1. **Evidence-backed**: Every claim is grounded in real tool output, not LLM hallucination.
2. **Closed-loop**: The Skeptic agent controls iteration — the system refines until convergence or max iterations.
3. **Active learning**: NSGA-II optimisation uses a neural surrogate model for fast evaluation, with real MD reserved for top/uncertain candidates.
4. **Self-improving**: Every research cycle produces instruction-tuning data for future LoRA fine-tuning.
5. **Fault-tolerant**: Graceful degradation when tools are unavailable, with failure memory and retry logic.

---
*Last updated: 2026-07-01*
---
```