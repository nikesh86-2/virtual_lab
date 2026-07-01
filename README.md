<p align="center">
  <h1 align="center">🧬 BioAgent — Autonomous Multi-Agent Biophysics Research Platform</h1>
  <p align="center">
    <em>AI-driven hypothesis generation, structural simulation, and peer review for viral RNA packaging research</em>
  </p>
  <p align="center">
    <a href="#quickstart">Quickstart</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#agents">Agents</a> •
    <a href="#configuration">Configuration</a> •
    <a href="#documentation">Documentation</a> •
    <a href="#roadmap">Roadmap</a>
  </p>
</p>

---

## Overview

**BioAgent** is an autonomous research platform that orchestrates multiple specialised AI agents to conduct end-to-end biophysics research — from literature discovery through molecular simulation to critical peer review. Built on [LangGraph](https://github.com/langchain-ai/langgraph), it coordinates six domain-expert agents backed by production scientific software (ViennaRNA, SimRNA, Rosetta, MAFFT) to iteratively refine research hypotheses with real empirical evidence.

The system runs on GPU-accelerated HPC infrastructure, serving a local [vLLM](https://github.com/vllm-project/vllm) instance (Qwen/Qwen2.5-32B-Instruct) as the reasoning backbone. Each research cycle produces structured, auditable outputs and instruction-tuning datasets for downstream model fine-tuning.

### Key Capabilities

- **Closed-loop research**: Hypothesise → gather evidence → simulate → critique → refine
- **Software-backed evidence**: Every claim is grounded in real tool output (RNAfold, SimRNA, Rosetta, MAFFT), not LLM hallucination
- **Two operating modes**: Guided verification of known motifs, or open-ended gap discovery
- **Self-improving**: Every research cycle is captured as instruction-tuning data (JSONL) for future model specialisation
- **Fault-tolerant**: Automatic checkpointing, retry logic, and graceful degradation when tools are unavailable

---

## Quickstart

### Prerequisites

| Dependency | Version | Purpose |
|---|---|---|
| CUDA | 12.6.2 | GPU inference |
| Conda / Miniforge | any | Environment management |
| Rosetta | 2025+ | Protein energy scoring |
| SimRNA | any | Coarse-grained RNA MD |
| SFold | any | RNA structure sampling |
| HDOCKlite | v1.1 | Protein-RNA docking |
| OpenBabel | any | PDB format conversion |
| PyMOL | any | Docking snapshot rendering |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/bioagent.git
cd bioagent

# 2. Create the conda environment
conda env create -f environment.yml
conda activate biophysics-research-agent

# 3. Verify tool availability
RNAfold --version          # ViennaRNA
mafft --version            # MAFFT
which SimRNA               # SimRNA binary
rosetta score --help 2>&1 | head -1   # Rosetta (module loaded)
obabel -V                  # OpenBabel
which pymol                # PyMOL (optional, for docking snapshots)

# 4. Configure Environment Variables
cp .env.example .env
# Edit .env and add your API keys (see Configuration section)
```

### Run a Research Session

```bash
# Guided mode — verify HPeV1 packaging signal integration (topic index 5)
sbatch --export=ALL,TOPIC_INDEX=5,MAX_ITERATIONS=3 run_virtual_lab_conda.sh

# Discovery mode — autonomous gap identification (topic index 6)
sbatch --export=ALL,TOPIC_INDEX=6,MAX_ITERATIONS=5 run_virtual_lab_conda.sh

# Local execution (interactive, no SLURM)
python virtual_lab_orchestrator.py --topic-index 0 --max-iterations 3
```

### Outputs

| Path | Description |
|---|---|
| `lab_results_*.json` | Full structured results (hypothesis, evidence, critique per iteration) |
| `lab_checkpoint.json` | Resumable state snapshot |
| `training_data/*.jsonl` | Instruction-tuning pairs for fine-tuning |
| `logs/lab_<jobid>.log` | Execution log |
| `logs/vllm_<jobid>.log` | vLLM server log |
| `output_data/docking_snapshots/` | PyMOL docking snapshot images |
| `output_data/debug_hdock/` | HDOCK debugging output |
| `output_data/hdock_work/` | HDOCK working directory |
| `pdb_cache/` | Cached protein structures |
| `rna_hdock_cache/` | Cached RNA structures for docking |

---

<a id="architecture"></a>
## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      LangGraph State Machine                    │
│                                                                 │
│  ┌──────┐    ┌────────────┐   ┌─────────────┐   ┌───────────┐  │
│  │  PI  │───▶│ Researcher │──▶│             │   │  Protein  │  │
│  │Agent │    └────────────┘   │             │   │   Agent   │  │
│  │      │    ┌────────────┐   │   Skeptic   │   └─────┬─────┘  │
│  │      │───▶│  Bioinfo   │──▶│   (Peer     │◀────────┘        │
│  │      │    └─────┬──────┘   │   Review)   │                  │
│  │      │          │          │             │                  │
│  │      │    ┌─────▼──────┐   │             │                  │
│  │      │    │ Structural │──▶│             │                  │
│  │      │    └────────────┘   │             │                  │
│  │      │    ┌────────────┐   │             │                  │
│  │      │───▶│     MD     │──▶│             │                  │
│  └──┬───┘    └────────────┘   └──────┬──────┘                  │
│     ▲                                │                         │
│     └────────── continue ◀───────────┘                         │
│                   or END                                        │
└─────────────────────────────────────────────────────────────────┘
        │                                        │
   ┌────▼────┐                            ┌──────▼──────┐
   │  vLLM   │                            │  Scientific │
   │ Llama-3 │                            │  Software   │
   │  70B    │                            │  Stack      │
   └─────────┘                            └─────────────┘
```

### Data Flow

1. **PI Agent** reads the research topic and seed questions, proposes a hypothesis
2. **Fan-out**: Researcher, Bioinfo, MD, and Protein agents execute in parallel
3. **Bioinfo → Structural**: MSA data feeds into RNAalifold for conservation-aware structure prediction
4. **Fan-in**: All evidence converges at the **Skeptic** for critical review
5. **Loop or terminate**: Skeptic critique feeds back to the PI for refinement, up to `max_iterations`

---

<a id="agents"></a>
## Agent Reference

| Agent | Role | Software Tools | Temperature |
|---|---|---|---|
| **PI** | Hypothesis generation & refinement | — | 0.7 |
| **Researcher** | Literature retrieval (local FAISS → Semantic Scholar fallback) | FAISS, Semantic Scholar API | 0.2 |
| **Bioinformatician** | Genome retrieval & multiple sequence alignment | NCBI Entrez, MAFFT | 0.1 |
| **Structural** | RNA 2D thermodynamic analysis & conservation | SFold, ViennaRNA (RNAfold, RNAalifold) | 0.2 / 0.3 |
| **MD Specialist** | Coarse-grained 3D RNA folding | SimRNA, rna-tools | 0.2 |
| **Protein Expert** | Capsid structure & electrostatic analysis | BioPython, Rosetta (score_jd2), HDOCKlite, OpenBabel | 0.1 / 0.3 |
| **Skeptic** | Critical peer review & iteration control | — | 0.7 |

---

<a id="configuration"></a>
## Configuration

### API Setup

To use the full capabilities of the Researcher agent, you should configure a Semantic Scholar API key.

1. **Semantic Scholar**: Request an API key at [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api). This increases your rate limit from 100 requests per 5 minutes to 1 request per second.
2. **NCBI Entrez**: Provide an email address for PubMed access to avoid being blocked by NCBI.

### Environment Variables

The system loads configuration from a `.env` file using `python-dotenv`.

### Research Topics

Topics and seed questions are defined in [`research_topics.yaml`](research_topics.yaml). Each topic includes:

```yaml
- name: "Viral Packaging Signals"
  description: "Identify cis-acting RNA elements in Picornaviridae"
  seed_questions:
    - "What conserved RNA structural motifs are shared across Picornaviridae?"
    - "Can a minimal stem-loop element direct heterologous genome packaging?"
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VLLM_URL` | `http://localhost:8000/v1` | vLLM OpenAI-compatible endpoint |
| `VLLM_MODEL` | `Qwen/Qwen2.5-32B-Instruct` | Model served by vLLM |
| `S2_API_KEY` | `None` | Semantic Scholar API Key |
| `ENTREZ_EMAIL` | `fbsnpat@leeds.ac.uk` | Email for NCBI Entrez API |
| `SFOLD_BIN` | `~/Sfold-main/bin/sfold` | Path to SFold binary |
| `ROSETTA_BIN` | auto-detect | Rosetta binary or bin directory |
| `ROSETTA_BIN_DIR` | auto-detect | Rosetta installation bin directory |
| `HF_HOME` | `./hf_cache` | HuggingFace cache directory |
| `TOPIC_INDEX` | `0` | Index into `research_topics.yaml` |
| `MAX_ITERATIONS` | `3` | Research refinement cycles |
| `VLAB_DOCKING_BACKEND` | `hdock` | Docking backend (hdock or vina) |
| `HDOCK_HOME` | `./HDOCKlite-v1.1` | HDOCKlite installation directory |
| `HDOCK_TIMEOUT` | `420` | HDOCK timeout in seconds |
| `VLAB_MIN_VALID_HDOCK_SCORE` | `-30` | Minimum valid HDOCK score |
| `VLAB_RENDER_DOCKING_SNAPSHOTS` | `1` | Enable PyMOL docking snapshot rendering |
| `OBABEL_BIN` | auto-detect | OpenBabel binary path |
| `PYMOL_BIN` | auto-detect | PyMOL binary path |

---

<a id="documentation"></a>
## Documentation

Detailed documentation is available in the [`docs/`](docs/) directory:

| Document | Description |
|---|---|
| [`docs/setup.md`](docs/setup.md) | Environment setup and configuration instructions |
| [`docs/training.md`](docs/training.md) | Training examples and LoRA fine-tuning guide |
| [`docs/agents/bioinfo_agent.md`](docs/agents/bioinfo_agent.md) | Bioinformatics agent documentation |
| [`docs/faiss_db.md`](docs/faiss_db.md) | FAISS database setup and management |
| [`docs/inhibitor_agent.md`](docs/inhibitor_agent.md) | Inhibitor agent documentation |
| [`docs/lora_finetuning.md`](docs/lora_finetuning.md) | LoRA fine-tuning detailed guide |
| [`docs/peptide_production.md`](docs/peptide_production.md) | Peptide production workflow |

---

## Project Structure

```
bioagent/
├── virtual_lab_orchestrator.py   # Core LangGraph workflow & agent definitions
├── research_agent_starter.py     # Literature search (PubMed, Semantic Scholar, FAISS)
├── sfold_wrapper.py              # SFold RNA structure prediction interface
├── viennarna_wrapper.py          # ViennaRNA (RNAfold, RNAalifold) interface
├── md_wrapper.py                 # SimRNA coarse-grained MD interface
├── protein_wrapper.py            # Rosetta & BioPython protein analysis
├── bioinfo_wrapper.py            # NCBI genome retrieval & MAFFT alignment
├── training_data_collector.py    # Instruction-tuning dataset generation
├── research_topics.yaml          # Configurable research topics & seed questions
├── environment.yml               # Conda environment specification
├── run_virtual_lab_conda.sh      # SLURM job submission script
└── logs/                         # Runtime logs (vLLM, job output)
```

---

<a id="roadmap"></a>
## Improvements & Roadmap

### Short-Term Improvements

- [ ] **Add `.gitignore`** — exclude `hf_cache/`, `pdb_cache/`, `model_cache/`, `logs/`, `__pycache__/`, `training_data/`, `*.pdb`, `*.trafl`, checkpoint files
- [ ] **Structured logging** — replace `print()` statements with proper logging; add JSON-structured log output for observability
- [ ] **Unit tests** — add pytest suite for each wrapper (mock subprocess calls) and integration tests for the LangGraph state machine
- [ ] **Type safety** — add `py.typed` marker and run `mypy --strict`; the current `type: ignore` comments indicate typing gaps
- [ ] **Remove duplicate `_biopython_analysis`** — `protein_wrapper.py` defines this method twice (lines 68 and 131); the second silently shadows the first
- [ ] **Error handling in wrappers** — standardise error return types (currently mix of strings and exceptions); consider a `Result[T, E]` pattern
- [ ] **Pin all dependencies** — `environment.yml` has several unpinned packages (`requests`, `beautifulsoup4`, etc.) which risks reproducibility

### Medium-Term Enhancements

- [ ] **Containerisation** — Dockerfile / Apptainer definition for portable deployment beyond SLURM clusters
- [ ] **API server** — wrap the orchestrator in a FastAPI service with WebSocket progress streaming
- [ ] **Checkpoint resume** — the `save_checkpoint` function writes state but there's no `load_checkpoint` path to resume interrupted jobs
- [ ] **Async agent execution** — use LangGraph's async capabilities for true parallel agent execution instead of sequential fan-out
- [ ] **Observability** — integrate OpenTelemetry tracing for agent execution spans; add Prometheus metrics for iteration times and tool success rates
- [ ] **Knowledge base management** — CLI tooling to build/rebuild the FAISS index, inspect indexed papers, and manage the local literature corpus
- [ ] **Model flexibility** — abstract the LLM backend to support multiple providers (OpenAI, Anthropic, local models) via a configuration layer

### Future Directions

- [ ] **Wet-lab integration** — generate machine-readable experimental protocols (e.g., OpenTrons Python scripts) from validated hypotheses
- [ ] **Active learning loop** — use the Skeptic's scores to prioritise which hypotheses warrant further simulation budget
- [ ] **Multi-GPU scale-out** — distribute agent workloads across multiple GPUs for larger models or parallel simulation campaigns
- [ ] **Domain expansion** — generalise the agent framework beyond viral packaging to other structural biology domains (protein design, drug-target interaction)
- [ ] **Fine-tuned specialist models** — use the collected `training_data/*.jsonl` to fine-tune smaller, domain-specific models that replace the general-purpose LLM for individual agent roles
- [ ] **Continuous literature monitoring** — scheduled PubMed/bioRxiv scans that automatically update the FAISS knowledge base and trigger re-evaluation of stale hypotheses
- [ ] **Provenance & reproducibility** — hash all tool inputs/outputs and store in a provenance graph (e.g., W3C PROV) for full auditability of scientific claims

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/async-agents`)
3. Ensure tests pass and code compiles cleanly (`python -m py_compile *.py`)
4. Submit a pull request with a clear description of changes

---

## License

This project is developed at the University of Leeds. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with LangGraph • vLLM • ViennaRNA • SimRNA • Rosetta • BioPython</sub>
</p>
