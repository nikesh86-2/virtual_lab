# Environment & Configuration Documentation

## Overview

The **Environment & Configuration** system manages all external dependencies, environment variables, paths, and runtime settings for VLAB2. It ensures the system can run consistently across different environments (SLURM HPC, local development, CI/CD).

## Environment Variables

### Core Runtime Variables

| Variable | Default | Description |
|---|---|---|
| `VLLM_API_BASE` | `http://localhost:8000/v1` | vLLM server endpoint |
| `VLLM_MODEL` | `Qwen/Qwen2.5-32B-Instruct` | Model name or path |
| `HF_HOME` | `/mnt/scratch/fbsnpat/bot/VLAB2/hf_cache` | HuggingFace cache directory |
| `HF_HUB_OFFLINE` | `1` | Offline mode for model loading |
| `TOKENIZERS_PARALLELISM` | `false` | Disable tokenizer parallelism |

### Optimisation Controls

| Variable | Default | Description |
|---|---|---|
| `VLAB_RNA_ENFORCE_MIN_FOLD` | `1` | Enforce RNA fold quality thresholds |
| `VLAB_ACCEPT_BINDING_SCORE` | `-50.0` | Threshold for favourable HDOCK relative score |
| `VLAB_MAX_ACCEPT_SCORE_SPREAD` | `10.0` | Maximum acceptable HDOCK relative score spread |
| `VLAB_MIN_VALID_HDOCK_SCORE` | `-30.0` | Minimum valid HDOCK relative score |
| `VLAB_TARGET_RESET_ON_HIGH_SPREAD` | `1` | Enable automatic target reset on high score spread |
| `VLAB_TARGET_RESET_REQUIRE_WEAK_BINDING` | `1` | Require weak binding to trigger target reset |
| `VLAB_TARGET_RESET_EXTREME_SPREAD_MULTIPLIER` | `2.0` | Multiplier for extreme spread detection |
| `VLAB_TARGET_MARK_RESET_AS_FAILED` | `0` | Mark reset targets as failed |
| `VLAB_TARGET_RESET_MIN_VALID_N` | `3` | Minimum valid samples before considering target reset |

### Convergence Controls

| Variable | Default | Description |
|---|---|---|
| `VLAB_MIN_VALID_DOCKINGS_FOR_CONVERGENCE` | `3` | Minimum validated dockings for convergence |
| `VLAB_MIN_VALID_DOCKINGS_PER_TARGET` | `3` | Minimum dockings per target |
| `VLAB_CONVERGENCE_SPREAD_THRESHOLD` | `2.0` | Maximum score spread for convergence |
| `VLAB_REQUIRE_INTERFACE_FOR_CONVERGENCE` | `1` | Require interface-clean poses for convergence |
| `VLAB_REQUIRE_CURRENT_TARGET_FOR_CONVERGENCE` | `1` | Require current target for convergence |
| `VLAB_REQUIRE_CONSERVATION_FOR_CONVERGENCE` | `0` | Require conservation signal for convergence |
| `VLAB_MIN_CONSERVATION_FOR_ACCEPT` | `0.2` | Minimum conservation fitness for acceptance |
| `VLAB_ALLOW_CONVERGENCE_ON_REVISE` | `0` | Allow convergence even if Skeptic recommends revision |

### Inhibitor Screening Controls

| Variable | Default | Description |
|---|---|---|
| `VLAB_INHIBITOR_ENABLED` | `0` | Enable inhibitor screening |
| `VLAB_INHIBITOR_MAX_SMALL_MOLECULES` | `100` | Max small molecules to screen |
| `VLAB_INHIBITOR_MAX_PEPTIDES` | `50` | Max peptides to screen |
| `VLAB_INHIBITOR_VINA_SEED` | `None` | AutoDock Vina reproducibility seed |

### Training Pipeline Controls

| Variable | Default | Description |
|---|---|---|
| `VLAB_LORA_TRAIN` | `0` | Enable LoRA fine-tuning (set to `1`) |
| `VLAB_MIN_TRAIN_ROWS` | `20` | Minimum training examples required |
| `VLAB_MIN_TRAIN_BYTES` | `5000` | Minimum dataset size in bytes |
| `VLAB_MIN_INTERFACE_VALID` | `5` | Minimum interface-valid examples |
| `VLAB_MIN_LITERATURE_EVIDENCE` | `2` | Minimum literature evidence examples |
| `VLAB_AUTO_MODEL_REPLACE` | `0` | Automatically replace vLLM model after training |
| `VLAB_MAX_MODEL_VERSIONS` | `3` | Maximum stored model versions |

### Path Configuration

| Variable | Default | Description |
|---|---|---|
| `BASE_DIR` | (script directory) | Base directory for output files |
| `HDOCK_HOME` | `./HDOCKlite-v1.1` | HDOCKlite installation path |
| `SIMRNA_HOME` | (conda env) | SimRNA installation path |
| `VIENNARNA_HOME` | (conda env) | ViennaRNA installation path |
| `PYMOL_HOME` | (conda env) | PyMOL installation path |
| `OPENBABEL_HOME` | (conda env) | OpenBabel installation path |

## Configuration Files

### Research Topics (`research_topics.yaml`)

Defines research topics with metadata:

```yaml
topics:
  - name: "SARS-CoV-2 RNA Inhibition"
    description: "Design RNA aptamers against SARS-CoV-2 spike protein"
    virus_name: "SARS-CoV-2"
    virus_family: "Coronaviridae"
    virus_genus: "Betacoronavirus"
    seed_questions:
      - "Which regions of the spike protein are most druggable?"
      - "Can RNA aptamers block receptor binding?"
    target_pdbs:
      - "6M71"  # Spike protein RBD
      - "7DM0"  # Full spike trimer
```

### Environment Template (`.env.example`)

```bash
# vLLM Configuration
VLLM_API_BASE=http://localhost:8000/v1
VLLM_MODEL=/path/to/Qwen2.5-32B-Instruct

# HuggingFace Configuration
HF_HOME=/path/to/hf_cache
HF_HUB_OFFLINE=1

# Path Configuration
BASE_DIR=/path/to/VLAB2
HDOCK_HOME=/path/to/HDOCKlite-v1.1

# Training Configuration
VLAB_LORA_TRAIN=0
VLAB_AUTO_MODEL_REPLACE=0

# Optimisation Controls
VLAB_RNA_ENFORCE_MIN_FOLD=1
VLAB_ACCEPT_BINDING_SCORE=-50.0
VLAB_CONVERGENCE_SPREAD_THRESHOLD=2.0

# Inhibitor Screening
VLAB_INHIBITOR_ENABLED=0
```

### Training Configuration (`training/training_config.yaml`)

```yaml
model_name: /path/to/Qwen2.5-32B-Instruct
output_dir: training/output_model
max_length: 4096

training:
  batch_size: 1
  gradient_accumulation_steps: 8
  epochs: 1
  learning_rate: 2.0e-5
  save_steps: 100
  logging_steps: 5

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
```

## Runtime Configuration

### Bootstrap Process

Located in: `orchestration/config.py`

```python
def bootstrap_runtime(anchor_file: str | None = None) -> None:
    """Convenience setup for CLI scripts."""
    configure_paths(anchor_file)      # Add VLAB2 to sys.path
    configure_environment()           # Load .env, set defaults
    configure_logging()               # Configure logging
```

### Path Configuration

```python
def configure_paths(anchor_file: str | None = None) -> None:
    """Ensure VLAB2 and repo root are importable."""
    vlab2_root = anchor.parents[1]    # VLAB2/orchestration/config.py → VLAB2/
    repo_root = anchor.parents[2]     # → repository root
    
    for path in [str(vlab2_root), str(repo_root)]:
        if path not in sys.path:
            sys.path.insert(0, path)
```

### Environment Configuration

```python
def configure_environment() -> None:
    """Load environment variables and apply safe defaults."""
    load_dotenv()  # Load .env file
    
    os.environ.setdefault("HF_HOME", "/path/to/hf_cache")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
```

### Logging Configuration

```python
def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
```

## SLURM Environment

### Run Script (`run_virtual_lab_conda.sh`)

The SLURM submission script handles:

1. **Module loading**: CUDA, Miniforge
2. **Environment activation**: Conda environment
3. **Cache setup**: HF_HOME, TMPDIR, etc.
4. **Path configuration**: HDOCK, OpenBabel, PyMOL
5. **vLLM startup**: Launch model server
6. **Readiness check**: Wait for vLLM to be ready
7. **Orchestrator execution**: Run with topic index and iterations
8. **Cleanup**: Terminate vLLM on job end

```bash
#!/bin/bash
#SBATCH --job-name=vlab2
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Load modules
module load cuda/12.6.2
module load miniforge

# Activate environment
source /path/to/conda/etc/profile.d/conda.sh
conda activate biophysics-research-agent-new

# Set up paths
export HF_HOME=${BASE_DIR}/hf_cache
export TMPDIR=${BASE_DIR}/tmp

# Start vLLM
python -m vllm.entrypoints.openai.api_server \
    --model ${VLLM_MODEL} \
    --port 8000 \
    --tensor-parallel-size 1 &
VLLM_PID=$!

# Wait for readiness
until curl -s http://localhost:8000/v1/models > /dev/null; do
    sleep 5
done

# Run orchestrator
python orchestration/virtual_lab_orchestrator.py \
    --topic-index ${TOPIC_INDEX} \
    --max-iterations ${MAX_ITERATIONS}

# Cleanup
kill ${VLLM_PID}
```

### SLURM Environment Variables

Passed via `--export=ALL`:

| Variable | Description |
|---|---|
| `TOPIC_INDEX` | Index into research_topics.yaml |
| `MAX_ITERATIONS` | Maximum research loop iterations |
| `BASE_DIR` | Base directory for outputs |
| `VLLM_MODEL` | Model path or name |

## Local Development Environment

### Prerequisites

- Python 3.9+
- Conda/Mamba
- CUDA 12.x (for GPU acceleration)
- vLLM server running on port 8000

### Setup Steps

```bash
# 1. Clone repository
git clone https://github.com/username/VLAB2.git
cd VLAB2

# 2. Create conda environment
conda create -n vlab2 python=3.11
conda activate vlab2

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and configure .env
cp .env.example .env
# Edit .env with your paths

# 5. Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct \
    --port 8000

# 6. Run orchestrator
python orchestration/virtual_lab_orchestrator.py \
    --topic-index 0 \
    --max-iterations 3
```

### Docker Environment (Optional)

```dockerfile
FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN python3.11 -m venv venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install -r requirements.txt

CMD ["python", "orchestration/virtual_lab_orchestrator.py", "--topic-index", "0", "--max-iterations", "3"]
```

## Configuration Precedence

Environment variables are loaded in this order (later overrides earlier):

1. **System environment**: Shell variables
2. **`.env` file**: Loaded by python-dotenv
3. **Defaults**: Set by `os.environ.setdefault()` in `configure_environment()`
4. **Runtime overrides**: Set by script arguments or SLURM exports

## Troubleshooting

### vLLM Connection Failed

- Check `VLLM_API_BASE` matches your vLLM server
- Ensure vLLM is running on the specified port
- Check firewall rules if running on remote server

### Model Loading Failed

- Verify `HF_HOME` points to valid cache
- Check `HF_HUB_OFFLINE` setting
- Ensure model files exist in cache directory

### HDOCK Not Found

- Verify `HDOCK_HOME` points to HDOCKlite installation
- Check executable permissions: `chmod +x ${HDOCK_HOME}/HDOCKlite`
- Ensure all dependencies are installed

### GPU Out of Memory

- Reduce `VLAB_INHIBITOR_MAX_SMALL_MOLECULES`
- Reduce population size in optimisation
- Use smaller batch sizes in training

### Path Errors

- Verify `BASE_DIR` is set correctly
- Check all tool paths in `.env` file
- Ensure directories exist and are writable

## Key Design Principles

- **Centralised configuration**: All settings in one place (`.env` + config files)
- **Sensible defaults**: System works out-of-the-box with reasonable defaults
- **Environment-specific**: Different settings for SLURM vs local development
- **Type-safe**: Environment variables are parsed with type conversion (`_env_int`, `_env_float`, `_env_bool`)
- **Documented**: All variables documented with defaults and descriptions
- **Version-controlled**: `.env.example` tracked in git, `.env` in `.gitignore`
- **Runtime-flexible**: Variables can be overridden at runtime via SLURM exports

---
*Last updated: 2026-07-01*