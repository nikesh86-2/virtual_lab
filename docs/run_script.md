# SLURM Run Script (`run_virtual_lab_conda.sh`)

## Overview

The `run_virtual_lab_conda.sh` script is the primary entry point for executing the Virtual Lab on SLURM-based HPC clusters. It handles environment setup, GPU configuration, vLLM server startup, and orchestrator execution.

## SLURM Directives

```bash
#SBATCH --job-name=virtual-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/fbsnpat/bot/VLAB2/output_data/logs/lab_%j.log
#SBATCH --error=/scratch/fbsnpat/bot/VLAB2/output_data/logs/lab_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fbsnpat@leeds.ac.uk
```

| Directive | Value | Purpose |
|---|---|---|
| `--job-name` | `virtual-lab` | Job name in SLURM queue |
| `--nodes` | `1` | Single node |
| `--ntasks` | `1` | Single task |
| `--partition` | `gpu` | GPU-enabled partition |
| `--gres` | `gpu:2` | 2 GPUs (0 and 1) |
| `--cpus-per-task` | `16` | 16 CPU cores |
| `--mem` | `128G` | 128 GB RAM |
| `--time` | `03:00:00` | 3-hour time limit |
| `--gres` | `gpu:2` | 2 GPUs for tensor parallelism |

## Script Sections

### 0. Root Paths & Directories

Sets up base directories:
- `BASE_DIR` — `/scratch/fbsnpat/bot/VLAB2`
- `SRC_ENV` — Conda environment path
- Creates: `logs/`, `output_data/logs/`, `output_data/debug_rna_prep/`, `hf_cache/`, `tmp/`
- Exports `PYTHONPATH` to include parent directory for `from VLAB2...` imports
- Loads `.env` file for API keys and configuration

### 1. Modules + Conda

Loads HPC modules:
- `cuda/12.6.2` — CUDA toolkit
- `miniforge` — Conda package manager
- Activates the biophysics research conda environment

**Note**: GROMACS is explicitly NOT loaded to avoid shared library conflicts.

### 2. Environment Setup: Cache + Temp Paths

Configures high-performance I/O paths:
- **With NVMe** (`TMP_SHARED`): Uses NVMe SSD for hot cache directories
- **Without NVMe**: Falls back to scratch directories

Key environment variables:
| Variable | Purpose |
|---|---|
| `PYTHONPYCACHEPREFIX` | Python bytecode cache |
| `TMPDIR` | Temporary working directory |
| `TRITON_CACHE_DIR` | Triton compiler cache |
| `TORCHINDUCTOR_CACHE_DIR` | TorchInductor graph cache |
| `HF_HOME` | HuggingFace model cache |
| `HF_HUB_CACHE` | HuggingFace hub cache |
| `HF_HUB_OFFLINE` | `1` — Offline mode, no network downloads |
| `TRANSFORMERS_OFFLINE` | `1` — Transformers offline mode |
| `CUDA_MODULE_LOADING` | `LAZY` — Delayed CUDA module loading |

### 3. Diagnostics

Runs pre-flight checks:
- Python version and executable path
- PyTorch version and CUDA availability
- SentenceTransformers import check
- GPU diagnostics via `nvidia-smi` and Python torch.cuda checks

### 4. Scientific Tools Setup

#### OpenBabel
- Verifies `OBABEL_BIN` is executable
- Checks PDB format plugin availability
- Uses `LD_LIBRARY_PATH` scoping to avoid global conflicts

#### HDOCKlite Docking
- Sets `VLAB_DOCKING_BACKEND=hdock`
- Configures HDOCK paths, timeouts, and model counts
- Creates debug and work directories
- Sets convergence controls:
  - `VLAB_MIN_VALID_HDOCK_SCORE=-30`
  - `VLAB_MIN_VALID_DOCKINGS_PER_TARGET=2`
  - `VLAB_ACCEPT_BINDING_SCORE=-50`
  - `VLAB_MAX_TARGET_SCORE_SPREAD=25`

#### Inhibitor Screening
- `VLAB_INHIBITOR_ENABLED=1` — Enable inhibitor screening
- `VLAB_INHIBITOR_MAX_SMALL_MOLECULES=10` — Max small molecules
- `VLAB_INHIBITOR_MAX_PEPTIDES=5` — Max peptides
- `VLAB_PUBCHEM_QUERIES="ribavirin|remdesivir"` — PubChem search terms

#### LoRA Fine-tuning Controls
- `VLAB_LORA_TRAIN=0` — Disabled by default in run script
- `VLAB_MIN_TRAIN_ROWS=20` — Minimum training rows
- `VLAB_MAX_MODEL_VERSIONS=3` — Max model versions to keep

#### AutoDock Vina
- Configured as alternative docking backend
- `VLAB_VINA_TIMEOUT=300` seconds

#### PyMOL Rendering
- `VLAB_RENDER_DOCKING_SNAPSHOTS=1` — Enable snapshot rendering
- Preflight check ensures PyMOL is functional

#### Interface Contact Analysis
- `VLAB_ANALYSE_INTERFACE_CONTACTS=1`
- `VLAB_INTERFACE_CONTACT_CUTOFF=5.0` Å
- `VLAB_MIN_INTERFACE_RESIDUE_CONTACTS=5`

#### Convergence Controls
- `VLAB_ALLOW_CONVERGENCE_ON_REVISE=0` — Require fresh convergence
- `VLAB_REQUIRE_CONSERVATION_FOR_CONVERGENCE=1` — Conservation required
- `VLAB_MIN_CONSERVATION_FOR_ACCEPT=0.2` — Minimum conservation threshold

### 5. Pre-warm Imports

Pre-warms critical Python imports to avoid cold-start delays during the run:
- PyTorch + CUDA availability check
- vLLM import check

Fails fast if CUDA is unavailable.

### 6. vLLM Configuration

Sets up the local LLM server:
| Variable | Value |
|---|---|
| `VLLM_HOST` | `127.0.0.1` |
| `VLLM_PORT` | `8000` |
| `VLLM_URL` | `http://127.0.0.1:8000/v1` |
| `VLLM_MODEL` | `Qwen/Qwen2.5-32B-Instruct` |
| `OPENAI_API_KEY` | `dummy` (local server) |

### 7. Cleanup Trap

Ensures vLLM process is killed on script exit:
```bash
trap cleanup EXIT
```

### 8. Start vLLM

Launches vLLM as a background process:
```bash
python -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 \
    --port 8000 \
    --model <path> \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --trust-remote-code
```

Key flags:
- `--tensor-parallel-size 2` — Splits model across 2 GPUs
- `--gpu-memory-utilization 0.95` — Uses 95% of GPU memory
- `--max-model-len 8192` — Max sequence length
- `--enable-prefix-caching` — Caches KV prefixes for efficiency

### 9. Readiness Check + LLM Preflight

Waits for vLLM to become ready:
- Polls `/models` endpoint every 10 seconds
- Max wait: 180 steps (30 minutes)
- On failure: tails last 100 lines of vLLM log

Once ready, runs an LLM preflight test:
```python
client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Return exactly: OK"}],
    temperature=0,
    max_tokens=8,
)
```

### 10. Orchestrator Execution

Runs the main Virtual Lab orchestrator:
```bash
python orchestration/virtual_lab_orchestrator.py \
    --topic-index ${TOPIC_INDEX} \
    --max-iterations ${MAX_ITERATIONS}
```

Parameters:
- `TOPIC_INDEX` — Index into `research_topics.yaml` (default: 0)
- `MAX_ITERATIONS` — Maximum refinement cycles (default: 3)

### 11. Normal Exit

Prints completion message and exits with the orchestrator's exit code.

## Usage

### Submit a Job

```bash
# Default topic, 3 iterations
sbatch run_virtual_lab_conda.sh

# Custom topic and iterations
sbatch --export=ALL,TOPIC_INDEX=5,MAX_ITERATIONS=5 run_virtual_lab_conda.sh
```

### Monitor a Job

```bash
# Check job status
squeue -u $(whoami)

# View output log
tail -f /scratch/fbsnpat/bot/VLAB2/output_data/logs/lab_<jobid>.log

# View vLLM log
tail -f /scratch/fbsnpat/bot/VLAB2/logs/vllm_<jobid>.log
```

## Troubleshooting

| Issue | Solution |
|---|---|
| vLLM fails to start | Check CUDA module load order, verify GPU availability with `nvidia-smi` |
| OpenBabel PDB format not found | Verify `OBABEL_BIN` path and `LD_LIBRARY_PATH` |
| HDOCK docking fails | Check `HDOCK_HOME` and `HDOCK_BIN` paths, verify createpl is executable |
| No results in output | Check `lab_results_*.json` files in the working directory |
| Coverage report missing | Ensure `coverage` is installed in the conda environment |

## Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `TOPIC_INDEX` | `0` | Which research topic to run |
| `MAX_ITERATIONS` | `3` | Max refinement cycles |
| `CUDA_VISIBLE_DEVICES` | `0,1` | Which GPUs to use |
| `HF_HUB_OFFLINE` | `1` | Offline mode for HuggingFace |
| `VLAB_INHIBITOR_ENABLED` | `1` | Enable inhibitor screening |
| `VLAB_LORA_TRAIN` | `0` | Enable LoRA training |
| `VLAB_RENDER_DOCKING_SNAPSHOTS` | `1` | Enable PyMOL snapshot rendering |

---
*Last updated: 2025-01-01*
---
```