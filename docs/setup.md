# Environment Setup

The main entry point is `run_virtual_lab_conda.sh`, which sets up the environment and runs the virtual lab orchestrator.

## Running the System

```bash
# Submit to SLURM (recommended for GPU jobs)
sbatch --export=ALL,TOPIC_INDEX=0,MAX_ITERATIONS=3 run_virtual_lab_conda.sh

# Or run locally (for testing)
bash run_virtual_lab_conda.sh
```

## Script Configuration

The script automatically:
- Loads required modules (CUDA 12.6.2, miniforge)
- Activates the conda environment
- Sets up cache directories (HF_HOME, TMPDIR, etc.)
- Configures HDOCKlite, OpenBabel, and PyMOL paths
- Starts the vLLM server (Qwen/Qwen2.5-32B-Instruct)
- Runs the virtual lab orchestrator

## Environment Details

- The script uses a pre-configured conda environment at `/mnt/scratch/fbsnpat/envs/biophysics-research-agent-new`
- HuggingFace models are loaded in offline mode (HF_HUB_OFFLINE=1)
- Cache directories are set up in ${BASE_DIR}/hf_cache, ${BASE_DIR}/tmp, etc.

## Prerequisites

Ensure the following are available:
- HDOCKlite installed at ${HDOCK_HOME} (default: ./HDOCKlite-v1.1)
- OpenBabel available in the conda environment for PDB format conversion
- PyMOL (optional but recommended for docking snapshot rendering)

## Additional Configuration

See the main [README.md](../README.md) for full environment variable configuration options.

---
*Last updated: 2026-07-01*
