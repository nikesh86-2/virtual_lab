#!/bin/bash
#SBATCH --job-name=virtual-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/fbsnpat/bot/VLAB2/output_data/logs/lab_%j.log
#SBATCH --error=/scratch/fbsnpat/bot/VLAB2/output_data/logs/lab_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fbsnpat@leeds.ac.uk

set -euo pipefail

# ===========================================================================
# 0. ROOT PATHS
# ===========================================================================

BASE_DIR="/scratch/fbsnpat/bot/VLAB2"
SRC_ENV="/mnt/scratch/fbsnpat/envs/biophysics-research-agent-new"
PY="${SRC_ENV}/bin/python3"

HF_SCRATCH="${BASE_DIR}/hf_cache"
PROJECT_DIR="${BASE_DIR}"

export CUDA_VISIBLE_DEVICES=0,1

mkdir -p \
    "${BASE_DIR}/logs" \
    "${BASE_DIR}/output_data/logs" \
    "${BASE_DIR}/output_data/debug_rna_prep" \
    "${HF_SCRATCH}" \
    "${BASE_DIR}/tmp"

cd "${PROJECT_DIR}" || exit 1

# Make parent package importable for `from VLAB2...`
export PYTHONPATH="/scratch/fbsnpat/bot:${PYTHONPATH:-}"

echo "Project: ${PROJECT_DIR}"
echo "Scratch root: ${BASE_DIR}"
echo "Node: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}"
echo "PYTHONPATH=${PYTHONPATH}"

# Load secrets/defaults: HF_TOKEN, S2_API_KEY, etc.
# Runtime-critical values are re-exported below after this load.
if [ -f "${PROJECT_DIR}/.env" ]; then
    echo "Loading .env from ${PROJECT_DIR}/.env"
    set -a
    # shellcheck source=/dev/null
    source "${PROJECT_DIR}/.env"
    set +a
else
    echo "No .env found at ${PROJECT_DIR}/.env"
fi

# ===========================================================================
# 1. MODULES + CONDA
# ===========================================================================

module load cuda/12.6.2
module load miniforge

# GROMACS is not required and can introduce incompatible shared libraries.
# module load gromacs/2024.4/gcc-13.2.0_cuda-12.6.2

eval "$(conda shell.bash hook)"

set +u
conda activate "${SRC_ENV}"
set -u

echo "===== MODULES ====="
module list 2>&1 || true
echo "==================="

# ===========================================================================
# 2. ENV SETUP: CACHE + TMP PATHS
# ===========================================================================

if [ -n "${TMP_SHARED:-}" ] && [ -d "${TMP_SHARED}" ]; then
    echo "Using NVMe for hot I/O: ${TMP_SHARED}"

    export PYTHONPYCACHEPREFIX="${TMP_SHARED}/pycache"
    export TMPDIR="${TMP_SHARED}/tmp"
    export TRITON_CACHE_DIR="${TMP_SHARED}/triton_cache"
    export TORCHINDUCTOR_CACHE_DIR="${TMP_SHARED}/torchinductor_cache"
else
    echo "No NVMe detected — using scratch only"

    export PYTHONPYCACHEPREFIX="${BASE_DIR}/pycache"
    export TMPDIR="${BASE_DIR}/tmp"
    export TRITON_CACHE_DIR="${BASE_DIR}/triton_cache"
    export TORCHINDUCTOR_CACHE_DIR="${BASE_DIR}/torchinductor_cache"
fi

# Persistent Hugging Face/cache paths
export HF_HOME="${HF_SCRATCH}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export XDG_CACHE_HOME="${HF_HOME}/xdg"

# Offline mode: model files must already exist in HF cache.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export CUDA_MODULE_LOADING=LAZY

mkdir -p \
    "${PYTHONPYCACHEPREFIX}" \
    "${TMPDIR}" \
    "${TRITON_CACHE_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${HF_HUB_CACHE}" \
    "${XDG_CACHE_HOME}"

echo "===== PATHS ====="
echo "BASE_DIR=${BASE_DIR}"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "SRC_ENV=${SRC_ENV}"
echo "PY=${PY}"
echo "HF_HOME=${HF_HOME}"
echo "HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
echo "TMPDIR=${TMPDIR}"
echo "PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX}"
echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "==============="

# ===========================================================================
# 3. PYTHON + GPU DIAGNOSTICS
# ===========================================================================

echo "===== PYTHON DIAGNOSTICS ====="
echo "which python: $(which python || true)"
echo "which python3: $(which python3 || true)"
echo "PY=${PY}"

"${PY}" -V
"${PY}" -c "import sys; print('sys.executable:', sys.executable)"
"${PY}" -c "import torch; print('torch', torch.__version__)"
"${PY}" -c "from sentence_transformers import CrossEncoder; print('CrossEncoder import OK')"
echo "=============================="

echo "===== GPU DIAGNOSTICS ====="
hostname
nvidia-smi || true

"${PY}" - <<'PY'
import torch

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY

echo "==========================="

# ===========================================================================
# 4. OPENBABEL / HDOCK / RNA PREP DEBUG SETUP
# ===========================================================================

export RNA_PREP_DEBUG_DIR="${BASE_DIR}/output_data/debug_rna_prep"
mkdir -p "${RNA_PREP_DEBUG_DIR}"

echo "===== OPENBABEL DIAGNOSTICS ====="
echo "RNA_PREP_DEBUG_DIR=${RNA_PREP_DEBUG_DIR}"

export OBABEL_ENV="/mnt/scratch/fbsnpat/envs/biophysics-research-agent"
export OBABEL_BIN="${OBABEL_ENV}/bin/obabel"

echo "OBABEL_ENV=${OBABEL_ENV}"
echo "OBABEL_BIN=${OBABEL_BIN}"

if [ ! -x "${OBABEL_BIN}" ]; then
    echo "ERROR: OBABEL_BIN is not executable: ${OBABEL_BIN}"
    find /users/fbsnpat/.conda/envs /mnt/scratch/fbsnpat/envs -path "*/bin/obabel" -type f 2>/dev/null || true
    exit 1
fi

# Avoid exporting OpenBabel/conda lib path globally. Only apply to obabel checks.
env LD_LIBRARY_PATH="${OBABEL_ENV}/lib:${LD_LIBRARY_PATH:-}" "${OBABEL_BIN}" -V

env LD_LIBRARY_PATH="${OBABEL_ENV}/lib:${LD_LIBRARY_PATH:-}" "${OBABEL_BIN}" -L formats | grep -i pdb || {
    echo "ERROR: OpenBabel cannot see PDB format plugin"
    env LD_LIBRARY_PATH="${OBABEL_ENV}/lib:${LD_LIBRARY_PATH:-}" "${OBABEL_BIN}" -L formats || true
    exit 1
}

echo "================================="

# ------------------------------------------------------------
# HDOCKlite backend
# ------------------------------------------------------------

export VLAB_DOCKING_BACKEND=hdock
export VLAB_REQUIRE_DOCKING=1
export VLAB_REQUIRE_VINA=0

export HDOCK_HOME="/mnt/scratch/fbsnpat/bot/VLAB2/HDOCKlite-v1.1"
export HDOCK_BIN="/mnt/scratch/fbsnpat/bot/VLAB2/bin/hdock_env.sh"
export CREATEPL_BIN="/mnt/scratch/fbsnpat/bot/VLAB2/bin/createpl_env.sh"

export HDOCK_TIMEOUT=420
export HDOCK_N_MODELS=5
export CREATEPL_TIMEOUT=120

export HDOCK_DEBUG_DIR="/mnt/scratch/fbsnpat/bot/VLAB2/output_data/debug_hdock"
export HDOCK_WORK_DIR="/mnt/scratch/fbsnpat/bot/VLAB2/output_data/hdock_work"
export PROTEIN_CACHE_DIR="/mnt/scratch/fbsnpat/bot/VLAB2/pdb_cache"
export RNA_HDOCK_CACHE_DIR="/mnt/scratch/fbsnpat/bot/VLAB2/output_data/rna_hdock_cache"
export RNA_PREP_DEBUG_DIR="/mnt/scratch/fbsnpat/bot/VLAB2/output_data/debug_rna_prep"

mkdir -p \
    "${HDOCK_DEBUG_DIR}" \
    "${HDOCK_WORK_DIR}" \
    "${PROTEIN_CACHE_DIR}" \
    "${RNA_HDOCK_CACHE_DIR}" \
    "${RNA_PREP_DEBUG_DIR}"

# Docking convergence / debug controls.
# These override .env if it has older values.
export VLAB_AGENT_EVAL_TOP_N=2
export VLAB_MAX_DOCKINGS_PER_TARGET=2
export VLAB_ABORT_TARGET_ON_FIRST_TIMEOUT=1

export VLAB_MIN_VALID_HDOCK_SCORE=-30
export VLAB_MIN_VALID_DOCKINGS_PER_TARGET=2
export VLAB_MIN_VALID_DOCKINGS_FOR_CONVERGENCE=2

export VLAB_REJECT_TARGET_ON_SCORE_SPREAD=1
export VLAB_MAX_TARGET_SCORE_SPREAD=25

export VLAB_ACCEPT_BINDING_SCORE=-50
export VLAB_MAX_ACCEPT_SCORE_SPREAD=15.0
export VLAB_CONVERGENCE_SPREAD_THRESHOLD=5.0

export VLAB_MAX_RECEPTOR_ATOMS=60000
export VLAB_TARGET_BLACKLIST="5TC1,8PNF"

export PYMOL_BIN="/mnt/scratch/fbsnpat/bot/VLAB2/bin/pymol_render_env.sh"
export VLAB_RENDER_DOCKING_SNAPSHOTS=1
export VLAB_DOCKING_SNAPSHOT_DIR="${BASE_DIR}/output_data/docking_snapshots"
export VLAB_RENDER_TIMEOUT=180

export VLAB_ANALYSE_INTERFACE_CONTACTS=1
export VLAB_INTERFACE_CONTACT_DIR="${BASE_DIR}/output_data/docking_contacts"
export VLAB_INTERFACE_CONTACT_CUTOFF=5.0
export VLAB_MIN_INTERFACE_RESIDUE_CONTACTS=5
export VLAB_MIN_INTERFACE_BASIC_CONTACTS=1

mkdir -p "${VLAB_INTERFACE_CONTACT_DIR}"

mkdir -p "${VLAB_DOCKING_SNAPSHOT_DIR}"

echo "PyMOL render preflight:"
"${PYMOL_BIN}" -cq -d "quit" || {
    echo "WARNING: PyMOL render preflight failed; snapshots may fail."
}

# Scientific convergence controls.
# For smoke tests, set:
#   VLAB_ALLOW_CONVERGENCE_ON_REVISE=1
#   VLAB_REQUIRE_CONSERVATION_FOR_CONVERGENCE=0
export VLAB_ALLOW_CONVERGENCE_ON_REVISE="${VLAB_ALLOW_CONVERGENCE_ON_REVISE:-0}"
export VLAB_REQUIRE_CONSERVATION_FOR_CONVERGENCE="${VLAB_REQUIRE_CONSERVATION_FOR_CONVERGENCE:-1}"
export VLAB_MIN_CONSERVATION_FOR_ACCEPT="${VLAB_MIN_CONSERVATION_FOR_ACCEPT:-0.2}"

export VLAB_STREAM_WARMUP_SECONDS="${VLAB_STREAM_WARMUP_SECONDS:-6}"

echo "===== DOCKING ENV ====="
echo "VLAB_DOCKING_BACKEND=${VLAB_DOCKING_BACKEND}"
echo "VLAB_REQUIRE_DOCKING=${VLAB_REQUIRE_DOCKING}"
echo "HDOCK_TIMEOUT=${HDOCK_TIMEOUT}"
echo "HDOCK_N_MODELS=${HDOCK_N_MODELS}"
echo "CREATEPL_TIMEOUT=${CREATEPL_TIMEOUT}"
echo "VLAB_AGENT_EVAL_TOP_N=${VLAB_AGENT_EVAL_TOP_N}"
echo "VLAB_MIN_VALID_HDOCK_SCORE=${VLAB_MIN_VALID_HDOCK_SCORE}"
echo "VLAB_MIN_VALID_DOCKINGS_PER_TARGET=${VLAB_MIN_VALID_DOCKINGS_PER_TARGET}"
echo "VLAB_MAX_RECEPTOR_ATOMS=${VLAB_MAX_RECEPTOR_ATOMS}"
echo "======================="

# ===========================================================================
# 5. PRE-WARM IMPORTS
# ===========================================================================

echo "Pre-warming imports..."

"${PY}" - <<'PY'
import sys
import torch

cuda_ok = torch.cuda.is_available()

print(f"Python executable: {sys.executable}")
print(f"PyTorch {torch.__version__}")
print(f"CUDA build: {torch.version.cuda}")
print(f"GPU available: {cuda_ok}")

if not cuda_ok:
    print("ERROR: CUDA not available. Check module load order and pytorch-cuda version.")
    sys.exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")

import vllm
print("vLLM import OK")
PY

if [ $? -ne 0 ]; then
    echo "CUDA/PyTorch/vLLM pre-warm failed. Diagnostics:"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
    module list 2>&1 || true
    exit 1
fi

# ===========================================================================
# 6. vLLM CONFIG
# ===========================================================================

export VLLM_HOST=127.0.0.1
export VLLM_PORT=8000
export VLLM_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

# Separate path used to start vLLM from name used by OpenAI-compatible clients.
export VLLM_MODEL_PATH="${BASE_DIR}/hf_cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
export VLLM_SERVED_MODEL="Qwen/Qwen2.5-32B-Instruct"

# What the app/client should request.
export VLLM_MODEL="${VLLM_SERVED_MODEL}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export OPENAI_BASE_URL="${VLLM_URL}"
export OPENAI_API_BASE="${VLLM_URL}"

VLLM_LOG="${BASE_DIR}/logs/vllm_${SLURM_JOB_ID:-manual}.log"
VLLM_PID=""

# ===========================================================================
# 7. CLEANUP TRAP
# ===========================================================================

cleanup() {
    echo "Cleaning up..."

    if [ -n "${VLLM_PID:-}" ]; then
        if kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "Stopping vLLM PID ${VLLM_PID}"
            kill "${VLLM_PID}" 2>/dev/null || true
            wait "${VLLM_PID}" 2>/dev/null || true
        fi
    fi

    echo "Cleanup complete."
}

trap cleanup EXIT

# ===========================================================================
# 8. START vLLM
# ===========================================================================

HOME_CACHE_CHECK="${HOME}/.triton"

if [ -d "${HOME_CACHE_CHECK}" ]; then
    echo "Removing stale Triton cache in home directory..."
    rm -rf "${HOME_CACHE_CHECK}"/* 2>/dev/null || true
fi

export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export NCCL_ASYNC_ERROR_HANDLING=1

echo "Starting vLLM..."
echo "VLLM_MODEL_PATH=${VLLM_MODEL_PATH}"
echo "VLLM_SERVED_MODEL=${VLLM_SERVED_MODEL}"
echo "VLLM_LOG=${VLLM_LOG}"

"${PY}" -m vllm.entrypoints.openai.api_server \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --model "${VLLM_MODEL_PATH}" \
    --served-model-name "${VLLM_SERVED_MODEL}" \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --trust-remote-code \
    --tokenizer "${VLLM_MODEL_PATH}" \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!

echo "vLLM PID: ${VLLM_PID}"
echo "vLLM log: ${VLLM_LOG}"

# ===========================================================================
# 9. READINESS CHECK + LLM PREFLIGHT
# ===========================================================================

echo "Waiting for vLLM..."

MAX_WAIT_STEPS=180
WAIT=0

until false; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "vLLM crashed"
        tail -n 100 "${VLLM_LOG}" || true
        exit 1
    fi

    if curl -sf "${VLLM_URL}/models" | grep -q '"object"'; then
        echo "vLLM READY"
        break
    fi

    sleep 10
    WAIT=$((WAIT + 1))

    echo "Waiting... ${WAIT}/${MAX_WAIT_STEPS}"

    if [ "${WAIT}" -ge "${MAX_WAIT_STEPS}" ]; then
        echo "Timeout waiting for vLLM"
        tail -n 100 "${VLLM_LOG}" || true
        exit 1
    fi
done

echo "===== vLLM MODELS ====="
curl -s "${VLLM_URL}/models" | "${PY}" -m json.tool || true
echo "======================="

echo "Running LLM preflight..."

"${PY}" - <<'PY'
import os
from openai import OpenAI

base_url = os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1")
model = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-32B-Instruct")

print("LLM preflight base_url:", base_url)
print("LLM preflight model:", model)

client = OpenAI(
    base_url=base_url,
    api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
)

resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Return exactly: OK"}],
    temperature=0,
    max_tokens=8,
)

print("LLM preflight response:", resp.choices[0].message.content)
PY

echo "LLM preflight complete."

# ===========================================================================
# 10. ORCHESTRATOR
# ===========================================================================

echo "Starting Virtual Lab..."

TOPIC_INDEX="${TOPIC_INDEX:-0}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3}"

if [ "${TOPIC_INDEX}" -lt 0 ]; then
    echo "Invalid TOPIC_INDEX=${TOPIC_INDEX}, falling back to 0"
    TOPIC_INDEX=0
fi

if [ "${MAX_ITERATIONS}" -le 0 ]; then
    echo "Invalid MAX_ITERATIONS=${MAX_ITERATIONS}, falling back to 3"
    MAX_ITERATIONS=3
fi

echo "Using:"
echo "  TOPIC_INDEX=${TOPIC_INDEX}"
echo "  MAX_ITERATIONS=${MAX_ITERATIONS}"
echo "  VLLM_URL=${VLLM_URL}"
echo "  VLLM_MODEL=${VLLM_MODEL}"
echo "  OBABEL_BIN=${OBABEL_BIN}"
echo "  RNA_PREP_DEBUG_DIR=${RNA_PREP_DEBUG_DIR}"
echo "  HDOCK_TIMEOUT=${HDOCK_TIMEOUT}"
echo "  HDOCK_N_MODELS=${HDOCK_N_MODELS}"

"${PY}" orchestration/virtual_lab_orchestrator.py \
    --topic-index "${TOPIC_INDEX}" \
    --max-iterations "${MAX_ITERATIONS}"

# ===========================================================================
# 11. NORMAL EXIT
# ===========================================================================

echo "Virtual Lab run complete."
echo "Job complete."