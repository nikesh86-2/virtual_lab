#!/bin/bash
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

set -euo pipefail

# ===========================================================================
# 0. ROOT PATHS
# ===========================================================================

BASE_DIR="/scratch/fbsnpat/bot/VLAB2"
PROJECT_DIR="${BASE_DIR}"
SRC_ENV="/mnt/scratch/fbsnpat/envs/biophysics-research-agent-new"
PY="${SRC_ENV}/bin/python3"
HF_SCRATCH="${BASE_DIR}/hf_cache"

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="/scratch/fbsnpat/bot:${PYTHONPATH:-}"

mkdir -p \
    "${BASE_DIR}/logs" \
    "${BASE_DIR}/output_data/logs" \
    "${BASE_DIR}/output_data/debug_rna_prep" \
    "${BASE_DIR}/output_data/debug_hdock" \
    "${BASE_DIR}/output_data/hdock_work" \
    "${BASE_DIR}/output_data/rna_hdock_cache" \
    "${BASE_DIR}/output_data/docking_snapshots" \
    "${BASE_DIR}/output_data/docking_contacts" \
    "${BASE_DIR}/output_data/vina_result_cache" \
    "${BASE_DIR}/pdb_cache" \
    "${HF_SCRATCH}" \
    "${BASE_DIR}/tmp"

cd "${PROJECT_DIR}"

echo "Project: ${PROJECT_DIR}"
echo "Scratch root: ${BASE_DIR}"
echo "Node: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}"
echo "PYTHONPATH=${PYTHONPATH}"

# Load credentials and non-critical defaults first. Runtime-critical values are
# deliberately re-exported later so stale .env values cannot override them.
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
# 1. MODULES AND CONDA
# ===========================================================================

module load cuda/12.6.2
module load miniforge

set +u
conda activate "${SRC_ENV}"
set -u

echo "===== MODULES ====="
module list 2>&1 || true
echo "==================="

if [ ! -x "${PY}" ]; then
    echo "ERROR: Python is not executable: ${PY}"
    exit 1
fi

# ===========================================================================
# 2. COVERAGE SETUP
# ===========================================================================

"${PY}" -m coverage --version >/dev/null 2>&1 || "${PY}" -m pip install coverage

# Per-job coverage avoids historical runs making files look active. Use a
# separate cumulative workflow if aggregate coverage across jobs is required.
COVERAGE_DATA_DIR="${BASE_DIR}/output_data/coverage/${SLURM_JOB_ID:-manual}"
mkdir -p "${COVERAGE_DATA_DIR}"

cat > "${PROJECT_DIR}/.coveragerc" <<EOF
[run]
data_file = ${COVERAGE_DATA_DIR}/.coverage
parallel = True
branch = True
source = ${PROJECT_DIR}
omit =
    */output_data/*
    */hf_cache/*
    */conda_pkgs/*
    */tmp/*
    */.venv/*
    */site-packages/*
    */HDOCKlite*/*
    ${PROJECT_DIR}/.coveragerc

[report]
show_missing = True
skip_covered = False
precision = 1
EOF

echo "===== COVERAGE CONFIG ====="
cat "${PROJECT_DIR}/.coveragerc"
echo "COVERAGE_DATA_DIR=${COVERAGE_DATA_DIR}"
echo "============================"

# ===========================================================================
# 3. CACHE AND TEMP PATHS
# ===========================================================================

if [ -n "${TMP_SHARED:-}" ] && [ -d "${TMP_SHARED}" ]; then
    echo "Using NVMe for hot I/O: ${TMP_SHARED}"
    export PYTHONPYCACHEPREFIX="${TMP_SHARED}/pycache"
    export TMPDIR="${TMP_SHARED}/tmp"
    export TRITON_CACHE_DIR="${TMP_SHARED}/triton_cache"
    export TORCHINDUCTOR_CACHE_DIR="${TMP_SHARED}/torchinductor_cache"
else
    echo "No NVMe detected; using scratch only"
    export PYTHONPYCACHEPREFIX="${BASE_DIR}/pycache"
    export TMPDIR="${BASE_DIR}/tmp"
    export TRITON_CACHE_DIR="${BASE_DIR}/triton_cache"
    export TORCHINDUCTOR_CACHE_DIR="${BASE_DIR}/torchinductor_cache"
fi

export HF_HOME="${HF_SCRATCH}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export XDG_CACHE_HOME="${HF_HOME}/xdg"
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
# 4. PYTHON AND GPU DIAGNOSTICS
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
    for index in range(torch.cuda.device_count()):
        print(index, torch.cuda.get_device_name(index))
PY

echo "==========================="

# ===========================================================================
# 5. OPENBABEL, HDOCK, VINA, AND RENDERING
# ===========================================================================

export OBABEL_ENV="/mnt/scratch/fbsnpat/envs/biophysics-research-agent"
export OBABEL_BIN="${OBABEL_ENV}/bin/obabel"
export RNA_PREP_DEBUG_DIR="${BASE_DIR}/output_data/debug_rna_prep"

if [ ! -x "${OBABEL_BIN}" ]; then
    echo "ERROR: OBABEL_BIN is not executable: ${OBABEL_BIN}"
    find /users/fbsnpat/.conda/envs /mnt/scratch/fbsnpat/envs \
        -path "*/bin/obabel" -type f 2>/dev/null || true
    exit 1
fi

echo "===== OPENBABEL DIAGNOSTICS ====="
echo "OBABEL_ENV=${OBABEL_ENV}"
echo "OBABEL_BIN=${OBABEL_BIN}"
env LD_LIBRARY_PATH="${OBABEL_ENV}/lib:${LD_LIBRARY_PATH:-}" "${OBABEL_BIN}" -V
env LD_LIBRARY_PATH="${OBABEL_ENV}/lib:${LD_LIBRARY_PATH:-}" \
    "${OBABEL_BIN}" -L formats | grep -i pdb || {
        echo "ERROR: OpenBabel cannot see PDB format plugin"
        exit 1
    }
echo "================================="

# RNA-protein HDOCK
export VLAB_DOCKING_BACKEND=hdock
export VLAB_REQUIRE_DOCKING=1
export HDOCK_HOME="${BASE_DIR}/HDOCKlite-v1.1"
export HDOCK_BIN="${BASE_DIR}/bin/hdock_env.sh"
export CREATEPL_BIN="${BASE_DIR}/bin/createpl_env.sh"
export HDOCK_TIMEOUT=420
export HDOCK_N_MODELS=5
export CREATEPL_TIMEOUT=120
export HDOCK_DEBUG_DIR="${BASE_DIR}/output_data/debug_hdock"
export HDOCK_WORK_DIR="${BASE_DIR}/output_data/hdock_work"
export PROTEIN_CACHE_DIR="${BASE_DIR}/pdb_cache"
export RNA_HDOCK_CACHE_DIR="${BASE_DIR}/output_data/rna_hdock_cache"

for binary in "${HDOCK_BIN}" "${CREATEPL_BIN}"; do
    if [ ! -x "${binary}" ]; then
        echo "ERROR: required docking wrapper is not executable: ${binary}"
        exit 1
    fi
done

# AutoDock Vina executable and compatibility aliases
export VLAB_REQUIRE_VINA=0
export VINA_BIN="${SRC_ENV}/bin/vina"
export VLAB_VINA_BIN="${VINA_BIN}"
export VINA_TIMEOUT=300
export VLAB_VINA_TIMEOUT="${VINA_TIMEOUT}"
export VLAB_VINA_EXHAUSTIVENESS=8
export VINA_EXHAUSTIVENESS="${VLAB_VINA_EXHAUSTIVENESS}"
export VLAB_VINA_SEED=1

# Persistent Vina result cache
export VLAB_VINA_CACHE_ENABLED=1
export VLAB_VINA_FORCE_REDOCK=0
export VLAB_VINA_CACHE_VERSION="vina-cache-v1"
export VLAB_VINA_RESULT_CACHE_DIR="${BASE_DIR}/output_data/vina_result_cache"
mkdir -p "${VLAB_VINA_RESULT_CACHE_DIR}"

if [ ! -x "${VINA_BIN}" ]; then
    echo "ERROR: Vina binary is not executable: ${VINA_BIN}"
    exit 1
fi

"${VINA_BIN}" --version || {
    echo "ERROR: Vina preflight failed"
    exit 1
}

# PyMOL rendering
export PYMOL_BIN="${BASE_DIR}/bin/pymol_render_env.sh"
export VLAB_RENDER_DOCKING_SNAPSHOTS=1
export VLAB_DOCKING_SNAPSHOT_DIR="${BASE_DIR}/output_data/docking_snapshots"
export VLAB_RENDER_TIMEOUT=180
mkdir -p "${VLAB_DOCKING_SNAPSHOT_DIR}"

if [ -x "${PYMOL_BIN}" ]; then
    echo "PyMOL render preflight:"
    "${PYMOL_BIN}" -cq -d "quit" || {
        echo "WARNING: PyMOL render preflight failed; snapshots may fail."
    }
else
    echo "WARNING: PyMOL wrapper is not executable: ${PYMOL_BIN}"
fi

# ===========================================================================
# 6. SCIENTIFIC AND PIPELINE CONTROLS
# ===========================================================================

# RNA candidate evaluation
export VLAB_AGENT_EVAL_TOP_N=3
export VLAB_MAX_DOCKINGS_PER_TARGET=3
export VLAB_ABORT_TARGET_ON_FIRST_TIMEOUT=1
export VLAB_MIN_VALID_HDOCK_SCORE=-30
export VLAB_MIN_VALID_DOCKINGS_PER_TARGET=2
export VLAB_REJECT_TARGET_ON_SCORE_SPREAD=1
export VLAB_MAX_TARGET_SCORE_SPREAD=25
export VLAB_MAX_RECEPTOR_ATOMS=60000
export VLAB_TARGET_BLACKLIST="5TC1,8PNF"

# Interface analysis and target acceptance
export VLAB_ANALYSE_INTERFACE_CONTACTS=1
export VLAB_INTERFACE_CONTACT_DIR="${BASE_DIR}/output_data/docking_contacts"
export VLAB_INTERFACE_CONTACT_CUTOFF=5.0
export VLAB_MIN_INTERFACE_RESIDUE_CONTACTS=5
export VLAB_MIN_INTERFACE_BASIC_CONTACTS=1
export VLAB_REQUIRE_INTERFACE_FOR_TARGET_ACCEPTANCE=1
export VLAB_ALLOW_PARTIAL_INTERFACE_TARGET=1
export VLAB_MIN_PARTIAL_INTERFACE_CLEAN=1
mkdir -p "${VLAB_INTERFACE_CONTACT_DIR}"

# Convergence
export VLAB_REQUIRE_CURRENT_TARGET_FOR_CONVERGENCE=1
export VLAB_REQUIRE_ACCEPTED_TARGET_FOR_CONVERGENCE=1
export VLAB_REQUIRE_INTERFACE_FOR_CONVERGENCE=1
export VLAB_MIN_VALID_DOCKINGS_FOR_CONVERGENCE=2
export VLAB_MIN_CLEAN_INTERFACES_FOR_CONVERGENCE=2
export VLAB_ACCEPT_BINDING_SCORE=-50
export VLAB_CONVERGENCE_SPREAD_THRESHOLD=5.0
export VLAB_MAX_ACCEPT_SCORE_SPREAD="${VLAB_CONVERGENCE_SPREAD_THRESHOLD}"
export VLAB_ALLOW_CONVERGENCE_ON_REVISE="${VLAB_ALLOW_CONVERGENCE_ON_REVISE:-0}"

# Fold and motif settings
export VLAB_RNA_ENFORCE_MIN_FOLD=1
export VLAB_RNA_MIN_ACCEPT_PAIR_DENSITY=0.24
export VLAB_RNA_MIN_ACCEPT_MFE_PER_NT=-0.08
export VLAB_RNA_REQUIRE_TARGET_MOTIF=0
export VLAB_RNA_TARGET_MOTIFS="RGAG:1.0,RAAG:1.0,GAG:0.45,AAG:0.45,CAG:0.25,GUC:0.25,AGU:0.25"

# Bioinformatics quality and RNAalifold policy
export VLAB_BIOINFO_FETCH_LIMIT=20
export VLAB_BIOINFO_MIN_MSA_SEQUENCES=3
export VLAB_BIOINFO_MIN_ALIGNMENT_LENGTH=20
export VLAB_REQUIRE_CONSERVATION_FOR_CONVERGENCE="${VLAB_REQUIRE_CONSERVATION_FOR_CONVERGENCE:-1}"
export VLAB_MIN_CONSERVATION_FOR_ACCEPT="${VLAB_MIN_CONSERVATION_FOR_ACCEPT:-0.2}"
export VLAB_ALIFOLD_MIN_CONSERVATION_FITNESS=0.5
export VLAB_ALIFOLD_ADVISORY_WHEN_LOW_CONSERVATION=1

# Inhibitor screening
export VLAB_INHIBITOR_ENABLED=1
export VLAB_ALLOW_INHIBITOR_ON_PARTIAL_TARGET=1
export VLAB_INHIBITOR_MAX_SMALL_MOLECULES=10
export VLAB_INHIBITOR_MAX_PEPTIDES=5
export VLAB_PUBCHEM_QUERIES="ribavirin|remdesivir"

# LoRA controls
export VLAB_LORA_TRAIN=0
export VLAB_MIN_TRAIN_ROWS=20
export VLAB_MIN_TRAIN_BYTES=5000
export VLAB_MIN_INTERFACE_VALID=5
export VLAB_MIN_LITERATURE_EVIDENCE=2
export VLAB_AUTO_MODEL_REPLACE=0
export VLAB_MAX_MODEL_VERSIONS=3

export VLAB_STREAM_WARMUP_SECONDS="${VLAB_STREAM_WARMUP_SECONDS:-6}"

# ===========================================================================
# 7. ENVIRONMENT SUMMARY AND IMPORT PREFLIGHT
# ===========================================================================

echo "===== PIPELINE ENV ====="
echo "VLAB_AGENT_EVAL_TOP_N=${VLAB_AGENT_EVAL_TOP_N}"
echo "VLAB_MAX_DOCKINGS_PER_TARGET=${VLAB_MAX_DOCKINGS_PER_TARGET}"
echo "VLAB_MIN_VALID_DOCKINGS_PER_TARGET=${VLAB_MIN_VALID_DOCKINGS_PER_TARGET}"
echo "VLAB_MIN_VALID_DOCKINGS_FOR_CONVERGENCE=${VLAB_MIN_VALID_DOCKINGS_FOR_CONVERGENCE}"
echo "VLAB_MIN_CLEAN_INTERFACES_FOR_CONVERGENCE=${VLAB_MIN_CLEAN_INTERFACES_FOR_CONVERGENCE}"
echo "VLAB_REQUIRE_ACCEPTED_TARGET_FOR_CONVERGENCE=${VLAB_REQUIRE_ACCEPTED_TARGET_FOR_CONVERGENCE}"
echo "VLAB_REQUIRE_CONSERVATION_FOR_CONVERGENCE=${VLAB_REQUIRE_CONSERVATION_FOR_CONVERGENCE}"
echo "VINA_BIN=${VINA_BIN}"
echo "VINA_TIMEOUT=${VINA_TIMEOUT}"
echo "VLAB_VINA_EXHAUSTIVENESS=${VLAB_VINA_EXHAUSTIVENESS}"
echo "VLAB_VINA_SEED=${VLAB_VINA_SEED}"
echo "VLAB_VINA_CACHE_ENABLED=${VLAB_VINA_CACHE_ENABLED}"
echo "VLAB_VINA_RESULT_CACHE_DIR=${VLAB_VINA_RESULT_CACHE_DIR}"
echo "========================"

# Compile critical edited modules before allocating the vLLM server.
CRITICAL_MODULES=(
    "orchestration/agents/structural_agent.py"
    "orchestration/agents/protein_agent.py"
    "orchestration/agents/bioinfo_agent.py"
    "orchestration/agents/skeptic_agent.py"
    "orchestration/agents/inhibitor_agent.py"
    "orchestration/routing.py"
    "orchestration/state_schema.py"
    "orchestration/state_factory.py"
    "core/inhibitor_docking.py"
    "core/vina_wrapper.py"
)

for module_path in "${CRITICAL_MODULES[@]}"; do
    if [ -f "${PROJECT_DIR}/${module_path}" ]; then
        "${PY}" -m py_compile "${PROJECT_DIR}/${module_path}"
    else
        echo "WARNING: critical module path not found: ${module_path}"
    fi
done

"${PY}" - <<'PY'
import sys
import torch

print(f"Python executable: {sys.executable}")
print(f"PyTorch {torch.__version__}")
print(f"CUDA build: {torch.version.cuda}")
print(f"GPU available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")

print(f"GPU: {torch.cuda.get_device_name(0)}")
import vllm
print("vLLM import OK")
PY

# ===========================================================================
# 8. vLLM CONFIGURATION AND CLEANUP
# ===========================================================================

export VLLM_HOST=127.0.0.1
export VLLM_PORT=8000
export VLLM_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
export VLLM_MODEL_PATH="${BASE_DIR}/hf_cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
export VLLM_SERVED_MODEL="Qwen/Qwen2.5-32B-Instruct"
export VLLM_MODEL="${VLLM_SERVED_MODEL}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export OPENAI_BASE_URL="${VLLM_URL}"
export OPENAI_API_BASE="${VLLM_URL}"

VLLM_LOG="${BASE_DIR}/logs/vllm_${SLURM_JOB_ID:-manual}.log"
VLLM_PID=""

cleanup() {
    echo "Cleaning up..."

    if [ -n "${VLLM_PID:-}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "Stopping vLLM PID ${VLLM_PID}"
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi

    echo "Cleanup complete."
}

trap cleanup EXIT INT TERM

HOME_CACHE_CHECK="${HOME}/.triton"
if [ -d "${HOME_CACHE_CHECK}" ]; then
    echo "Removing stale Triton cache in home directory..."
    rm -rf "${HOME_CACHE_CHECK}"/* 2>/dev/null || true
fi

export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export NCCL_ASYNC_ERROR_HANDLING=1

# ===========================================================================
# 9. START AND CHECK vLLM
# ===========================================================================

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

echo "Waiting for vLLM..."
MAX_WAIT_STEPS=180
WAIT=0

while true; do
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
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Return exactly: OK"}],
    temperature=0,
    max_tokens=8,
)
print("LLM preflight response:", response.choices[0].message.content)
PY

echo "LLM preflight complete."

# ===========================================================================
# 10. RUN ORCHESTRATOR UNDER COVERAGE
# ===========================================================================

echo "Starting Virtual Lab..."

TOPIC_INDEX="${TOPIC_INDEX:-0}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3}"

if [ "${TOPIC_INDEX}" -lt 0 ]; then
    echo "Invalid TOPIC_INDEX=${TOPIC_INDEX}; falling back to 0"
    TOPIC_INDEX=0
fi

if [ "${MAX_ITERATIONS}" -le 0 ]; then
    echo "Invalid MAX_ITERATIONS=${MAX_ITERATIONS}; falling back to 3"
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

set +e
"${PY}" -m coverage run --rcfile="${PROJECT_DIR}/.coveragerc" \
    orchestration/virtual_lab_orchestrator.py \
    --topic-index "${TOPIC_INDEX}" \
    --max-iterations "${MAX_ITERATIONS}"
ORCH_EXIT=$?
set -e

echo "Orchestrator exited with code ${ORCH_EXIT}"

# ===========================================================================
# 11. COVERAGE REPORTS, INCLUDING ZERO-COVERAGE FILES
# ===========================================================================

echo "===== COVERAGE REPORT ====="
cd "${PROJECT_DIR}"
"${PY}" -m coverage combine --rcfile="${PROJECT_DIR}/.coveragerc" || true
"${PY}" -m coverage report --rcfile="${PROJECT_DIR}/.coveragerc" -m || true
"${PY}" -m coverage html --rcfile="${PROJECT_DIR}/.coveragerc" \
    -d "${COVERAGE_DATA_DIR}/htmlcov" || true
"${PY}" -m coverage json --rcfile="${PROJECT_DIR}/.coveragerc" \
    -o "${COVERAGE_DATA_DIR}/coverage.json" || true

if [ -f "${COVERAGE_DATA_DIR}/coverage.json" ]; then
    "${PY}" - "${COVERAGE_DATA_DIR}/coverage.json" <<'PY'
import json
import sys

report_path = sys.argv[1]
with open(report_path, encoding="utf-8") as handle:
    report = json.load(handle)

zero_coverage = []
for filename, data in report.get("files", {}).items():
    summary = data.get("summary", {})
    covered = int(summary.get("covered_lines", 0) or 0)
    statements = int(summary.get("num_statements", 0) or 0)
    if statements > 0 and covered == 0:
        zero_coverage.append((filename, statements))

print("\n===== ZERO-COVERAGE PYTHON FILES =====")
for filename, statements in sorted(zero_coverage):
    print(f"{statements:5d} statements  {filename}")
print(f"Total zero-coverage files: {len(zero_coverage)}")
print("Note: zero coverage in this run does not prove a file is globally dead.")
PY
fi

echo "Coverage data: ${COVERAGE_DATA_DIR}/.coverage"
echo "Coverage JSON: ${COVERAGE_DATA_DIR}/coverage.json"
echo "Coverage HTML: ${COVERAGE_DATA_DIR}/htmlcov/index.html"
echo "==========================="
# ===========================================================================
# 12. EXIT
# ===========================================================================

echo "Virtual Lab run complete."
echo "Job complete."
exit "${ORCH_EXIT}"
