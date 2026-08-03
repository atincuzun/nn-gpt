#!/bin/bash

#SBATCH --job-name=moe-gate-showcase
#SBATCH --output=gslurm-moe-gate-showcase-%j.out
#SBATCH --error=gslurm-moe-gate-showcase-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1,tmp:100G
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=standard
#SBATCH --nodelist=jn004

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-/data/42-julia-hpc-ai-cv-students/s497179/nn-gpt-moe-gate-experiment}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual}}"
RUN_DATETIME="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/outputs/moe_gate_showcase_${RUN_TAG}}"
LOCAL_ROOT="${TMPDIR:-/tmp}/nngpt_moe_gate_${RUN_TAG}"
VENV_DIR="${LOCAL_ROOT}/venv"
HF_HOME="${HF_HOME:-${DATA_ROOT}/huggingface}"
LOG_DIR="${PROJECT_DIR}/logs/moe_gate_showcase"
LOG_FILE="${LOG_DIR}/moe_gate_showcase_${RUN_DATETIME}_${RUN_TAG}.log"

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}" "${LOCAL_ROOT}/pip" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "Job ID       : ${SLURM_JOB_ID:-manual}"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "Project      : ${PROJECT_DIR}"
echo "Node storage : ${LOCAL_ROOT}"
echo "Output       : ${OUTPUT_DIR}"
echo "Log          : ${LOG_FILE}"
echo "=========================================="

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --no-cache-dir --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install --no-cache-dir \
    transformers==4.46.3 \
    accelerate \
    datasets \
    pandas \
    overrides \
    nn-dataset \
    --extra-index-url https://download.pytorch.org/whl/cu130

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PIP_CACHE_DIR="${LOCAL_ROOT}/pip"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"

"${VENV_DIR}/bin/python" -c \
    "import torch, transformers; print('torch=', torch.__version__, 'cuda=', torch.version.cuda, 'available=', torch.cuda.is_available()); print('transformers=', transformers.__version__)"

"${VENV_DIR}/bin/python" train_moe_gates.py \
    --model deepseek-ai/DeepSeek-V2-Lite \
    --data-source nngenprompt \
    --prompt-config ab/gpt/conf/prompt/train/NN_gen.json \
    --gate low_rank \
    --max-prompts 8 \
    --max-length 4096 \
    --max-new-tokens 4096 \
    --validation-fraction 0.25 \
    --validation-steps 1 \
    --steps 2 \
    --batch-size 1 \
    --learning-rate 1e-3 \
    --dtype bfloat16 \
    --device-map auto \
    --gradient-checkpointing \
    --output "${OUTPUT_DIR}"

echo "Showcase completed: ${OUTPUT_DIR}"
