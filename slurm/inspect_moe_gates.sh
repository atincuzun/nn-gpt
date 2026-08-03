#!/bin/bash

#SBATCH --job-name=inspect-moe-gates
#SBATCH --output=gslurm-inspect-moe-gates-%j.out
#SBATCH --error=gslurm-inspect-moe-gates-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,tmp:80G
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_computervision
#SBATCH --account=computervision
#SBATCH --qos=computervision

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/data/42-julia-hpc-ai-cv-students/s497179/nn-gpt-moe-gate-experiment}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual}}"
LOCAL_ROOT="${TMPDIR:-/tmp}/inspect_moe_gates_${RUN_TAG}"
VENV_DIR="${LOCAL_ROOT}/venv"
HF_HOME="${HF_HOME:-${DATA_ROOT}/huggingface}"

mkdir -p "${LOCAL_ROOT}" "${HF_HOME}"
exec > >(tee -a "gslurm-inspect-moe-gates-${SLURM_JOB_ID:-manual}.out") 2>&1

echo "=========================================="
echo "Job ID       : ${SLURM_JOB_ID:-manual}"
echo "Node         : $(hostname)"
echo "GPU(s)       : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tr '\n' ';')"
echo "=========================================="

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    python3 -m venv "${VENV_DIR}"
fi
export PATH="${VENV_DIR}/bin:${PATH}"
"${VENV_DIR}/bin/python" -m pip install --no-cache-dir --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install --no-cache-dir \
    transformers==4.46.3 accelerate datasets pandas overrides nn-dataset \
    peft==0.14.0 trl==0.12.2 deepspeed==0.18.3 tqdm \
    --extra-index-url https://download.pytorch.org/whl/cu130

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"

"${VENV_DIR}/bin/python" inspect_moe_gates.py \
    --model "${MODEL:-deepseek-ai/DeepSeek-V2-Lite-Chat}" \
    --gate "${GATE:-linear}" \
    --dtype "${DTYPE:-bfloat16}" \
    --device-map auto \
    --prompt "${PROMPT:-What is 2+2?}" \
    ${INIT_FROM_ORIG:+--initialize-from-original} \
    ${TEACHER_STUDENT:+--teacher-student}

echo "Inspection completed."
