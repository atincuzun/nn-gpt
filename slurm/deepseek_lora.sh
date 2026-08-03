#!/bin/bash

#SBATCH --job-name=deepseek-nngpt-lora
#SBATCH --output=gslurm-deepseek-lora-%j.out
#SBATCH --error=gslurm-deepseek-lora-%j.out
#SBATCH --account=computervision
#SBATCH --qos=computervision
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2,tmp:100G
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_computervision

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/data/42-julia-hpc-ai-cv-students/s497179/nn-gpt-moe-gate-experiment}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual}}"
RUN_DATETIME="$(date +%Y%m%d_%H%M%S)"
LOCAL_ROOT="${TMPDIR:-/tmp}/nngpt_moe_gate_${RUN_TAG}"
VENV_DIR="${LOCAL_ROOT}/venv"
HF_HOME="${HF_HOME:-${DATA_ROOT}/huggingface}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/outputs/deepseek_v2_lite_chat_lora_${RUN_DATETIME}_${RUN_TAG}}"
LOG_DIR="${PROJECT_DIR}/logs/deepseek_lora"
LOG_FILE="${LOG_DIR}/deepseek_lora_${RUN_DATETIME}_${RUN_TAG}.log"

mkdir -p "${LOCAL_ROOT}" "${HF_HOME}" "${OUTPUT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

du -sh /home/s497179 "${DATA_ROOT}" || true
df -h /home /data "${LOCAL_ROOT}"

echo "Node   : $(hostname)"
echo "Output : ${OUTPUT_DIR}"
echo "Log    : ${LOG_FILE}"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    python3 -m venv "${VENV_DIR}"
fi
export PATH="${VENV_DIR}/bin:${PATH}"
"${VENV_DIR}/bin/python" -m pip install --no-cache-dir \
    transformers==4.46.3 accelerate datasets pandas overrides nn-dataset peft==0.14.0 \
    --extra-index-url https://download.pytorch.org/whl/cu130

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"
"${VENV_DIR}/bin/python" finetune_deepseek_lora.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --output "${OUTPUT_DIR}" \
    --steps "${LORA_STEPS:-20}" \
    --gradient-accumulation "${GRADIENT_ACCUMULATION:-4}" \
    --max-prompts "${MAX_PROMPTS:-128}" \
    --max-length "${MAX_LENGTH:-4096}" \
    --batch-size 1 \
    --rank "${LORA_RANK:-16}" \
    --alpha "${LORA_ALPHA:-32}" \
    --learning-rate "${LEARNING_RATE:-2e-4}" \
    --device-map auto
