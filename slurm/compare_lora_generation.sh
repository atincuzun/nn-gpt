#!/bin/bash

#SBATCH --job-name=compare-lora-codegen
#SBATCH --output=gslurm-compare-codegen-%j.out
#SBATCH --error=gslurm-compare-codegen-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2,tmp:100G
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --partition=standard

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-/data/42-julia-hpc-ai-cv-students/s497179/nn-gpt-moe-gate-experiment}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual}}"
LOCAL_ROOT="${TMPDIR:-/tmp}/nngpt_compare_codegen_${RUN_TAG}"
VENV_DIR="${LOCAL_ROOT}/venv"
HF_HOME="${HF_HOME:-${DATA_ROOT}/huggingface}"
ADAPTER_DIR="${ADAPTER_DIR:?Set ADAPTER_DIR to the fine-tuned LoRA adapter directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/outputs/lora_codegen_comparison_${RUN_TAG}}"

mkdir -p "${LOCAL_ROOT}" "${HF_HOME}" "${OUTPUT_DIR}"
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --no-cache-dir \
    transformers==4.46.3 accelerate datasets pandas overrides nn-dataset peft==0.14.0 \
    --extra-index-url https://download.pytorch.org/whl/cu130

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"
"${VENV_DIR}/bin/python" compare_lora_generation.py \
    --adapter "${ADAPTER_DIR}" \
    --output "${OUTPUT_DIR}" \
    --num-prompts "${NUM_PROMPTS:-4}" \
    --max-source-prompts "${MAX_SOURCE_PROMPTS:-64}" \
    --max-length "${MAX_LENGTH:-4096}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-4096}" \
    --device-map auto
