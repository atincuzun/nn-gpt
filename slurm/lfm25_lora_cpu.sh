#!/bin/bash

#SBATCH --job-name=lfm25-nngpt-lora
#SBATCH --output=slurm-lfm25-lora-%j.out
#SBATCH --error=slurm-lfm25-lora-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=tmp:50G
#SBATCH --time=1-00:00:00
#SBATCH --partition=cpu_standard
#SBATCH --account=computervision
#SBATCH --qos=computervision

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/data/42-julia-hpc-ai-cv-students/s497179/nn-gpt-moe-gate-experiment}"
LOCAL_ROOT="${TMPDIR:-/tmp}/lfm25_lora_${SLURM_JOB_ID}"
VENV_DIR="${LOCAL_ROOT}/venv"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-lfm25_nngpt_lora}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/outputs/${OUTPUT_PREFIX}_${SLURM_JOB_ID}}"

mkdir -p "${LOCAL_ROOT}" "${DATA_ROOT}/huggingface" "${OUTPUT_DIR}"

export HF_HOME="${DATA_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --no-cache-dir --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install --no-cache-dir \
    torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cpu
"${VENV_DIR}/bin/python" -m pip install --no-cache-dir \
    transformers==5.9.0 accelerate datasets pandas overrides nn-dataset \
    peft==0.20.0 pytest tqdm \
    --extra-index-url https://download.pytorch.org/whl/cpu

cd "${PROJECT_DIR}"
"${VENV_DIR}/bin/python" -m pytest test/test_nngpt_data.py -q
"${VENV_DIR}/bin/python" finetune_deepseek_lora.py \
    --model "${MODEL:-LiquidAI/LFM2.5-8B-A1B}" \
    --output "${OUTPUT_DIR}" \
    --prompt-config "${PROMPT_CONFIG:-ab/gpt/conf/prompt/train/NN_gen.json}" \
    --device-map none \
    --local-files-only \
    --steps "${LORA_STEPS:-20}" \
    --learning-rate "${LORA_LR:-1e-4}" \
    --rank "${LORA_RANK:-16}" \
    --alpha "${LORA_ALPHA:-32}" \
    --dropout "${LORA_DROPOUT:-0.05}" \
    --batch-size 1 \
    --gradient-accumulation "${GRADIENT_ACCUMULATION:-4}" \
    --max-length "${MAX_LENGTH:-4096}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-4096}" \
    --max-prompts "${MAX_PROMPTS:-128}" \
    --validation-steps "${VALIDATION_STEPS:-5}"
