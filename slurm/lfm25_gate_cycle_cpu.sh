#!/bin/bash

#SBATCH --job-name=lfm25-gate-cycle
#SBATCH --output=slurm-lfm25-gate-cycle-%j.out
#SBATCH --error=slurm-lfm25-gate-cycle-%j.out
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
LOCAL_ROOT="${TMPDIR:-/tmp}/lfm25_gate_cycle_${SLURM_JOB_ID}"
VENV_DIR="${LOCAL_ROOT}/venv"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/outputs/lfm25_gate_cycle_${SLURM_JOB_ID}}"

: "${ADAPTER:?ADAPTER must point to a completed LFM2.5 LoRA adapter}"
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
export NNGPT_STRIP_THINK_OUTPUT=1

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --no-cache-dir --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install --no-cache-dir \
    torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cpu
"${VENV_DIR}/bin/python" -m pip install --no-cache-dir \
    transformers==5.9.0 accelerate datasets pandas overrides nn-dataset \
    peft==0.20.0 trl==1.9.2 deepspeed==0.18.3 pytest tqdm \
    --extra-index-url https://download.pytorch.org/whl/cpu

cd "${PROJECT_DIR}"
"${VENV_DIR}/bin/python" -m pytest \
    test/test_gate_morphism.py test/test_nngpt_data.py -q
"${VENV_DIR}/bin/python" run_moe_gate_cycle.py \
    --model "${MODEL:-LiquidAI/LFM2.5-8B-A1B}" \
    --adapter "${ADAPTER}" \
    --output "${OUTPUT_DIR}" \
    --device-map none \
    --dtype bfloat16 \
    --local-files-only \
    --gate-mode direct \
    --gate-implementation exact_residual_mlp \
    --layers 23 \
    --epochs "${EPOCHS:-4}" \
    --test-nn "${TEST_NN:-20}" \
    --nn-train-epochs "${NN_TRAIN_EPOCHS:-5}" \
    --gate-train-steps "${GATE_TRAIN_STEPS:-20}" \
    --gate-learning-rate "${GATE_LR:-5e-6}" \
    --max-prompts "${MAX_PROMPTS:-128}" \
    --max-length "${MAX_LENGTH:-4096}" \
    --generation-max-new-tokens "${GENERATION_MAX_NEW_TOKENS:-8192}" \
    --generation-backend pipeline \
    --temperature 0.2 \
    --top-k 80 \
    --top-p 0.9 \
    --repetition-penalty 1.05 \
    --batch-size 1 \
    --validation-steps 5 \
    --fixed-evaluation-prompts
