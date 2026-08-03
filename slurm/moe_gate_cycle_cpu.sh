#!/bin/bash

#SBATCH --job-name=moe-gate-cpu
#SBATCH --output=slurm-moe-gate-cpu-%j.out
#SBATCH --error=slurm-moe-gate-cpu-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=tmp:100G
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu_standard
#SBATCH --account=computervision
#SBATCH --qos=computervision

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/data/42-julia-hpc-ai-cv-students/s497179/nn-gpt-moe-gate-experiment}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID}}"
LOCAL_ROOT="${TMPDIR:-/tmp}/nngpt_moe_gate_cpu_${RUN_TAG}"
VENV_DIR="${LOCAL_ROOT}/venv"
HF_HOME="${HF_HOME:-${DATA_ROOT}/huggingface}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/outputs/moe_gate_cycle_cpu_${RUN_TAG}}"

mkdir -p "${LOCAL_ROOT}" "${HF_HOME}" "${OUTPUT_DIR}"

export HF_HOME
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
    transformers==4.46.3 accelerate datasets pandas overrides nn-dataset \
    peft==0.14.0 trl==0.12.2 deepspeed==0.18.3 pytest tqdm \
    --extra-index-url https://download.pytorch.org/whl/cpu

cd "${PROJECT_DIR}"

if [ "${RUN_GATE_TESTS:-0}" = "1" ]; then
    "${VENV_DIR}/bin/python" -m pytest \
        test/test_gate_morphism.py test/test_nngpt_data.py -q
fi

CYCLE_CMD=(
    "${VENV_DIR}/bin/python" run_moe_gate_cycle.py
    --model "${MODEL:-deepseek-ai/DeepSeek-V2-Lite-Chat}"
    --output "${OUTPUT_DIR}"
    --device-map none
    --dtype "${DTYPE:-bfloat16}"
    --gate-mode "${GATE_MODE:-direct}"
    --gate-implementation "${GATE_IMPLEMENTATION:-svd_signed_pair_silu}"
    --epochs "${EPOCHS:-1}"
    --test-nn "${TEST_NN:-1}"
    --nn-train-epochs "${NN_TRAIN_EPOCHS:-1}"
    --gate-train-steps "${GATE_TRAIN_STEPS:-1}"
    --gate-learning-rate "${GATE_LR:-1e-4}"
    --max-prompts "${MAX_PROMPTS:-2}"
    --max-length "${MAX_LENGTH:-512}"
    --generation-max-new-tokens "${GENERATION_MAX_NEW_TOKENS:-4096}"
    --generation-backend "${GENERATION_BACKEND:-pipeline}"
    --temperature "${TEMPERATURE:-1.0}"
    --top-k "${TOP_K:-50}"
    --top-p "${TOP_P:-0.9}"
    --batch-size 1
    --validation-steps "${VALIDATION_STEPS:-1}"
)

if [ -n "${GATE_LAYERS:-}" ]; then
    read -r -a LAYER_ARGS <<< "${GATE_LAYERS}"
    CYCLE_CMD+=(--layers "${LAYER_ARGS[@]}")
fi
if [ -n "${GENERATION_MAX_INPUT_LENGTH:-}" ]; then
    CYCLE_CMD+=(--generation-max-input-length "${GENERATION_MAX_INPUT_LENGTH}")
fi
if [ "${LOCAL_FILES_ONLY:-0}" = "1" ]; then
    CYCLE_CMD+=(--local-files-only)
fi
if [ "${VERIFY_MORPHISM_ONLY:-0}" = "1" ]; then
    CYCLE_CMD+=(--verify-morphism-only)
fi
if [ "${PROGRESSIVE_UNFREEZE_DESCENDING:-0}" = "1" ]; then
    CYCLE_CMD+=(--progressive-unfreeze-descending)
fi
if [ "${GRADIENT_CHECKPOINTING:-0}" = "1" ]; then
    CYCLE_CMD+=(--gradient-checkpointing)
fi
if [ "${FIXED_EVALUATION_PROMPTS:-0}" = "1" ]; then
    CYCLE_CMD+=(--fixed-evaluation-prompts)
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Output: ${OUTPUT_DIR}"
echo "Model: ${MODEL:-deepseek-ai/DeepSeek-V2-Lite-Chat}"
echo "CPU threads: ${SLURM_CPUS_PER_TASK}"
"${CYCLE_CMD[@]}"
