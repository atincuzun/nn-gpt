#!/bin/bash

#SBATCH --job-name=moe-gate-cycle
#SBATCH --output=gslurm-moe-gate-cycle-%j.out
#SBATCH --error=gslurm-moe-gate-cycle-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2,tmp:100G
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_computervision
#SBATCH --account=computervision
#SBATCH --qos=computervision

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/data/42-julia-hpc-ai-cv-students/s497179/nn-gpt-moe-gate-experiment}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual}}"
RUN_DATETIME="$(date +%Y%m%d_%H%M%S)"
LOCAL_ROOT="${TMPDIR:-/tmp}/nngpt_moe_gate_${RUN_TAG}"
VENV_DIR="${LOCAL_ROOT}/venv"
HF_HOME="${HF_HOME:-${DATA_ROOT}/huggingface}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/outputs/moe_gate_cycle_${RUN_DATETIME}_${RUN_TAG}}"
LOG_DIR="${PROJECT_DIR}/logs/moe_gate_cycle"
LOG_FILE="${LOG_DIR}/moe_gate_cycle_${RUN_DATETIME}_${RUN_TAG}.log"

mkdir -p "${LOCAL_ROOT}" "${HF_HOME}" "${OUTPUT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

du -sh /home/s497179 "${DATA_ROOT}" || true
df -h /home /data "${LOCAL_ROOT}"

echo "=========================================="
echo "Job ID       : ${SLURM_JOB_ID:-manual}"
echo "Node         : $(hostname)"
echo "GPU(s)       : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tr '\n' ';')"
echo "Output       : ${OUTPUT_DIR}"
echo "Log          : ${LOG_FILE}"
echo "=========================================="

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    python3 -m venv "${VENV_DIR}"
fi
export PATH="${VENV_DIR}/bin:${PATH}"
"${VENV_DIR}/bin/python" -m pip install --no-cache-dir --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install --no-cache-dir \
    transformers==4.46.3 accelerate datasets pandas overrides nn-dataset \
    peft==0.14.0 trl==0.12.2 deepspeed==0.18.3 pytest tqdm \
    --extra-index-url https://download.pytorch.org/whl/cu130

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

MODEL_ID="${MODEL:-deepseek-ai/DeepSeek-V2-Lite-Chat}"
MODEL_PATH="${MODEL_ID}"
if [ "${STAGE_MODEL_TO_TMP:-1}" = "1" ] && [[ "${MODEL_ID}" != /* ]]; then
    export MODEL_ID
    SNAPSHOT_DIR="$("${VENV_DIR}/bin/python" -c 'import os; from huggingface_hub import snapshot_download; print(snapshot_download(os.environ["MODEL_ID"]))')"
    LOCAL_MODEL_DIR="${LOCAL_ROOT}/model"
    echo "Staging model from ${SNAPSHOT_DIR} to ${LOCAL_MODEL_DIR}"
    df -h "${LOCAL_ROOT}"
    mkdir -p "${LOCAL_MODEL_DIR}"
    cp -aL "${SNAPSHOT_DIR}/." "${LOCAL_MODEL_DIR}/"
    MODEL_PATH="${LOCAL_MODEL_DIR}"
fi

cd "${PROJECT_DIR}"
if [ "${VERIFY_MORPHISM_ONLY:-0}" = "1" ]; then
    "${VENV_DIR}/bin/python" -m pytest test/test_gate_morphism.py -q
fi
CYCLE_CMD=(
    "${VENV_DIR}/bin/python" run_moe_gate_cycle.py
    --model "${MODEL_PATH}"
    --output "${OUTPUT_DIR}"
    --test-nn "${TEST_NN:-10}"
    --nn-train-epochs "${NN_TRAIN_EPOCHS:-3}"
    --gate-train-steps "${GATE_TRAIN_STEPS:-50}"
    --gate-learning-rate "${GATE_LR:-1e-4}"
    --gate-mode "${GATE_MODE:-direct}"
    --gate-implementation "${GATE_IMPLEMENTATION:-svd_signed_pair_silu}"
    --gate-init-noise-scale "${GATE_INIT_NOISE_SCALE:-0}"
    --distillation-weight "${DISTILLATION_WEIGHT:-1.0}"
    --student-weight-step "${STUDENT_WEIGHT_STEP:-0.1}"
    --handoff-mode "${HANDOFF_MODE:-guarded}"
    --max-prompts "${MAX_PROMPTS:-4096}"
    --max-length "${MAX_LENGTH:-4096}"
    --generation-max-new-tokens "${MAX_NEW_TOKENS:-16384}"
    --epochs "${EPOCHS:-5}"
    --batch-size "${BATCH_SIZE:-1}"
    --validation-fraction "${VAL_FRACTION:-0.1}"
    --validation-steps "${VAL_STEPS:-16}"
    --dtype bfloat16
    --device-map auto
    --gradient-checkpointing
)
if [ -n "${ADAPTER_PATH:-}" ]; then
    CYCLE_CMD+=(--adapter "${ADAPTER_PATH}")
fi
if [ -n "${GATE_LAYERS:-}" ]; then
    read -r -a LAYER_ARGS <<< "${GATE_LAYERS}"
    CYCLE_CMD+=(--layers "${LAYER_ARGS[@]}")
fi
if [ "${VERIFY_MORPHISM_ONLY:-0}" = "1" ]; then
    CYCLE_CMD+=(--verify-morphism-only)
fi
"${CYCLE_CMD[@]}"

echo "MoE gate cycle completed: ${OUTPUT_DIR}"
