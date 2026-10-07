#!/bin/bash
# KTPO — Cloudcast Broadcast Optimization — Qwen3-4B-Instruct-2507
# Config: 16×16, k=8, max=32768, improvement-rate eviction, 50 rollouts
#
# Usage:
#   SEED=2 nohup bash scripts_launch/Qwen3-4B-Instruct-2507/ktpo_4b_16x16x8_cloudcast.sh \
#       > logs/ktpo_4b_cloudcast_k8_32k_seed2.log 2>&1 &
# ================================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

eval "$(conda shell.bash hook)" 2>/dev/null || true

pip install git+https://github.com/fzyzcjy/torch_memory_saver.git \
    --no-cache-dir --force-reinstall --quiet 2>/dev/null || true

########################### SHARED CONFIG ###########################

export SAVE_PATH="${SAVE_PATH:-ckpts}"
export LEARNX_NUM_ROLLOUT="${LEARNX_NUM_ROLLOUT:-50}"
export LEARNX_SAVE_INTERVAL="${LEARNX_SAVE_INTERVAL:-10}"
SEED="${SEED:-1}"

MODEL_NAME="Qwen3-4B-Instruct-2507"
HF_MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
models_file_name="qwen3-4B-instruct-2507.sh"
SMALL_MODEL_NAME="qwen3_4b"

TASK="cloudcast"
IS_TRAINING=True
REWARD_PROCESS_TYPE="improve_reward"
LAZY_OUTPUT_PENALTY=1

WANDB_API_KEY="${WANDB_API_KEY:-}"   # optional; W&B logging is on only when set
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-KTPO}"

# ── Batch: 16×16 = 256 samples/rollout ──
export LEARNX_ROLLOUT_BATCH_SIZE="${LEARNX_ROLLOUT_BATCH_SIZE:-16}"

export LEARNX_N_SAMPLES_PER_PROMPT="${LEARNX_N_SAMPLES_PER_PROMPT:-16}"

export LEARNX_NUM_STEPS_PER_ROLLOUT="${LEARNX_NUM_STEPS_PER_ROLLOUT:-1}"

export LEARNX_OVER_SAMPLING_BATCH_SIZE="${LEARNX_OVER_SAMPLING_BATCH_SIZE:-16}"


# ── k=8 (8-step refinement) ──
export LEARNX_CONTEXT_HORIZON="${LEARNX_CONTEXT_HORIZON:-0}"

export LEARNX_MULTISTEP_K="${LEARNX_MULTISTEP_K:-8}"

export LEARNX_REWARD_AGG="${LEARNX_REWARD_AGG:-max_improve}"


# ── Evaluator (subprocess, high parallelism) ──
export LEARNX_EVAL_OMP_THREADS="${LEARNX_EVAL_OMP_THREADS:-1}"

export LEARNX_EVAL_PARALLEL="${LEARNX_EVAL_PARALLEL:-192}"

export LEARNX_EVAL_N_REPEAT="${LEARNX_EVAL_N_REPEAT:-1}"

export LEARNX_CLOUDCAST_CFG_TIMEOUT="${LEARNX_CLOUDCAST_CFG_TIMEOUT:-60}"

export LEARNX_CLOUDCAST_EVAL_TIMEOUT="${LEARNX_CLOUDCAST_EVAL_TIMEOUT:-120}"


# ── Parallelism: TP=8, CP=1 — 4B+26k context fits without sequence-dim sharding ──
export LEARNX_TP_SIZE="${LEARNX_TP_SIZE:-8}"

export LEARNX_CP_SIZE="${LEARNX_CP_SIZE:-1}"

export LEARNX_ACTOR_NUM_GPUS="${LEARNX_ACTOR_NUM_GPUS:-8}"

export LEARNX_ROLLOUT_NUM_GPUS="${LEARNX_ROLLOUT_NUM_GPUS:-8}"


# ── Generation ──
export LEARNX_ROLLOUT_MAX_RESPONSE_LEN="${LEARNX_ROLLOUT_MAX_RESPONSE_LEN:-32768}"


# ── Pool update: improvement-rate eviction ──
export LEARNX_ACDS_RETIRE_THR="${LEARNX_ACDS_RETIRE_THR:-0.30}"

export LEARNX_ACDS_RETIRE_THR_END="${LEARNX_ACDS_RETIRE_THR_END:-0.00}"

export LEARNX_ACDS_MASTERED_WINDOW="${LEARNX_ACDS_MASTERED_WINDOW:-10}"


# ── Safety ──
export LEARNX_MAX_ERR_CHARS="${LEARNX_MAX_ERR_CHARS:-10000}"

export LEARNX_MAX_FEEDBACK_TOKENS="${LEARNX_MAX_FEEDBACK_TOKENS:-2048}"


# ── Performance ──
export LEARNX_MAX_TOKENS_PER_GPU="${LEARNX_MAX_TOKENS_PER_GPU:-4096}"

export LEARNX_SGLANG_MEM_FRACTION="${LEARNX_SGLANG_MEM_FRACTION:-0.8}"


########################### DERIVED PATHS ###########################

INITIAL_PROGRAM="openevolve_adapted/examples/${TASK}/initial_programs/initial_program.py"
EVALUATOR_FILE="openevolve_adapted/examples/${TASK}/evaluators/evaluator_modular.py"
CONFIG_YAML="openevolve_adapted/examples/${TASK}/configs/config_${TASK}.yaml"

_AGG_SHORT="${LEARNX_REWARD_AGG//_/}"
RUN_NAME="${SMALL_MODEL_NAME}_ktpo_${TASK}_16x16x8_32k_${_AGG_SHORT}_seed${SEED}_r${LEARNX_NUM_ROLLOUT}"

########################### ENVIRONMENT SETUP #######################

mkdir -p $SAVE_PATH/{tmp,hf,wandb,shm} logs
mkdir -p /tmp/triton

export TMPDIR=/tmp
export HF_HOME=$SAVE_PATH/hf
export HUGGINGFACE_HUB_CACHE=$SAVE_PATH/hf/hub
export TRANSFORMERS_CACHE=$SAVE_PATH/hf/hub
export HF_DATASETS_CACHE=$SAVE_PATH/hf/datasets
export SAVE_SHM_DIR=$SAVE_PATH/shm
export TRITON_CACHE_DIR=/tmp/triton
export WANDB_CACHE_DIR=$SAVE_PATH/wandb
export WANDB_DIR=$SAVE_PATH/wandb
export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_ENTITY=$WANDB_ENTITY
export WANDB_PROJECT=$WANDB_PROJECT

########################### MODEL SETUP #############################

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  KTPO — Cloudcast 4B — 16×16×k8, 32k, improvement-rate eviction, 50r           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Seed     : ${SEED}"
echo "║  Rollouts : ${LEARNX_NUM_ROLLOUT}"
echo "║  Batch    : ${LEARNX_ROLLOUT_BATCH_SIZE}×${LEARNX_N_SAMPLES_PER_PROMPT}×${LEARNX_MULTISTEP_K} = $((LEARNX_ROLLOUT_BATCH_SIZE*LEARNX_N_SAMPLES_PER_PROMPT*LEARNX_MULTISTEP_K)) evals/rollout"
echo "║  MaxLen   : 32768"
echo "║  Eval     : 192 parallel, 60s per-config timeout"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

FORCE_DOWNLOAD=0
if [ -d "$SAVE_SHM_DIR/$MODEL_NAME" ] && [ -f "$SAVE_SHM_DIR/$MODEL_NAME/config.json" ] && [ $FORCE_DOWNLOAD -eq 0 ]; then
    echo "Model already in shm, skipping download"
else
    [ -d "$SAVE_SHM_DIR/$MODEL_NAME" ] && rm -rf "$SAVE_SHM_DIR/$MODEL_NAME"
    echo "Downloading $HF_MODEL_ID..."
    huggingface-cli download $HF_MODEL_ID --local-dir ./$MODEL_NAME 2>/dev/null \
        || hf download $HF_MODEL_ID --local-dir ./$MODEL_NAME
    mkdir -p $SAVE_SHM_DIR
    cp -r $MODEL_NAME $SAVE_SHM_DIR/
fi

source scripts_launch/models/${models_file_name}
if [ ! -d "$SAVE_SHM_DIR/${MODEL_NAME}_torch_dist" ] || [ $FORCE_DOWNLOAD -eq 1 ]; then
    echo "Converting to torch_dist..."
    PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint $SAVE_SHM_DIR/$MODEL_NAME \
        --save $SAVE_SHM_DIR/${MODEL_NAME}_torch_dist
fi

########################### RUN ######################################

CKPT_DIR="${SAVE_PATH}/${RUN_NAME}"
LOGFILE="${CKPT_DIR}/train_log.txt"
mkdir -p "${CKPT_DIR}"

echo "Starting: ${RUN_NAME}"
echo "Config  : ${CONFIG_YAML}"

bash "${SCRIPT_DIR}/_general.sh" \
    "${WANDB_PROJECT}" "${RUN_NAME}" \
    "${INITIAL_PROGRAM}" "${EVALUATOR_FILE}" "${CONFIG_YAML}" \
    "${SAVE_PATH}" "${IS_TRAINING}" "${LAZY_OUTPUT_PENALTY}" \
    "${REWARD_PROCESS_TYPE}" "${SEED}" \
    2>&1 | tee -a "${LOGFILE}"

echo "KTPO Cloudcast seed=${SEED} completed!"
