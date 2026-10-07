#!/bin/bash
# KTPO — TXN Scheduling — Qwen3.5-9B — COLOCATED — NO THINKING
# Config: 16×16, k=8, improvement-rate eviction, 50 rollouts, 8xH100 colocated
# All training params identical to circle_packing 3.5-9B colocate, task=txn_scheduling
#
# Usage:
#   SEED=1 nohup bash scripts_launch/Qwen3.5-9B/ktpo_qwen3.5_9B_txn_scheduling_nothink_colocate.sh \
#       > logs/ktpo_qwen3.5_9b_txn_scheduling_nothink_seed1.log 2>&1 &
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

MODEL_NAME="Qwen3.5-9B"
HF_MODEL_ID="Qwen/Qwen3.5-9B"
models_file_name="qwen3.5-9B.sh"
SMALL_MODEL_NAME="qwen3.5_9b"

TASK="txn_scheduling"
IS_TRAINING=True
REWARD_PROCESS_TYPE="improve_reward"
LAZY_OUTPUT_PENALTY=1

WANDB_API_KEY="${WANDB_API_KEY:-}"   # optional; W&B logging is on only when set
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-KTPO}"

export LEARNX_ENABLE_THINKING="${LEARNX_ENABLE_THINKING:-false}"
case "${LEARNX_ENABLE_THINKING,,}" in
    1|true|yes)
        THINK_TAG="think"
        THINK_BANNER="AUTO ON (--enable-thinking)"
        ;;
    0|false|no)
        THINK_TAG="nothink"
        THINK_BANNER="AUTO OFF (--no-enable-thinking)"
        ;;
    *)
        echo "Invalid LEARNX_ENABLE_THINKING=${LEARNX_ENABLE_THINKING} (expected 0/1/false/true)" >&2
        exit 1
        ;;
esac

# ── Batch: 16×16 = 256 samples/rollout ──
export LEARNX_ROLLOUT_BATCH_SIZE="${LEARNX_ROLLOUT_BATCH_SIZE:-16}"

export LEARNX_N_SAMPLES_PER_PROMPT="${LEARNX_N_SAMPLES_PER_PROMPT:-16}"

export LEARNX_NUM_STEPS_PER_ROLLOUT="${LEARNX_NUM_STEPS_PER_ROLLOUT:-1}"

export LEARNX_OVER_SAMPLING_BATCH_SIZE="${LEARNX_OVER_SAMPLING_BATCH_SIZE:-16}"


# ── k=8 (8-step refinement) ──
export LEARNX_CONTEXT_HORIZON="${LEARNX_CONTEXT_HORIZON:-0}"

export LEARNX_MULTISTEP_K="${LEARNX_MULTISTEP_K:-8}"

export LEARNX_REWARD_AGG="${LEARNX_REWARD_AGG:-max_improve}"


# ── Evaluator ──
export LEARNX_EVAL_OMP_THREADS="${LEARNX_EVAL_OMP_THREADS:-4}"

export LEARNX_EVAL_PARALLEL="${LEARNX_EVAL_PARALLEL:-48}"

export LEARNX_EVAL_N_REPEAT="${LEARNX_EVAL_N_REPEAT:-1}"


# ── Parallelism: COLOCATED, TP=8, 8 GPUs shared ──
# Actor: TP=8, DP=1 on all 8 GPUs
# sglang: TP=8, 1 engine on all 8 GPUs (sglang handles GQA natively)
# memory_saver offloads sglang weights during train, actor weights during rollout
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
# offload releases sglang memory before train, so 0.8 is safe and maximizes rollout throughput
export LEARNX_MAX_TOKENS_PER_GPU="${LEARNX_MAX_TOKENS_PER_GPU:-4096}"

export LEARNX_SGLANG_MEM_FRACTION="${LEARNX_SGLANG_MEM_FRACTION:-0.8}"

# Keep NCCL override, but let train-memory-margin-bytes fall back to the framework default (1GB).
export OPENAI_EXTRA_ARGS="${OPENAI_EXTRA_ARGS:+${OPENAI_EXTRA_ARGS} }--train-env-vars {\"NCCL_NVLS_ENABLE\":\"0\"}"

########################### DERIVED PATHS ###########################

INITIAL_PROGRAM="openevolve_adapted/examples/${TASK}/initial_programs/initial_program.py"
EVALUATOR_FILE="openevolve_adapted/examples/${TASK}/evaluators/evaluator_modular.py"
CONFIG_YAML="openevolve_adapted/examples/${TASK}/configs/config_${TASK}.yaml"

_AGG_SHORT="${LEARNX_REWARD_AGG//_/}"
BASE_RUN_NAME="${RUN_NAME_OVERRIDE:-db_${SMALL_MODEL_NAME}_${THINK_TAG}_ktpo_${TASK}_colocate_${LEARNX_ROLLOUT_BATCH_SIZE}x${LEARNX_N_SAMPLES_PER_PROMPT}x${LEARNX_MULTISTEP_K}_${_AGG_SHORT}_seed${SEED}_r${LEARNX_NUM_ROLLOUT}}"
RUN_SESSION_TAG="${RUN_SESSION_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
GIT_COMMIT_SHORT="${GIT_COMMIT_SHORT_OVERRIDE:-$(git -c safe.directory="$(pwd)" rev-parse --short=12 HEAD 2>/dev/null || echo 'unknown')}"
RUN_SESSION_NAME="${RUN_SESSION_NAME_OVERRIDE:-${BASE_RUN_NAME}__${RUN_SESSION_TAG}__${GIT_COMMIT_SHORT}}"
export LEARNX_WANDB_GROUP="${LEARNX_WANDB_GROUP:-${BASE_RUN_NAME}}"
export LEARNX_WANDB_RUN_NAME="${LEARNX_WANDB_RUN_NAME:-${RUN_SESSION_NAME}}"
export LEARNX_SAVE_CKPT_DIR="${LEARNX_SAVE_CKPT_DIR:-${SAVE_PATH}/${RUN_SESSION_NAME}}"
export LEARNX_LOAD_CKPT_DIR="${LEARNX_LOAD_CKPT_DIR:-}"
export LEARNX_RUN_ARTIFACT_DIR="${LEARNX_RUN_ARTIFACT_DIR:-${SAVE_PATH}/${RUN_SESSION_NAME}}"
RUN_NAME="${RUN_SESSION_NAME}"

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
echo "║  KTPO — TXN Scheduling 9B NOTHINK — COLOCATED 8xH100║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Seed     : ${SEED}"
echo "║  Rollouts : ${LEARNX_NUM_ROLLOUT}"
echo "║  Batch    : ${LEARNX_ROLLOUT_BATCH_SIZE}×${LEARNX_N_SAMPLES_PER_PROMPT}×${LEARNX_MULTISTEP_K} = $((LEARNX_ROLLOUT_BATCH_SIZE*LEARNX_N_SAMPLES_PER_PROMPT*LEARNX_MULTISTEP_K)) evals/rollout"
echo "║  Thinking : ${THINK_BANNER}"
echo "║  Mode     : COLOCATE (TP=8 actor, TP=8 sglang)"
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
    PYTHONPATH=$(pwd):/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint $SAVE_SHM_DIR/$MODEL_NAME \
        --save $SAVE_SHM_DIR/${MODEL_NAME}_torch_dist
fi

########################### RUN ######################################

CKPT_DIR="${LEARNX_SAVE_CKPT_DIR}"
ARTIFACT_DIR="${LEARNX_RUN_ARTIFACT_DIR}"
LOGFILE="${ARTIFACT_DIR}/train_log.txt"
mkdir -p "${CKPT_DIR}" "${ARTIFACT_DIR}"

echo "Starting base run : ${BASE_RUN_NAME}"
echo "Starting session  : ${RUN_SESSION_NAME}"
echo "Git commit        : ${GIT_COMMIT_SHORT}"
echo "Checkpoint dir    : ${CKPT_DIR}"
echo "Resume dir        : ${LEARNX_LOAD_CKPT_DIR:-<none>}"
echo "Config            : ${CONFIG_YAML}"

# Use colocate general.sh (with --colocate flag)
bash "${SCRIPT_DIR}/_general_colocate.sh" \
    "${WANDB_PROJECT}" "${RUN_NAME}" \
    "${INITIAL_PROGRAM}" "${EVALUATOR_FILE}" "${CONFIG_YAML}" \
    "${SAVE_PATH}" "${IS_TRAINING}" "${LAZY_OUTPUT_PENALTY}" \
    "${REWARD_PROCESS_TYPE}" "${SEED}" \
    2>&1 | tee -a "${LOGFILE}"

echo "KTPO TXN Scheduling 9B nothink seed=${SEED} completed!"
