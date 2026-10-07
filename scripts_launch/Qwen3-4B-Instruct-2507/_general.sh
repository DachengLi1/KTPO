#!/bin/bash
# General training script with configurable parameters
# Usage: ./scripts_launch/${MODEL_NAME}/_general.sh WANDB_PROJECT RUN_NAME INITIAL_PROGRAM EVALUATOR_FILE CONFIG_YAML SAVE_PATH IS_TRAINING LAZY_OUTPUT_PENALTY REWARD_PROCESS_TYPE SEED

if [ $# -lt 10 ]; then
  echo "Usage: $0 WANDB_PROJECT RUN_NAME INITIAL_PROGRAM EVALUATOR_FILE CONFIG_YAML SAVE_PATH IS_TRAINING LAZY_OUTPUT_PENALTY REWARD_PROCESS_TYPE SEED"
  echo ""
  echo "Required parameters:"
  echo "  WANDB_PROJECT        - Weights & Biases project name"
  echo "  RUN_NAME             - Experiment run name"
  echo "  INITIAL_PROGRAM      - Path to initial program file"
  echo "  EVALUATOR_FILE       - Path to evaluator file"
  echo "  CONFIG_YAML          - Path to config YAML file"
  echo "  SAVE_PATH            - Save directory path"
  echo "  IS_TRAINING          - True for training, False for inference-only"
  echo "  LAZY_OUTPUT_PENALTY  - Lazy output penalty level (1 or 2)"
  echo "  REWARD_PROCESS_TYPE  - Reward processing type (original_reward, rl_normalized_reward, etc.)"
  echo "  SEED                 - Random seed for reproducibility"
  exit 1
fi

WANDB_PROJECT=$1
RUN_NAME=$2
INITIAL_PROGRAM=$3
EVALUATOR_FILE=$4
CONFIG_YAML=$5
SAVE_PATH=$6
IS_TRAINING=$7
LAZY_OUTPUT_PENALTY=$8
REWARD_PROCESS_TYPE=$9
SEED=${10}

SAVE_SHM_DIR="${SAVE_PATH}/shm"
CKPT_DIR="${SAVE_PATH}/${RUN_NAME}"
RECORD_PATH="${SAVE_PATH}/${RUN_NAME}/records"
MODEL_NAME="Qwen3-4B-Instruct-2507"

# Determine debug-rollout-only mode based on IS_TRAINING
if [ "$IS_TRAINING" = "False" ] || [ "$IS_TRAINING" = "false" ]; then
    DEBUG_ROLLOUT_ONLY="--debug-rollout-only"
    echo "Inference-only mode enabled (IS_TRAINING=$IS_TRAINING)"
else
    DEBUG_ROLLOUT_ONLY=""
    echo "Normal training mode (IS_TRAINING=$IS_TRAINING)"
fi

# OpenAI external API backend (set OPENAI_API_BASE env to enable)
OPENAI_EXTRA_ARGS="${OPENAI_EXTRA_ARGS:-}"
if [ -n "${OPENAI_API_BASE:-}" ]; then
    OPENAI_EXTRA_ARGS="${OPENAI_EXTRA_ARGS:+${OPENAI_EXTRA_ARGS} }--openai-api-base ${OPENAI_API_BASE} --openai-model ${OPENAI_MODEL:-${HF_MODEL_ID}}"
    DEBUG_ROLLOUT_ONLY="--debug-rollout-only"
    echo "OpenAI backend mode: base=${OPENAI_API_BASE} model=${OPENAI_MODEL:-${HF_MODEL_ID}}"
fi

# Create checkpoint directory
mkdir -p "${CKPT_DIR}"

# Skip cleanup & Ray init when orchestrated by an outer script (SKIP_INIT=1)
if [ "${SKIP_INIT:-0}" != "1" ]; then
    pkill -9 sglang || true
sleep 3
ray stop --force
    pkill -9 ray || true
# pkill -9 python
sleep 3
    pkill -9 ray || true
# pkill -9 python
fi

set -ex

export PYTHONBUFFERED=16
export TOKENIZERS_PARALLELISM=false
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"

case "${LEARNX_ACDS_IMPROVING_ONLY:-0}" in
  1|true|TRUE|yes|YES)
    export LEARNX_ACDS_IMPROVING_ONLY=1
    ;;
  0|false|FALSE|no|NO)
    export LEARNX_ACDS_IMPROVING_ONLY=0
    ;;
  *)
    echo "Invalid LEARNX_ACDS_IMPROVING_ONLY=${LEARNX_ACDS_IMPROVING_ONLY} (expected 0/1/false/true)" >&2
    exit 1
    ;;
esac

# Evaluator threading control (prevents BLAS over-threading in parallel evals)
export OMP_NUM_THREADS="${LEARNX_EVAL_OMP_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${LEARNX_EVAL_OMP_THREADS:-1}"
export MKL_NUM_THREADS="${LEARNX_EVAL_OMP_THREADS:-1}"

# NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l || echo 0)
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
echo "NVLINK_COUNT: $NVLINK_COUNT"
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
# HAS_NVLINK=0
# echo "Overriding HAS_NVLINK to $HAS_NVLINK for compatibility"

source scripts_launch/models/qwen3-4B-instruct-2507.sh

# Allow caller to override the HF checkpoint used for actor + sglang weight init.
# When LEARNX_HF_CKPT is set, the training run starts *fresh* (no distcp resume)
# but with actor weights loaded from the specified HF checkpoint.
# Leave CKPT_DIR empty so Megatron falls back to --hf-checkpoint instead of
# trying to resume from a distcp snapshot (which would hit DB state bugs).
HF_CKPT_PATH="${LEARNX_HF_CKPT:-${SAVE_SHM_DIR}/${MODEL_NAME}}"
echo "HF_CKPT_PATH: ${HF_CKPT_PATH}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT_PATH}"
   --ref-load "${SAVE_SHM_DIR}/${MODEL_NAME}_torch_dist"
   --load "${CKPT_DIR}/"
   --save "${CKPT_DIR}/"
   --save-interval "${LEARNX_SAVE_INTERVAL:-5}"
)

ROLLOUT_ARGS=(
  --disable-rollout-global-dataset
  --evolving-gym
  --evolving-gym-initial-program "${INITIAL_PROGRAM}"
  --evolving-gym-evaluator-file "${EVALUATOR_FILE}"
  --evolving-gym-config-path "${CONFIG_YAML}"
  --evolving-gym-max-concurrent-evals "${LEARNX_EVAL_PARALLEL:-16}"
  --evolving-gym-log-prompts
  --evolving-gym-record
  --evolving-gym-record-dir "${RECORD_PATH}"
  --evolving-gym-lazy-output-penalty-level "${LAZY_OUTPUT_PENALTY}"
  --evolving-gym-seed ${SEED}
  --evolving-gym-reward-process-type "${REWARD_PROCESS_TYPE}"

  --apply-chat-template

  --rm-type evolving-gym
  --reward-key reward

  --num-rollout "${LEARNX_NUM_ROLLOUT:-1000000}"
  --rollout-batch-size "${LEARNX_ROLLOUT_BATCH_SIZE:-32}"
  --n-samples-per-prompt "${LEARNX_N_SAMPLES_PER_PROMPT:-16}"
  --rollout-max-response-len "${LEARNX_ROLLOUT_MAX_RESPONSE_LEN:-32768}"
  --rollout-temperature 1.0

  --over-sampling-batch-size "${LEARNX_OVER_SAMPLING_BATCH_SIZE:-32}"
  # --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

  --num-steps-per-rollout "${LEARNX_NUM_STEPS_PER_ROLLOUT:-1}"
  --wandb-always-use-train-step
  --balance-data
)

# NOTE: AWRS logging block removed — slime/utils/arguments.py has no
# `--evolving-gym-awrs-log[-dir]` parser, so enabling the env-gate would
# crash train.py at argparse. Reintroduce only with parser + consumer support.

# Optional: Multi-step refinement mode.
# Enable by: export LEARNX_MULTISTEP_K=4  (flat K)
#        or: export LEARNX_MULTISTEP_K_SCHEDULE="1:31,4:23"  (curriculum)
echo ""
echo "┌─────────────── KTPO Feature Config ─────────────────┐"
MULTISTEP_K="${LEARNX_MULTISTEP_K:-0}"
MULTISTEP_ENABLED=false
if [ "$MULTISTEP_K" -gt 0 ]; then
  ROLLOUT_ARGS+=( --multistep-refine-k "${MULTISTEP_K}" )
  echo "│  multistep_k           = ${MULTISTEP_K}"
  MULTISTEP_ENABLED=true
else
  echo "│  multistep_k           = 0 (disabled)"
fi
# Curriculum K schedule: e.g. "1:31,4:23" → K=1 for 31 rollouts, then K=4 for 23
if [ -n "${LEARNX_MULTISTEP_K_SCHEDULE:-}" ]; then
  ROLLOUT_ARGS+=( --multistep-k-schedule "${LEARNX_MULTISTEP_K_SCHEDULE}" )
  echo "│  k_schedule            = ${LEARNX_MULTISTEP_K_SCHEDULE}"
  MULTISTEP_ENABLED=true
else
  echo "│  k_schedule            = <none>"
fi
# Cross-rollout trajectory context horizon.
# When >0, parent's refinement history (code+score) is carried into the prompt.
if [ "${LEARNX_CONTEXT_HORIZON:-0}" -gt 0 ]; then
  ROLLOUT_ARGS+=( --context-horizon "${LEARNX_CONTEXT_HORIZON}" )
  echo "│  context_horizon       = ${LEARNX_CONTEXT_HORIZON}"
else
  echo "│  context_horizon       = 0 (disabled)"
fi
# Multi-step reward aggregation (KTPO default: max_improve = improvement reward).
REWARD_AGG_VAL="${LEARNX_REWARD_AGG:-max_improve}"
if [ -n "${LEARNX_REWARD_AGG:-}" ]; then
  ROLLOUT_ARGS+=( --multistep-reward-agg "${LEARNX_REWARD_AGG}" )
fi
echo "│  reward_agg            = ${REWARD_AGG_VAL}"
# Multi-epoch training per rollout: reuse rollout data for multiple gradient steps.
# 0 = "follow k" (epochs = effective_k at each rollout), 1 = default, >1 = fixed.
N_TRAIN_EPOCHS="${LEARNX_N_TRAIN_EPOCHS_PER_ROLLOUT:-1}"
if [ "$N_TRAIN_EPOCHS" -eq 0 ]; then
  ROLLOUT_ARGS+=( --n-train-epochs-per-rollout 0 )
  echo "│  train_epochs/rollout  = follow_k (dynamic)"
elif [ "$N_TRAIN_EPOCHS" -gt 1 ]; then
  ROLLOUT_ARGS+=( --n-train-epochs-per-rollout "${N_TRAIN_EPOCHS}" )
  echo "│  train_epochs/rollout  = ${N_TRAIN_EPOCHS}"
else
  echo "│  train_epochs/rollout  = 1 (default)"
fi
# Adaptive-k early termination: stop chain if N consecutive steps don't beat db_worst_score.
ADAPTIVE_K_PATIENCE="${LEARNX_ADAPTIVE_K_PATIENCE:-0}"
ADAPTIVE_K_PATIENCE_MODE="${LEARNX_ADAPTIVE_K_PATIENCE_MODE:-consecutive}"
if [ "$ADAPTIVE_K_PATIENCE" -gt 0 ]; then
  ROLLOUT_ARGS+=( --adaptive-k-patience "${ADAPTIVE_K_PATIENCE}" )
  ROLLOUT_ARGS+=( --adaptive-k-patience-mode "${ADAPTIVE_K_PATIENCE_MODE}" )
  echo "│  adaptive_k_patience   = ${ADAPTIVE_K_PATIENCE} (mode=${ADAPTIVE_K_PATIENCE_MODE})"
else
  echo "│  adaptive_k_patience   = 0 (disabled)"
fi
# Sample budget: stop the whole run after N total LLM eval calls.
# When set, num_rollout is used as an upper bound only; the real stopping criterion is the budget.
if [ -n "${LEARNX_SAMPLE_BUDGET:-}" ] && [ "${LEARNX_SAMPLE_BUDGET}" -gt 0 ]; then
  ROLLOUT_ARGS+=( --sample-budget "${LEARNX_SAMPLE_BUDGET}" )
  echo "│  sample_budget         = ${LEARNX_SAMPLE_BUDGET}"
else
  echo "│  sample_budget         = <none> (use num_rollout)"
fi
# Single-reference mode: show only the parent program and instruct model to improve it.
if [ "${LEARNX_SINGLE_REFERENCE:-0}" = "1" ]; then
  ROLLOUT_ARGS+=( --single-reference )
  echo "│  single_reference      = true"
else
  echo "│  single_reference      = false (multi-ref, default)"
fi
# Pool update with improvement-rate eviction: always on.
ROLLOUT_ARGS+=( --evolving-gym-beat-rate-eviction )
ROLLOUT_ARGS+=( --evolving-gym-acds-retire-thr "${LEARNX_ACDS_RETIRE_THR:-0.30}" )
ROLLOUT_ARGS+=( --evolving-gym-acds-retire-thr-end "${LEARNX_ACDS_RETIRE_THR_END:-0.00}" )
ROLLOUT_ARGS+=( --evolving-gym-acds-mastered-window "${LEARNX_ACDS_MASTERED_WINDOW:-10}" )
case "${LEARNX_ACDS_VALID_ONLY_BEAT_RATE:-0}" in
  1|true|TRUE|yes|YES)
    ROLLOUT_ARGS+=( --evolving-gym-acds-valid-only-beat-rate )
    ACDS_VALID_ONLY_BEAT_RATE=true
    ;;
  0|false|FALSE|no|NO)
    ACDS_VALID_ONLY_BEAT_RATE=false
    ;;
  *)
    echo "Invalid LEARNX_ACDS_VALID_ONLY_BEAT_RATE=${LEARNX_ACDS_VALID_ONLY_BEAT_RATE} (expected 0/1/false/true)" >&2
    exit 1
    ;;
esac
ROLLOUT_ARGS+=( --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_evolving_gym_acds_trainable )
echo "│  ir_eviction           = ON (improvement-rate eviction)"
echo "│  ir_valid_only         = ${ACDS_VALID_ONLY_BEAT_RATE}"
echo "│  ir_improving_only     = ${LEARNX_ACDS_IMPROVING_ONLY}"
echo "│  retire_thr            = ${LEARNX_ACDS_RETIRE_THR:-0.30} → ${LEARNX_ACDS_RETIRE_THR_END:-0.00} (decay)"
echo "│  mastered_window       = ${LEARNX_ACDS_MASTERED_WINDOW:-10}"
echo "│  rollout_batch_size    = ${LEARNX_ROLLOUT_BATCH_SIZE:-32}"
echo "│  n_samples_per_prompt  = ${LEARNX_N_SAMPLES_PER_PROMPT:-16}"
echo "│  over_sampling_batch   = ${LEARNX_OVER_SAMPLING_BATCH_SIZE:-32}"
if [ "${LEARNX_OVER_SAMPLING_BATCH_SIZE:-32}" = "${LEARNX_ROLLOUT_BATCH_SIZE:-32}" ]; then
  echo "│  ⚠ over_sampling == rollout_batch → over-sampling mechanism is a no-op (set OVER_SAMPLING > BATCH to enable filter rejection headroom)"
fi
echo "│  eval_parallel         = ${LEARNX_EVAL_PARALLEL:-16}"
echo "│  eval_omp_threads      = ${LEARNX_EVAL_OMP_THREADS:-1}"
echo "│  eval_n_repeat         = ${LEARNX_EVAL_N_REPEAT:-1}"
echo "│  max_tokens_per_gpu    = ${LEARNX_MAX_TOKENS_PER_GPU:-2048}"
echo "│  num_rollout           = ${LEARNX_NUM_ROLLOUT:-1000000}"
echo "│  TP=${LEARNX_TP_SIZE:-4}  CP=${LEARNX_CP_SIZE:-2}"
echo "└────────────────────────────────────────────────────────┘"
echo ""


PERF_ARGS=(
  --tensor-model-parallel-size "${LEARNX_TP_SIZE:-4}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size "${LEARNX_CP_SIZE:-2}"
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1

  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1

  --use-dynamic-batch-size
  --max-tokens-per-gpu "${LEARNX_MAX_TOKENS_PER_GPU:-2048}"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28

  --use-tis
)


OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LEARNX_LR:-1e-6}"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98

  --optimizer-cpu-offload                 # Offload optimizer state (e.g. moments) to CPU memory to save GPU VRAM
  --overlap-cpu-optimizer-d2h-h2d         # Overlap CPU<->GPU data transfer with computation to reduce waiting time
  --use-precision-aware-optimizer         # Enable precision-aware optimizer to dynamically handle numeric ranges and reduce overflow/instability risks

)

# W&B is opt-in: enabled only when WANDB_API_KEY is set in the environment.
WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
WANDB_ARGS=(
  --use-wandb
  ${WANDB_ENTITY:+--wandb-team ${WANDB_ENTITY}}
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${RUN_NAME}"
  --disable-wandb-random-suffix
)
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${LEARNX_ROLLOUT_NUM_GPUS:-8}"
  --sglang-mem-fraction-static "${LEARNX_SGLANG_MEM_FRACTION:-0.8}"
  --sglang-context-length 131072
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256) # increase throughput
)

MISC_ARGS=(
  ${DEBUG_ROLLOUT_ONLY}
  --seed ${SEED}
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

# OpenAI mode: override GPU-heavy settings with lightweight defaults
if [ -n "${OPENAI_API_BASE:-}" ]; then
    SGLANG_ARGS=( --rollout-num-gpus-per-engine 1 )
    PERF_ARGS=(
        --tensor-model-parallel-size 1 --sequence-parallel
        --pipeline-model-parallel-size 1 --context-parallel-size 1
        --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
    )
    RAY_NUM_GPUS=0
else
    # Support custom GPU count via environment variable
    RAY_NUM_GPUS="${LEARNX_RAY_NUM_GPUS:-8}"
fi

# Start Ray (skip when orchestrated by outer script)
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
if [ "${SKIP_INIT:-0}" != "1" ]; then
    ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${RAY_NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi


# Disable Triton
export TRITON_DISABLE=1



export FAST_MOUNT=$SAVE_PATH/fast_mount
export HF_DATASETS_CACHE=$FAST_MOUNT/hf/datasets
export DATASETS_CACHE=$HF_DATASETS_CACHE
export DATASETS_TMPDIR=$FAST_MOUNT/tmp
export PYARROW_TMP_DIR=$FAST_MOUNT/tmp

mkdir -p "$HF_DATASETS_CACHE" "$DATASETS_TMPDIR"
echo "[disk] HF_DATASETS_CACHE=$HF_DATASETS_CACHE TMPDIR=$TMPDIR"



# Build the runtime environment JSON with proper variable substitution
# Build the runtime environment JSON. Explicit non-LEARNX_ vars are listed
# below; every `LEARNX_*` that the launcher has exported is auto-forwarded so
# Python/Ray workers see launcher overrides. This closes the previous gap
# where only 5 LEARNX_* reached the Python side.
RUNTIME_ENV_JSON="$(python3 - <<PY
import json, os
env = {
    "PYTHONPATH": "/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "HF_HOME": "${HF_HOME}",
    "HUGGINGFACE_HUB_CACHE": "${HUGGINGFACE_HUB_CACHE}",
    "TRANSFORMERS_CACHE": "${TRANSFORMERS_CACHE}",
    "HF_DATASETS_CACHE": "${HF_DATASETS_CACHE}",
    "DATASETS_CACHE": "${DATASETS_CACHE}",
    "DATASETS_TMPDIR": "${DATASETS_TMPDIR}",
    "PYARROW_TMP_DIR": "${PYARROW_TMP_DIR}",
    "TMPDIR": "${TMPDIR}",
    "WANDB_CACHE_DIR": "${WANDB_CACHE_DIR}",
    "WANDB_DIR": "${WANDB_DIR}",
    "WANDB_GROUP": "${RUN_NAME}",
    "TRITON_DISABLE": "1",
    "RAY_memory_monitor_refresh_ms": "${RAY_memory_monitor_refresh_ms}",
    "OMP_NUM_THREADS": "${LEARNX_EVAL_OMP_THREADS:-1}",
    "OPENBLAS_NUM_THREADS": "${LEARNX_EVAL_OMP_THREADS:-1}",
    "MKL_NUM_THREADS": "${LEARNX_EVAL_OMP_THREADS:-1}",
    # Defensive defaults for LEARNX_* vars no launcher currently exports.
    # The auto-forward below would skip them (os.environ has no key); these
    # bash fallbacks guarantee Python reads the historical default value.
    "LEARNX_EVAL_N_REPEAT": "${LEARNX_EVAL_N_REPEAT:-1}",
    "LEARNX_EVAL_PARALLEL": "${LEARNX_EVAL_PARALLEL:-16}",
    "LEARNX_ACDS_BEAT_PARENT": "${LEARNX_ACDS_BEAT_PARENT:-0}",
    "LEARNX_ACDS_ENTRY_MODE": "${LEARNX_ACDS_ENTRY_MODE:-}",
    "LEARNX_ACDS_IMPROVING_ONLY": "${LEARNX_ACDS_IMPROVING_ONLY:-0}",
}
# Auto-forward catches every other LEARNX_* the launcher exports (so new
# knobs land in Ray env without editing this template). When a launcher does
# export one of the 5 vars above, this update overrides the defensive default.
env.update({k: v for k, v in os.environ.items() if k.startswith("LEARNX_")})
print(json.dumps({"env_vars": env}))
PY
)"

echo "RUNTIME_ENV_JSON = $RUNTIME_ENV_JSON"

ray job submit --address="${RAY_ADDRESS:-http://127.0.0.1:8265}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${LEARNX_ACTOR_NUM_GPUS:-8}" \
   --colocate \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${DISTRIBUTED_ARGS[@]} \
  ${WANDB_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${MISC_ARGS[@]} \
  ${OPENAI_EXTRA_ARGS}
