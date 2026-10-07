#!/bin/bash
# Docker runner for ADRS 9B tasks (EPLB, Prism, TXN scheduling)
# Applies the same Megatron + sglang patches as the other 9B entry scripts,
# then launches the per-task training script.
#
# Usage: bash docker/launchers/run_adrs.sh <eplb|prism|txn> [seed]
#   (or via launcher: SEED=N bash docker/launchers/launch.sh run_adrs.sh <eplb|prism|txn>)
set -euxo pipefail

TASK="${1:?Usage: bash docker/launchers/run_adrs.sh <eplb|prism|txn> [seed]}"
SEED="${2:-${SEED:-1}}"

case "$TASK" in
    eplb)        SCRIPT="scripts_launch/Qwen3.5-9B/ktpo_qwen3.5_9B_eplb_nothink_colocate.sh" ;;
    prism)       SCRIPT="scripts_launch/Qwen3.5-9B/ktpo_qwen3.5_9B_prism_nothink_colocate.sh" ;;
    txn)         SCRIPT="scripts_launch/Qwen3.5-9B/ktpo_qwen3.5_9B_txn_nothink_colocate.sh" ;;
    *)           echo "Unknown task: $TASK (options: eplb, prism, txn)"; exit 1 ;;
esac

echo "Task: $TASK, Seed: $SEED, Script: $SCRIPT"

export PIP_BREAK_SYSTEM_PACKAGES=1

# ── Megatron: align to dev commit, overlay local changes, clear bytecode cache ──
cd /root/Megatron-LM && git fetch --all && git checkout 3714d81d4

bash /root/KTPO/docker/apply_megatron_patch.sh

pip install --no-deps -e . --quiet

# ── SGLang: backport upstream idle/offload fix ──
bash /root/KTPO/docker/apply_sglang_runtime_patch.sh

# ── Install KTPO dependencies ──
cd /root/KTPO
pip install --no-deps -e openevolve_adapted/ --quiet
pip install --no-deps -e . --quiet
pip install tensorboard "numpy<2" --quiet
pip install "transformers==5.3.0" --quiet
pip install git+https://github.com/fzyzcjy/torch_memory_saver.git --no-cache-dir --force-reinstall --quiet 2>/dev/null || true

# ── Launch training ──
export SAVE_PATH="/root/ckpts"
SEED="$SEED" bash "$SCRIPT"
