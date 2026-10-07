#!/bin/bash
# Docker entry script for cloudcast 3.5-9B nothink training
# Pins Megatron-LM to commit 3714d81d4 and applies the overlay in docker/patch/megatron_files
# Then launches the training script
set -euxo pipefail

export PIP_BREAK_SYSTEM_PACKAGES=1

# ── Megatron: align to dev commit, overlay local changes, clear bytecode cache ──
cd /root/Megatron-LM && git fetch --all && git checkout 3714d81d4

bash /root/KTPO/docker/apply_megatron_patch.sh

# Reinstall Megatron with patches
pip install --no-deps -e . --quiet

# ── SGLang: backport upstream idle/offload fix ──
bash /root/KTPO/docker/apply_sglang_runtime_patch.sh

# ── Install KTPO dependencies ──
cd /root/KTPO
pip install --no-deps -e openevolve_adapted/ --quiet 2>/dev/null || true
pip install --no-deps -e . --quiet 2>/dev/null || true
pip install tensorboard "numpy<2" --quiet 2>/dev/null || true
pip install "transformers==5.3.0" --quiet 2>/dev/null || true

# ── Launch training ──
export SAVE_PATH="/root/ckpts"
export SEED="${SEED:-1}"

bash scripts_launch/Qwen3.5-9B/ktpo_qwen3.5_9B_cloudcast_nothink_colocate.sh
