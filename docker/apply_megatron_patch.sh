#!/bin/bash
# Apply the Megatron file-overlay from docker/patch/megatron_files/megatron/.
#
# Mirror of apply_sglang_runtime_patch.sh: a single-step, idempotent helper
# that the per-task `docker/launchers/run_*.sh` entry scripts call after
# `git checkout <pinned commit>` resets /root/Megatron-LM. The overlay covers
# 19 files including 8 net-new files that don't appear in the upstream commit.
#
# Why `cp -r SRC/.` rather than `cp -r SRC`:
# `cp -r SRC/megatron/ DST/megatron/` with an existing DST/megatron/ silently
# nests to DST/megatron/megatron/... (POSIX behaviour). The trailing `/.` makes
# cp copy SRC's contents into DST instead of nesting, which is the only correct
# form when DST already exists. The previous launcher cp masked this with two
# explicit cp -v lines for the two most load-bearing files; those are dropped
# now that the recursive copy lands at the right path.

set -euo pipefail

MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
PATCH_SRC="${PATCH_SRC:-/root/KTPO/docker/patch/megatron_files/megatron}"

echo "=== Applying Megatron patches (overlay from ${PATCH_SRC}) ==="
cp -rv "${PATCH_SRC}/." "${MEGATRON_ROOT}/megatron/"

find "${MEGATRON_ROOT}" -name "*.pyc" -delete
find "${MEGATRON_ROOT}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "=== VERIFY: Megatron overlay landed at correct path ==="

# Canary 1: arguments.py (megatron/training arm). The `--use-gated-attention`
# flag is added by the overlay and absent from upstream Megatron at the pinned
# commit, so its presence proves the overlay reached megatron/training/.
grep -q -- "--use-gated-attention" "${MEGATRON_ROOT}/megatron/training/arguments.py" \
    && echo "PATCH OK: arguments.py contains --use-gated-attention" \
    || { echo "PATCH FAILED: arguments.py missing --use-gated-attention (overlay misnested?)"; exit 1; }

# Canary 2: fused_mla_yarn_rope_apply.py (megatron/core arm). Net-new file in
# the overlay (does not exist in upstream), so file-presence is a sufficient
# canary for the megatron/core arm.
[ -f "${MEGATRON_ROOT}/megatron/core/fusions/fused_mla_yarn_rope_apply.py" ] \
    && echo "PATCH OK: fused_mla_yarn_rope_apply.py present (overlay-only net-new file)" \
    || { echo "PATCH FAILED: fused_mla_yarn_rope_apply.py missing (overlay misnested?)"; exit 1; }
