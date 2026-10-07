#!/bin/bash
set -euo pipefail

SGLANG_ROOT="${SGLANG_ROOT:-/sgl-workspace/sglang}"
PATCH_FILE="${PATCH_FILE:-/root/KTPO/docker/patch/latest/sglang_pr20296.patch}"

echo "=== Applying SGLang patches (upstream PR #20296 idle/offload fix) ==="

if (cd "$SGLANG_ROOT" && patch -p1 -R --dry-run < "$PATCH_FILE" >/dev/null 2>&1); then
    echo "SGLang patch already applied, skipping."
else
    (cd "$SGLANG_ROOT" && patch -p1 --forward < "$PATCH_FILE")
fi

find "$SGLANG_ROOT" -name "*.pyc" -delete
find "$SGLANG_ROOT" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "=== VERIFY: SGLang idle/offload patch applied ==="
grep -n "def is_fully_idle" "$SGLANG_ROOT/python/sglang/srt/managers/scheduler.py"
grep -n "release_memory_occupation should be called only when server is idle" \
    "$SGLANG_ROOT/python/sglang/srt/managers/scheduler_update_weights_mixin.py"
