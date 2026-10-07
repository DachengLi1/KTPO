#!/bin/bash
# Local Docker launcher — mounts KTPO + ckpts, then runs a docker/launchers/run_*.sh entry script.
#
# Usage:
#   SEED=1 bash docker/launchers/launch.sh run_llm_sql.sh
#   SEED=2 bash docker/launchers/launch.sh run_cloudcast.sh
#   SEED=3 bash docker/launchers/launch.sh run_adrs.sh eplb        # extra args forwarded
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENTRY_SCRIPT="${1:?Usage: SEED=N bash docker/launchers/launch.sh <run_xxx.sh> [args...]}"
shift
SEED="${SEED:-1}"
IMAGE="slimerl/slime@sha256:1fa4695649e8369ea70ddb87e9d6388c71c1fed7a4381bac0d605ff004556a40"

KTPO_DIR="${REPO_ROOT}"
CKPTS_DIR="${CKPTS_DIR:-${REPO_ROOT}/ckpts}"
TASK_NAME="$(basename "$ENTRY_SCRIPT" .sh | sed 's/^run_//')"
CONTAINER_NAME="${TASK_NAME}_seed${SEED}_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${CKPTS_DIR}" logs

echo "════════════════════════════════════════"
echo "  Task       : ${TASK_NAME}"
echo "  Seed       : ${SEED}"
echo "  Entry      : ${ENTRY_SCRIPT}"
echo "  Image      : ${IMAGE}"
echo "  Container  : ${CONTAINER_NAME}"
echo "  Mounts     : ${KTPO_DIR} -> /root/KTPO"
echo "               ${CKPTS_DIR}  -> /root/ckpts"
echo "════════════════════════════════════════"

# Forward every LEARNX_*, OPENAI_*, HF_*, WANDB_* from the host environment so
# launch/wrapper-level knobs (LEARNX_SAVE_INTERVAL, LEARNX_ENABLE_THINKING,
# LEARNX_LOAD_CKPT_DIR, LEARNX_HF_CKPT, OPENAI_API_BASE, etc.) actually reach
# the container. Anything unset on the host stays unset inside.
ENV_FORWARD=()
while IFS='=' read -r _k _v; do
  [ -z "$_k" ] && continue
  ENV_FORWARD+=(-e "$_k=$_v")
done < <(env | grep -E '^(LEARNX_|OPENAI_|HF_|WANDB_)')

docker run --gpus all --ipc=host --net=host \
  --name "${CONTAINER_NAME}" \
  -v "${KTPO_DIR}:/root/KTPO" \
  -v "${CKPTS_DIR}:/root/ckpts" \
  -e SEED="${SEED}" \
  "${ENV_FORWARD[@]}" \
  "${IMAGE}" \
  bash "/root/KTPO/docker/launchers/${ENTRY_SCRIPT}" "$@" \
  2>&1 | tee "logs/${TASK_NAME}_seed${SEED}.log"
