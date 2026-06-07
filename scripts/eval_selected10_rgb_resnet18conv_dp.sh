#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MANIFEST="${MANIFEST:-${REPO_ROOT}/manifests/selected10_fulltask_heuristic_waypoints_train100_val25_test25_from_train450_stratified_20260606.jsonl}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/selected10_rgb_resnet18conv_clip_dp_h1}"
POLICY_DIR="${POLICY_DIR:-${RUN_ROOT}}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/online_eval_${SPLIT:-val}}"
PYTHON="${PYTHON:-python3}"

: "${LOWDIM_ROOT_200:?Set LOWDIM_ROOT_200 to the all200 low-dim metadata root}"
: "${LOWDIM_ROOT_400:?Set LOWDIM_ROOT_400 to the all400 low-dim metadata root}"

TASK_ARGS=()
if [[ -n "${TASKS:-}" ]]; then
  for task in ${TASKS}; do
    TASK_ARGS+=(--task "${task}")
  done
fi

OPTIONAL_ARGS=()
if [[ "${NO_EMA:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--no-use-ema)
fi
if [[ "${RECORD_VIDEO:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--record-video)
fi
if [[ "${SAVE_STEP_LOGS:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--save-step-logs)
fi
if [[ "${CONTINUE_AFTER_INVALID:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--continue-after-invalid)
fi
if [[ "${COLLISION_CHECKING:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--collision-checking)
fi
if [[ "${MAX_EPISODES:-0}" -gt 0 ]]; then
  OPTIONAL_ARGS+=(--max-episodes "${MAX_EPISODES}")
fi
if [[ "${MAX_EPISODES_PER_TASK:-0}" -gt 0 ]]; then
  OPTIONAL_ARGS+=(--max-episodes-per-task "${MAX_EPISODES_PER_TASK}")
fi
ROOT_ARGS=(
  --lowdim-root-200 "${LOWDIM_ROOT_200}"
  --lowdim-root-400 "${LOWDIM_ROOT_400}"
)
if [[ -n "${RGB_ROOT_200:-}" ]]; then
  ROOT_ARGS+=(--rgb-root-200 "${RGB_ROOT_200}")
fi
if [[ -n "${RGB_ROOT_400:-}" ]]; then
  ROOT_ARGS+=(--rgb-root-400 "${RGB_ROOT_400}")
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
"${PYTHON}" -m rlbench_multiview_dp.eval_online \
  --manifest "${MANIFEST}" \
  "${ROOT_ARGS[@]}" \
  --policy-dir "${POLICY_DIR}" \
  --checkpoint "${CHECKPOINT:-latest}" \
  --out-dir "${OUT_DIR}" \
  --split "${SPLIT:-val}" \
  --max-policy-steps "${MAX_POLICY_STEPS:-0}" \
  --extra-policy-steps "${EXTRA_POLICY_STEPS:-2}" \
  --sample-steps "${SAMPLE_STEPS:-100}" \
  --device "${DEVICE:-cuda}" \
  --arm-mode "${ARM_MODE:-planning}" \
  --image-size "${IMAGE_SIZE:-0}" \
  --write-every "${WRITE_EVERY:-1}" \
  "${TASK_ARGS[@]}" \
  "${OPTIONAL_ARGS[@]}" \
  "$@"
