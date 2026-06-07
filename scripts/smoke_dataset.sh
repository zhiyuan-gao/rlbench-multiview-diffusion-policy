#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MANIFEST="${MANIFEST:-${REPO_ROOT}/manifests/selected10_fulltask_heuristic_waypoints_train100_val25_test25_from_train450_stratified_20260606.jsonl}"
PYTHON="${PYTHON:-python3}"

: "${RGB_ROOT_200:?Set RGB_ROOT_200 to the all200 RGB root}"
: "${RGB_ROOT_400:?Set RGB_ROOT_400 to the all400 RGB root}"
: "${LOWDIM_ROOT_200:?Set LOWDIM_ROOT_200 to the all200 low-dim metadata root}"
: "${LOWDIM_ROOT_400:?Set LOWDIM_ROOT_400 to the all400 low-dim metadata root}"

TASK_ARGS=()
if [[ -n "${TASKS:-}" ]]; then
  for task in ${TASKS}; do
    TASK_ARGS+=(--task "${task}")
  done
fi

OPTIONAL_ARGS=()
if [[ "${VALIDATE_IMAGE_PATHS:-1}" == "1" ]]; then
  OPTIONAL_ARGS+=(--validate-image-paths)
fi
if [[ "${MAX_EPISODES:-2}" -gt 0 ]]; then
  OPTIONAL_ARGS+=(--max-episodes "${MAX_EPISODES:-2}")
fi
if [[ "${MAX_EPISODES_PER_TASK:-0}" -gt 0 ]]; then
  OPTIONAL_ARGS+=(--max-episodes-per-task "${MAX_EPISODES_PER_TASK}")
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
"${PYTHON}" -m rlbench_multiview_dp.smoke_data \
  --manifest "${MANIFEST}" \
  --rgb-root-200 "${RGB_ROOT_200}" \
  --rgb-root-400 "${RGB_ROOT_400}" \
  --lowdim-root-200 "${LOWDIM_ROOT_200}" \
  --lowdim-root-400 "${LOWDIM_ROOT_400}" \
  --split "${SPLIT:-train}" \
  --obs-horizon "${OBS_HORIZON:-2}" \
  --sample-every-n "${SAMPLE_EVERY_N:-0}" \
  --view-names "${VIEW_NAMES:-front,left_shoulder,right_shoulder}" \
  --image-size "${IMAGE_SIZE:-256}" \
  --crop-size "${CROP_SIZE:-256}" \
  --num-samples "${NUM_SAMPLES:-4}" \
  "${TASK_ARGS[@]}" \
  "${OPTIONAL_ARGS[@]}" \
  "$@"
