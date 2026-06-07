#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MANIFEST="${MANIFEST:-${REPO_ROOT}/manifests/selected10_fulltask_heuristic_waypoints_train100_val25_test25_from_train450_stratified_20260606.jsonl}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/selected10_rgb_resnet18conv_clip_dp_h1}"
PYTHON="${PYTHON:-python3}"

: "${RGB_ROOT_200:?Set RGB_ROOT_200 to the all200 RGB root}"
: "${RGB_ROOT_400:?Set RGB_ROOT_400 to the all400 RGB root}"
: "${LOWDIM_ROOT_200:?Set LOWDIM_ROOT_200 to the all200 low-dim metadata root}"
: "${LOWDIM_ROOT_400:?Set LOWDIM_ROOT_400 to the all400 low-dim metadata root}"

OBS_HORIZON="${OBS_HORIZON:-2}"
SAMPLE_EVERY_N="${SAMPLE_EVERY_N:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
CROP_SIZE="${CROP_SIZE:-256}"
NUM_GPUS="${NUM_GPUS:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-$(( (GLOBAL_BATCH_SIZE + NUM_GPUS - 1) / NUM_GPUS ))}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_STEPS="${MAX_STEPS:-40000}"
LR="${LR:-1e-4}"
VISUAL_LR="${VISUAL_LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-100}"
EVAL_EVERY="${EVAL_EVERY:-0}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
VISUAL_BACKBONE="${VISUAL_BACKBONE:-resnet18}"
VISUAL_FEATURE_DIM="${VISUAL_FEATURE_DIM:-64}"
GLOBAL_COND_DIM="${GLOBAL_COND_DIM:-512}"
FUSION_HIDDEN_DIM="${FUSION_HIDDEN_DIM:-512}"
UNET_DIMS="${UNET_DIMS:-256,512,1024}"
CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-large-patch14}"

TASK_ARGS=()
if [[ -n "${TASKS:-}" ]]; then
  for task in ${TASKS}; do
    TASK_ARGS+=(--task "${task}")
  done
fi

OPTIONAL_ARGS=()
if [[ "${IMAGENET_PRETRAINED:-1}" == "0" ]]; then
  OPTIONAL_ARGS+=(--no-imagenet-pretrained)
fi
if [[ "${SHARE_VISUAL_ENCODER:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--share-visual-encoder)
fi
if [[ "${CLIP_LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--clip-local-files-only)
fi
if [[ -n "${DUMMY_TEXT_DIM:-}" ]]; then
  OPTIONAL_ARGS+=(--dummy-text-dim "${DUMMY_TEXT_DIM}")
fi

TRAIN_ARGS=(
  --manifest "${MANIFEST}"
  --rgb-root-200 "${RGB_ROOT_200}"
  --rgb-root-400 "${RGB_ROOT_400}"
  --lowdim-root-200 "${LOWDIM_ROOT_200}"
  --lowdim-root-400 "${LOWDIM_ROOT_400}"
  --out-dir "${RUN_ROOT}"
  --train-split "${TRAIN_SPLIT:-train}"
  --val-split "${VAL_SPLIT:-val}"
  --obs-horizon "${OBS_HORIZON}"
  --sample-every-n "${SAMPLE_EVERY_N}"
  --view-names "${VIEW_NAMES:-front,left_shoulder,right_shoulder}"
  --image-size "${IMAGE_SIZE}"
  --crop-size "${CROP_SIZE}"
  --proprio-mode "${PROPRIO_MODE:-ee_rotvec}"
  --clip-model "${CLIP_MODEL}"
  --visual-backbone "${VISUAL_BACKBONE}"
  --visual-feature-dim "${VISUAL_FEATURE_DIM}"
  --global-cond-dim "${GLOBAL_COND_DIM}"
  --fusion-hidden-dim "${FUSION_HIDDEN_DIM}"
  --unet-dims "${UNET_DIMS}"
  --diffusion-steps "${DIFFUSION_STEPS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --max-steps "${MAX_STEPS}"
  --lr "${LR}"
  --visual-lr "${VISUAL_LR}"
  --warmup-steps "${WARMUP_STEPS}"
  --eval-every "${EVAL_EVERY}"
  --save-every "${SAVE_EVERY}"
  --seed "${SEED:-0}"
  "${TASK_ARGS[@]}"
  "${OPTIONAL_ARGS[@]}"
)

mkdir -p "${RUN_ROOT}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="${NUM_GPUS}" -m rlbench_multiview_dp.train "${TRAIN_ARGS[@]}" "$@"
else
  "${PYTHON}" -m rlbench_multiview_dp.train "${TRAIN_ARGS[@]}" "$@"
fi
