#!/bin/bash
#SBATCH --job-name=diffuser-train
#SBATCH --time=2-00:00:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_diffuser_mujoco_train_%x_%j.out
#SBATCH --error=slurm_diffuser_mujoco_train_%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

TASK="${TASK:-hopper-medium-v2}"
SEED="${SEED:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
RUN_NAME="${RUN_NAME:-${TASK}_seed${SEED}}"
DATASET_H5PATH="${DATASET_H5PATH:-}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: venv python not found at ${PYTHON}"
  exit 1
fi

export WANDB_MODE="${WANDB_MODE}"

echo "============================================================"
echo "diffuser_d4rl_mujoco full training"
echo "============================================================"
echo "hostname: $(hostname)"
echo "repo: ${REPO_DIR}"
echo "branch: $(git branch --show-current 2>/dev/null || echo unknown)"
echo "venv: ${VENV_DIR}"
echo "task: ${TASK}"
echo "seed: ${SEED}"
echo "wandb_mode: ${WANDB_MODE}"
echo "run_suffix: ${RUN_SUFFIX:-<default>}"
echo "run_name: ${RUN_NAME}"
echo "dataset_h5path: ${DATASET_H5PATH:-<default>}"
echo "diffusion_gradient_steps: 1000000 (config default)"
echo "save_interval: 100000"
echo "log_interval: 1000"
echo "============================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

TRAIN_ARGS=(
  task="${TASK}"
  mode=train
  seed="${SEED}"
  device=cuda:0
)
if [[ -n "${RUN_SUFFIX}" ]]; then
  TRAIN_ARGS+=(run_suffix="${RUN_SUFFIX}")
fi
if [[ -n "${DATASET_H5PATH}" ]]; then
  TRAIN_ARGS+=("dataset_h5path=${DATASET_H5PATH}")
fi

"${PYTHON}" pipelines/diffuser_d4rl_mujoco.py \
  "${TRAIN_ARGS[@]}" \
  enable_wandb=true \
  wandb_mode="${WANDB_MODE}" \
  project=CleanDiffuser \
  group=diffuser_d4rl_mujoco \
  name="${RUN_NAME}"

echo ""
echo "Training completed."
