#!/bin/bash
#SBATCH --job-name=diffuser_unicycle_full64
#SBATCH --time=1-00:00:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_diffuser_unicycle_train_full64_%j.out
#SBATCH --error=slurm_diffuser_unicycle_train_full64_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

SEED="${SEED:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
RUN_SUFFIX="${RUN_SUFFIX:-full64_6k}"
RUN_NAME="${RUN_NAME:-unicycle_full64_6k_seed${SEED}}"
DATASET_H5PATH="${DATASET_H5PATH:-${REPO_DIR}/results/unicycle_offline/full64_6k/unicycle_offline.hdf5}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: venv python not found at ${PYTHON}"
  exit 1
fi

if [[ ! -f "${DATASET_H5PATH}" ]]; then
  echo "ERROR: dataset not found at ${DATASET_H5PATH}"
  echo "Run data collection first (see run_collect_unicycle_full64.sbatch)."
  exit 1
fi

export WANDB_MODE="${WANDB_MODE}"

echo "============================================================"
echo "diffuser_unicycle training — full 64-step dataset (6k eps)"
echo "============================================================"
echo "hostname: $(hostname)"
echo "repo: ${REPO_DIR}"
echo "seed: ${SEED}"
echo "run_suffix: ${RUN_SUFFIX}"
echo "dataset: ${DATASET_H5PATH}"
echo "checkpoints: results/diffuser_unicycle/Unicycle-v0/${RUN_SUFFIX}/"
echo "diffusion_gradient_steps: 500000"
echo "save_interval: 50000"
echo "============================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

"${PYTHON}" pipelines/diffuser_unicycle.py \
  task=unicycle-v0 \
  mode=train \
  seed="${SEED}" \
  device=cuda:0 \
  "dataset_h5path=${DATASET_H5PATH}" \
  run_suffix="${RUN_SUFFIX}" \
  enable_wandb=true \
  wandb_mode="${WANDB_MODE}" \
  project=CleanDiffuser \
  group=diffuser_unicycle \
  name="${RUN_NAME}"

echo ""
echo "Training completed."
