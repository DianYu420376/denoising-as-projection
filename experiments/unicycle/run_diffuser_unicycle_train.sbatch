#!/bin/bash
#SBATCH --job-name=diffuser_unicycle_train
#SBATCH --time=1-00:00:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_diffuser_unicycle_train_%j.out
#SBATCH --error=slurm_diffuser_unicycle_train_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

SEED="${SEED:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
RUN_NAME="${RUN_NAME:-unicycle-v0_seed${SEED}}"
DATASET_H5PATH="${DATASET_H5PATH:-${REPO_DIR}/results/unicycle_offline/unicycle_offline.hdf5}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: venv python not found at ${PYTHON}"
  exit 1
fi

export WANDB_MODE="${WANDB_MODE}"

echo "============================================================"
echo "diffuser_unicycle training (diffusion only, 4000-ep dataset)"
echo "============================================================"
echo "hostname: $(hostname)"
echo "repo: ${REPO_DIR}"
echo "venv: ${VENV_DIR}"
echo "seed: ${SEED}"
echo "dataset: ${DATASET_H5PATH}"
echo "diffusion_gradient_steps: 500000"
echo "save_interval: 50000"
echo "============================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

TRAIN_ARGS=(
  task=unicycle-v0
  mode=train
  seed="${SEED}"
  device=cuda:0
  "dataset_h5path=${DATASET_H5PATH}"
)
if [[ -n "${RUN_SUFFIX}" ]]; then
  TRAIN_ARGS+=(run_suffix="${RUN_SUFFIX}")
fi

"${PYTHON}" pipelines/diffuser_unicycle.py \
  "${TRAIN_ARGS[@]}" \
  enable_wandb=true \
  wandb_mode="${WANDB_MODE}" \
  project=CleanDiffuser \
  group=diffuser_unicycle \
  name="${RUN_NAME}"

echo ""
echo "Training completed."
