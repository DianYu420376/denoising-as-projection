#!/bin/bash
#SBATCH --job-name=guid_synth_train
#SBATCH --time=08:00:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_guidance_synthetic_subspace_train_%j.out
#SBATCH --error=slurm_guidance_synthetic_subspace_train_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

SEED="${SEED:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-80000}"
NUM_DATASET_SAMPLES="${NUM_DATASET_SAMPLES:-200000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/guidance_synthetic}"

echo "============================================================"
echo "Synthetic subspace diffusion — pretrain Test A & Test B"
echo "============================================================"
echo "hostname: $(hostname)"
echo "seed: ${SEED}"
echo "train_steps: ${TRAIN_STEPS}"
echo "num_dataset_samples: ${NUM_DATASET_SAMPLES}"
echo "output: ${OUTPUT_ROOT}"
echo "============================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

COMMON=(
  --mode train
  --seed "${SEED}"
  --device cuda:0
  --output-root "${OUTPUT_ROOT}"
  --train-steps "${TRAIN_STEPS}"
  --num-dataset-samples "${NUM_DATASET_SAMPLES}"
  --batch-size 256
  --diffusion-steps 50
  --training-diffusion-steps 50
  --log-interval 1000
  --save-interval 20000
)

echo ""
echo "========== Test A: 2D square subspace in R^10 =========="
"${PYTHON}" pipelines/guidance_synthetic_subspace.py \
  --test test_a \
  "${COMMON[@]}"

echo ""
echo "========== Test B: 5D ellipsoid subspace in R^10 =========="
"${PYTHON}" pipelines/guidance_synthetic_subspace.py \
  --test test_b \
  "${COMMON[@]}"

echo ""
echo "Pretraining completed."
echo "Checkpoints:"
echo "  ${OUTPUT_ROOT}/test_a/diffusion_ckpt_latest.pt"
echo "  ${OUTPUT_ROOT}/test_b/diffusion_ckpt_latest.pt"
