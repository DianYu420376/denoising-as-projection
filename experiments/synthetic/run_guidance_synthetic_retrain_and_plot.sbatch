#!/bin/bash
#SBATCH --job-name=guid_retrain
#SBATCH --time=24:00:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_guidance_synthetic_retrain_plot_%j.out
#SBATCH --error=slurm_guidance_synthetic_retrain_plot_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

SEED="${SEED:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-240000}"
NUM_DATASET_SAMPLES="${NUM_DATASET_SAMPLES:-200000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/guidance_synthetic}"

echo "============================================================"
echo "Recreate datasets + retrain (3x) + figure plots"
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
  --recreate-dataset
)

echo ""
echo "========== Test A: recreate dataset + train =========="
"${PYTHON}" pipelines/guidance_synthetic_subspace.py \
  --mode train \
  --test test_a \
  "${COMMON[@]}"

echo ""
echo "========== Test B: recreate dataset + train =========="
"${PYTHON}" pipelines/guidance_synthetic_subspace.py \
  --mode train \
  --test test_b \
  "${COMMON[@]}"

echo ""
echo "========== figures (paths + convergence) =========="
"${PYTHON}" pipelines/guidance_synthetic_visualize_subspace.py \
  --test both \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda:0 \
  --diffusion-steps 50 \
  --training-diffusion-steps 50 \
  --sampling-steps 50 \
  --optimization-guidance-last-steps 50 \
  --n-paths 5 \
  --w-cg 0.6 \
  --opt-scale 8 \
  --w-cg-test-b 10 \
  --opt-scale-test-b 3 \
  --plot-limit-a 2.0 \
  --plot-limit-b 1.8

echo ""
echo "Done."
echo "Checkpoints:"
echo "  ${OUTPUT_ROOT}/test_a/diffusion_ckpt_latest.pt"
echo "  ${OUTPUT_ROOT}/test_b/diffusion_ckpt_latest.pt"
