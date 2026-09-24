#!/bin/bash
#SBATCH --job-name=heart_plan_unguided
#SBATCH --time=00:45:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_heart_plan_unguided_%j.out
#SBATCH --error=slurm_heart_plan_unguided_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

CKPT_PATH="${CKPT_PATH:-results/diffuser_unicycle/Unicycle-v0/full64_6k/diffusion_ckpt_latest.pt}"
SEED="${SEED:-0}"
CANDIDATES="${CANDIDATES:-100}"
SAMPLING_STEPS="${SAMPLING_STEPS:-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/unicycle_eval/heart_plan_unguided_100_candidates}"

echo "Heart plan unguided baseline (DDIM ${SAMPLING_STEPS} reverse steps, prior=init state)"
echo "checkpoint: ${CKPT_PATH}"
echo "candidates: ${CANDIDATES}"

"${PYTHON}" pipelines/unicycle_eval.py \
  --eval-task heart_plan \
  --ckpt-path "${CKPT_PATH}" \
  --seed "${SEED}" \
  --device cuda:0 \
  --solver ddim \
  --sample-step-schedule uniform \
  --sampling-steps "${SAMPLING_STEPS}" \
  --diffusion-steps 20 \
  --temperature 1.0 \
  --use-ema \
  --heart-config-names standard_w_cg0 \
  --heart-plan-candidates "${CANDIDATES}" \
  --heart-reference-steps 256 \
  --output-dir "${OUTPUT_ROOT}"

echo ""
echo "Done. Results under ${OUTPUT_ROOT}/"
