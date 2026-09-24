#!/bin/bash
#SBATCH --job-name=heart_plan_wcg_sweep
#SBATCH --time=02:00:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_heart_plan_wcg_sweep_%j.out
#SBATCH --error=slurm_heart_plan_wcg_sweep_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

CKPT_PATH="${CKPT_PATH:-results/diffuser_unicycle/Unicycle-v0/diffusion_ckpt_latest.pt}"
SEED="${SEED:-0}"
CANDIDATES="${CANDIDATES:-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/unicycle_eval/heart_plan_wcg_sweep}"

CONFIGS=(
  standard_w_cg0p3
  standard_w_cg0p5
  standard_w_cg1
  standard_w_cg2
  standard_w_cg5
)

echo "Heart plan w_cg sweep: ${CONFIGS[*]}"
echo "checkpoint: ${CKPT_PATH}"
echo "candidates per w_cg: ${CANDIDATES}"

for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "========== ${cfg} =========="
  "${PYTHON}" pipelines/unicycle_eval.py \
    --eval-task heart_plan \
    --ckpt-path "${CKPT_PATH}" \
    --seed "${SEED}" \
    --device cuda:0 \
    --solver ddim \
    --sample-step-schedule uniform \
    --sampling-steps 20 \
    --diffusion-steps 20 \
    --temperature 1.0 \
    --use-ema \
    --heart-config-names "${cfg}" \
    --heart-plan-candidates "${CANDIDATES}" \
    --heart-reference-steps 256 \
    --output-dir "${OUTPUT_ROOT}/${cfg}"
done

echo ""
echo "Sweep completed. Results under ${OUTPUT_ROOT}/"
