#!/bin/bash
#SBATCH --job-name=heart_plan_opt_s100
#SBATCH --time=00:45:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_heart_plan_opt_sweep_s100_%j.out
#SBATCH --error=slurm_heart_plan_opt_sweep_s100_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

CKPT_PATH="${CKPT_PATH:-results/diffuser_unicycle/Unicycle-v0/full64_6k/diffusion_ckpt_latest.pt}"
SEED="${SEED:-0}"
CANDIDATES="${CANDIDATES:-100}"
SAMPLING_STEPS="${SAMPLING_STEPS:-100}"
OPT_GUIDANCE_LAST_STEPS="${OPT_GUIDANCE_LAST_STEPS:-50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/unicycle_eval/heart_plan_opt_sweep_s100_100_candidates}"

# Optimization guidance: w_cg=0, shift x_t by eta * grad(R) on the last N reverse steps.
CONFIGS=(
  optimization_scale_30
  optimization_scale_50
  optimization_scale_100
  optimization_scale_200
)

echo "Heart plan optimization-guidance sweep (DDIM ${SAMPLING_STEPS} reverse steps)"
echo "checkpoint: ${CKPT_PATH}"
echo "candidates per scale: ${CANDIDATES}"
echo "optimization_guidance_last_steps: ${OPT_GUIDANCE_LAST_STEPS}"
echo "configs: ${CONFIGS[*]}"

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
    --sampling-steps "${SAMPLING_STEPS}" \
    --diffusion-steps 100 \
    --temperature 1.0 \
    --use-ema \
    --heart-config-names "${cfg}" \
    --heart-plan-candidates "${CANDIDATES}" \
    --heart-reference-steps 256 \
    --optimization-guidance-last-steps "${OPT_GUIDANCE_LAST_STEPS}" \
    --output-dir "${OUTPUT_ROOT}/${cfg}"
done

echo ""
echo "Sweep completed. Results under ${OUTPUT_ROOT}/"
