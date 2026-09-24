#!/bin/bash
#SBATCH --job-name=ref_plan_subset
#SBATCH --time=24:00:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_reference_plan_subset_%j.out
#SBATCH --error=slurm_reference_plan_subset_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

CKPT_PATH="${CKPT_PATH:-results/diffuser_unicycle/Unicycle-v0/full64_6k/diffusion_ckpt_latest.pt}"
SEED="${SEED:-0}"
CANDIDATES="${CANDIDATES:-100}"
SAMPLING_STEPS="${SAMPLING_STEPS:-20}"
PLAN_PLOT_TOP_K="${PLAN_PLOT_TOP_K:-5}"
REFERENCE_STEPS="${REFERENCE_STEPS:-64}"
OPT_GUIDANCE_LAST_STEPS="${OPT_GUIDANCE_LAST_STEPS:-10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/unicycle_eval/reference_plan_v1}"

TRAJECTORIES=(
  circle
  random_smooth_1
  random_smooth_3
  random_smooth_7
)

WCG_CONFIGS=(
  standard_w_cg0
  standard_w_cg10
  standard_w_cg20
  standard_w_cg30
  standard_w_cg50
)

OPT_CONFIGS=(
  optimization_scale_10
  optimization_scale_20
  optimization_scale_30
  optimization_scale_40
)

COMMON_ARGS=(
  --eval-task reference_plan
  --ckpt-path "${CKPT_PATH}"
  --seed "${SEED}"
  --device cuda:0
  --solver ddim
  --sample-step-schedule uniform
  --sampling-steps "${SAMPLING_STEPS}"
  --temperature 1.0
  --use-ema
  --heart-plan-candidates "${CANDIDATES}"
  --plan-plot-top-k "${PLAN_PLOT_TOP_K}"
  --reference-steps "${REFERENCE_STEPS}"
)

echo "Reference plan subset (DDIM ${SAMPLING_STEPS}, ${CANDIDATES} candidates, top-${PLAN_PLOT_TOP_K} plots)"
echo "checkpoint: ${CKPT_PATH}"
echo "trajectories: ${TRAJECTORIES[*]}"
echo "output: ${OUTPUT_ROOT}"

for traj in "${TRAJECTORIES[@]}"; do
  for cfg in "${WCG_CONFIGS[@]}"; do
    echo ""
    echo "========== ${traj} | ${cfg} =========="
    "${PYTHON}" pipelines/unicycle_eval.py \
      "${COMMON_ARGS[@]}" \
      --reference-name "${traj}" \
      --heart-config-names "${cfg}" \
      --diffusion-steps 20 \
      --output-dir "${OUTPUT_ROOT}/${traj}/${cfg}"
  done

  for cfg in "${OPT_CONFIGS[@]}"; do
    echo ""
    echo "========== ${traj} | ${cfg} =========="
    "${PYTHON}" pipelines/unicycle_eval.py \
      "${COMMON_ARGS[@]}" \
      --reference-name "${traj}" \
      --heart-config-names "${cfg}" \
      --diffusion-steps 100 \
      --training-diffusion-steps 20 \
      --optimization-guidance-last-steps "${OPT_GUIDANCE_LAST_STEPS}" \
      --output-dir "${OUTPUT_ROOT}/${traj}/${cfg}"
  done
done

echo ""
echo "Aggregating analysis log..."
"${PYTHON}" pipelines/reference_plan_aggregate_log.py \
  --root "${OUTPUT_ROOT}"

echo "Done. Results under ${OUTPUT_ROOT}/"
