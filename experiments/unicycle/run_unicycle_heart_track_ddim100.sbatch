#!/bin/bash
#SBATCH --job-name=heart_track_ddim100
#SBATCH --time=06:00:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_heart_track_ddim100_%j.out
#SBATCH --error=slurm_heart_track_ddim100_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

CKPT_PATH="${CKPT_PATH:-results/diffuser_unicycle/Unicycle-v0/diffusion_ckpt_latest.pt}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/unicycle_eval/heart_track_ddim100_rawgrad}"

CONFIGS=(
  standard_w_cg0p3
  standard_w_cg0p5
  standard_w_cg1
)

echo "Heart MPC tracking: DDIM 100 reverse steps, raw reward gradient"
echo "checkpoint: ${CKPT_PATH}"
echo "configs: ${CONFIGS[*]}"

for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "========== ${cfg} =========="
  "${PYTHON}" pipelines/unicycle_eval.py \
    --eval-task heart \
    --ckpt-path "${CKPT_PATH}" \
    --seed "${SEED}" \
    --device cuda:0 \
    --solver ddim \
    --sample-step-schedule uniform \
    --sampling-steps 100 \
    --diffusion-steps 20 \
    --temperature 1.0 \
    --use-ema \
    --tracking-steps 256 \
    --init-from-heart-start \
    --heart-mpc-execute-steps 10 \
    --heart-config-names "${cfg}" \
    --heart-checkpoint-steps 0 64 128 192 240 \
    --output-dir "${OUTPUT_ROOT}/${cfg}"
done

echo ""
echo "Done. Results under ${OUTPUT_ROOT}/"
