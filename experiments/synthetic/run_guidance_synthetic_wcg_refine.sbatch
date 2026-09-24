#!/bin/bash
#SBATCH --job-name=guid_wcg_ref
#SBATCH --time=01:00:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_guidance_synthetic_wcg_refine_%j.out
#SBATCH --error=slurm_guidance_synthetic_wcg_refine_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/guidance_synthetic}"

echo "Refined w_cg sweep in [0, 1] step=0.05 | matched steps 50/50/50"
nvidia-smi || true

"${PYTHON}" pipelines/guidance_synthetic_grid_sweep.py \
  --test both \
  --w-cg-only \
  --output-tag wcg_refine_0_1_s50 \
  --device cuda:0 \
  --output-root "${OUTPUT_ROOT}" \
  --diffusion-steps 50 \
  --training-diffusion-steps 50 \
  --sampling-steps 50 \
  --optimization-guidance-last-steps 25 \
  --num-samples 512 \
  --grid-min 0 \
  --grid-max 1 \
  --grid-step 0.05

echo "Done."
echo "  ${OUTPUT_ROOT}/wcg_refine_0_1_s50_summary.json"
