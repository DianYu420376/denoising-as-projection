#!/bin/bash
#SBATCH --job-name=guid_grid_s50
#SBATCH --time=03:00:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_guidance_synthetic_grid_s50_%j.out
#SBATCH --error=slurm_guidance_synthetic_grid_s50_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/guidance_synthetic}"

echo "Grid sweep: w_cg & opt_scale in [0,10], matched steps 50/50/50"
nvidia-smi || true

COMMON=(
  --device cuda:0
  --output-root "${OUTPUT_ROOT}"
  --diffusion-steps 50
  --training-diffusion-steps 50
  --sampling-steps 50
  --optimization-guidance-last-steps 25
  --num-samples 512
  --grid-min 0
  --grid-max 10
  --grid-step 1
)

"${PYTHON}" pipelines/guidance_synthetic_grid_sweep.py --test both "${COMMON[@]}"

echo ""
echo "Test A visualization (contour + feasible region + diffusion paths)"
"${PYTHON}" pipelines/guidance_synthetic_visualize_test_a.py \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda:0 \
  --diffusion-steps 50 \
  --training-diffusion-steps 50 \
  --sampling-steps 50 \
  --optimization-guidance-last-steps 25 \
  --n-paths 8

echo ""
echo "Done."
echo "  ${OUTPUT_ROOT}/grid_s50_summary.json"
echo "  ${OUTPUT_ROOT}/test_a/sweep/grid_s50.json"
echo "  ${OUTPUT_ROOT}/test_a/figures/test_a_subspace_contour_s50.png"
