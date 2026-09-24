#!/bin/bash
#SBATCH --job-name=guid_synth_sweep
#SBATCH --time=04:00:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_guidance_synthetic_sweep_%j.out
#SBATCH --error=slurm_guidance_synthetic_sweep_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/guidance_synthetic}"

echo "============================================================"
echo "Synthetic guidance adaptive sweep — Test A & B"
echo "============================================================"
echo "hostname: $(hostname)"
echo "output: ${OUTPUT_ROOT}"
nvidia-smi || true

"${PYTHON}" pipelines/guidance_synthetic_sweep.py \
  --test both \
  --device cuda:0 \
  --output-root "${OUTPUT_ROOT}" \
  --num-samples 512 \
  --validation-samples 2048 \
  --sampling-steps 20 \
  --diffusion-steps 50 \
  --training-diffusion-steps 50 \
  --inference-diffusion-steps 100

echo ""
echo "Sweep complete. Results:"
echo "  ${OUTPUT_ROOT}/test_a/sweep/sweep_results.json"
echo "  ${OUTPUT_ROOT}/test_b/sweep/sweep_results.json"
echo "  ${OUTPUT_ROOT}/sweep_combined_summary.json"
