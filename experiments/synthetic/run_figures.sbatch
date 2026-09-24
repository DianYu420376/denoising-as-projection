#!/bin/bash
#SBATCH --job-name=guid_synth_fig
#SBATCH --time=01:00:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_guidance_synthetic_figures_%j.out
#SBATCH --error=slurm_guidance_synthetic_figures_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/guidance_synthetic}"

echo "Figure generation: Test A & B, last_steps=50 (resample + cache trajectories)"
nvidia-smi || true

"${PYTHON}" pipelines/guidance_synthetic_visualize_subspace.py \
  --test both \
  --resample \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda:0 \
  --diffusion-steps 50 \
  --training-diffusion-steps 50 \
  --sampling-steps 50 \
  --optimization-guidance-last-steps 50 \
  --n-paths 5 \
  --w-cg 0.6 \
  --opt-scale 15 \
  --w-cg-test-b 10 \
  --opt-scale-test-b 10 \
  --plot-limit-a 2.0 \
  --plot-limit-b 1.8

echo "Done."
