#!/bin/bash
#SBATCH --job-name=guid_wcg_resweep
#SBATCH --time=02:00:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_guidance_synthetic_wcg_resweep_%j.out
#SBATCH --error=slurm_guidance_synthetic_wcg_resweep_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

echo "Balanced w_cg resweep (score = reward - 2.0 * violation)"
nvidia-smi || true

"${PYTHON}" pipelines/guidance_synthetic_wcg_resweep.py \
  --test both \
  --device cuda:0 \
  --violation-penalty 2.0 \
  --num-samples 1024 \
  --validation-samples 4096 \
  --coarse-step 0.5 \
  --refine-step 0.1 \
  --refine-radius 2.5

echo "Done."
