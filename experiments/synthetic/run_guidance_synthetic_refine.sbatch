#!/bin/bash
#SBATCH --job-name=guid_synth_refine
#SBATCH --time=02:00:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_guidance_synthetic_refine_%j.out
#SBATCH --error=slurm_guidance_synthetic_refine_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

"${PYTHON}" pipelines/guidance_synthetic_refine.py
