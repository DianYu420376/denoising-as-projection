#!/bin/bash
#SBATCH --job-name=collect_unicycle_full64
#SBATCH --time=01:00:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_collect_unicycle_full64_%j.out
#SBATCH --error=slurm_collect_unicycle_full64_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

TARGET_FULL="${TARGET_FULL:-6000}"
SEED="${SEED:-0}"

echo "Collecting ${TARGET_FULL} full-horizon (64-step) unicycle episodes..."
"${PYTHON}" pipelines/collect_unicycle_offline_data.py \
  --output-dir results/unicycle_offline/full64_6k \
  --require-full-horizon \
  --target-full-episodes "${TARGET_FULL}" \
  --horizon 64 \
  --seed "${SEED}" \
  --num-plots 12

echo "Done. Dataset: results/unicycle_offline/full64_6k/unicycle_offline.hdf5"
