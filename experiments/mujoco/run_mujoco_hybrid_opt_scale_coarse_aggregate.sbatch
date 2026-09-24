#!/bin/bash
#SBATCH --job-name=hyb-meta-agg
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/hybrid_opt_scale_sweep/coarse_meta_agg_%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

: "${SWEEP_ROOT:?Set SWEEP_ROOT}"

MANIFEST="${MANIFEST:-${SWEEP_ROOT}/sweep_manifest.json}"

echo "Meta-aggregate coarse sweep: ${SWEEP_ROOT}"

"${PYTHON}" -u pipelines/aggregate_hybrid_opt_scale_sweep.py \
  --sweep-root "${SWEEP_ROOT}" \
  --manifest "${MANIFEST}" \
  --output-json "${SWEEP_ROOT}/coarse_sweep_summary.json"

echo ">>> Meta-aggregate finished"
