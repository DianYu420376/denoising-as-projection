#!/bin/bash
#SBATCH --job-name=hopper-sweep-agg
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_hopper_opt_seed_sweep_aggregate_%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

: "${ARRAY_JOB_ID:?Set ARRAY_JOB_ID to the SLURM array job id}"

SEED_MIN="${SEED_MIN:-1}"
SEED_MAX="${SEED_MAX:-15}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs/slurm/hybrid_guidance_seed_sweep}"
RESULTS_DIR="${RESULTS_DIR:-}"
OUTPUT_JSON="${OUTPUT_JSON:-${REPO_DIR}/seed_sweep_aggregate_${ARRAY_JOB_ID}.json}"

echo "============================================================"
echo "Seed sweep aggregate"
echo "============================================================"
echo "array_job_id: ${ARRAY_JOB_ID}"
echo "log_dir:      ${LOG_DIR}"
echo "results_dir:  ${RESULTS_DIR:-<none>}"
echo "seed_range:   ${SEED_MIN}-${SEED_MAX}"
echo "output_json:  ${OUTPUT_JSON}"
echo "============================================================"

AGG_ARGS=(
  --repo-dir "${REPO_DIR}"
  --seed-min "${SEED_MIN}"
  --seed-max "${SEED_MAX}"
  --job-ids "${ARRAY_JOB_ID}"
  --log-dir "${LOG_DIR}"
  --no-merge-families
  --output-json "${OUTPUT_JSON}"
)
if [[ -n "${RESULTS_DIR}" ]]; then
  AGG_ARGS+=(--results-dir "${RESULTS_DIR}")
fi

"${PYTHON}" -u pipelines/aggregate_seed_sweep_from_out.py "${AGG_ARGS[@]}"

echo ""
echo ">>> Aggregate finished"
