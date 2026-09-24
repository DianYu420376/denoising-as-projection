#!/bin/bash
#SBATCH --job-name=mujoco-hybrid-opt
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-15
#SBATCH --output=logs/slurm/hybrid_opt_scale_sweep/%A_%a.out
#SBATCH --error=logs/slurm/hybrid_opt_scale_sweep/%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

SWEEP_LOG_DIR="${REPO_DIR}/logs/slurm/hybrid_opt_scale_sweep"
mkdir -p "${SWEEP_LOG_DIR}"

# Heartbeat immediately so we can tell prolog vs batch-script start.
echo "=== $(date -Is) batch start host=$(hostname) job=${SLURM_ARRAY_JOB_ID:-na}_${SLURM_ARRAY_TASK_ID:-na} ==="

: "${TASK:?Set TASK (walker2d-medium-v2 or halfcheetah-medium-v2)}"
: "${OPT_SCALE:?Set OPT_SCALE for this array job}"
: "${SWEEP_ROOT:?Set SWEEP_ROOT to coarse sweep output directory}"

CKPT="${CKPT:-latest}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"
SEED="${SLURM_ARRAY_TASK_ID}"
W_CG="${W_CG:-0}"
SOLVER="${SOLVER:-ddim}"
SAMPLING_STEPS="${SAMPLING_STEPS:-20}"
TEMPERATURE="${TEMPERATURE:-1.0}"
OPT_GUIDANCE_LAST_STEPS="${OPT_GUIDANCE_LAST_STEPS:-20}"
NOISE_SCHEDULE="${NOISE_SCHEDULE:-cosine}"
DDIM_ETA="${DDIM_ETA:-1}"
OPT_ALPHA_SIGMA_SCALE="${OPT_ALPHA_SIGMA_SCALE:-1}"
SEED_MIN="${SEED_MIN:-1}"
SEED_MAX="${SEED_MAX:-15}"
GUIDANCE_MODE="${GUIDANCE_MODE:-hybrid}"

AGG_JOB_NAME="${AGG_JOB_NAME:-mujoco-hybrid-opt-agg}"

case "${TASK}" in
  walker2d-medium-v2) DEFAULT_SIM_ENV="Walker2d-v2" ;;
  halfcheetah-medium-v2) DEFAULT_SIM_ENV="HalfCheetah-v2" ;;
  hopper-medium-v2) DEFAULT_SIM_ENV="Hopper-v2" ;;
  *) DEFAULT_SIM_ENV="${TASK}" ;;
esac
SIM_ENV_NAME="${SIM_ENV_NAME:-${DEFAULT_SIM_ENV}}"

if [[ "${SIM_ENV_NAME}" == *"-v4"* ]]; then
  echo "ERROR: refusing v4 sim env '${SIM_ENV_NAME}'"
  exit 1
fi

mkdir -p "${SWEEP_LOG_DIR}"

SCALE_LABEL="${OPT_SCALE//./p}"
WCG_LABEL="${W_CG//./p}"
DEFAULT_SCALE_DIR="${SWEEP_ROOT}/by_scale/opt${SCALE_LABEL}_array${SLURM_ARRAY_JOB_ID}"
SCALE_DIR="${SCALE_DIR:-${DEFAULT_SCALE_DIR}}"
mkdir -p "${SCALE_DIR}"
echo "scale_dir=${SCALE_DIR}" > "${SCALE_DIR}/.job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.started"

RUN_NAME="seed${SEED}_hybrid_wcg${WCG_LABEL}_opt${SCALE_LABEL}"
OUT_FILE="${SCALE_DIR}/${RUN_NAME}.json"
AGG_OUTPUT_JSON="${AGG_OUTPUT_JSON:-${SCALE_DIR}/seed_sweep_aggregate.json}"

if [[ "${SLURM_ARRAY_TASK_ID:-}" == "${SEED_MIN}" ]] && [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  AGG_OUT="${SWEEP_LOG_DIR}/${SLURM_ARRAY_JOB_ID}_aggregate_%j.out"
  AGG_JSON="${AGG_OUTPUT_JSON}"
  sbatch --parsable \
    --job-name="${AGG_JOB_NAME}" \
    ${SBATCH_ACCOUNT:+--account=${SBATCH_ACCOUNT}} \
    --partition=cpu \
    --cpus-per-task=1 \
    --mem=4G \
    --time=00:15:00 \
    --output="${AGG_OUT}" \
    --dependency=afterany:"${SLURM_ARRAY_JOB_ID}" \
      --export=ALL,ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID}",LOG_DIR="${SWEEP_LOG_DIR}",RESULTS_DIR="${SCALE_DIR}",SEED_MIN="${SEED_MIN}",SEED_MAX="${SEED_MAX}",OUTPUT_JSON="${AGG_JSON}" \
    ${SCRIPT_DIR}/run_hopper_opt_seed_sweep_aggregate.sbatch
fi

ALPHA_SIGMA_FLAG=()
if [[ "${OPT_ALPHA_SIGMA_SCALE}" == "1" ]]; then
  ALPHA_SIGMA_FLAG=(--optimization-guidance-alpha-sigma-scale)
fi

echo "============================================================"
echo "hybrid opt_scale seed sweep"
echo "============================================================"
echo "task: ${TASK}"
echo "sim_env: ${SIM_ENV_NAME}"
echo "array_job_id: ${SLURM_ARRAY_JOB_ID}"
echo "array_task_id: ${SLURM_ARRAY_TASK_ID}"
echo "ckpt: ${CKPT}"
echo "guidance_mode: ${GUIDANCE_MODE}"
echo "w_cg: ${W_CG}"
echo "opt_scale: ${OPT_SCALE}"
echo "opt_guidance_last_steps: ${OPT_GUIDANCE_LAST_STEPS}"
echo "optimization_guidance_alpha_sigma_scale: ${OPT_ALPHA_SIGMA_SCALE}"
echo "solver: ${SOLVER}, temp: ${TEMPERATURE}, eta: ${DDIM_ETA}"
echo "sweep_root: ${SWEEP_ROOT}"
echo "scale_dir: ${SCALE_DIR}"
echo "output_file: ${OUT_FILE}"
echo "============================================================"

"${PYTHON}" -u pipelines/eval_hopper_trajectory.py \
  --task "${TASK}" \
  --ckpt "${CKPT}" \
  --seed "${SEED}" \
  --num_episodes "${NUM_EPISODES}" \
  --max_steps "${MAX_STEPS}" \
  --log_interval "${LOG_INTERVAL}" \
  --sim-env-name "${SIM_ENV_NAME}" \
  --solver "${SOLVER}" \
  --sampling-steps "${SAMPLING_STEPS}" \
  --temperature "${TEMPERATURE}" \
  --optimization-guidance-last-steps "${OPT_GUIDANCE_LAST_STEPS}" \
  --noise-schedule "${NOISE_SCHEDULE}" \
  --ddim-eta "${DDIM_ETA}" \
  --guidance_mode "${GUIDANCE_MODE}" \
  --optimization_guidance_scale "${OPT_SCALE}" \
  --w_cg "${W_CG}" \
  "${ALPHA_SIGMA_FLAG[@]}" \
  --output "${OUT_FILE}" \
  2>&1 | tee "${SCALE_DIR}/${RUN_NAME}.log"

echo ""
echo ">>> Finished ${RUN_NAME}"
echo "Results: ${OUT_FILE}"
