#!/bin/bash
#SBATCH --job-name=dyn-std-vs-opt
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/dynamic_feasibility/%x_%j.out
#SBATCH --error=logs/slurm/dynamic_feasibility/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

: "${TASK:?Set TASK (hopper-medium-v2, halfcheetah-medium-v2, or walker2d-medium-v2)}"

CKPT="${CKPT:-latest}"
SEED_START="${SEED_START:-0}"
NUM_SEEDS="${NUM_SEEDS:-10}"
NUM_CANDIDATES="${NUM_CANDIDATES:-64}"
SAMPLING_STEPS="${SAMPLING_STEPS:-20}"
CONFIG_PRESET="${CONFIG_PRESET:-ep150_std_vs_opt}"
OPT_SCALE="${OPT_SCALE:-}"
OPT_W_CG="${OPT_W_CG:-}"

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

JOB_ID="${SLURM_JOB_ID:-local}"
RUN_TAG="${RUN_TAG:-${CONFIG_PRESET}_besthybrid_$(date +%Y%m%d_%H%M%S)_job${JOB_ID}}"
OUTPUT_DIR="${REPO_DIR}/results/diffuser_d4rl_mujoco/${TASK}/dynamic_feasibility/${RUN_TAG}"
SLURM_LOG_DIR="${REPO_DIR}/logs/slurm/dynamic_feasibility"

mkdir -p "${OUTPUT_DIR}" "${SLURM_LOG_DIR}"

OPT_SCALE_FLAG=()
if [[ -n "${OPT_SCALE}" ]]; then
  OPT_SCALE_FLAG=(--opt-scale "${OPT_SCALE}")
fi
OPT_W_CG_FLAG=()
if [[ -n "${OPT_W_CG}" ]]; then
  OPT_W_CG_FLAG=(--opt-w-cg "${OPT_W_CG}")
fi

echo "============================================================"
echo "Dynamic feasibility: monte carlo + standard + hybrid (raw + normalized L2)"
echo "============================================================"
echo "hostname: $(hostname)"
echo "job_id: ${JOB_ID}"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>/dev/null || true
echo "task: ${TASK}"
echo "ckpt: ${CKPT}"
echo "sim_env: ${SIM_ENV_NAME}"
echo "config_preset: ${CONFIG_PRESET}"
echo "configs: monte_carlo_w_cg0 + standard_repo + best_hybrid (from task yaml)"
echo "opt_scale override: ${OPT_SCALE:-<task preset>}"
echo "opt_w_cg override: ${OPT_W_CG:-<task preset>}"
echo "num_seeds: ${NUM_SEEDS} (seed_start=${SEED_START})"
echo "num_candidates: ${NUM_CANDIDATES}"
echo "sampling_steps: ${SAMPLING_STEPS}"
echo "output_dir: ${OUTPUT_DIR}"
echo "============================================================"

"${PYTHON}" -u pipelines/dynamic_feasibility_hopper_v2_comparison.py \
  --repo-dir "${REPO_DIR}" \
  --task "${TASK}" \
  --ckpt "${CKPT}" \
  --sim-env-name "${SIM_ENV_NAME}" \
  --seed-start "${SEED_START}" \
  --num-seeds "${NUM_SEEDS}" \
  --num-candidates "${NUM_CANDIDATES}" \
  --sampling-steps "${SAMPLING_STEPS}" \
  --config-preset "${CONFIG_PRESET}" \
  "${OPT_SCALE_FLAG[@]}" \
  "${OPT_W_CG_FLAG[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-json "${OUTPUT_DIR}/summary.json"

echo ""
echo "Done. Results in ${OUTPUT_DIR}/"
