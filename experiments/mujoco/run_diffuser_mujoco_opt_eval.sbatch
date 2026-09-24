#!/bin/bash
#SBATCH --job-name=diffuser-opt-eval
#SBATCH --time=08:00:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_diffuser_mujoco_opt_eval_%j.out
#SBATCH --error=slurm_diffuser_mujoco_opt_eval_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

D4RL_DIR="${D4RL_DIR:-${D4RL_DIR}}"

TASK="${TASK:-hopper-medium-v2}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
CKPT="${CKPT:-latest}"
SEED="${SEED:-0}"

# Full eval uses config defaults unless overridden below.
NUM_EPISODES="${NUM_EPISODES:-3}"
NUM_CANDIDATES="${NUM_CANDIDATES:-64}"
NUM_ENVS="${NUM_ENVS:-1}"
OPT_SCALES="${OPT_SCALES:-0.01 0.05 0.1}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: venv python not found at ${PYTHON}"
  exit 1
fi

export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${D4RL_DIR}:${PYTHONPATH:-}"

# Separate eval artifacts from training checkpoints and prior slurm logs.
RUN_TAG="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
if [[ -n "${RUN_SUFFIX}" ]]; then
  EVAL_BASE="${REPO_DIR}/results/diffuser_d4rl_mujoco/${TASK}/${RUN_SUFFIX}"
else
  EVAL_BASE="${REPO_DIR}/results/diffuser_d4rl_mujoco/${TASK}"
fi
EVAL_DIR="${EVAL_BASE}/opt_guidance_eval/${RUN_TAG}"
mkdir -p "${EVAL_DIR}"

SUMMARY_FILE="${EVAL_DIR}/summary.txt"

echo "============================================================" | tee "${SUMMARY_FILE}"
echo "diffuser_d4rl_mujoco optimization-guidance full eval" | tee -a "${SUMMARY_FILE}"
echo "============================================================" | tee -a "${SUMMARY_FILE}"
echo "hostname: $(hostname)" | tee -a "${SUMMARY_FILE}"
echo "repo: ${REPO_DIR}" | tee -a "${SUMMARY_FILE}"
echo "task: ${TASK}" | tee -a "${SUMMARY_FILE}"
echo "ckpt: ${CKPT}" | tee -a "${SUMMARY_FILE}"
echo "seed: ${SEED}" | tee -a "${SUMMARY_FILE}"
echo "num_episodes: ${NUM_EPISODES}" | tee -a "${SUMMARY_FILE}"
echo "num_candidates: ${NUM_CANDIDATES}" | tee -a "${SUMMARY_FILE}"
echo "eval_dir: ${EVAL_DIR}" | tee -a "${SUMMARY_FILE}"
echo "checkpoints (read-only): results/diffuser_d4rl_mujoco/${TASK}/" | tee -a "${SUMMARY_FILE}"
echo "============================================================" | tee -a "${SUMMARY_FILE}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee -a "${SUMMARY_FILE}" || true
fi

run_inference() {
  local run_name="$1"
  shift
  local log_file="${EVAL_DIR}/${run_name}.log"
  echo "" | tee -a "${SUMMARY_FILE}"
  echo ">>> Running ${run_name}" | tee -a "${SUMMARY_FILE}"
  echo "    log: ${log_file}" | tee -a "${SUMMARY_FILE}"
  INFER_ARGS=(
    task="${TASK}"
    mode=inference
    device=cuda:0
    seed="${SEED}"
    ckpt="${CKPT}"
  )
  if [[ -n "${RUN_SUFFIX}" ]]; then
    INFER_ARGS+=(run_suffix="${RUN_SUFFIX}")
  fi

  "${PYTHON}" pipelines/diffuser_d4rl_mujoco.py \
    "${INFER_ARGS[@]}" \
    num_episodes="${NUM_EPISODES}" \
    num_candidates="${NUM_CANDIDATES}" \
    num_envs="${NUM_ENVS}" \
    render_video=false \
    render_ignore_termination=false \
    max_render_steps=1000 \
    enable_wandb=false \
    "$@" \
    2>&1 | tee "${log_file}"
  echo ">>> Finished ${run_name}" | tee -a "${SUMMARY_FILE}"
}

# 1) Standard classifier-guidance baseline (existing behavior).
run_inference "standard_w_cg_default" \
  guidance_mode=standard \
  optimization_guidance_scale=0.0

# 2) Optimization-guidance sweeps (requires w_cg=0).
for scale in ${OPT_SCALES}; do
  scale_label="${scale//./p}"
  run_inference "optimization_scale_${scale_label}" \
    guidance_mode=optimization \
    optimization_guidance_scale="${scale}" \
    task.w_cg=0.0
done

echo "" | tee -a "${SUMMARY_FILE}"
echo "Full optimization-guidance eval completed." | tee -a "${SUMMARY_FILE}"
echo "All per-run logs and summary are under: ${EVAL_DIR}" | tee -a "${SUMMARY_FILE}"
echo "Training checkpoints and prior slurm logs were not modified." | tee -a "${SUMMARY_FILE}"
