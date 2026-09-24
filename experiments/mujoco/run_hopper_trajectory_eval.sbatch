#!/bin/bash
#SBATCH --job-name=hopper-traj-eval
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-15
#SBATCH --output=slurm_hopper_traj_eval_%A_%a.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

CKPT="${CKPT:-latest}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
TEMPERATURE="${TEMPERATURE:-1.0}"
W_CG="${W_CG:-0.5}"

OUTPUT="results/diffuser_d4rl_mujoco/hopper-medium-v2/trajectory_eval_${CKPT}_seed${SEED}.json"

${PYTHON} -u pipelines/eval_hopper_trajectory.py \
  --task hopper-medium-v2 \
  --ckpt "${CKPT}" \
  --num_episodes "${NUM_EPISODES}" \
  --max_steps "${MAX_STEPS}" \
  --log_interval "${LOG_INTERVAL}" \
  --seed "${SEED}" \
  --temperature "${TEMPERATURE}" \
  --output "${OUTPUT}"\
  --w_cg "${W_CG}"
