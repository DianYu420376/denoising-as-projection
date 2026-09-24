#!/bin/bash
#SBATCH --job-name=unicycle_eval
#SBATCH --time=04:00:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_unicycle_eval_%j.out
#SBATCH --error=slurm_unicycle_eval_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

EVAL_TASK="${EVAL_TASK:-feasibility}"
CKPT="${CKPT:-latest}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
SEED="${SEED:-0}"

echo "============================================================"
echo "unicycle_eval: dynamic feasibility (unguided) + optional heart tracking"
echo "============================================================"
echo "eval_task: ${EVAL_TASK}"
echo "ckpt: ${CKPT}"
echo "run_suffix: ${RUN_SUFFIX:-<default>}"
echo "============================================================"

EVAL_ARGS=(
  --eval-task "${EVAL_TASK}"
  --ckpt "${CKPT}"
  --seed "${SEED}"
  --device cuda:0
)
if [[ -n "${RUN_SUFFIX}" ]]; then
  EVAL_ARGS+=(--run-suffix "${RUN_SUFFIX}")
fi

"${PYTHON}" pipelines/unicycle_eval.py "${EVAL_ARGS[@]}"

echo ""
echo "Evaluation completed."
