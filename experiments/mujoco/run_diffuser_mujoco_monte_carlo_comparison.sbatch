#!/bin/bash
#SBATCH --job-name=diffuser-mc-cmp
#SBATCH --time=2-00:00:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_diffuser_mujoco_monte_carlo_comparison_%j.out
#SBATCH --error=slurm_diffuser_mujoco_monte_carlo_comparison_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

D4RL_DIR="${D4RL_DIR:-${D4RL_DIR}}"

TASK="${TASK:-hopper-medium-v2}"
CKPT="${CKPT:-latest}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-50}"
NUM_EPISODES="${NUM_EPISODES:-1}"
NUM_CANDIDATES="${NUM_CANDIDATES:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
# Default output goes to guidance_comparison_monte_carlo/<timestamp>/ via eval script.

echo "============================================================"
echo "Monte Carlo baseline comparison (w_cg=0, standard mode)"
echo "============================================================"
echo "hostname: $(hostname)"
echo "task: ${TASK} ckpt: ${CKPT}"
echo "seeds: ${SEED_START}..${SEED_END}"
echo "num_episodes: ${NUM_EPISODES} num_candidates: ${NUM_CANDIDATES}"
echo "config: monte_carlo_w_cg0 (guidance_mode=standard, w_cg=0)"
echo "Same params as standard_w_cg0p3 except w_cg=0"
echo "============================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

ARGS=(
  --repo-dir "${REPO_DIR}"
  --python "${PYTHON}"
  --task "${TASK}"
  --ckpt "${CKPT}"
  --run-suffix "${RUN_SUFFIX}"
  --seed-start "${SEED_START}"
  --seed-end "${SEED_END}"
  --num-episodes "${NUM_EPISODES}"
  --num-candidates "${NUM_CANDIDATES}"
  --configs "monte_carlo_w_cg0"
)
if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

"${PYTHON}" -u pipelines/guidance_comparison_eval.py "${ARGS[@]}"

echo ""
echo "Monte Carlo comparison completed."
