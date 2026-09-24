#!/bin/bash
#SBATCH --job-name=diffuser-guidance-smoke
#SBATCH --time=01:00:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_diffuser_mujoco_guidance_comparison_smoke_%j.out
#SBATCH --error=slurm_diffuser_mujoco_guidance_comparison_smoke_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

D4RL_DIR="${D4RL_DIR:-${D4RL_DIR}}"

TASK="${TASK:-hopper-medium-v2}"
CKPT="${CKPT:-latest}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

echo "============================================================"
echo "Guidance comparison SMOKE test"
echo "============================================================"
echo "hostname: $(hostname)"
echo "task: ${TASK} ckpt: ${CKPT}"
echo "seeds: 0 only | num_episodes: 1 | num_candidates: 4"
echo "configs: standard + optimization scales 0.1"
echo "============================================================"

"${PYTHON}" -u pipelines/guidance_comparison_eval.py \
  --repo-dir "${REPO_DIR}" \
  --python "${PYTHON}" \
  --task "${TASK}" \
  --ckpt "${CKPT}" \
  --run-suffix "${RUN_SUFFIX}" \
  --seed-start 0 \
  --seed-end 0 \
  --num-episodes 1 \
  --num-candidates 4 \
  --opt-scales "0.1"

echo ""
echo "SMOKE PASS: guidance comparison completed without errors."
