#!/bin/bash
#SBATCH --job-name=mujoco-ep150-std
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/ep150_thorough_sweep/%x_%j.out
#SBATCH --error=logs/slurm/ep150_thorough_sweep/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

# 150 seeds (0-149), 1 episode per seed, repo default standard guidance.
: "${TASK:?Set TASK (e.g. walker2d-medium-v2 or halfcheetah-medium-v2)}"

CKPT="${CKPT:-latest}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-149}"
STATS_INTERVAL="${STATS_INTERVAL:-10}"
MAX_STEPS="${MAX_STEPS:-1000}"

# Rollout on standalone Gym v2 physics envs (mujoco_py), not Gymnasium v4.
case "${TASK}" in
  walker2d-medium-v2) DEFAULT_SIM_ENV="Walker2d-v2" ;;
  halfcheetah-medium-v2) DEFAULT_SIM_ENV="HalfCheetah-v2" ;;
  hopper-medium-v2) DEFAULT_SIM_ENV="Hopper-v2" ;;
  *) DEFAULT_SIM_ENV="${TASK}" ;;
esac
SIM_ENV_NAME="${SIM_ENV_NAME:-${DEFAULT_SIM_ENV}}"

if [[ "${SIM_ENV_NAME}" == *"-v4"* ]]; then
  echo "ERROR: refusing v4 sim env '${SIM_ENV_NAME}'; use Walker2d-v2 / HalfCheetah-v2 / Hopper-v2"
  exit 1
fi

# Repo defaults from configs/diffuser/mujoco/mujoco.yaml (+ task w_cg from task yaml).
SOLVER="${SOLVER:-ddpm}"
SAMPLING_STEPS="${SAMPLING_STEPS:-20}"
TEMPERATURE="${TEMPERATURE:-0.5}"
GUIDANCE_MODE="${GUIDANCE_MODE:-standard}"
OPT_SCALE="${OPT_SCALE:-0.0}"

JOB_ID="${SLURM_JOB_ID:-local}"
NUM_SEEDS=$((SEED_END - SEED_START + 1))
RUN_TAG="${RUN_TAG:-standard_repo_seeds${SEED_START}-${SEED_END}_job${JOB_ID}}"

SWEEP_DIR="${REPO_DIR}/results/diffuser_d4rl_mujoco/${TASK}/ep150_thorough_sweep/${RUN_TAG}"
LOG_DIR="${SWEEP_DIR}/logs"
JSON_DIR="${SWEEP_DIR}/json"
INTERIM_DIR="${SWEEP_DIR}/interim"
SLURM_LOG_DIR="${REPO_DIR}/logs/slurm/ep150_thorough_sweep"
RUN_NAME="${RUN_NAME:-standard_repo}"

mkdir -p "${LOG_DIR}" "${JSON_DIR}" "${INTERIM_DIR}" "${SLURM_LOG_DIR}"

echo "============================================================"
echo "MuJoCo ep150 standard guidance: ${TASK}"
echo "============================================================"
echo "hostname: $(hostname)"
echo "job_id: ${JOB_ID}"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>/dev/null || true
echo "task: ${TASK}"
echo "ckpt: ${CKPT}"
echo "seeds: ${SEED_START}..${SEED_END} (${NUM_SEEDS} total)"
echo "stats_interval: every ${STATS_INTERVAL} seeds"
echo "max_steps: ${MAX_STEPS}"
echo "sim_env: ${SIM_ENV_NAME}"
echo "guidance_mode: ${GUIDANCE_MODE}"
echo "solver: ${SOLVER}, sampling_steps: ${SAMPLING_STEPS}, temperature: ${TEMPERATURE}"
echo "w_cg: from task yaml (repo default)"
echo "sweep_dir: ${SWEEP_DIR}"
echo "============================================================"

LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
OUT_JSON="${JSON_DIR}/${RUN_NAME}.json"
GPU_LOG="${LOG_DIR}/${RUN_NAME}.gpu0.stdout"

echo ""
echo ">>> Starting ${RUN_NAME}"
echo "    method log: ${LOG_FILE}"
echo "    gpu stdout: ${GPU_LOG}"
echo "    output json: ${OUT_JSON}"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u pipelines/run_ep150_config_seed_sweep.py \
  --task "${TASK}" \
  --ckpt "${CKPT}" \
  --run-name "${RUN_NAME}" \
  --log-file "${LOG_FILE}" \
  --output "${OUT_JSON}" \
  --interim-dir "${INTERIM_DIR}" \
  --seed-start "${SEED_START}" \
  --seed-end "${SEED_END}" \
  --stats-interval "${STATS_INTERVAL}" \
  --max-steps "${MAX_STEPS}" \
  --sim-env-name "${SIM_ENV_NAME}" \
  --solver "${SOLVER}" \
  --sampling-steps "${SAMPLING_STEPS}" \
  --temperature "${TEMPERATURE}" \
  --guidance_mode "${GUIDANCE_MODE}" \
  --optimization_guidance_scale "${OPT_SCALE}" \
  > "${GPU_LOG}" 2>&1

echo ">>> Finished ${RUN_NAME}"

SWEEP_DIR="${SWEEP_DIR}" JSON_DIR="${JSON_DIR}" SEED_START="${SEED_START}" SEED_END="${SEED_END}" TASK="${TASK}" "${PYTHON}" -u - <<'PY'
import json
import math
import os
from pathlib import Path

sweep_dir = Path(os.environ["SWEEP_DIR"])
json_dir = Path(os.environ["JSON_DIR"])
task = os.environ["TASK"]
seed_start = int(os.environ["SEED_START"])
seed_end = int(os.environ["SEED_END"])
num_seeds = seed_end - seed_start + 1

rows = []
for path in sorted(json_dir.glob("*.json")):
    with open(path) as f:
        data = json.load(f)
    std_reward = float(data.get("std_reward", float("nan")))
    std_norm = float(data.get("std_normalized_score_x100", float("nan")))
    se_reward = float(data.get("se_reward", std_reward / math.sqrt(num_seeds) if num_seeds else float("nan")))
    se_norm = float(data.get("se_normalized_score_x100", std_norm / math.sqrt(num_seeds) if num_seeds else float("nan")))
    rows.append(
        {
            "run": data.get("run_name", path.stem),
            "guidance_mode": data.get("guidance_mode"),
            "optimization_guidance_scale": data.get("optimization_guidance_scale"),
            "w_cg": data.get("w_cg"),
            "solver": data.get("solver"),
            "sampling_steps": data.get("sampling_steps"),
            "temperature": data.get("temperature"),
            "num_seeds": int(data.get("num_seeds", num_seeds)),
            "mean_reward": float(data.get("mean_reward", float("nan"))),
            "std_reward": std_reward,
            "se_reward": se_reward,
            "mean_norm_x100": float(data.get("mean_normalized_score_x100", float("nan"))),
            "std_norm_x100": std_norm,
            "se_norm_x100": se_norm,
            "mean_survival_steps": float(data.get("mean_survival_steps", float("nan"))),
            "result_file": str(path),
        }
    )

summary = {
    "task": task,
    "sweep_dir": str(sweep_dir),
    "seed_start": seed_start,
    "seed_end": seed_end,
    "num_seeds": num_seeds,
    "episodes_per_seed": 1,
    "metric_primary": "mean_normalized_score_x100",
    "ranking": rows,
    "best": rows[0] if rows else None,
}

out = sweep_dir / "sweep_summary.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Ep150 Standard Guidance Summary ===")
for i, row in enumerate(rows, 1):
    print(
        f"{i}. {row['run']:22s} "
        f"mode={row['guidance_mode']:<12} "
        f"w_cg={row['w_cg']!s:>8} "
        f"reward={row['mean_reward']:8.2f} ± {row['se_reward']:6.2f} "
        f"(std={row['std_reward']:7.2f}) "
        f"norm_x100={row['mean_norm_x100']:6.2f} ± {row['se_norm_x100']:5.2f} "
        f"(std={row['std_norm_x100']:5.2f})"
    )
print(f"\nSaved {out}")
PY

echo ""
echo ">>> Sweep complete."
echo "    method logs: ${LOG_DIR}/"
echo "    json results: ${JSON_DIR}/"
echo "    interim snapshots: ${INTERIM_DIR}/"
echo "    slurm logs: ${SLURM_LOG_DIR}/"
