#!/bin/bash
#SBATCH --job-name=mujoco-ep150-hybrid
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/ep150_thorough_sweep/%x_%j.out
#SBATCH --error=logs/slurm/ep150_thorough_sweep/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

: "${TASK:?Set TASK (e.g. walker2d-medium-v2)}"

CKPT="${CKPT:-latest}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-149}"
STATS_INTERVAL="${STATS_INTERVAL:-10}"
MAX_STEPS="${MAX_STEPS:-1000}"

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

SOLVER="${SOLVER:-ddim}"
SAMPLING_STEPS="${SAMPLING_STEPS:-20}"
TEMPERATURE="${TEMPERATURE:-1.0}"
W_CG="${W_CG:-0}"
OPT_GUIDANCE_LAST_STEPS="${OPT_GUIDANCE_LAST_STEPS:-20}"
NOISE_SCHEDULE="${NOISE_SCHEDULE:-cosine}"
DDIM_ETA="${DDIM_ETA:-1}"
OPT_ALPHA_SIGMA_SCALE="${OPT_ALPHA_SIGMA_SCALE:-1}"
GUIDANCE_MODE="${GUIDANCE_MODE:-hybrid}"

# opt_scale values for this sweep (override via OPT_SCALES env: "0.01 0.03 0.05")
if [[ -n "${OPT_SCALES:-}" ]]; then
  read -r -a HYBRID_OPT_SCALES <<< "${OPT_SCALES}"
else
  HYBRID_OPT_SCALES=(0.01 0.03 0.05)
fi

JOB_ID="${SLURM_JOB_ID:-local}"
NUM_SEEDS=$((SEED_END - SEED_START + 1))
RUN_TAG="${RUN_TAG:-hybrid_wcg${W_CG}_optlast${OPT_GUIDANCE_LAST_STEPS}_seeds${SEED_START}-${SEED_END}_job${JOB_ID}}"

SWEEP_DIR="${REPO_DIR}/results/diffuser_d4rl_mujoco/${TASK}/ep150_thorough_sweep/${RUN_TAG}"
LOG_DIR="${SWEEP_DIR}/logs"
JSON_DIR="${SWEEP_DIR}/json"
INTERIM_DIR="${SWEEP_DIR}/interim"
SLURM_LOG_DIR="${REPO_DIR}/logs/slurm/ep150_thorough_sweep"

mkdir -p "${LOG_DIR}" "${JSON_DIR}" "${INTERIM_DIR}" "${SLURM_LOG_DIR}"

WCG_LABEL="${W_CG//./p}"

ALPHA_SIGMA_FLAG=()
if [[ "${OPT_ALPHA_SIGMA_SCALE}" == "1" ]]; then
  ALPHA_SIGMA_FLAG=(--optimization-guidance-alpha-sigma-scale)
fi

echo "============================================================"
echo "MuJoCo ep150 hybrid sweep: ${TASK}"
echo "============================================================"
echo "hostname: $(hostname)"
echo "job_id: ${JOB_ID}"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>/dev/null || true
echo "task: ${TASK}"
echo "sim_env: ${SIM_ENV_NAME}"
echo "ckpt: ${CKPT}"
echo "seeds: ${SEED_START}..${SEED_END} (${NUM_SEEDS} total)"
echo "stats_interval: every ${STATS_INTERVAL} seeds"
echo "max_steps: ${MAX_STEPS}"
echo "guidance_mode: ${GUIDANCE_MODE}"
echo "w_cg: ${W_CG}"
echo "opt_scales: ${HYBRID_OPT_SCALES[*]} (${#HYBRID_OPT_SCALES[@]} total)"
echo "num_gpus: ${NUM_GPUS}"
echo "opt_last: ${OPT_GUIDANCE_LAST_STEPS}"
echo "alpha_sigma_scale: ${OPT_ALPHA_SIGMA_SCALE}"
echo "solver: ${SOLVER}, temp: ${TEMPERATURE}, eta: ${DDIM_ETA}"
echo "sweep_dir: ${SWEEP_DIR}"
echo "============================================================"

run_config() {
  local gpu_id="$1"
  local run_name="$2"
  local opt_scale="$3"
  shift 3

  local log_file="${LOG_DIR}/${run_name}.log"
  local out_json="${JSON_DIR}/${run_name}.json"
  local gpu_log="${LOG_DIR}/${run_name}.gpu${gpu_id}.stdout"

  echo ""
  echo ">>> [GPU ${gpu_id}] Starting ${run_name} (w_cg=${W_CG}, opt_scale=${opt_scale})"
  echo "    method log: ${log_file}"
  echo "    gpu stdout: ${gpu_log}"
  echo "    output json: ${out_json}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON}" -u pipelines/run_ep150_config_seed_sweep.py \
    --task "${TASK}" \
    --ckpt "${CKPT}" \
    --run-name "${run_name}" \
    --log-file "${log_file}" \
    --output "${out_json}" \
    --interim-dir "${INTERIM_DIR}" \
    --seed-start "${SEED_START}" \
    --seed-end "${SEED_END}" \
    --stats-interval "${STATS_INTERVAL}" \
    --max-steps "${MAX_STEPS}" \
    --sim-env-name "${SIM_ENV_NAME}" \
    --solver "${SOLVER}" \
    --sampling-steps "${SAMPLING_STEPS}" \
    --temperature "${TEMPERATURE}" \
    --optimization-guidance-last-steps "${OPT_GUIDANCE_LAST_STEPS}" \
    --noise-schedule "${NOISE_SCHEDULE}" \
    --ddim-eta "${DDIM_ETA}" \
    --guidance_mode "${GUIDANCE_MODE}" \
    --optimization_guidance_scale "${opt_scale}" \
    --w_cg "${W_CG}" \
    "${ALPHA_SIGMA_FLAG[@]}" \
    "$@" \
    > "${gpu_log}" 2>&1

  echo ">>> [GPU ${gpu_id}] Finished ${run_name}"
}

run_config_bg() {
  run_config "$@" &
}

scale_label() {
  echo "${1//./p}"
}

NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}"
if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "ERROR: invalid NUM_GPUS='${NUM_GPUS}'"
  exit 1
fi

FAIL=0
total="${#HYBRID_OPT_SCALES[@]}"
wave_start=0
wave_idx=0
while [[ "${wave_start}" -lt "${total}" ]]; do
  wave_idx=$((wave_idx + 1))
  wave_end=$((wave_start + NUM_GPUS))
  if [[ "${wave_end}" -gt "${total}" ]]; then
    wave_end="${total}"
  fi
  wave_count=$((wave_end - wave_start))
  echo ""
  echo ">>> Wave ${wave_idx}: launching ${wave_count} configs (${wave_start}..$((wave_end - 1))) on GPUs 0-$((wave_count - 1))"

  PIDS=()
  gpu=0
  for ((idx = wave_start; idx < wave_end; idx++)); do
    opt_scale="${HYBRID_OPT_SCALES[idx]}"
    label="$(scale_label "${opt_scale}")"
    run_name="hybrid_wcg${WCG_LABEL}_opt${label}_optlast${OPT_GUIDANCE_LAST_STEPS}"
    run_config_bg "${gpu}" "${run_name}" "${opt_scale}"
    PIDS+=($!)
    gpu=$((gpu + 1))
  done

  echo ">>> Wave ${wave_idx} pids: ${PIDS[*]}"
  for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
      echo ">>> ERROR: pid ${pid} failed"
      FAIL=1
    fi
  done
  wave_start="${wave_end}"
done

SWEEP_DIR="${SWEEP_DIR}" JSON_DIR="${JSON_DIR}" SEED_START="${SEED_START}" SEED_END="${SEED_END}" TASK="${TASK}" "${PYTHON}" -u - <<'PY'
import json
import math
import os
from pathlib import Path

sweep_dir = Path(os.environ["SWEEP_DIR"])
json_dir = Path(os.environ["JSON_DIR"])
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
            "task": data.get("task", os.environ.get("TASK")),
            "guidance_mode": data.get("guidance_mode"),
            "optimization_guidance_scale": data.get("optimization_guidance_scale"),
            "w_cg": data.get("w_cg"),
            "solver": data.get("solver"),
            "optimization_guidance_last_steps": data.get("optimization_guidance_last_steps"),
            "optimization_guidance_alpha_sigma_scale": data.get("optimization_guidance_alpha_sigma_scale"),
            "num_seeds": int(data.get("num_seeds", num_seeds)),
            "mean_reward": float(data.get("mean_reward", float("nan"))),
            "std_reward": std_reward,
            "se_reward": se_reward,
            "mean_norm_x100": float(data.get("mean_normalized_score_x100", float("nan"))),
            "std_norm_x100": std_norm,
            "se_norm_x100": se_norm,
            "result_file": str(path),
        }
    )

rows.sort(key=lambda r: r["mean_norm_x100"], reverse=True)
summary = {
    "task": os.environ.get("TASK"),
    "sweep_dir": str(sweep_dir),
    "seed_start": seed_start,
    "seed_end": seed_end,
    "num_seeds": num_seeds,
    "stats_interval": 10,
    "episodes_per_seed": 1,
    "metric_primary": "mean_normalized_score_x100",
    "ranking": rows,
    "best": rows[0] if rows else None,
}

out = sweep_dir / "sweep_summary.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Ep150 Hybrid Sweep Summary ===")
for i, row in enumerate(rows, 1):
    print(
        f"{i}. {row['run']:36s} "
        f"opt_scale={row['optimization_guidance_scale']!s:>6} w_cg={row['w_cg']!s:>4} "
        f"reward={row['mean_reward']:8.2f} ± {row['se_reward']:6.2f} "
        f"(std={row['std_reward']:7.2f}) "
        f"norm_x100={row['mean_norm_x100']:6.2f} ± {row['se_norm_x100']:5.2f} "
        f"(std={row['std_norm_x100']:5.2f})"
    )
print(f"\nSaved {out}")
PY

echo ""
if [[ "${FAIL}" -ne 0 ]]; then
  echo ">>> Sweep finished with errors. See ${LOG_DIR}/ and ${SLURM_LOG_DIR}/"
  exit 1
fi
echo ">>> Sweep complete."
echo "    method logs: ${LOG_DIR}/"
echo "    json results: ${JSON_DIR}/"
echo "    interim snapshots: ${INTERIM_DIR}/"
echo "    slurm logs: ${SLURM_LOG_DIR}/"
