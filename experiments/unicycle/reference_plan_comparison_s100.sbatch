#!/bin/bash
#SBATCH --job-name=ref_plan_cmp_s100
#SBATCH --time=48:00:00
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_reference_plan_comparison_s100_%j.out
#SBATCH --error=slurm_reference_plan_comparison_s100_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

CKPT_PATH="${CKPT_PATH:-results/diffuser_unicycle/Unicycle-v0/full64_6k/diffusion_ckpt_latest.pt}"
SEED="${SEED:-0}"
CANDIDATES="${CANDIDATES:-100}"
SAMPLING_STEPS="${SAMPLING_STEPS:-20}"
PLAN_PLOT_TOP_K="${PLAN_PLOT_TOP_K:-5}"
REFERENCE_STEPS="${REFERENCE_STEPS:-64}"
OPT_GUIDANCE_LAST_STEPS="${OPT_GUIDANCE_LAST_STEPS:-10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/unicycle_eval/reference_plan_comparison_s100_v4}"

TRAJECTORIES=(
  circle
  half_heart
  random_smooth_1
  random_smooth_2
  random_smooth_3
  random_smooth_4
  random_smooth_5
  random_smooth_6
  random_smooth_7
  random_smooth_8
  random_smooth_9
  random_smooth_10
  random_smooth_11
)
if [[ -n "${TRAJECTORIES_OVERRIDE:-}" ]]; then
  read -r -a TRAJECTORIES <<< "${TRAJECTORIES_OVERRIDE}"
fi

WCG_CONFIGS=(
  standard_w_cg0
  standard_w_cg10
  standard_w_cg20
  standard_w_cg30
  standard_w_cg50
)

OPT_CONFIGS=(
  optimization_scale_10
  optimization_scale_20
  optimization_scale_30
  optimization_scale_40
)

COMMON_ARGS=(
  --eval-task reference_plan
  --ckpt-path "${CKPT_PATH}"
  --seed "${SEED}"
  --device cuda:0
  --solver ddim
  --sample-step-schedule uniform
  --sampling-steps "${SAMPLING_STEPS}"
  --temperature 1.0
  --use-ema
  --heart-plan-candidates "${CANDIDATES}"
  --plan-plot-top-k "${PLAN_PLOT_TOP_K}"
  --reference-steps "${REFERENCE_STEPS}"
)

echo "Reference plan comparison (DDIM ${SAMPLING_STEPS}, ${CANDIDATES} candidates, top-${PLAN_PLOT_TOP_K} plots)"
echo "checkpoint: ${CKPT_PATH}"
echo "trajectories: ${TRAJECTORIES[*]}"
echo "w_cg configs: ${WCG_CONFIGS[*]}"
echo "opt configs: ${OPT_CONFIGS[*]} (last_steps=${OPT_GUIDANCE_LAST_STEPS}, diffusion_steps=100)"
echo "output: ${OUTPUT_ROOT}"

for traj in "${TRAJECTORIES[@]}"; do
  for cfg in "${WCG_CONFIGS[@]}"; do
    echo ""
    echo "========== ${traj} | ${cfg} =========="
    "${PYTHON}" pipelines/unicycle_eval.py \
      "${COMMON_ARGS[@]}" \
      --reference-name "${traj}" \
      --heart-config-names "${cfg}" \
      --diffusion-steps 20 \
      --output-dir "${OUTPUT_ROOT}/${traj}/${cfg}"
  done

  for cfg in "${OPT_CONFIGS[@]}"; do
    echo ""
    echo "========== ${traj} | ${cfg} =========="
    "${PYTHON}" pipelines/unicycle_eval.py \
      "${COMMON_ARGS[@]}" \
      --reference-name "${traj}" \
      --heart-config-names "${cfg}" \
      --diffusion-steps 100 \
      --training-diffusion-steps 20 \
      --optimization-guidance-last-steps "${OPT_GUIDANCE_LAST_STEPS}" \
      --output-dir "${OUTPUT_ROOT}/${traj}/${cfg}"
  done
done

echo ""
echo "Aggregating comparison summary..."
OUTPUT_ROOT="${OUTPUT_ROOT}" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
rows = []
for summary_path in sorted(root.glob("*/*/summary.json")):
    traj = summary_path.parent.parent.name
    cfg = summary_path.parent.name
    with open(summary_path) as f:
        data = json.load(f)
    block = data.get("reference_plan", {})
    stats = block.get("candidate_metrics_summary", {})
    rows.append({
        "trajectory": traj,
        "config": cfg,
        "plan_reward_mean": stats.get("plan_reward", {}).get("mean"),
        "plan_reward_std": stats.get("plan_reward", {}).get("std"),
        "rollout_reward_mean": stats.get("rollout_reward", {}).get("mean"),
        "rollout_reward_std": stats.get("rollout_reward", {}).get("std"),
        "feasibility_l2_mean": stats.get("feasibility_mean_l2_future", {}).get("mean"),
        "feasibility_l2_std": stats.get("feasibility_mean_l2_future", {}).get("std"),
        "summary_path": str(summary_path),
    })

out_path = root / "comparison_summary.json"
with open(out_path, "w") as f:
    json.dump({"runs": rows}, f, indent=2)
print(f"Wrote {out_path} ({len(rows)} runs)")
PY

echo "Done. Results under ${OUTPUT_ROOT}/"
