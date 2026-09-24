#!/bin/bash
# Fast replot from saved candidates.npz (no GPU, ~1 min for full v1 set).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/unicycle_eval/reference_plan_v1}"
PLAN_PLOT_TOP_K="${PLAN_PLOT_TOP_K:-5}"
TRAJECTORIES="${TRAJECTORIES:-}"

ARGS=(--root "${OUTPUT_ROOT}" --top-k "${PLAN_PLOT_TOP_K}")
if [[ -n "${TRAJECTORIES}" ]]; then
  read -r -a TRAJ_ARR <<< "${TRAJECTORIES}"
  ARGS+=(--trajectories "${TRAJ_ARR[@]}")
fi

"${PYTHON}" pipelines/reference_plan_replot.py "${ARGS[@]}"
