#!/bin/bash
#SBATCH --job-name=hybrid-opt-launch
# Pass site-specific --account/--partition on the sbatch command line.
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/hybrid_opt_scale_sweep/coarse_launcher_%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

: "${TASK:?Set TASK (walker2d-medium-v2 or halfcheetah-medium-v2)}"

W_CG="${W_CG:-0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
OPT_GUIDANCE_LAST_STEPS="${OPT_GUIDANCE_LAST_STEPS:-20}"
DDIM_ETA="${DDIM_ETA:-1.0}"
OPT_ALPHA_SIGMA_SCALE="${OPT_ALPHA_SIGMA_SCALE:-1}"
GUIDANCE_MODE="${GUIDANCE_MODE:-hybrid}"
LAUNCHER_JOB_ID="${SLURM_JOB_ID:-local}"

# Coarse grids: smallest scale ~= task standard w_cg (walker 0.007, cheetah 0.0001).
case "${TASK}" in
  walker2d-medium-v2)
    STD_W_CG="0.007"
    OPT_SCALES=(0.007 0.01 0.03 0.1 0.3 1.0 1.1)
    ;;
  halfcheetah-medium-v2)
    STD_W_CG="0.0001"
    OPT_SCALES=(0.0001 0.001 0.01 0.1 0.3 1.0 1.1)
    ;;
  *)
    echo "ERROR: no coarse scale grid defined for TASK=${TASK}"
    exit 1
    ;;
esac

RUN_TAG="${RUN_TAG:-coarse_wcg${W_CG}_optlast${OPT_GUIDANCE_LAST_STEPS}_alphasig${OPT_ALPHA_SIGMA_SCALE}_job${LAUNCHER_JOB_ID}}"
SWEEP_ROOT="${REPO_DIR}/results/diffuser_d4rl_mujoco/${TASK}/hybrid_opt_scale_coarse_sweep/${RUN_TAG}"
mkdir -p "${SWEEP_ROOT}/by_scale"
mkdir -p "${REPO_DIR}/logs/slurm/hybrid_opt_scale_sweep"

MANIFEST="${SWEEP_ROOT}/sweep_manifest.json"

echo "============================================================"
echo "Launch hybrid opt_scale coarse sweep"
echo "============================================================"
echo "task: ${TASK}"
echo "std_w_cg (reference): ${STD_W_CG}"
echo "fixed w_cg: ${W_CG}"
echo "temp: ${TEMPERATURE}, opt_last: ${OPT_GUIDANCE_LAST_STEPS}, eta: ${DDIM_ETA}"
echo "alpha_sigma_scale: ${OPT_ALPHA_SIGMA_SCALE}"
echo "opt_scales: ${OPT_SCALES[*]}"
echo "sweep_root: ${SWEEP_ROOT}"
echo "============================================================"

ARRAY_JOB_IDS=()
SCALE_MANIFEST_ENTRIES=()

for OPT_SCALE in "${OPT_SCALES[@]}"; do
  SCALE_LABEL="${OPT_SCALE//./p}"
  JID="$(
    sbatch --parsable \
      --job-name="hyb-${TASK%%-*}-opt${SCALE_LABEL}" \
      --export=ALL,TASK="${TASK}",OPT_SCALE="${OPT_SCALE}",W_CG="${W_CG}",TEMPERATURE="${TEMPERATURE}",OPT_GUIDANCE_LAST_STEPS="${OPT_GUIDANCE_LAST_STEPS}",DDIM_ETA="${DDIM_ETA}",OPT_ALPHA_SIGMA_SCALE="${OPT_ALPHA_SIGMA_SCALE}",GUIDANCE_MODE="${GUIDANCE_MODE}",SWEEP_ROOT="${SWEEP_ROOT}" \
      "${REPO_DIR}/run_mujoco_hybrid_opt_scale_seed_sweep.sbatch"
  )"
  ARRAY_JOB_IDS+=("${JID}")
  SCALE_MANIFEST_ENTRIES+=("{\"opt_scale\": ${OPT_SCALE}, \"opt_scale_label\": \"opt${SCALE_LABEL}\", \"array_job_id\": \"${JID}\", \"scale_dir\": \"opt${SCALE_LABEL}_array${JID}\"}")
  echo ">>> Submitted opt_scale=${OPT_SCALE} array job ${JID}"
done

DEP=""
for JID in "${ARRAY_JOB_IDS[@]}"; do
  DEP="${DEP:+$DEP:}${JID}"
done

META_AGG_OUT="${REPO_DIR}/logs/slurm/hybrid_opt_scale_sweep/coarse_meta_agg_${LAUNCHER_JOB_ID}_%j.out"
META_JID="$(
  sbatch --parsable \
    --job-name="hyb-meta-agg" \
    ${SBATCH_ACCOUNT:+--account=${SBATCH_ACCOUNT}} \
    --partition=cpu \
    --cpus-per-task=1 \
    --mem=4G \
    --time=00:30:00 \
    --output="${META_AGG_OUT}" \
    --dependency=afterany:${DEP} \
    --export=ALL,SWEEP_ROOT="${SWEEP_ROOT}",MANIFEST="${MANIFEST}" \
    "${REPO_DIR}/run_mujoco_hybrid_opt_scale_coarse_aggregate.sbatch"
)"
echo ">>> Submitted meta-aggregate job ${META_JID} (after arrays: ${DEP})"

# Write manifest with all scale entries.
SCALES_CSV=$(IFS=,; echo "${OPT_SCALES[*]}")
JOBS_CSV=$(IFS=,; echo "${ARRAY_JOB_IDS[*]}")
"${PYTHON}" -u - <<PY
import json
from pathlib import Path

opt_scales = [float(x) for x in "${SCALES_CSV}".split(",")]
job_ids = "${JOBS_CSV}".split(",")
scales = []
for opt_scale, jid in zip(opt_scales, job_ids):
    label = str(opt_scale).replace(".", "p")
    scales.append({
        "opt_scale": opt_scale,
        "opt_scale_label": f"opt{label}",
        "array_job_id": jid,
        "scale_dir": f"opt{label}_array{jid}",
    })

manifest = {
    "task": "${TASK}",
    "launcher_job_id": "${LAUNCHER_JOB_ID}",
    "meta_aggregate_job_id": "${META_JID}",
    "sweep_root": "${SWEEP_ROOT}",
    "guidance_mode": "${GUIDANCE_MODE}",
    "w_cg": float("${W_CG}"),
    "std_w_cg_reference": float("${STD_W_CG}"),
    "temperature": float("${TEMPERATURE}"),
    "opt_guidance_last_steps": int("${OPT_GUIDANCE_LAST_STEPS}"),
    "ddim_eta": float("${DDIM_ETA}"),
    "optimization_guidance_alpha_sigma_scale": bool(int("${OPT_ALPHA_SIGMA_SCALE}")),
    "solver": "ddim",
    "sampling_steps": 20,
    "seeds_per_scale": 15,
    "seed_range": [1, 15],
    "opt_scales": [s["opt_scale"] for s in scales],
    "array_job_ids": [s["array_job_id"] for s in scales],
    "scales": scales,
}
path = Path("${MANIFEST}")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(manifest, indent=2))
print(f"Wrote {path}")
PY

echo ""
echo ">>> Coarse sweep launched."
echo "    sweep_root: ${SWEEP_ROOT}"
echo "    manifest:   ${MANIFEST}"
echo "    meta_agg:   ${META_JID}"
