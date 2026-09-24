#!/usr/bin/env bash
# Shared environment setup for experiment launchers.
# Source from any launcher under experiments/<family>/:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common/env.sh"

_EXPERIMENTS_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${_EXPERIMENTS_COMMON_DIR}/../.." && pwd)}"

# Prefer an activated venv; otherwise honor VENV_DIR / VIRTUAL_ENV.
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${PYTHON:-${VIRTUAL_ENV}/bin/python}"
elif [[ -n "${VENV_DIR:-}" && -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON="${PYTHON:-${VENV_DIR}/bin/python}"
else
  PYTHON="${PYTHON:-python}"
fi

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/pipelines:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

echo "[env] REPO_DIR=${REPO_DIR}"
echo "[env] PYTHON=${PYTHON}"
