# Experiment launchers

Optional SLURM wrappers. Prefer the Python commands in the root `README.md` for local runs.

| Directory | Experiment family |
|-----------|-------------------|
| `synthetic/` | Convex constrained Test A / Test B |
| `unicycle/` | Offline data, training, reference-plan / heart tracking |
| `mujoco/` | D4RL Diffuser train, 150-seed PDG/DCG evals, dynamic feasibility |
| `common/env.sh` | Shared `REPO_DIR` / `PYTHON` / `PYTHONPATH` setup |

All launchers source `common/env.sh`. Pass `--account` / `--partition` on the `sbatch` command line for your cluster.
