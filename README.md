# Denoising-Corrected Gradient (DCG) Guidance — Code

Anonymous code release accompanying an ICLR 2027 submission.

This repository provides an inference-time **denoising-corrected gradient (DCG)** guidance method for diffusion models, built on top of the [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) library (`cleandiffuser/`). We thank the CleanDiffuser authors for the modular diffusion / Diffuser infrastructure.

**Guidance terminology in this codebase**

| Paper / method name | Code flag | Meaning |
|---------------------|-----------|---------|
| PDG / standard classifier guidance | `guidance_mode=standard`, `w_cg=…` | Gradient / classifier guidance applied in the usual reverse step |
| DCG | `guidance_mode=optimization` (or `hybrid`), `optimization_guidance_scale=…` | Gradient step, then denoiser (Stein posterior mean) as geometric correction |
| Monte Carlo (unguided) | `guidance_mode=standard`, `w_cg=0` | Samples from the learned prior without objective guidance |

Core implementation: `cleandiffuser/diffusion/guidance.py` (used by DDIM / `DiscreteDiffusionSDE` sampling).

---

## Repository layout

```
cleandiffuser/          # Core library (diffusion, guidance, envs, datasets)
configs/                # Hydra / task configs (MuJoCo, unicycle)
pipelines/              # Python entry points for train / eval / plots
experiments/
  common/env.sh         # Shared REPO_DIR / PYTHON / PYTHONPATH setup
  synthetic/            # Convex constrained synthetic (Test A / Test B)
  unicycle/             # Unicycle data, train, reference-plan eval
  mujoco/               # D4RL MuJoCo Diffuser train / eval / sweeps
docs/                   # Extra notes (best configs, etc.)
tests/                  # Unit tests
```

Launch scripts under `experiments/` are optional SLURM wrappers. Every experiment can also be run directly with `python` (preferred for local reproduction).

---

## Setup

```bash
# Create / activate a Python 3.10 env, then:
pip install -e .
# or: pip install -r requirements.txt && pip install -e .

export PYTHONPATH="pipelines:.:${PYTHONPATH:-}"
# MuJoCo rendering (headless):
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
```

**Dependencies of note:** PyTorch, MuJoCo / `mujoco_py`, Gym, D4RL (for MuJoCo offline datasets).

**SLURM (optional):** pass your site account/partition on the command line, e.g.

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  --export=ALL,TASK=hopper-medium-v2 \
  experiments/mujoco/run_mujoco_ep150_standard_guidance.sbatch
```

Or set `VENV_DIR=/path/to/venv` / activate a venv before submitting. Launchers source `experiments/common/env.sh` and resolve `REPO_DIR` automatically.

---

## Experiments

The numerical experiments fall into three families, matching the appendix of the submission.

### 1. Synthetic convex constrained optimization (Test A / Test B)

Data live on a low-dimensional convex set in $\mathbb{R}^{10}$ (Test A: 2D square; Test B: 5D ellipsoid). Compare standard guidance vs DCG on objective value and constraint violation.

**Train denoisers**

```bash
# SLURM
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  experiments/synthetic/run_guidance_synthetic_subspace_train.sbatch

# Or directly (see script for full flags):
python pipelines/guidance_synthetic_subspace.py --test a --mode train ...
python pipelines/guidance_synthetic_subspace.py --test b --mode train ...
```

**Generate figures / path visualizations** (Test A & B)

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  experiments/synthetic/run_figures.sbatch

# Direct:
python pipelines/guidance_synthetic_visualize_subspace.py \
  --test both --resample \
  --output-root results/guidance_synthetic \
  --device cuda:0 \
  --diffusion-steps 50 --training-diffusion-steps 50 --sampling-steps 50 \
  --optimization-guidance-last-steps 50 \
  --n-paths 5 \
  --w-cg 0.6 --opt-scale 15 \
  --w-cg-test-b 10 --opt-scale-test-b 10
```

Related helpers: `pipelines/guidance_synthetic_sweep.py`, `guidance_synthetic_grid_sweep.py`, `guidance_synthetic_refine.py`, launchers under `experiments/synthetic/`.

Outputs default to `results/guidance_synthetic/`.

---

### 2. Unicycle motion planning / reference tracking

Planar unicycle with horizon $H=64$. Offline diffusion prior + inference-time tracking cost. Metrics: planned cost, rollout cost, dynamic feasibility error between planned observations and open-loop execution.

**Collect offline data → train → evaluate**

```bash
# Data
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  experiments/unicycle/run_collect_unicycle_full64.sbatch
# or: python pipelines/collect_unicycle_offline_data.py ...

# Train
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  experiments/unicycle/run_diffuser_unicycle_train_full64.sbatch
# or: python pipelines/diffuser_unicycle.py ...

# Reference-plan comparison (Monte Carlo / standard guidance / DCG)
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  experiments/unicycle/reference_plan_comparison_s100.sbatch
```

**Direct eval example** (after a checkpoint exists):

```bash
python pipelines/unicycle_eval.py \
  --eval-task reference_plan \
  --reference-name circle \
  --ckpt-path results/diffuser_unicycle/Unicycle-v0/full64_6k/diffusion_ckpt_latest.pt \
  --num-candidates 100 \
  --sampling-steps 20 \
  --output-root results/unicycle_eval/reference_plan
```

Useful configs live under `configs/diffuser/unicycle/`. Heart-track and feasibility ablations: other launchers in `experiments/unicycle/`. Aggregation: `pipelines/scrape_reference_plan_comparison.py`, `pipelines/reference_plan_aggregate_log.py`.

---

### 3. D4RL MuJoCo Diffuser benchmarks

Tasks: `hopper-medium-v2`, `walker2d-medium-v2`, `halfcheetah-medium-v2`. Same Diffuser-style planner; compare **standard guidance (PDG)** vs **DCG** over 150 evaluation seeds.

**Train**

```bash
python pipelines/diffuser_d4rl_mujoco.py task=hopper-medium-v2 mode=train seed=0 device=cuda:0

# Or:
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  --export=ALL,TASK=hopper-medium-v2 \
  experiments/mujoco/run_diffuser_mujoco_train.sbatch
```

**Standard guidance (PDG) — 150 seeds**

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  --export=ALL,TASK=hopper-medium-v2 \
  experiments/mujoco/run_mujoco_ep150_standard_guidance.sbatch
```

**DCG / hybrid opt-scale sweep — 150 seeds**

```bash
# Example: hopper best scale from config is 0.9; walker 0.05; halfcheetah 0.00003
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  --export=ALL,TASK=hopper-medium-v2,OPT_SCALES="0.9",W_CG=0 \
  experiments/mujoco/run_mujoco_ep150_hybrid_sweep_3gpu.sbatch
```

Single-seed trajectory eval:

```bash
python pipelines/eval_hopper_trajectory.py \
  --task hopper-medium-v2 --ckpt latest \
  --guidance_mode hybrid \
  --optimization_guidance_scale 0.9 \
  --w_cg 0.0 \
  --solver ddim --sampling-steps 20 --temperature 1.0 \
  --optimization-guidance-last-steps 20 \
  --seed 0
```

**Recommended DCG scales** (also stored as `best_hybrid` in each task yaml under `configs/diffuser/mujoco/task/`):

| Task | `optimization_guidance_scale` | `w_cg` |
|------|-------------------------------|--------|
| hopper-medium-v2 | 0.9 | 0.0 |
| walker2d-medium-v2 | 0.05 | 0.0 |
| halfcheetah-medium-v2 | 0.00003 | 0.0 |

With `sampling_steps=20` and `optimization_guidance_last_steps=20`, classifier guidance is inactive on those reverse steps, so `w_cg` is set to `0`. See `docs/BEST_CONFIGS.md`.

**Dynamic feasibility (optional analysis)** — open-loop plan vs rollout L2 for Monte Carlo / standard / best DCG:

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  --export=ALL,TASK=hopper-medium-v2,SIM_ENV_NAME=Hopper-v2 \
  experiments/mujoco/run_dynamic_feasibility_ep150_std_vs_opt.sbatch
```

Sim envs use Gym **v2** physics: `Hopper-v2`, `Walker2d-v2`, `HalfCheetah-v2`.

---

## Key Python entry points

| Purpose | Script |
|---------|--------|
| MuJoCo train / infer | `pipelines/diffuser_d4rl_mujoco.py` |
| MuJoCo trajectory eval | `pipelines/eval_hopper_trajectory.py` |
| MuJoCo 150-seed sweeps | `pipelines/run_ep150_config_seed_sweep.py` |
| MuJoCo dynamic feasibility | `pipelines/dynamic_feasibility_hopper_v2_comparison.py` |
| Synthetic train / sample | `pipelines/guidance_synthetic_subspace.py` |
| Synthetic figures | `pipelines/guidance_synthetic_visualize_subspace.py` |
| Unicycle train | `pipelines/diffuser_unicycle.py` |
| Unicycle eval | `pipelines/unicycle_eval.py` |
| Offline unicycle data | `pipelines/collect_unicycle_offline_data.py` |

---

## Tests

```bash
pytest tests/test_d4rl_render_utils_v2.py tests/test_optimization_guidance.py -q
```

---

## Citation / upstream

Please cite CleanDiffuser when using the base library:

- CleanDiffuser: https://github.com/CleanDiffuserTeam/CleanDiffuser
- Docs: https://cleandiffuserteam.github.io/CleanDiffuserDocs/

This anonymous release will be linked from the camera-ready version of the submission.
