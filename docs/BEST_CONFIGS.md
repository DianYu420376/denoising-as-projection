# Best MuJoCo configs (150-seed evaluations)

Canonical source: `configs/diffuser/mujoco/task/<task>.yaml` → `best_hybrid` block.

Metric: mean normalized score ×100 (higher is better), over 150 evaluation seeds.

## Standard guidance baseline (PDG)

| Task | w_cg | solver | temperature | norm×100 |
|------|------|--------|-------------|----------|
| hopper-medium-v2 | 0.3 | ddpm | 0.5 | 93.03 |
| walker2d-medium-v2 | 0.007 | ddpm | 0.5 | 76.70 |
| halfcheetah-medium-v2 | 0.0001 | ddpm | 0.5 | 44.47 |

## Best DCG / hybrid (`best_hybrid`)

With `optimization_guidance_last_steps=20` and `sampling_steps=20`, classifier guidance (`w_cg`) is inactive on those steps → **`w_cg: 0.0`**.

| Task | opt_scale | norm×100 |
|------|-----------|----------|
| hopper-medium-v2 | 0.9 | 94.82 |
| walker2d-medium-v2 | 0.05 | 78.14 |
| halfcheetah-medium-v2 | 0.00003 | 44.80 |

Solver for DCG runs: `ddim`, temperature `1.0`, `ddim_eta=1.0`, `optimization_guidance_alpha_sigma_scale=true`.
