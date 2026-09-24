#!/usr/bin/env python3
"""Phase-5 ultra-refinement around best w_cg / low opt scales."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from guidance_synthetic_sweep import (
    SweepConfig,
    SweepRunner,
    _label_opt,
    _w_cg_configs,
)
from utils import set_seed


def _args(test: str):
    class A:
        pass

    a = A()
    a.test = test
    a.output_root = "results/guidance_synthetic"
    a.seed = 0
    a.num_dataset_samples = 200_000
    a.square_half_width = 1.0
    a.ellipsoid_axes = [1.0, 0.85, 0.7, 0.55, 0.4]
    a.model_dim = 64
    a.diffusion_steps = 50
    a.training_diffusion_steps = 50
    a.inference_diffusion_steps = 100
    a.predict_noise = False
    a.ema_rate = 0.9999
    a.num_samples = 1024
    a.validation_samples = 4096
    a.solver = "ddim"
    a.sampling_steps = 20
    a.sample_step_schedule = "uniform"
    a.temperature = 1.0
    a.use_ema = True
    a.violation_penalty = 0.5
    a.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return a


def refine_test(test: str, w_vals: list[float], opt_vals: list[float], last_steps: list[int]) -> None:
    args = _args(test)
    set_seed(args.seed)
    runner = SweepRunner(args)
    _, geometry, normalizer = runner._load_bundle(test)
    sweep_path = Path(args.output_root) / test / "sweep" / "sweep_results.json"
    results = json.loads(sweep_path.read_text())["all_runs"]

    cfgs = _w_cg_configs(w_vals)
    for scale in opt_vals:
        for ls in last_steps:
            cfgs.append(
                SweepConfig(
                    _label_opt(scale, ls),
                    "optimization",
                    0.0,
                    float(scale),
                    int(ls),
                )
            )
    runner.run_phase("phase5_ultra_refine", cfgs, geometry, normalizer, results)

    args.num_samples = 4096
    runner._agent_cache.clear()
    top = runner.top_k(results, "standard", 3) + runner.top_k(results, "optimization", 3)
    val_cfgs = [
        SweepConfig(
            r["name"] + "_final",
            r["guidance_mode"],
            r["w_cg"],
            r["optimization_guidance_scale"],
            r["optimization_guidance_last_steps"],
        )
        for r in top
    ]
    runner.run_phase("phase5_final_validate", val_cfgs, geometry, normalizer, results)

    best_w = max((r for r in results if r["guidance_mode"] == "standard"), key=lambda r: r["reward_mean"])
    best_o = max((r for r in results if r["guidance_mode"] == "optimization"), key=lambda r: r["reward_mean"])
    summary = {
        "test": test,
        "best_by_reward": {"standard_w_cg": best_w, "optimization": best_o},
        "all_runs": results,
    }
    with open(sweep_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"FINAL {test} | w_cg {best_w['name']}: reward={best_w['reward_mean']:.4f} "
        f"viol={best_w['violation_mean']:.4f}"
    )
    print(
        f"FINAL {test} | opt {best_o['name']}: reward={best_o['reward_mean']:.4f} "
        f"viol={best_o['violation_mean']:.4f}"
    )


if __name__ == "__main__":
    refine_test(
        "test_a",
        w_vals=[19.0 + 0.25 * i for i in range(36)],
        opt_vals=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
        last_steps=[15, 20, 25, 30, 40, 50],
    )
    refine_test(
        "test_b",
        w_vals=[48.0 + 0.25 * i for i in range(40)],
        opt_vals=[10.0, 15.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 35.0, 40.0],
        last_steps=[15, 20, 25, 30, 40, 50],
    )
