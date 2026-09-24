#!/usr/bin/env python3
"""Grid sweep w_cg and optimization_guidance_scale in [0, 10] with matched 50/50/50 steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cleandiffuser.classifier.runtime_reward import RuntimeRewardClassifier
from cleandiffuser.diffusion.guidance import validate_guidance_config
from guidance_synthetic_subspace import (
    AMBIENT_DIM,
    HORIZON,
    _build_agent,
    _load_geometry,
    _load_dataset_bundle,
    _make_reward_fn,
    _resolve_paths,
    _sample_batch_metrics,
)
from utils import set_seed


def _grid_values(vmin: float, vmax: float, step: float) -> list[float]:
    vals = np.arange(vmin, vmax + step * 0.5, step)
    return [float(round(v, 6)) for v in vals]


def run_grid_sweep(args) -> dict:
    set_seed(args.seed)
    paths = _resolve_paths(args.test, Path(args.output_root))
    _, geometry, normalizer = _load_dataset_bundle(args)

    w_vals = _grid_values(args.grid_min, args.grid_max, args.grid_step)
    opt_vals = _grid_values(args.grid_min, args.grid_max, args.grid_step)

    reward_fn = _make_reward_fn(geometry, normalizer, args.device)
    agent = _build_agent(args, classifier=None, diffusion_steps=args.diffusion_steps)
    agent.load(str(paths["ckpt"]))
    agent.eval()

    rows: list[dict] = []
    prior = torch.zeros((args.num_samples, HORIZON, AMBIENT_DIM), device=args.device)

    def eval_cfg(guidance_mode: str, w_cg: float, opt_scale: float) -> dict:
        validate_guidance_config(guidance_mode, w_cg, opt_scale)
        if w_cg == 0.0 and opt_scale == 0.0:
            agent.classifier = None
        else:
            agent.classifier = RuntimeRewardClassifier(reward_fn, device=args.device)
        traj, _ = agent.sample(
            prior,
            solver=args.solver,
            n_samples=args.num_samples,
            sample_steps=args.sampling_steps,
            sample_step_schedule=args.sample_step_schedule,
            use_ema=args.use_ema,
            w_cg=w_cg,
            guidance_mode=guidance_mode,
            optimization_guidance_scale=opt_scale,
            optimization_guidance_last_steps=args.optimization_guidance_last_steps,
            temperature=args.temperature,
        )
        metrics = _sample_batch_metrics(traj, geometry, normalizer)
        return {
            "test": args.test,
            "guidance_mode": guidance_mode,
            "w_cg": w_cg,
            "optimization_guidance_scale": opt_scale,
            "diffusion_steps": args.diffusion_steps,
            "training_diffusion_steps": args.training_diffusion_steps,
            "sampling_steps": args.sampling_steps,
            "optimization_guidance_last_steps": args.optimization_guidance_last_steps,
            "num_samples": args.num_samples,
            **metrics,
        }

    print(
        f"[{args.test}] grid sweep w_cg in [{args.grid_min}, {args.grid_max}] "
        f"step={args.grid_step} | steps={args.sampling_steps}/{args.diffusion_steps}/"
        f"{args.training_diffusion_steps}"
        + ("" if not args.w_cg_only else " | w_cg only")
    )

    for w in w_vals:
        row = eval_cfg("standard", w, 0.0)
        rows.append(row)
        print(
            f"  w_cg={w:5g}  reward={row['reward_mean']:9.4f}  "
            f"violation={row['violation_mean']:.4f}"
        )

    if not args.w_cg_only:
        print(
            f"[{args.test}] opt_scale in [{args.grid_min}, {args.grid_max}] step={args.grid_step}"
        )
        for s in opt_vals:
            if s == 0.0:
                continue  # same as w_cg=0 unguided
            row = eval_cfg("optimization", 0.0, s)
            rows.append(row)
            print(
                f"  opt={s:5g}  reward={row['reward_mean']:9.4f}  "
                f"violation={row['violation_mean']:.4f}"
            )

    w_cg_rows = [r for r in rows if r["guidance_mode"] == "standard"]
    opt_rows = [r for r in rows if r["guidance_mode"] == "optimization"]
    summary = {
        "test": args.test,
        "grid_min": args.grid_min,
        "grid_max": args.grid_max,
        "grid_step": args.grid_step,
        "w_cg_only": args.w_cg_only,
        "diffusion_steps": args.diffusion_steps,
        "training_diffusion_steps": args.training_diffusion_steps,
        "sampling_steps": args.sampling_steps,
        "runs": rows,
        "best_w_cg": max(w_cg_rows, key=lambda r: r["reward_mean"]) if w_cg_rows else None,
        "best_opt": max(opt_rows, key=lambda r: r["reward_mean"]) if opt_rows else None,
    }

    out_dir = paths["root"] / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.output_tag or f"grid_s{args.sampling_steps}"
    out_path = out_dir / f"{tag}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path}")
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test", choices=["test_a", "test_b", "both"], default="both")
    p.add_argument("--output-root", default="results/guidance_synthetic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-dataset-samples", type=int, default=200_000)
    p.add_argument("--square-half-width", type=float, default=1.0)
    p.add_argument("--ellipsoid-axes", nargs=5, type=float, default=[1.0, 0.85, 0.7, 0.55, 0.4])
    p.add_argument("--model-dim", type=int, default=64)
    p.add_argument("--diffusion-steps", type=int, default=50)
    p.add_argument("--training-diffusion-steps", type=int, default=50)
    p.add_argument("--sampling-steps", type=int, default=50)
    p.add_argument("--optimization-guidance-last-steps", type=int, default=25)
    p.add_argument("--predict-noise", action="store_true")
    p.add_argument("--ema-rate", type=float, default=0.9999)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--solver", default="ddim")
    p.add_argument("--sample-step-schedule", default="uniform")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grid-min", type=float, default=0.0)
    p.add_argument("--grid-max", type=float, default=10.0)
    p.add_argument("--grid-step", type=float, default=1.0)
    p.add_argument("--w-cg-only", action="store_true")
    p.add_argument("--output-tag", default=None)
    args = p.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    return args


def main():
    args = parse_args()
    tests = ["test_a", "test_b"] if args.test == "both" else [args.test]
    combined = {}
    tag = args.output_tag or f"grid_s{args.sampling_steps}"
    for test in tests:
        args.test = test
        combined[test] = run_grid_sweep(args)
    out = Path(args.output_root) / f"{tag}_summary.json"
    with open(out, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
