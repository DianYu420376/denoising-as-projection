#!/usr/bin/env python3
"""Re-sweep w_cg only, selecting configs by balanced reward–violation score.

Score = reward_mean - violation_penalty * violation_mean

Phases:
  1 — dense coarse grid (w_cg from 0 to w_max)
  2 — refine ±radius around top-K balanced configs (fine step)
  3 — high-sample validation of top balanced picks (+ raw-reward reference)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from guidance_synthetic_sweep import (
    SweepConfig,
    SweepRunner,
    _refine_values,
    _w_cg_configs,
)
from utils import set_seed


def _balanced_score(reward: float, violation: float, penalty: float) -> float:
    return reward - penalty * violation


def _update_scores(results: list[dict], penalty: float) -> None:
    for r in results:
        r["violation_penalty"] = penalty
        r["balanced_score"] = _balanced_score(r["reward_mean"], r["violation_mean"], penalty)


def _top_balanced(results: list[dict], k: int, penalty: float) -> list[dict]:
    pool = [r for r in results if r["guidance_mode"] == "standard" and r["w_cg"] > 0]
    for r in pool:
        r["_pick_score"] = _balanced_score(r["reward_mean"], r["violation_mean"], penalty)
    pool.sort(key=lambda r: (r["_pick_score"], r["reward_mean"]), reverse=True)
    return pool[:k]


def run_wcg_resweep(args) -> dict:
    set_seed(args.seed)
    runner = SweepRunner(args)
    runner.args.violation_penalty = args.violation_penalty
    _, geometry, normalizer = runner._load_bundle(args.test)

    sweep_path = Path(args.output_root) / args.test / "sweep" / "wcg_resweep.json"
    results: list[dict] = []
    if sweep_path.is_file() and args.resume:
        results = json.loads(sweep_path.read_text()).get("all_runs", [])

    # Phase 1 — dense coarse
    w_coarse = np.unique(
        np.concatenate(
            [
                np.arange(0.0, min(6.0, args.w_max) + 1e-9, 0.25),
                np.arange(0.0, args.w_max + 1e-9, args.coarse_step),
            ]
        )
    ).tolist()
    runner.run_phase("wcg_p1_coarse", _w_cg_configs(w_coarse), geometry, normalizer, results)
    _update_scores(results, args.violation_penalty)

    # Phase 2 — refine around top balanced
    refine_cfgs: list[SweepConfig] = []
    for row in _top_balanced(results, args.refine_top_k, args.violation_penalty):
        refine_cfgs.extend(
            _w_cg_configs(
                _refine_values(
                    row["w_cg"],
                    radius=args.refine_radius,
                    step=args.refine_step,
                    lo=0.0,
                )
            )
        )
    runner.run_phase("wcg_p2_refine", refine_cfgs, geometry, normalizer, results)
    _update_scores(results, args.violation_penalty)

    # Phase 3 — validate top balanced + best raw reward (for comparison)
    old_n = args.num_samples
    args.num_samples = args.validation_samples
    runner._agent_cache.clear()
    validate_rows = _top_balanced(results, args.validate_top_k, args.violation_penalty)
    raw_best = max(
        (r for r in results if r["guidance_mode"] == "standard" and r["w_cg"] > 0),
        key=lambda r: r["reward_mean"],
    )
    if raw_best["w_cg"] not in {r["w_cg"] for r in validate_rows}:
        validate_rows.append(raw_best)
    val_cfgs = [
        SweepConfig(
            f"{r['name']}_balanced_val",
            "standard",
            r["w_cg"],
            0.0,
        )
        for r in validate_rows
    ]
    runner.run_phase("wcg_p3_validate", val_cfgs, geometry, normalizer, results)
    args.num_samples = old_n
    _update_scores(results, args.violation_penalty)

    best_balanced = _top_balanced(results, 1, args.violation_penalty)[0]
    unguided = next(r for r in results if r["w_cg"] == 0.0 and r["optimization_guidance_scale"] == 0.0)

    # Report under alternate penalties for transparency
    alt_picks = {}
    for pen in args.report_penalties:
        pick = _top_balanced(results, 1, pen)[0]
        alt_picks[str(pen)] = {
            "w_cg": pick["w_cg"],
            "name": pick["name"],
            "reward_mean": pick["reward_mean"],
            "violation_mean": pick["violation_mean"],
            "balanced_score": _balanced_score(pick["reward_mean"], pick["violation_mean"], pen),
        }

    summary = {
        "test": args.test,
        "violation_penalty_used": args.violation_penalty,
        "unguided": unguided,
        "best_balanced_w_cg": best_balanced,
        "best_raw_reward_w_cg": raw_best,
        "best_by_alternate_penalties": alt_picks,
        "all_runs": results,
    }
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sweep_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n[{args.test}] BEST balanced w_cg (penalty={args.violation_penalty}): "
        f"w_cg={best_balanced['w_cg']:.4g} reward={best_balanced['reward_mean']:.4f} "
        f"viol={best_balanced['violation_mean']:.4f} "
        f"score={best_balanced['balanced_score']:.4f}"
    )
    print(
        f"[{args.test}] Best raw reward w_cg: w_cg={raw_best['w_cg']:.4g} "
        f"reward={raw_best['reward_mean']:.4f} viol={raw_best['violation_mean']:.4f}"
    )
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
    p.add_argument("--inference-diffusion-steps", type=int, default=100)
    p.add_argument("--predict-noise", action="store_true")
    p.add_argument("--ema-rate", type=float, default=0.9999)
    p.add_argument("--num-samples", type=int, default=1024)
    p.add_argument("--validation-samples", type=int, default=4096)
    p.add_argument("--solver", default="ddim")
    p.add_argument("--sampling-steps", type=int, default=20)
    p.add_argument("--sample-step-schedule", default="uniform")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--violation-penalty",
        type=float,
        default=2.0,
        help="Balanced score = reward - penalty * violation (default 2.0)",
    )
    p.add_argument("--report-penalties", nargs="+", type=float, default=[1.0, 2.0, 5.0])
    p.add_argument("--w-max", type=float, default=None, help="Max w_cg in coarse grid (per-test default)")
    p.add_argument("--coarse-step", type=float, default=1.0)
    p.add_argument("--refine-top-k", type=int, default=5)
    p.add_argument("--refine-radius", type=float, default=2.0)
    p.add_argument("--refine-step", type=float, default=0.1)
    p.add_argument("--validate-top-k", type=int, default=3)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    return args


def main():
    args = parse_args()
    tests = ["test_a", "test_b"] if args.test == "both" else [args.test]
    summaries = {}
    defaults = {"test_a": 28.0, "test_b": 58.0}
    for test in tests:
        args.test = test
        if args.w_max is None:
            args.w_max = defaults[test]
        summaries[test] = run_wcg_resweep(args)

    out = Path(args.output_root) / "wcg_balanced_resweep_summary.json"
    with open(out, "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
