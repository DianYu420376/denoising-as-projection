#!/usr/bin/env python3
"""Adaptive multi-phase hyperparameter sweep for synthetic guidance tests A & B.

Phase 1 — coarse grid over w_cg and optimization_guidance_scale
Phase 2 — refine around top candidates (± window, finer step)
Phase 3 — sweep optimization_guidance_last_steps for best opt config
Phase 4 — re-evaluate top configs with more samples for stable ranking
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cleandiffuser.classifier.runtime_reward import RuntimeRewardClassifier
from cleandiffuser.diffusion.guidance import validate_guidance_config
from guidance_synthetic_subspace import (
    AMBIENT_DIM,
    HORIZON,
    GeometryMeta,
    _build_agent,
    _load_dataset_bundle,
    _make_reward_fn,
    _resolve_paths,
    _sample_batch_metrics,
)
from utils import set_seed


@dataclass(frozen=True)
class SweepConfig:
    name: str
    guidance_mode: str
    w_cg: float
    optimization_guidance_scale: float
    optimization_guidance_last_steps: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "guidance_mode": self.guidance_mode,
            "w_cg": self.w_cg,
            "optimization_guidance_scale": self.optimization_guidance_scale,
            "optimization_guidance_last_steps": self.optimization_guidance_last_steps,
        }


def _label_w_cg(w: float) -> str:
    if w == 0.0:
        return "standard_w_cg0"
    s = f"{w:g}".replace(".", "p")
    return f"standard_w_cg{s}"


def _label_opt(scale: float, last_steps: int | None = None) -> str:
    s = f"{scale:g}".replace(".", "p")
    if last_steps is None or last_steps == 10:
        return f"optimization_scale_{s}"
    return f"optimization_scale_{s}_last{last_steps}"


def _w_cg_configs(values: list[float], last_steps: int = 10) -> list[SweepConfig]:
    out = []
    for w in values:
        out.append(
            SweepConfig(
                name=_label_w_cg(w),
                guidance_mode="standard",
                w_cg=float(w),
                optimization_guidance_scale=0.0,
                optimization_guidance_last_steps=last_steps,
            )
        )
    return out


def _opt_configs(
    scales: list[float],
    last_steps: int = 10,
) -> list[SweepConfig]:
    out = []
    for scale in scales:
        out.append(
            SweepConfig(
                name=_label_opt(scale, last_steps),
                guidance_mode="optimization",
                w_cg=0.0,
                optimization_guidance_scale=float(scale),
                optimization_guidance_last_steps=last_steps,
            )
        )
    return out


def _refine_values(center: float, *, radius: float, step: float, lo: float = 0.0) -> list[float]:
    if step <= 0:
        return [center]
    start = max(lo, center - radius)
    end = center + radius
    n = int(math.floor((end - start) / step + 1e-9)) + 1
    vals = [round(start + i * step, 6) for i in range(n)]
    vals = sorted(set(max(lo, v) for v in vals))
    return vals


def _dedupe_configs(configs: list[SweepConfig]) -> list[SweepConfig]:
    seen: set[tuple] = set()
    out: list[SweepConfig] = []
    for cfg in configs:
        key = (
            cfg.guidance_mode,
            round(cfg.w_cg, 8),
            round(cfg.optimization_guidance_scale, 8),
            cfg.optimization_guidance_last_steps,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(cfg)
    return out


class SweepRunner:
    def __init__(self, args):
        self.args = args
        self._agent_cache: dict[tuple, Any] = {}

    def _load_bundle(self, test: str):
        class _A:
            pass

        a = _A()
        a.test = test
        a.output_root = self.args.output_root
        a.seed = self.args.seed
        a.num_dataset_samples = self.args.num_dataset_samples
        a.square_half_width = self.args.square_half_width
        a.ellipsoid_axes = self.args.ellipsoid_axes
        return _load_dataset_bundle(a)

    def _get_agent(self, cfg: SweepConfig, reward_fn, diffusion_steps: int):
        key = diffusion_steps
        if key not in self._agent_cache:
            agent = _build_agent(
                self.args,
                classifier=None,
                diffusion_steps=diffusion_steps,
            )
            paths = _resolve_paths(self.args.test, Path(self.args.output_root))
            agent.load(str(paths["ckpt"]))
            agent.eval()
            self._agent_cache[key] = agent
        agent = self._agent_cache[key]
        if cfg.w_cg == 0.0 and cfg.optimization_guidance_scale == 0.0:
            agent.classifier = None
        else:
            agent.classifier = RuntimeRewardClassifier(reward_fn, device=self.args.device)
        return agent

    def evaluate(self, cfg: SweepConfig, geometry: GeometryMeta, normalizer) -> dict[str, Any]:
        validate_guidance_config(cfg.guidance_mode, cfg.w_cg, cfg.optimization_guidance_scale)
        reward_fn = _make_reward_fn(geometry, normalizer, self.args.device)
        diff_steps = (
            self.args.inference_diffusion_steps
            if cfg.guidance_mode == "optimization"
            else self.args.diffusion_steps
        )
        agent = self._get_agent(cfg, reward_fn, diff_steps)

        prior = torch.zeros((self.args.num_samples, HORIZON, AMBIENT_DIM), device=self.args.device)
        traj, _ = agent.sample(
            prior,
            solver=self.args.solver,
            n_samples=self.args.num_samples,
            sample_steps=self.args.sampling_steps,
            sample_step_schedule=self.args.sample_step_schedule,
            use_ema=self.args.use_ema,
            w_cg=cfg.w_cg,
            guidance_mode=cfg.guidance_mode,
            optimization_guidance_scale=cfg.optimization_guidance_scale,
            optimization_guidance_last_steps=cfg.optimization_guidance_last_steps,
            temperature=self.args.temperature,
        )
        metrics = _sample_batch_metrics(traj, geometry, normalizer)
        row = {**cfg.to_dict(), **metrics, "num_samples": self.args.num_samples}
        row["score"] = row["reward_mean"] - self.args.violation_penalty * row["violation_mean"]
        return row

    def run_phase(
        self,
        phase: str,
        configs: list[SweepConfig],
        geometry: GeometryMeta,
        normalizer,
        results: list[dict],
    ) -> list[dict]:
        configs = _dedupe_configs(configs)
        print(f"\n[{self.args.test}] {phase}: {len(configs)} configs, n={self.args.num_samples}")
        for cfg in configs:
            key = (
                cfg.guidance_mode,
                round(cfg.w_cg, 8),
                round(cfg.optimization_guidance_scale, 8),
                cfg.optimization_guidance_last_steps,
            )
            if any(
                (
                    r["guidance_mode"],
                    round(r["w_cg"], 8),
                    round(r["optimization_guidance_scale"], 8),
                    r["optimization_guidance_last_steps"],
                )
                == key
                for r in results
            ):
                continue
            row = self.evaluate(cfg, geometry, normalizer)
            row["phase"] = phase
            results.append(row)
            print(
                f"  {cfg.name}: reward={row['reward_mean']:.4f} "
                f"viol={row['violation_mean']:.4f} score={row['score']:.4f}"
            )
        return results

    def top_k(self, results: list[dict], guidance_mode: str, k: int = 3) -> list[dict]:
        pool = [r for r in results if r["guidance_mode"] == guidance_mode]
        pool.sort(key=lambda r: (r["score"], r["reward_mean"]), reverse=True)
        return pool[:k]


def run_adaptive_sweep(args) -> dict:
    set_seed(args.seed)
    args.test = args.test  # set by caller
    runner = SweepRunner(args)
    _, geometry, normalizer = runner._load_bundle(args.test)
    results: list[dict] = []

    # Phase 1 — coarse
    w_coarse = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0]
    opt_coarse = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 150.0, 200.0]
    runner.run_phase("phase1_coarse", _w_cg_configs(w_coarse) + _opt_configs(opt_coarse), geometry, normalizer, results)

    # Phase 2 — refine w_cg
    w_refine_configs: list[SweepConfig] = []
    for row in runner.top_k(results, "standard", k=4):
        w_refine_configs.extend(
            _w_cg_configs(
                _refine_values(row["w_cg"], radius=args.w_cg_refine_radius, step=args.w_cg_refine_step),
            )
        )
    runner.run_phase("phase2_w_cg_refine", w_refine_configs, geometry, normalizer, results)

    # Phase 2 — refine opt scale
    opt_refine_configs: list[SweepConfig] = []
    for row in runner.top_k(results, "optimization", k=4):
        opt_refine_configs.extend(
            _opt_configs(
                _refine_values(
                    row["optimization_guidance_scale"],
                    radius=args.opt_refine_radius,
                    step=args.opt_refine_step,
                    lo=0.1,
                ),
            )
        )
    runner.run_phase("phase2_opt_refine", opt_refine_configs, geometry, normalizer, results)

    # Phase 3 — last_steps for top opt configs
    last_step_configs: list[SweepConfig] = []
    for row in runner.top_k(results, "optimization", k=3):
        scale = row["optimization_guidance_scale"]
        for ls in args.last_steps_candidates:
            last_step_configs.append(
                SweepConfig(
                    name=_label_opt(scale, ls),
                    guidance_mode="optimization",
                    w_cg=0.0,
                    optimization_guidance_scale=scale,
                    optimization_guidance_last_steps=int(ls),
                )
            )
    runner.run_phase("phase3_last_steps", last_step_configs, geometry, normalizer, results)

    # Phase 4 — high-sample validation of top configs
    top_validate = runner.top_k(results, "standard", k=2) + runner.top_k(results, "optimization", k=2)
    validate_configs = [
        SweepConfig(
            name=r["name"] + "_validate",
            guidance_mode=r["guidance_mode"],
            w_cg=r["w_cg"],
            optimization_guidance_scale=r["optimization_guidance_scale"],
            optimization_guidance_last_steps=r["optimization_guidance_last_steps"],
        )
        for r in top_validate
    ]
    old_n = args.num_samples
    args.num_samples = args.validation_samples
    runner._agent_cache.clear()
    runner.run_phase("phase4_validate", validate_configs, geometry, normalizer, results)
    args.num_samples = old_n

    best_w = runner.top_k(results, "standard", k=1)[0]
    best_opt = runner.top_k(results, "optimization", k=1)[0]
    best_reward_w = max(
        (r for r in results if r["guidance_mode"] == "standard"),
        key=lambda r: r["reward_mean"],
    )
    best_reward_opt = max(
        (r for r in results if r["guidance_mode"] == "optimization"),
        key=lambda r: r["reward_mean"],
    )

    summary = {
        "test": args.test,
        "num_runs": len(results),
        "violation_penalty_for_score": args.violation_penalty,
        "best_by_score": {"standard_w_cg": best_w, "optimization": best_opt},
        "best_by_raw_reward": {"standard_w_cg": best_reward_w, "optimization": best_reward_opt},
        "all_runs": results,
    }

    out_dir = Path(args.output_root) / args.test / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{args.test}] wrote {out_path}")
    print(f"[{args.test}] BEST w_cg (score): {best_w['name']} reward={best_w['reward_mean']:.4f} "
          f"viol={best_w['violation_mean']:.4f}")
    print(f"[{args.test}] BEST opt (score): {best_opt['name']} reward={best_opt['reward_mean']:.4f} "
          f"viol={best_opt['violation_mean']:.4f}")
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
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--validation-samples", type=int, default=2048)
    p.add_argument("--solver", default="ddim")
    p.add_argument("--sampling-steps", type=int, default=20)
    p.add_argument("--sample-step-schedule", default="uniform")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--violation-penalty", type=float, default=0.5,
                   help="Score = reward_mean - penalty * violation_mean")
    p.add_argument("--w-cg-refine-radius", type=float, default=5.0)
    p.add_argument("--w-cg-refine-step", type=float, default=0.5)
    p.add_argument("--opt-refine-radius", type=float, default=20.0)
    p.add_argument("--opt-refine-step", type=float, default=2.0)
    p.add_argument("--last-steps-candidates", nargs="+", type=int,
                   default=[5, 8, 10, 12, 15, 20, 25, 30, 40, 50])
    args = p.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    return args


def main():
    args = parse_args()
    tests = ["test_a", "test_b"] if args.test == "both" else [args.test]
    all_summaries = {}
    for test in tests:
        args.test = test
        all_summaries[test] = run_adaptive_sweep(args)

    combined_path = Path(args.output_root) / "sweep_combined_summary.json"
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nCombined summary: {combined_path}")


if __name__ == "__main__":
    main()
