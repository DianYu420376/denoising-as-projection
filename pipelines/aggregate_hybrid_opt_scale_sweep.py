#!/usr/bin/env python3
"""Aggregate hybrid opt_scale coarse sweep results across scale subdirectories."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

SEED_JSON = re.compile(r"^seed(?P<seed>\d+)_.*\.json$")


def _mean_std(values: list[float]) -> tuple[float, float, int]:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(arr.mean()), 0.0, 1
    return float(arr.mean()), float(arr.std(ddof=1)), n


def _load_scale_summary(scale_dir: Path) -> dict | None:
    agg = scale_dir / "seed_sweep_aggregate.json"
    if agg.is_file():
        with open(agg) as f:
            report = json.load(f)
        primary = report.get("primary_summary") or report.get("json_summary")
        if primary:
            return {
                "source": "seed_sweep_aggregate",
                "scale_dir": str(scale_dir),
                "n_seeds": primary.get("n_seeds", 0),
                "seeds_found": primary.get("seeds_found", []),
                "seeds_missing": primary.get("seeds_missing", []),
                "mean_reward": float(primary.get("mean_reward", float("nan"))),
                "std_reward": float(primary.get("std_reward", float("nan"))),
                "mean_norm_x100": float(primary.get("mean_norm_x100", float("nan"))),
                "std_norm_x100": float(primary.get("std_norm_x100", float("nan"))),
                "per_seed": primary.get("per_seed", []),
            }

    rewards: list[float] = []
    norms: list[float] = []
    per_seed: list[dict] = []
    for path in sorted(scale_dir.glob("seed*.json")):
        if path.name == "seed_sweep_aggregate.json":
            continue
        m = SEED_JSON.match(path.name)
        if not m:
            continue
        with open(path) as f:
            data = json.load(f)
        seed = int(m.group("seed"))
        reward = float(data.get("mean_reward", float("nan")))
        norm = float(data.get("mean_normalized_score_x100", float("nan")))
        rewards.append(reward)
        norms.append(norm)
        per_seed.append({"seed": seed, "reward": reward, "norm_x100": norm, "file": str(path)})

    if not rewards:
        return None

    r_mean, r_std, n = _mean_std(rewards)
    n_mean, n_std, _ = _mean_std(norms)
    seeds_found = sorted(r["seed"] for r in per_seed)
    return {
        "source": "seed_json_files",
        "scale_dir": str(scale_dir),
        "n_seeds": n,
        "seeds_found": seeds_found,
        "seeds_missing": [],
        "mean_reward": r_mean,
        "std_reward": r_std,
        "mean_norm_x100": n_mean,
        "std_norm_x100": n_std,
        "per_seed": per_seed,
    }


def aggregate_sweep(sweep_root: Path, manifest: dict | None = None) -> dict:
    by_scale_dir = sweep_root / "by_scale"
    rows: list[dict] = []

    scale_entries = []
    if manifest and manifest.get("scales"):
        scale_entries = manifest["scales"]
    elif by_scale_dir.is_dir():
        scale_entries = [{"scale_dir": p.name} for p in sorted(by_scale_dir.iterdir()) if p.is_dir()]
    else:
        scale_entries = [{"scale_dir": p.name} for p in sorted(sweep_root.iterdir()) if p.is_dir()]

    for entry in scale_entries:
        rel = entry.get("scale_dir") or entry.get("dir")
        opt_scale = entry.get("opt_scale")
        array_job_id = entry.get("array_job_id")
        scale_path = by_scale_dir / rel if (by_scale_dir / rel).is_dir() else sweep_root / rel
        if not scale_path.is_dir():
            continue

        summary = _load_scale_summary(scale_path)
        if summary is None:
            rows.append(
                {
                    "opt_scale": opt_scale,
                    "scale_dir": str(scale_path),
                    "array_job_id": array_job_id,
                    "n_seeds": 0,
                    "status": "missing",
                }
            )
            continue

        if opt_scale is None:
            opt_scale = entry.get("opt_scale_label")
        if opt_scale is None:
            m = re.search(r"opt([^_]+)", scale_path.name)
            opt_scale = float(m.group(1).replace("p", ".")) if m else float("nan")

        rows.append(
            {
                "opt_scale": float(opt_scale),
                "scale_dir": str(scale_path),
                "array_job_id": array_job_id,
                "n_seeds": summary["n_seeds"],
                "seeds_found": summary["seeds_found"],
                "seeds_missing": summary.get("seeds_missing", []),
                "mean_reward": summary["mean_reward"],
                "std_reward": summary["std_reward"],
                "se_norm_x100": summary["std_norm_x100"] / math.sqrt(summary["n_seeds"])
                if summary["n_seeds"]
                else float("nan"),
                "se_reward": summary["std_reward"] / math.sqrt(summary["n_seeds"])
                if summary["n_seeds"]
                else float("nan"),
                "mean_norm_x100": summary["mean_norm_x100"],
                "std_norm_x100": summary["std_norm_x100"],
                "source": summary["source"],
                "status": "ok" if summary["n_seeds"] >= 15 else "partial",
                "per_seed": summary.get("per_seed", []),
            }
        )

    rows.sort(key=lambda r: r.get("opt_scale", float("nan")))
    complete = [r for r in rows if r.get("status") == "ok"]
    best = max(complete, key=lambda r: r["mean_norm_x100"]) if complete else None

    return {
        "sweep_root": str(sweep_root),
        "manifest": manifest,
        "num_scales": len(rows),
        "num_scales_complete": len(complete),
        "metric_primary": "mean_norm_x100",
        "by_opt_scale": rows,
        "best_by_norm_x100": best,
    }


def _print_summary(report: dict) -> None:
    print(f"=== Hybrid opt_scale coarse sweep ===")
    print(f"Sweep root: {report['sweep_root']}")
    print(f"Scales complete: {report['num_scales_complete']}/{report['num_scales']}")
    print()
    print(f"{'opt_scale':>10}  {'n':>3}  {'reward_mean':>10}  {'reward_std':>10}  {'norm_mean':>9}  {'norm_std':>9}  status")
    for row in report["by_opt_scale"]:
        if row.get("status") == "missing":
            print(f"{row.get('opt_scale', '?'):>10}  {'0':>3}  {'—':>10}  {'—':>10}  {'—':>9}  {'—':>9}  missing")
            continue
        print(
            f"{row['opt_scale']:>10.4g}  {row['n_seeds']:>3}  "
            f"{row['mean_reward']:>10.2f}  {row['std_reward']:>10.2f}  "
            f"{row['mean_norm_x100']:>9.2f}  {row['std_norm_x100']:>9.2f}  {row['status']}"
        )
    if report.get("best_by_norm_x100"):
        b = report["best_by_norm_x100"]
        print(
            f"\nBest (norm_x100): opt_scale={b['opt_scale']} "
            f"reward={b['mean_reward']:.2f}±{b['std_reward']:.2f} "
            f"norm={b['mean_norm_x100']:.2f}±{b['std_norm_x100']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--manifest", default=None, help="Optional sweep_manifest.json path.")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    manifest_path = Path(args.manifest) if args.manifest else sweep_root / "sweep_manifest.json"
    manifest = None
    if manifest_path.is_file():
        with open(manifest_path) as f:
            manifest = json.load(f)

    report = aggregate_sweep(sweep_root, manifest)
    _print_summary(report)

    out = Path(args.output_json) if args.output_json else sweep_root / "coarse_sweep_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
