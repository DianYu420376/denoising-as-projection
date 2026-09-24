#!/usr/bin/env python3
"""Aggregate mean/std from seed*.json in a hybrid scale directory."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

SEED_JSON = re.compile(r"^seed(?P<seed>\d+)_.*\.json$")


def aggregate(scale_dir: Path, seed_min: int, seed_max: int) -> dict:
    rewards, norms, per_seed = [], [], []
    for path in sorted(scale_dir.glob("seed*.json")):
        if path.name.startswith("seed_sweep"):
            continue
        m = SEED_JSON.match(path.name)
        if not m:
            continue
        seed = int(m.group("seed"))
        if seed < seed_min or seed_max < seed:
            continue
        with open(path) as f:
            data = json.load(f)
        reward = float(data["mean_reward"])
        norm = float(data["mean_normalized_score_x100"])
        rewards.append(reward)
        norms.append(norm)
        per_seed.append({"seed": seed, "reward": reward, "norm_x100": norm})

    n = len(rewards)
    if n == 0:
        return {"n_seeds": 0, "seed_range": [seed_min, seed_max]}

    r = np.asarray(rewards)
    z = np.asarray(norms)
    std_r = float(r.std(ddof=1)) if n > 1 else 0.0
    std_z = float(z.std(ddof=1)) if n > 1 else 0.0
    seeds_found = sorted(x["seed"] for x in per_seed)
    return {
        "scale_dir": str(scale_dir),
        "seed_range": [seed_min, seed_max],
        "n_seeds": n,
        "seeds_found": seeds_found,
        "seeds_missing": [s for s in range(seed_min, seed_max + 1) if s not in seeds_found],
        "mean_reward": float(r.mean()),
        "std_reward": std_r,
        "se_reward": std_r / math.sqrt(n),
        "mean_norm_x100": float(z.mean()),
        "std_norm_x100": std_z,
        "se_norm_x100": std_z / math.sqrt(n),
        "per_seed": sorted(per_seed, key=lambda x: x["seed"]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scale-dir", required=True)
    p.add_argument("--seed-min", type=int, required=True)
    p.add_argument("--seed-max", type=int, required=True)
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    scale_dir = Path(args.scale_dir)
    summary = aggregate(scale_dir, args.seed_min, args.seed_max)
    print(
        f"seeds {args.seed_min}-{args.seed_max}: n={summary.get('n_seeds', 0)} "
        f"reward={summary.get('mean_reward', float('nan')):.2f} ± {summary.get('std_reward', float('nan')):.2f} "
        f"norm_x100={summary.get('mean_norm_x100', float('nan')):.2f} ± {summary.get('std_norm_x100', float('nan')):.2f}"
    )
    if summary.get("seeds_missing"):
        print(f"missing: {summary['seeds_missing']}")

    if args.output_json:
        out = Path(args.output_json)
        out.write_text(json.dumps(summary, indent=2))
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
