"""Aggregate per-seed hopper trajectory eval JSON files into mean/std summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob",
        default="results/diffuser_d4rl_mujoco/hopper-medium-v2/trajectory_eval_latest_seed*.json",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    files = sorted(repo.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No eval files matched: {args.input_glob}")

    records = []
    for path in files:
        with open(path) as f:
            data = json.load(f)
        records.append(
            {
                "seed": int(data["seed"]),
                "raw_reward": float(data["mean_reward"]),
                "normalized_score_x100": float(data["mean_normalized_score_x100"]),
                "survival_steps": float(data["mean_survival_steps"]),
                "sim_env": data.get("sim_env"),
                "path": str(path),
            }
        )

    raw = np.array([r["raw_reward"] for r in records], dtype=np.float64)
    norm = np.array([r["normalized_score_x100"] for r in records], dtype=np.float64)
    steps = np.array([r["survival_steps"] for r in records], dtype=np.float64)

    summary = {
        "n_seeds": len(records),
        "seeds": [r["seed"] for r in records],
        "sim_env": records[0]["sim_env"],
        "mean_raw_reward": float(raw.mean()),
        "std_raw_reward": float(raw.std(ddof=1) if raw.size > 1 else 0.0),
        "mean_normalized_score_x100": float(norm.mean()),
        "std_normalized_score_x100": float(norm.std(ddof=1) if norm.size > 1 else 0.0),
        "mean_survival_steps": float(steps.mean()),
        "std_survival_steps": float(steps.std(ddof=1) if steps.size > 1 else 0.0),
        "per_seed": records,
    }

    print("=== Aggregated trajectory eval (seeds) ===")
    print(
        f"raw_reward: {summary['mean_raw_reward']:.2f} ± {summary['std_raw_reward']:.2f}"
    )
    print(
        "normalized_x100: "
        f"{summary['mean_normalized_score_x100']:.2f} ± "
        f"{summary['std_normalized_score_x100']:.2f}"
    )
    print(
        f"survival_steps: {summary['mean_survival_steps']:.1f} ± "
        f"{summary['std_survival_steps']:.1f}"
    )

    out = Path(args.output) if args.output else repo / (
        "results/diffuser_d4rl_mujoco/hopper-medium-v2/trajectory_eval_latest_seeds_summary.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
