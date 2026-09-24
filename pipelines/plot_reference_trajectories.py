#!/usr/bin/env python3
"""Generate and plot reference trajectories (circle + 6 seed-fixed random) for review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from reference_trajectories import (
    DATA_XY_LIMIT,
    build_reference_catalog,
    validate_trajectory,
)


def _plot_catalog(trajectories, output_path: Path) -> None:
    n = len(trajectories)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, traj in enumerate(trajectories):
        ax = axes[idx // ncols, idx % ncols]
        xy = traj.xy
        ax.plot(xy[:, 0], xy[:, 1], color="tab:blue", linewidth=1.5)
        ax.scatter(xy[0, 0], xy[0, 1], color="black", s=50, marker="*", zorder=5, label="start")
        ax.scatter(xy[-1, 0], xy[-1, 1], color="tab:red", s=40, marker="x", zorder=5, label="end")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-DATA_XY_LIMIT, DATA_XY_LIMIT)
        ax.set_ylim(-DATA_XY_LIMIT, DATA_XY_LIMIT)
        stats = validate_trajectory(xy)
        ax.set_title(
            f"{traj.name}\n{traj.family} | max step={stats['max_segment_length']:.3f}",
            fontsize=9,
        )
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    fig.suptitle(
        "Reference trajectories for multi-plan eval (64 steps = full path, trackable region)",
        fontsize=12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        default="results/unicycle_eval/reference_trajectories",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = build_reference_catalog(num_steps=args.num_steps)

    payload = {
        "num_steps": args.num_steps,
        "data_xy_limit": DATA_XY_LIMIT,
        "trajectories": [],
    }
    for traj in trajectories:
        entry = traj.to_dict()
        entry["validation"] = validate_trajectory(traj.xy)
        payload["trajectories"].append(entry)

    catalog_path = output_dir / "catalog.json"
    with open(catalog_path, "w") as f:
        json.dump(payload, f, indent=2)

    plot_path = output_dir / "reference_trajectories_review.png"
    _plot_catalog(trajectories, plot_path)

    print(f"Wrote {catalog_path}")
    print(f"Wrote {plot_path}")
    for entry in payload["trajectories"]:
        v = entry["validation"]
        print(
            f"  {entry['name']:18s}  "
            f"x=[{entry['xy_min'][0]:+.2f},{entry['xy_max'][0]:+.2f}]  "
            f"y=[{entry['xy_min'][1]:+.2f},{entry['xy_max'][1]:+.2f}]  "
            f"max_step={v['max_segment_length']:.3f}  ok={v['in_bounds']}"
        )


if __name__ == "__main__":
    main()
