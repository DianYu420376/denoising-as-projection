#!/usr/bin/env python3
"""Scrape reference_plan comparison results from per-run summary.json files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

METRICS = (
    ("plan_reward", "plan_reward"),
    ("rollout_reward", "rollout_reward"),
    ("feasibility_mean_l2_future", "feasibility_l2"),
)

DEFAULT_CONFIG_ORDER = [
    "standard_w_cg0",
    "standard_w_cg10",
    "standard_w_cg30",
    "standard_w_cg50",
    "standard_w_cg100",
    "optimization_scale_40",
    "optimization_scale_50",
    "optimization_scale_60",
]

TRAJECTORY_ORDER = [
    "circle",
    "sinusoid_freq1",
    "sinusoid_freq2",
    "sinusoid_freq3",
    "random_smooth_1",
    "random_smooth_2",
    "random_smooth_3",
    "random_smooth_4",
    "random_smooth_5",
    "random_smooth_6",
]


def _ordered(items: set[str], preferred: list[str]) -> list[str]:
    ordered = [name for name in preferred if name in items]
    extras = sorted(items - set(ordered))
    return ordered + extras


def _extract_plan_block(payload: dict) -> dict | None:
    for key in ("reference_plan", "heart_plan"):
        if key in payload:
            return payload[key]
    return None


def scrape_comparison_root(root: Path) -> dict:
    """Load all {trajectory}/{config}/summary.json under root."""
    runs: list[dict] = []
    missing: list[str] = []

    if not root.is_dir():
        raise FileNotFoundError(f"Comparison root not found: {root}")

    for summary_path in sorted(root.glob("*/*/summary.json")):
        trajectory = summary_path.parent.parent.name
        config = summary_path.parent.name
        with open(summary_path) as f:
            payload = json.load(f)

        block = _extract_plan_block(payload)
        if block is None:
            missing.append(f"{trajectory}/{config} (no reference_plan/heart_plan block)")
            continue

        stats = block.get("candidate_metrics_summary", {})
        row = {
            "trajectory": trajectory,
            "config": config,
            "plan_candidates": block.get("plan_candidates", block.get("heart_plan_candidates")),
            "summary_path": str(summary_path),
        }
        for stat_key, _label in METRICS:
            metric = stats.get(stat_key, {})
            row[f"{stat_key}_mean"] = metric.get("mean")
            row[f"{stat_key}_std"] = metric.get("std")
        runs.append(row)

    by_trajectory: dict[str, list[dict]] = {}
    for row in runs:
        by_trajectory.setdefault(row["trajectory"], []).append(row)

    trajectories = _ordered(set(by_trajectory), TRAJECTORY_ORDER)
    all_configs = {row["config"] for row in runs}
    config_order = _ordered(all_configs, DEFAULT_CONFIG_ORDER)

    return {
        "root": str(root),
        "n_runs": len(runs),
        "trajectories": trajectories,
        "config_order": config_order,
        "runs": runs,
        "by_trajectory": {
            traj: sorted(
                by_trajectory.get(traj, []),
                key=lambda r: (
                    config_order.index(r["config"])
                    if r["config"] in config_order
                    else len(config_order)
                ),
            )
            for traj in trajectories
        },
        "missing_or_invalid": missing,
    }


def _fmt_mean_std(mean, std, width: int = 18) -> str:
    if mean is None or std is None:
        return f"{'—':>{width}}"
    return f"{mean:+.4f} ± {std:.4f}".rjust(width)


def print_summary(summary: dict) -> None:
    print(f"Root: {summary['root']}")
    print(f"Runs found: {summary['n_runs']}")
    if summary["missing_or_invalid"]:
        print(f"Skipped: {len(summary['missing_or_invalid'])}")

    metric_headers = [label for _, label in METRICS]
    config_width = 26
    metric_width = 20
    header = f"{'config':<{config_width}}" + "".join(
        f"{name:>{metric_width}}" for name in metric_headers
    )

    for trajectory in summary["trajectories"]:
        rows = summary["by_trajectory"].get(trajectory, [])
        if not rows:
            continue
        print()
        print(f"=== {trajectory} ===")
        print(header)
        print("-" * len(header))
        for row in rows:
            line = f"{row['config']:<{config_width}}"
            for stat_key, _label in METRICS:
                line += _fmt_mean_std(
                    row.get(f"{stat_key}_mean"),
                    row.get(f"{stat_key}_std"),
                    width=metric_width,
                )
            print(line)

    expected = len(summary["trajectories"]) * len(summary["config_order"])
    if summary["n_runs"] < expected:
        print()
        print(
            f"Note: expected up to {expected} runs "
            f"({len(summary['trajectories'])} trajectories × {len(summary['config_order'])} configs), "
            f"found {summary['n_runs']}."
        )


def print_matrix(summary: dict) -> None:
    """Compact view: one metric at a time, trajectories × configs (mean only)."""
    trajectories = summary["trajectories"]
    configs = summary["config_order"]
    lookup = {(r["trajectory"], r["config"]): r for r in summary["runs"]}

    for stat_key, label in METRICS:
        print()
        print(f"=== {label} (mean) ===")
        header = f"{'trajectory':<18}" + "".join(f"{c:>14}" for c in configs)
        print(header)
        print("-" * len(header))
        for traj in trajectories:
            cells = [f"{traj:<18}"]
            for cfg in configs:
                row = lookup.get((traj, cfg))
                if row is None or row.get(f"{stat_key}_mean") is None:
                    cells.append(f"{'—':>14}")
                else:
                    cells.append(f"{row[f'{stat_key}_mean']:>14.4f}")
            print("".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape reference_plan comparison mean/std per trajectory and config."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/unicycle_eval/reference_plan_comparison_s100_v2"),
        help="Comparison output root ({trajectory}/{config}/summary.json).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write scraped summary JSON.",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Also print mean-only trajectory × config matrices per metric.",
    )
    args = parser.parse_args()

    summary = scrape_comparison_root(args.root.resolve())
    print_summary(summary)
    if args.matrix:
        print_matrix(summary)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print()
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
