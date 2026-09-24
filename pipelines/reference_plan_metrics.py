#!/usr/bin/env python3
"""Compute mean/std of plan reward, rollout reward, and feasibility L2 per reference task and config."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

METRICS = (
    ("plan_reward", "plan_reward"),
    ("rollout_reward", "rollout_reward"),
    ("feasibility_mean_l2_future", "feasibility_l2"),
)

DEFAULT_TRAJECTORY_ORDER = [
    "circle",
    "random_smooth_1",
    "random_smooth_2",
    "random_smooth_3",
    "random_smooth_4",
    "random_smooth_5",
    "random_smooth_6",
    "random_smooth_7",
    "random_smooth_8",
    "random_smooth_9",
    "random_smooth_10",
    "random_smooth_11",
    "half_heart",
]

DEFAULT_CONFIG_ORDER = [
    "standard_w_cg0",
    "standard_w_cg10",
    "standard_w_cg20",
    "standard_w_cg30",
    "standard_w_cg50",
    "optimization_scale_10",
    "optimization_scale_20",
    "optimization_scale_30",
    "optimization_scale_40",
]


def _ordered(items: set[str], preferred: list[str]) -> list[str]:
    ordered = [name for name in preferred if name in items]
    extras = sorted(items - set(ordered))
    return ordered + extras


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _load_block(summary_path: Path) -> dict | None:
    with open(summary_path) as f:
        payload = json.load(f)
    for key in ("reference_plan", "heart_plan"):
        if key in payload:
            return payload[key]
    return None


def _metrics_from_summary(block: dict) -> dict:
    stats = block.get("candidate_metrics_summary", {})
    row = {
        "n_candidates": block.get("plan_candidates"),
        "source": "summary",
    }
    for stat_key, label in METRICS:
        metric = stats.get(stat_key, {})
        row[f"{label}_mean"] = metric.get("mean")
        row[f"{label}_std"] = metric.get("std")
    return row


def _metrics_from_npz(traj_dir: Path) -> dict:
    npz_path = traj_dir / "candidates.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing trajectory cache: {npz_path}")
    data = np.load(npz_path)
    row = {"n_candidates": int(data["plan_reward"].shape[0]), "source": "npz"}
    mapping = {
        "plan_reward": "plan_reward",
        "rollout_reward": "rollout_reward",
        "feasibility_mean_l2_future": "feasibility_l2",
    }
    for npz_key, label in mapping.items():
        mean, std = _mean_std(data[npz_key].astype(float).tolist())
        row[f"{label}_mean"] = mean
        row[f"{label}_std"] = std
    return row


def collect_metrics(
    root: Path,
    *,
    prefer_npz: bool = False,
) -> dict:
    rows: list[dict] = []
    missing: list[str] = []

    if not root.is_dir():
        raise FileNotFoundError(f"Root not found: {root}")

    for summary_path in sorted(root.glob("*/*/summary.json")):
        trajectory = summary_path.parent.parent.name
        config = summary_path.parent.name
        block = _load_block(summary_path)
        if block is None:
            missing.append(f"{trajectory}/{config} (no plan block in summary.json)")
            continue

        traj_dir = summary_path.parent / "reference_plan" / "trajectories"
        try:
            if prefer_npz and traj_dir.is_dir():
                metrics = _metrics_from_npz(traj_dir)
            else:
                metrics = _metrics_from_summary(block)
        except FileNotFoundError:
            metrics = _metrics_from_summary(block)

        rows.append(
            {
                "trajectory": trajectory,
                "config": config,
                "summary_path": str(summary_path),
                "trajectory_dir": str(traj_dir) if traj_dir.is_dir() else None,
                **metrics,
            }
        )

    trajectories = _ordered({row["trajectory"] for row in rows}, DEFAULT_TRAJECTORY_ORDER)
    configs = _ordered({row["config"] for row in rows}, DEFAULT_CONFIG_ORDER)
    lookup = {(row["trajectory"], row["config"]): row for row in rows}

    return {
        "root": str(root),
        "n_runs": len(rows),
        "trajectories": trajectories,
        "configs": configs,
        "rows": rows,
        "lookup": lookup,
        "missing_or_invalid": missing,
    }


def _fmt(mean, std, width: int = 18) -> str:
    if mean is None or std is None or (isinstance(mean, float) and np.isnan(mean)):
        return f"{'—':>{width}}"
    return f"{mean:+.4f} ± {std:.4f}".rjust(width)


def print_tables(summary: dict) -> None:
    print(f"Root: {summary['root']}")
    print(f"Runs: {summary['n_runs']}")
    if summary["missing_or_invalid"]:
        print(f"Skipped: {len(summary['missing_or_invalid'])}")

    metric_labels = [label for _, label in METRICS]
    config_width = 24
    metric_width = 20
    header = f"{'trajectory':<18} {'config':<{config_width}}" + "".join(
        f"{name:>{metric_width}}" for name in metric_labels
    )

    for trajectory in summary["trajectories"]:
        print()
        print(f"=== {trajectory} ===")
        print(header)
        print("-" * len(header))
        for config in summary["configs"]:
            row = summary["lookup"].get((trajectory, config))
            if row is None:
                continue
            line = f"{trajectory:<18} {config:<{config_width}}"
            for stat_key, label in METRICS:
                line += _fmt(row.get(f"{label}_mean"), row.get(f"{label}_std"), width=metric_width)
            print(line)


def print_matrix(summary: dict, label: str) -> None:
    trajectories = summary["trajectories"]
    configs = summary["configs"]
    print()
    print(f"=== {label} mean ===")
    header = f"{'trajectory':<18}" + "".join(f"{c:>14}" for c in configs)
    print(header)
    print("-" * len(header))
    for trajectory in trajectories:
        cells = [f"{trajectory:<18}"]
        for config in configs:
            row = summary["lookup"].get((trajectory, config))
            value = None if row is None else row.get(f"{label}_mean")
            cells.append(f"{value:>14.4f}" if value is not None else f"{'—':>14}")
        print("".join(cells))


def write_csv(summary: dict, csv_path: Path) -> None:
    fieldnames = [
        "trajectory",
        "config",
        "n_candidates",
        "source",
        "plan_reward_mean",
        "plan_reward_std",
        "rollout_reward_mean",
        "rollout_reward_std",
        "feasibility_l2_mean",
        "feasibility_l2_std",
        "summary_path",
        "trajectory_dir",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_log(summary: dict, log_path: Path) -> None:
    lines = [
        "Reference plan metrics — mean ± std over all candidates",
        f"root: {summary['root']}",
        f"runs: {summary['n_runs']}",
        "",
        (
            f"{'trajectory':<18} {'config':<24} {'n':>4} "
            f"{'plan_reward':>20} {'rollout_reward':>20} {'feasibility_l2':>20}"
        ),
        "-" * 112,
    ]
    for trajectory in summary["trajectories"]:
        for config in summary["configs"]:
            row = summary["lookup"].get((trajectory, config))
            if row is None:
                continue
            lines.append(
                f"{trajectory:<18} {config:<24} {row.get('n_candidates', '—')!s:>4} "
                f"{_fmt(row.get('plan_reward_mean'), row.get('plan_reward_std'))} "
                f"{_fmt(row.get('rollout_reward_mean'), row.get('rollout_reward_std'))} "
                f"{_fmt(row.get('feasibility_l2_mean'), row.get('feasibility_l2_std'))}"
            )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mean/std of plan reward, rollout reward, and feasibility L2 per task and config."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/unicycle_eval/reference_plan_v1"),
        help="Comparison output root ({trajectory}/{config}/summary.json).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output (default: {root}/metrics_mean_std.json).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output (default: {root}/metrics_mean_std.csv).",
    )
    parser.add_argument(
        "--output-log",
        type=Path,
        default=None,
        help="Optional text log (default: {root}/metrics_mean_std.log).",
    )
    parser.add_argument(
        "--prefer-npz",
        action="store_true",
        help="Recompute mean/std from trajectories/candidates.npz instead of summary.json.",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Also print mean-only trajectory × config matrices.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    summary = collect_metrics(root, prefer_npz=args.prefer_npz)

    print_tables(summary)
    if args.matrix:
        for _, label in METRICS:
            print_matrix(summary, label)

    json_path = args.output_json or (root / "metrics_mean_std.json")
    csv_path = args.output_csv or (root / "metrics_mean_std.csv")
    log_path = args.output_log or (root / "metrics_mean_std.log")

    payload = {
        "root": summary["root"],
        "n_runs": summary["n_runs"],
        "trajectories": summary["trajectories"],
        "configs": summary["configs"],
        "rows": summary["rows"],
        "missing_or_invalid": summary["missing_or_invalid"],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    write_csv(summary, csv_path)
    write_log(summary, log_path)

    print()
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()
