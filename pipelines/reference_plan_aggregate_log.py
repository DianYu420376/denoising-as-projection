#!/usr/bin/env python3
"""Aggregate reference_plan candidate logs into a single analysis summary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _load_block(summary_path: Path) -> dict | None:
    with open(summary_path) as f:
        payload = json.load(f)
    for key in ("reference_plan", "heart_plan"):
        if key in payload:
            return payload[key]
    return None


def aggregate_root(root: Path) -> dict:
    runs: list[dict] = []
    for summary_path in sorted(root.glob("*/*/summary.json")):
        trajectory = summary_path.parent.parent.name
        config = summary_path.parent.name
        block = _load_block(summary_path)
        if block is None:
            continue

        best = block.get("best_candidate", {})
        stats = block.get("candidate_metrics_summary", {})
        runs.append(
            {
                "trajectory": trajectory,
                "config": config,
                "summary_path": str(summary_path),
                "trajectory_dir": block.get("trajectory_dir"),
                "plot_dir": block.get("plot_dir"),
                "plan_candidates": block.get("plan_candidates"),
                "best_candidate_idx": best.get("candidate_idx"),
                "best_seed": best.get("seed"),
                "best_plan_reward": best.get("plan_reward"),
                "best_rollout_reward": best.get("rollout_reward"),
                "best_feasibility_l2": best.get("feasibility_mean_l2_future"),
                "plan_reward_mean": stats.get("plan_reward", {}).get("mean"),
                "plan_reward_std": stats.get("plan_reward", {}).get("std"),
                "rollout_reward_mean": stats.get("rollout_reward", {}).get("mean"),
                "rollout_reward_std": stats.get("rollout_reward", {}).get("std"),
                "feasibility_l2_mean": stats.get("feasibility_mean_l2_future", {}).get("mean"),
                "feasibility_l2_std": stats.get("feasibility_mean_l2_future", {}).get("std"),
                "top_plots": block.get("top_candidates_by_plan_reward", []),
            }
        )

    trajectories = sorted({row["trajectory"] for row in runs})
    configs = sorted({row["config"] for row in runs})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "n_runs": len(runs),
        "trajectories": trajectories,
        "configs": configs,
        "runs": runs,
    }


def write_analysis_log(summary: dict, log_path: Path) -> None:
    lines = [
        "Reference plan evaluation — aggregate candidate index log",
        f"root: {summary['root']}",
        f"generated_at: {summary['generated_at']}",
        f"runs: {summary['n_runs']}",
        "",
        (
            f"{'trajectory':<18} {'config':<24} {'best_idx':>8} {'plan_r':>12} "
            f"{'rollout_r':>12} {'feas_l2':>10} {'plan_mean':>12} {'feas_mean':>10}"
        ),
        "-" * 110,
    ]
    for row in summary["runs"]:
        lines.append(
            f"{row['trajectory']:<18} {row['config']:<24} "
            f"{row.get('best_candidate_idx', '—')!s:>8} "
            f"{row.get('best_plan_reward', float('nan')):>12.4f} "
            f"{row.get('best_rollout_reward', float('nan')):>12.4f} "
            f"{row.get('best_feasibility_l2', float('nan')):>10.4f} "
            f"{row.get('plan_reward_mean', float('nan')):>12.4f} "
            f"{row.get('feasibility_l2_mean', float('nan')):>10.4f}"
        )

    lines.extend(["", "Plotted top candidates (by plan reward):", ""])
    for row in summary["runs"]:
        lines.append(f"=== {row['trajectory']} / {row['config']} ===")
        for plot in row.get("top_plots", []):
            lines.append(
                f"  rank={plot.get('rank')} candidate_idx={plot.get('candidate_idx')} "
                f"plan_reward={plot.get('plan_reward'):.6f} "
                f"rollout_reward={plot.get('rollout_reward'):.6f} "
                f"feasibility_l2={plot.get('feasibility_mean_l2_future'):.6f} "
                f"plot={plot.get('plot_path')}"
            )
        lines.append("")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate reference_plan candidate logs.")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Comparison output root ({trajectory}/{config}/summary.json).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON summary path (default: {root}/comparison_summary.json).",
    )
    parser.add_argument(
        "--output-log",
        type=Path,
        default=None,
        help="Optional text log path (default: {root}/analysis.log).",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    summary = aggregate_root(root)
    json_path = args.output_json or (root / "comparison_summary.json")
    log_path = args.output_log or (root / "analysis.log")

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    write_analysis_log(summary, log_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {log_path}")
    print(f"Aggregated {summary['n_runs']} runs")


if __name__ == "__main__":
    main()
