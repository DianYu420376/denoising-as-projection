"""Scrape guidance-comparison results from Slurm logs or per-run JSON files."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PER_SEED_HEADER = re.compile(
    r"--- Per-seed summary \(normalized x100\) seed=(\d+) ---"
)
PER_SEED_ROW = re.compile(
    r"^\s+(?P<config>\S+)\s+norm=\s*(?P<norm>[-\d.]+)\s+"
    r"raw=\s*(?P<raw>[-\d.]+)\s+steps=\s*(?P<steps>[-\d.]+)\s*$"
)

CONFIG_DISPLAY_ORDER = [
    "monte_carlo_w_cg0",
    "standard_w_cg0p3",
    "optimization_scale_0p1",
    "optimization_scale_0p25",
    "optimization_scale_0p75",
    "optimization_scale_1p0",
]

EXPECTED_CONFIGS = list(CONFIG_DISPLAY_ORDER)


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def scrape_slurm_output(path: Path) -> dict:
    lines = path.read_text().splitlines()
    records: list[dict] = []
    current_seed: int | None = None

    for line in lines:
        header = PER_SEED_HEADER.match(line)
        if header:
            current_seed = int(header.group(1))
            continue

        if current_seed is None:
            continue

        if line.startswith("--- Cumulative mean"):
            current_seed = None
            continue

        match = PER_SEED_ROW.match(line)
        if match:
            records.append(
                {
                    "seed": current_seed,
                    "config": match.group("config"),
                    "normalized_score_x100": float(match.group("norm")),
                    "raw_reward": float(match.group("raw")),
                    "survival_steps": float(match.group("steps")),
                }
            )

    return _summarize_records(records, source=str(path))


def scrape_slurm_outputs(paths: list[Path]) -> dict:
    records: list[dict] = []
    seen: set[tuple[int, str]] = set()
    for path in paths:
        summary = scrape_slurm_output(path)
        for cfg_stats in summary["configs"].values():
            for row in cfg_stats["per_seed"]:
                key = (row["seed"], row["config"])
                if key in seen:
                    continue
                seen.add(key)
                records.append(row)
    source = ", ".join(str(path) for path in paths)
    return _summarize_records(records, source=source)


def _load_json_records(path: Path) -> list[dict]:
    records: list[dict] = []
    for json_path in sorted(path.glob("seed_*_*.json")):
        stem = json_path.stem  # seed_000_standard_w_cg0p3
        parts = stem.split("_", 2)
        if len(parts) < 3 or parts[0] != "seed":
            continue
        seed = int(parts[1])
        config = parts[2]
        with open(json_path) as f:
            payload = json.load(f)
        records.append(
            {
                "seed": seed,
                "config": config,
                "normalized_score_x100": float(payload["mean_normalized_score_x100"]),
                "raw_reward": float(payload["mean_raw_reward"]),
                "survival_steps": float(payload["mean_survival_steps"]),
            }
        )
    return records


def scrape_json_dir(path: Path) -> dict:
    return _summarize_records(_load_json_records(path), source=str(path))


def scrape_json_dirs(paths: list[Path]) -> dict:
    records: list[dict] = []
    seen: set[tuple[int, str]] = set()
    for path in paths:
        for record in _load_json_records(path):
            key = (record["seed"], record["config"])
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
    source = ", ".join(str(path) for path in paths)
    return _summarize_records(records, source=source)


def discover_per_run_dirs(task_dir: Path) -> dict[str, Path]:
    """Find latest per_run dirs for main comparison and Monte Carlo runs."""
    candidates: list[tuple[float, Path, set[str]]] = []
    for per_run in task_dir.glob("guidance_comparison*/**/per_run"):
        if not per_run.is_dir():
            continue
        configs = {p.stem.split("_", 2)[-1] for p in per_run.glob("seed_*_*.json")}
        if not configs:
            continue
        candidates.append((per_run.stat().st_mtime, per_run, configs))

    discovered: dict[str, Path] = {}
    if not candidates:
        return discovered

    candidates.sort(key=lambda item: item[0], reverse=True)

    for _, per_run, configs in candidates:
        if "main" not in discovered and "standard_w_cg0p3" in configs:
            discovered["main"] = per_run
        if "monte_carlo" not in discovered and "monte_carlo_w_cg0" in configs:
            discovered["monte_carlo"] = per_run
        if len(discovered) == 2:
            break

    return discovered


def _ordered_configs(config_names: set[str]) -> list[str]:
    ordered = [c for c in CONFIG_DISPLAY_ORDER if c in config_names]
    ordered += sorted(c for c in config_names if c not in CONFIG_DISPLAY_ORDER)
    return ordered


def _paired_seeds(records: list[dict], config_names: list[str]) -> list[int]:
    by_seed: dict[int, set[str]] = defaultdict(set)
    for record in records:
        by_seed[record["seed"]].add(record["config"])
    wanted = set(config_names)
    return sorted(seed for seed, present in by_seed.items() if wanted.issubset(present))


def _summarize_records(records: list[dict], source: str) -> dict:
    by_config: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_config[record["config"]].append(record)

    summary = {
        "source": source,
        "n_records": len(records),
        "seeds_completed": sorted({r["seed"] for r in records}),
        "configs": {},
        "coverage": {},
        "paired_seeds_all_expected": [],
        "paired_summary_all_expected": {},
    }

    for config in _ordered_configs(set(by_config)):
        rows = sorted(by_config[config], key=lambda r: r["seed"])
        norm_vals = [r["normalized_score_x100"] for r in rows]
        raw_vals = [r["raw_reward"] for r in rows]
        step_vals = [r["survival_steps"] for r in rows]

        mean_norm, std_norm = _mean_std(norm_vals)
        mean_raw, std_raw = _mean_std(raw_vals)
        mean_steps, std_steps = _mean_std(step_vals)

        summary["configs"][config] = {
            "n_seeds": len(rows),
            "seeds": [r["seed"] for r in rows],
            "mean_normalized_score_x100": mean_norm,
            "std_normalized_score_x100": std_norm,
            "mean_raw_reward": mean_raw,
            "std_raw_reward": std_raw,
            "mean_survival_steps": mean_steps,
            "std_survival_steps": std_steps,
            "per_seed": rows,
        }

    present_configs = set(summary["configs"])
    for config in EXPECTED_CONFIGS:
        n = summary["configs"].get(config, {}).get("n_seeds", 0)
        summary["coverage"][config] = n

    paired = _paired_seeds(records, EXPECTED_CONFIGS)
    summary["paired_seeds_all_expected"] = paired
    if paired:
        paired_records = [r for r in records if r["seed"] in paired]
        paired_by_config: dict[str, list[dict]] = defaultdict(list)
        for record in paired_records:
            paired_by_config[record["config"]].append(record)
        for config in _ordered_configs(set(paired_by_config)):
            rows = paired_by_config[config]
            mean_norm, std_norm = _mean_std([r["normalized_score_x100"] for r in rows])
            mean_raw, std_raw = _mean_std([r["raw_reward"] for r in rows])
            summary["paired_summary_all_expected"][config] = {
                "n_seeds": len(rows),
                "mean_normalized_score_x100": mean_norm,
                "std_normalized_score_x100": std_norm,
                "mean_raw_reward": mean_raw,
                "std_raw_reward": std_raw,
            }

    return summary


def _print_coverage(summary: dict) -> None:
    print("Config coverage (number of seeds with data):")
    for config in EXPECTED_CONFIGS:
        count = summary["coverage"].get(config, 0)
        marker = "OK" if count > 0 else "MISSING"
        print(f"  {config:28s} {count:4d} seeds  [{marker}]")
    paired = summary["paired_seeds_all_expected"]
    if paired:
        print(
            f"Paired seeds with all {len(EXPECTED_CONFIGS)} configs: "
            f"{paired[0]}..{paired[-1]} ({len(paired)} seeds)"
        )
    else:
        print(f"Paired seeds with all {len(EXPECTED_CONFIGS)} configs: none yet")
    print()


def _print_summary(summary: dict) -> None:
    seeds = summary["seeds_completed"]
    if seeds:
        print(f"Seeds with data: {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds)")
    print(f"Source: {summary['source']}")
    print(f"Total per-seed records: {summary['n_records']}")
    print()
    _print_coverage(summary)
    print(f"{'Config':<28} {'Norm x100':>16} {'Raw reward':>22} {'Survival steps':>20}")
    print("-" * 90)

    for config, stats in summary["configs"].items():
        print(
            f"{config:<28} "
            f"{stats['mean_normalized_score_x100']:7.2f} ± {stats['std_normalized_score_x100']:5.2f}   "
            f"{stats['mean_raw_reward']:8.1f} ± {stats['std_raw_reward']:6.1f}   "
            f"{stats['mean_survival_steps']:7.1f} ± {stats['std_survival_steps']:5.1f}"
        )

    paired_summary = summary.get("paired_summary_all_expected", {})
    if paired_summary:
        print()
        print("Paired comparison (only seeds with all expected configs):")
        print(f"{'Config':<28} {'Norm x100':>16} {'Raw reward':>22}")
        print("-" * 70)
        for config in _ordered_configs(set(paired_summary)):
            stats = paired_summary[config]
            print(
                f"{config:<28} "
                f"{stats['mean_normalized_score_x100']:7.2f} ± {stats['std_normalized_score_x100']:5.2f}   "
                f"{stats['mean_raw_reward']:8.1f} ± {stats['std_raw_reward']:6.1f}"
            )

    print()
    print("Per-seed raw rewards:")
    configs = _ordered_configs(set(summary["configs"]))
    header = f"{'seed':>6}  " + "  ".join(f"{c:>24}" for c in configs)
    print(header)
    print("-" * len(header))

    lookup: dict[int, dict[str, dict]] = defaultdict(dict)
    for cfg, stats in summary["configs"].items():
        for row in stats["per_seed"]:
            lookup[row["seed"]][cfg] = row

    for seed in summary["seeds_completed"]:
        cells = [f"{seed:6d}"]
        for cfg in configs:
            row = lookup[seed].get(cfg)
            if row is None:
                cells.append(f"{'—':>24}")
            else:
                cells.append(f"{row['raw_reward']:24.1f}")
        print("  ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate guidance-comparison rewards from Slurm logs or JSON."
    )
    parser.add_argument("--repo-dir", type=Path, default=Path(str(Path(__file__).resolve().parents[1])))
    parser.add_argument("--task", default="hopper-medium-v2")
    parser.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Auto-discover and merge latest main guidance_comparison per_run dir "
            "with latest monte_carlo_w_cg0 per_run dir."
        ),
    )
    parser.add_argument(
        "--slurm-out",
        type=Path,
        action="append",
        help="Path to slurm_*.out log (repeatable; merges main + monte carlo logs).",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        action="append",
        help="Path to per_run/ directory with seed_*_*.json files (repeatable).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write aggregated summary JSON.",
    )
    args = parser.parse_args()

    json_dirs = list(args.json_dir or [])
    slurm_outs = list(args.slurm_out or [])

    if args.auto:
        task_dir = args.repo_dir / "results" / "diffuser_d4rl_mujoco" / args.task
        discovered = discover_per_run_dirs(task_dir)
        if "main" in discovered:
            print(f"[auto] main comparison: {discovered['main']}")
            if discovered["main"] not in json_dirs:
                json_dirs.append(discovered["main"])
        else:
            print(f"[auto] WARNING: no main comparison per_run found under {task_dir}")
        if "monte_carlo" in discovered:
            print(f"[auto] monte carlo: {discovered['monte_carlo']}")
            if discovered["monte_carlo"] not in json_dirs:
                json_dirs.append(discovered["monte_carlo"])
        else:
            print(
                "[auto] WARNING: no monte_carlo_w_cg0 results yet "
                f"(expected under {task_dir}/guidance_comparison_monte_carlo/...)"
            )
        print()

    if not slurm_outs and not json_dirs:
        parser.error("Provide --auto, --slurm-out, and/or --json-dir.")

    if slurm_outs and json_dirs:
        if len(slurm_outs) > 1:
            slurm_summary = scrape_slurm_outputs(slurm_outs)
        else:
            slurm_summary = scrape_slurm_output(slurm_outs[0])
        print("=== From Slurm log(s) ===")
        _print_summary(slurm_summary)
        print()

    if len(json_dirs) > 1:
        summary = scrape_json_dirs(json_dirs)
        source_label = "merged json-dirs"
    elif len(json_dirs) == 1:
        summary = scrape_json_dir(json_dirs[0])
        source_label = "json-dir"
    else:
        if len(slurm_outs) > 1:
            summary = scrape_slurm_outputs(slurm_outs)
        else:
            summary = scrape_slurm_output(slurm_outs[0])
        source_label = "slurm-out"

    if json_dirs or not slurm_outs:
        print(f"=== Aggregated results ({source_label}) ===")
        _print_summary(summary)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print()
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
