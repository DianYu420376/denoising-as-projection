#!/usr/bin/env python3
"""Aggregate hopper seed-sweep metrics from slurm .out logs.

Includes any ``*.out`` file whose name ends with ``_<seed>.out`` for
seed in [1, 15] (typical SLURM array task suffix). Groups results by run
prefix (everything before ``_<seed>.out``).

Known families (e.g. ``slurm_hopper_traj_eval`` split across array jobs)
can be merged into a single combined summary.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT_PATTERN = re.compile(r"^(?P<run_prefix>.+)_(?P<seed>[1-9]|1[0-5])\.out$")
JSON_SEED_PATTERN = re.compile(r"^seed(?P<seed>\d+)_.*\.json$")
EVAL_LINE = re.compile(
    r"\[eval\] reward=([\d.]+) ± [\d.]+ \| normalized_x100=([\d.]+)"
)
SAMPLING_CONFIG_LINE = re.compile(
    r"\[eval\] Sampling config: .*solver=(?P<solver>\S+), .*"
    r"sampling_steps=(?P<sampling_steps>\d+), .*"
    r"guidance_mode=(?P<guidance_mode>\S+), .*"
    r"optimization_guidance_scale=(?P<optimization_guidance_scale>\S+)"
)
HEADER_FIELDS = {
    "opt_scale": re.compile(r"^opt_scale:\s*(\S+)"),
    "opt_guidance_last_steps": re.compile(r"^opt_guidance_last_steps:\s*(\S+)"),
    "temperature": re.compile(r"^temperature:\s*(\S+)"),
    "sampling_steps": re.compile(r"^sampling_steps:\s*(\S+)"),
    "solver": re.compile(r"^solver:\s*(\S+)"),
    "output_dir": re.compile(r"^output_dir:\s*(.+)$"),
    "array_job_id": re.compile(r"^array_job_id:\s*(\S+)"),
    "w_cg": re.compile(r"^w_cg:\s*(\S+)"),
    "guidance_mode": re.compile(r"^guidance_mode:\s*(\S+)"),
}

# Merge multiple array jobs into one logical run (e.g. traj eval 0-4 + 5-15).
MERGE_FAMILIES: dict[str, re.Pattern[str]] = {
    "slurm_hopper_traj_eval": re.compile(r"^slurm_hopper_traj_eval_\d+$"),
}


def _mean_std(values: list[float]) -> tuple[float, float, int]:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(arr.mean()), 0.0, 1
    return float(arr.mean()), float(arr.std(ddof=1)), n


def _job_id_from_run_prefix(run_prefix: str) -> str | None:
    tail = run_prefix.rsplit("_", 1)[-1]
    return tail if tail.isdigit() else None


def _discover_seeds_from_logs(
    log_dir: Path,
    *,
    job_ids: list[str] | None,
    only_prefix_substrings: list[str] | None,
    run_prefixes: list[str] | None,
) -> set[int]:
    """Return seed indices present in .out filenames for the selected jobs."""
    seeds: set[int] = set()
    for path in log_dir.glob("*.out"):
        m = OUT_PATTERN.match(path.name)
        if not m:
            continue
        run_prefix = m.group("run_prefix")
        if only_prefix_substrings and not any(
            sub in run_prefix for sub in only_prefix_substrings
        ):
            continue
        if run_prefixes and run_prefix not in run_prefixes:
            continue
        if job_ids:
            job_id = _job_id_from_run_prefix(run_prefix)
            if job_id is None or job_id not in job_ids:
                continue
        seeds.add(int(m.group("seed")))
    return seeds


def _effective_seed_range(
    seed_min: int,
    seed_max: int,
    discovered_seeds: set[int],
) -> tuple[int, int, bool]:
    if not discovered_seeds:
        return seed_min, seed_max, False
    effective_min = min(seed_min, min(discovered_seeds))
    effective_max = max(seed_max, max(discovered_seeds))
    expanded = effective_min != seed_min or effective_max != seed_max
    return effective_min, effective_max, expanded


def _family_key(run_prefix: str, merge_families: bool) -> str | None:
    if not merge_families:
        return None
    for family, pat in MERGE_FAMILIES.items():
        if pat.match(run_prefix):
            return family
    return None


def _parse_out_file(path: Path, seed: int, run_prefix: str) -> dict | None:
    text = path.read_text(errors="replace")
    match = EVAL_LINE.search(text)
    if not match:
        return None

    meta: dict[str, str] = {"run_prefix": run_prefix}
    for line in text.splitlines()[:40]:
        for key, pat in HEADER_FIELDS.items():
            m = pat.match(line.strip())
            if m:
                meta[key] = m.group(1).strip()

    cfg = SAMPLING_CONFIG_LINE.search(text)
    if cfg:
        for key in ("solver", "sampling_steps", "guidance_mode", "optimization_guidance_scale"):
            if key not in meta and cfg.groupdict().get(key):
                meta[key] = cfg.group(key)

    job_id = meta.get("array_job_id")
    if job_id is None:
        tail = run_prefix.rsplit("_", 1)[-1]
        job_id = tail if tail.isdigit() else run_prefix

    return {
        "file": str(path),
        "run_prefix": run_prefix,
        "job_id": job_id,
        "seed": seed,
        "reward": float(match.group(1)),
        "norm_x100": float(match.group(2)),
        **meta,
    }


def _summarize_rows(
    group_key: str,
    rows: list[dict],
    seed_min: int,
    seed_max: int,
    *,
    merged: bool = False,
    source_prefixes: list[str] | None = None,
    requested_seed_range: tuple[int, int] | None = None,
) -> dict:
    rows = sorted(rows, key=lambda r: r["seed"])
    seeds_found = [r["seed"] for r in rows]
    missing = [s for s in range(seed_min, seed_max + 1) if s not in seeds_found]

    rewards = [r["reward"] for r in rows]
    norms = [r["norm_x100"] for r in rows]
    r_mean, r_std, n = _mean_std(rewards)
    n_mean, n_std, _ = _mean_std(norms)

    meta = rows[0] if rows else {}
    summary = {
        "run_prefix": group_key,
        "job_id": meta.get("job_id"),
        "merged": merged,
        "n_seeds": n,
        "seed_range": [seed_min, seed_max],
        "seeds_found": seeds_found,
        "seeds_missing": missing,
        "opt_scale": meta.get("opt_scale"),
        "opt_guidance_last_steps": meta.get("opt_guidance_last_steps"),
        "temperature": meta.get("temperature"),
        "sampling_steps": meta.get("sampling_steps"),
        "solver": meta.get("solver"),
        "guidance_mode": meta.get("guidance_mode"),
        "optimization_guidance_scale": meta.get("optimization_guidance_scale"),
        "output_dir": meta.get("output_dir"),
        "mean_reward": r_mean,
        "std_reward": r_std,
        "mean_norm_x100": n_mean,
        "std_norm_x100": n_std,
        "per_seed": rows,
    }
    if requested_seed_range is not None:
        summary["requested_seed_range"] = list(requested_seed_range)
    if merged and source_prefixes:
        summary["source_run_prefixes"] = sorted(source_prefixes)
        summary["source_job_ids"] = sorted({r["job_id"] for r in rows if r.get("job_id")})
    return summary


def aggregate_json_results(
    results_dir: Path,
    seed_min: int = 1,
    seed_max: int = 15,
    array_job_id: str | None = None,
) -> dict | None:
    """Aggregate per-seed eval JSON files under a sweep results directory."""
    if not results_dir.is_dir():
        return None

    rows: list[dict] = []
    for path in sorted(results_dir.glob("seed*.json")):
        if path.name == "seed_sweep_aggregate.json":
            continue
        m = JSON_SEED_PATTERN.match(path.name)
        if not m:
            continue
        seed = int(m.group("seed"))
        if seed < seed_min or seed > seed_max:
            continue
        with open(path) as f:
            data = json.load(f)
        reward = float(data.get("mean_reward", float("nan")))
        norm = float(data.get("mean_normalized_score_x100", float("nan")))
        rows.append(
            {
                "file": str(path),
                "seed": seed,
                "reward": reward,
                "norm_x100": norm,
                "guidance_mode": data.get("guidance_mode"),
                "w_cg": data.get("w_cg"),
                "optimization_guidance_scale": data.get("optimization_guidance_scale"),
                "opt_guidance_last_steps": data.get("optimization_guidance_last_steps"),
                "solver": data.get("solver"),
                "output_dir": str(results_dir),
                "job_id": array_job_id,
            }
        )

    if not rows:
        return None

    r_mean, r_std, n = _mean_std([r["reward"] for r in rows])
    n_mean, n_std, _ = _mean_std([r["norm_x100"] for r in rows])
    seeds_found = sorted(r["seed"] for r in rows)
    missing = [s for s in range(seed_min, seed_max + 1) if s not in seeds_found]
    meta = rows[0]

    return {
        "source": "json",
        "results_dir": str(results_dir),
        "job_id": array_job_id,
        "n_seeds": n,
        "seed_range": [seed_min, seed_max],
        "seeds_found": seeds_found,
        "seeds_missing": missing,
        "guidance_mode": meta.get("guidance_mode"),
        "w_cg": meta.get("w_cg"),
        "optimization_guidance_scale": meta.get("optimization_guidance_scale"),
        "opt_guidance_last_steps": meta.get("opt_guidance_last_steps"),
        "solver": meta.get("solver"),
        "mean_reward": r_mean,
        "std_reward": r_std,
        "mean_norm_x100": n_mean,
        "std_norm_x100": n_std,
        "per_seed": sorted(rows, key=lambda r: r["seed"]),
    }


def aggregate_out_files(
    repo_dir: Path,
    seed_min: int = 1,
    seed_max: int = 15,
    run_prefixes: list[str] | None = None,
    job_ids: list[str] | None = None,
    merge_families: bool = True,
    only_prefix_substrings: list[str] | None = None,
    log_dir: Path | None = None,
    results_dir: Path | None = None,
) -> dict:
    requested_seed_range = (seed_min, seed_max)
    search_dir = log_dir if log_dir is not None else repo_dir
    array_job_id = job_ids[0] if job_ids and len(job_ids) == 1 else None

    json_summary = None
    if results_dir is not None:
        json_summary = aggregate_json_results(
            results_dir, seed_min, seed_max, array_job_id=array_job_id
        )

    discovered_seeds = _discover_seeds_from_logs(
        search_dir,
        job_ids=job_ids,
        only_prefix_substrings=only_prefix_substrings,
        run_prefixes=run_prefixes,
    )
    seed_min, seed_max, range_expanded = _effective_seed_range(
        seed_min, seed_max, discovered_seeds
    )

    by_run: dict[str, list[dict]] = defaultdict(list)

    for path in sorted(search_dir.glob("*.out")):
        m = OUT_PATTERN.match(path.name)
        if not m:
            continue
        seed = int(m.group("seed"))
        if seed < seed_min or seed > seed_max:
            continue
        run_prefix = m.group("run_prefix")
        if only_prefix_substrings and not any(
            sub in run_prefix for sub in only_prefix_substrings
        ):
            continue
        if run_prefixes and run_prefix not in run_prefixes:
            continue

        row = _parse_out_file(path, seed, run_prefix)
        if row is None:
            continue
        if job_ids and row.get("job_id") not in job_ids:
            continue
        by_run[run_prefix].append(row)

    per_job_summaries = []
    for run_prefix in sorted(by_run):
        per_job_summaries.append(
            _summarize_rows(
                run_prefix,
                by_run[run_prefix],
                seed_min,
                seed_max,
                requested_seed_range=requested_seed_range,
            )
        )

    merged_summaries = []
    if merge_families:
        by_family: dict[str, list[dict]] = defaultdict(list)
        family_sources: dict[str, set[str]] = defaultdict(set)
        for run_prefix, rows in by_run.items():
            family = _family_key(run_prefix, merge_families=True)
            if family is None:
                continue
            by_family[family].extend(rows)
            family_sources[family].add(run_prefix)

        for family in sorted(by_family):
            # One row per seed; later file wins if duplicate seed across jobs.
            dedup: dict[int, dict] = {}
            for row in sorted(by_family[family], key=lambda r: (r["seed"], r["run_prefix"])):
                dedup[row["seed"]] = row
            merged_rows = list(dedup.values())
            merged_summaries.append(
                _summarize_rows(
                    f"{family} (combined)",
                    merged_rows,
                    seed_min,
                    seed_max,
                    merged=True,
                    source_prefixes=list(family_sources[family]),
                    requested_seed_range=requested_seed_range,
                )
            )

    report = {
        "seed_range": [seed_min, seed_max],
        "requested_seed_range": list(requested_seed_range),
        "seed_range_expanded_from_logs": range_expanded,
        "discovered_seeds_in_logs": sorted(discovered_seeds),
        "log_dir": str(search_dir),
        "results_dir": str(results_dir) if results_dir else None,
        "pattern": "*_<seed>.out with seed in [1, 15]",
        "json_summary": json_summary,
        "job_summaries": per_job_summaries,
        "merged_family_summaries": merged_summaries,
    }

    # Prefer JSON aggregation when .out discovery failed but JSON exists.
    if json_summary and not per_job_summaries and not merged_summaries:
        report["primary_summary"] = json_summary
    elif json_summary and per_job_summaries:
        report["primary_summary"] = per_job_summaries[0]
    elif json_summary:
        report["primary_summary"] = json_summary
    elif per_job_summaries:
        report["primary_summary"] = per_job_summaries[0]
    else:
        report["primary_summary"] = None

    return report


def _print_summary_item(item: dict, seed_max: int, seed_min: int) -> None:
    print()
    if item.get("run_prefix"):
        label = item["run_prefix"]
    elif item.get("source") == "json":
        label = f"json:{item.get('results_dir', 'results')}"
    else:
        label = item.get("job_id") or "summary"
    if item.get("merged"):
        print(f"{label}  (n={item['n_seeds']}/{seed_max - seed_min + 1})")
        if item.get("source_run_prefixes"):
            print(f"  sources: {', '.join(item['source_run_prefixes'])}")
    else:
        print(f"{label}  (n={item['n_seeds']}/{seed_max - seed_min + 1})")

    settings = []
    if item.get("guidance_mode"):
        settings.append(f"mode={item['guidance_mode']}")
    if item.get("w_cg") is not None:
        settings.append(f"w_cg={item['w_cg']}")
    opt_scale = item.get("opt_scale")
    if opt_scale is None:
        opt_scale = item.get("optimization_guidance_scale")
    if opt_scale is not None:
        settings.append(f"scale={opt_scale}")
    if item.get("solver"):
        settings.append(f"solver={item['solver']}")
    if item.get("sampling_steps"):
        settings.append(f"steps={item['sampling_steps']}")
    if item.get("temperature"):
        settings.append(f"temp={item['temperature']}")
    opt_last = item.get("opt_guidance_last_steps")
    if opt_last is None:
        opt_last = item.get("optimization_guidance_last_steps")
    if opt_last is not None:
        settings.append(f"opt_last={opt_last}")
    if settings:
        print(f"  settings: {' '.join(settings)}")
    if item.get("output_dir"):
        print(f"  output_dir: {item['output_dir']}")
    if item.get("results_dir"):
        print(f"  results_dir: {item['results_dir']}")
    if item["seeds_missing"]:
        print(f"  missing seeds: {item['seeds_missing']}")
    print(f"  reward:      {item['mean_reward']:8.2f} ± {item['std_reward']:7.2f}")
    print(f"  norm_x100:   {item['mean_norm_x100']:8.2f} ± {item['std_norm_x100']:7.2f}")
    print("  per-seed:")
    for row in sorted(item["per_seed"], key=lambda r: r["seed"]):
        src = ""
        if item.get("merged") and row.get("run_prefix"):
            src = f"  [{row['run_prefix']}]"
        print(
            f"    seed {row['seed']:2d}: "
            f"reward={row['reward']:8.2f}  norm_x100={row['norm_x100']:7.2f}{src}"
        )


def _print_report(report: dict) -> None:
    seed_min, seed_max = report["seed_range"]
    print(f"=== Seed aggregate (seeds {seed_min}-{seed_max}) ===")
    if report.get("log_dir"):
        print(f"Log dir: {report['log_dir']}")
    if report.get("results_dir"):
        print(f"Results dir: {report['results_dir']}")
    if report.get("seed_range_expanded_from_logs"):
        req_min, req_max = report["requested_seed_range"]
        print(
            f"Note: expanded seed range from logs "
            f"(requested {req_min}-{req_max}, "
            f"discovered {report.get('discovered_seeds_in_logs', [])})"
        )
    print(f"Pattern: {report['pattern']}")

    primary = report.get("primary_summary")
    if primary:
        print("\n--- Primary summary ---")
        _print_summary_item(primary, seed_max, seed_min)

    if not report["job_summaries"] and not report.get("merged_family_summaries") and not primary:
        print("No completed runs found.")
        return

    if report.get("json_summary") and report.get("json_summary") is not primary:
        print("\n--- JSON results summary ---")
        _print_summary_item(report["json_summary"], seed_max, seed_min)

    if report.get("merged_family_summaries"):
        print("\n--- Merged families ---")
        for item in report["merged_family_summaries"]:
            _print_summary_item(item, seed_max, seed_min)

    if report["job_summaries"]:
        print("\n--- Per array job ---")
        for item in report["job_summaries"]:
            _print_summary_item(item, seed_max, seed_min)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Directory containing slurm *.out files.",
    )
    parser.add_argument("--seed-min", type=int, default=1)
    parser.add_argument("--seed-max", type=int, default=15)
    parser.add_argument(
        "--run-prefixes",
        default=None,
        help="Comma-separated run prefixes to include (default: all).",
    )
    parser.add_argument(
        "--only-prefix-substrings",
        default=None,
        help="Comma-separated substrings; only matching run prefixes are included.",
    )
    parser.add_argument(
        "--job-ids",
        default=None,
        help="Comma-separated SLURM array job IDs to include (default: all).",
    )
    parser.add_argument(
        "--no-merge-families",
        action="store_true",
        help="Do not merge known split array families (e.g. traj eval).",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory containing slurm *.out files (default: --repo-dir).",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory with per-seed eval JSON files (seed<N>_*.json).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write JSON summary.",
    )
    args = parser.parse_args()

    run_prefixes = None
    if args.run_prefixes:
        run_prefixes = [x.strip() for x in args.run_prefixes.split(",") if x.strip()]
    only_substrings = None
    if args.only_prefix_substrings:
        only_substrings = [
            x.strip() for x in args.only_prefix_substrings.split(",") if x.strip()
        ]
    job_ids = None
    if args.job_ids:
        job_ids = [x.strip() for x in args.job_ids.split(",") if x.strip()]

    log_dir = Path(args.log_dir) if args.log_dir else None
    results_dir = Path(args.results_dir) if args.results_dir else None

    report = aggregate_out_files(
        Path(args.repo_dir),
        seed_min=args.seed_min,
        seed_max=args.seed_max,
        run_prefixes=run_prefixes,
        job_ids=job_ids,
        merge_families=not args.no_merge_families,
        only_prefix_substrings=only_substrings,
        log_dir=log_dir,
        results_dir=results_dir,
    )
    _print_report(report)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
