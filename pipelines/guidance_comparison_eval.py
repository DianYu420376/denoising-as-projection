"""Run standard vs optimization-guidance inference with per-seed reporting."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

MONTE_CARLO_CONFIG = {
    "name": "monte_carlo_w_cg0",
    "guidance_mode": "standard",
    "optimization_guidance_scale": 0.0,
    "w_cg": 0.0,
}

STANDARD_CONFIG = {
    "name": "standard_w_cg0p3",
    "guidance_mode": "standard",
    "optimization_guidance_scale": 0.0,
    "w_cg": 0.3,
}


def _opt_scale_label(scale: float) -> str:
    """Canonical config suffix: 0.5 -> 0p5, 2.0 -> 2, 10.0 -> 10."""
    scale_f = float(scale)
    if scale_f.is_integer():
        return str(int(scale_f))
    return str(scale_f).replace(".", "p")


def _standard_w_cg_config(name: str) -> dict | None:
    """Parse names like standard_w_cg0p3, standard_w_cg0p1, standard_w_cg1."""
    prefix = "standard_w_cg"
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix):]
    if suffix in ("", "0"):
        w_cg = 0.0
    else:
        w_cg = float(suffix.replace("p", "."))
    return {
        "name": name,
        "guidance_mode": "standard",
        "optimization_guidance_scale": 0.0,
        "w_cg": w_cg,
    }


def _optimization_config(name: str) -> dict | None:
    """Parse names like optimization_scale_0p5, optimization_scale_2."""
    prefix = "optimization_scale_"
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix):]
    scale = float(suffix.replace("p", "."))
    return {
        "name": name,
        "guidance_mode": "optimization",
        "optimization_guidance_scale": scale,
        "w_cg": 0.0,
    }


def build_configs(
    opt_scales: list[float],
    include_monte_carlo: bool = False,
    config_names: list[str] | None = None,
) -> list[dict]:
    configs: list[dict] = []
    if include_monte_carlo:
        configs.append(MONTE_CARLO_CONFIG)
    configs.append(STANDARD_CONFIG)
    for scale in opt_scales:
        label = _opt_scale_label(scale)
        configs.append(
            {
                "name": f"optimization_scale_{label}",
                "guidance_mode": "optimization",
                "optimization_guidance_scale": float(scale),
                "w_cg": 0.0,
            }
        )

    if config_names:
        allowed = set(config_names)
        extra_configs = []
        for name in config_names:
            if name in {cfg["name"] for cfg in configs}:
                continue
            parsed = _standard_w_cg_config(name) or _optimization_config(name)
            if parsed is not None:
                extra_configs.append(parsed)
        configs.extend(extra_configs)
        configs = [cfg for cfg in configs if cfg["name"] in allowed]
        missing = allowed - {cfg["name"] for cfg in configs}
        if missing:
            known = {
                MONTE_CARLO_CONFIG["name"],
                STANDARD_CONFIG["name"],
                *[f"optimization_scale_{_opt_scale_label(s)}" for s in opt_scales],
                *[
                    f"standard_w_cg{_opt_scale_label(w)}"
                    for w in (0.0, 0.1, 0.3, 1.0)
                ],
            }
            raise ValueError(f"Unknown config names: {sorted(missing)}. Known: {sorted(known)}")

    return configs


def _log(msg: str) -> None:
    print(msg, flush=True)


def _run_single_inference(
    repo_dir: Path,
    python_bin: Path,
    task: str,
    ckpt: str,
    seed: int,
    config: dict,
    eval_output: Path,
    num_episodes: int,
    num_candidates: int,
    run_suffix: str | None,
) -> dict:
    cmd = [
        str(python_bin),
        str(repo_dir / "pipelines" / "diffuser_d4rl_mujoco.py"),
        f"task={task}",
        "mode=inference",
        "device=cuda:0",
        f"seed={seed}",
        f"ckpt={ckpt}",
        f"num_episodes={num_episodes}",
        f"num_candidates={num_candidates}",
        "num_envs=1",
        "render_video=false",
        "enable_wandb=false",
        f"guidance_mode={config['guidance_mode']}",
        f"optimization_guidance_scale={config['optimization_guidance_scale']}",
        f"task.w_cg={config['w_cg']}",
        f"sim_env_name={task}",
        f"eval_output={eval_output}",
    ]
    if run_suffix:
        cmd.append(f"run_suffix={run_suffix}")

    _log(f"[run] seed={seed} config={config['name']}")
    _log(f"      eval_output={eval_output}")
    proc = subprocess.run(cmd, cwd=str(repo_dir), check=False, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Inference failed for seed={seed} config={config['name']} "
            f"(exit={proc.returncode})"
        )

    if not eval_output.exists():
        raise FileNotFoundError(f"Missing eval output: {eval_output}")

    with open(eval_output) as f:
        return json.load(f)


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _format_row(name: str, mean: float, std: float, metric: str = "norm_x100") -> str:
    if metric == "norm_x100":
        return f"{name:28s} {mean:8.2f} ± {std:5.2f}"
    return f"{name:28s} {mean:8.4f} ± {std:5.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--python", default="python")
    parser.add_argument("--task", default="hopper-medium-v2")
    parser.add_argument("--ckpt", default="latest")
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--opt-scales",
        default="0.1,0.25,0.75,1.0",
        help="Comma-separated optimization_guidance_scale values.",
    )
    parser.add_argument(
        "--include-monte-carlo",
        action="store_true",
        help="Include monte_carlo_w_cg0 (standard guidance, w_cg=0).",
    )
    parser.add_argument(
        "--configs",
        default="",
        help="Comma-separated config names to run (default: all built configs).",
    )
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    python_bin = Path(args.python)
    run_suffix = args.run_suffix or None

    opt_scales = [float(x.strip()) for x in args.opt_scales.split(",") if x.strip()]
    config_names = [x.strip() for x in args.configs.split(",") if x.strip()] or None
    include_monte_carlo = args.include_monte_carlo or (
        config_names is not None and MONTE_CARLO_CONFIG["name"] in config_names
    )
    configs = build_configs(
        opt_scales,
        include_monte_carlo=include_monte_carlo,
        config_names=config_names,
    )

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_suffix:
        base = repo_dir / "results" / "diffuser_d4rl_mujoco" / args.task / run_suffix
    else:
        base = repo_dir / "results" / "diffuser_d4rl_mujoco" / args.task
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif config_names == [MONTE_CARLO_CONFIG["name"]]:
        output_dir = base / "guidance_comparison_monte_carlo" / run_tag
    else:
        output_dir = base / "guidance_comparison" / run_tag
    per_run_dir = output_dir / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    _log("============================================================")
    _log("Guidance comparison eval")
    _log("============================================================")
    _log(f"task={args.task} ckpt={args.ckpt}")
    _log(f"seeds={args.seed_start}..{args.seed_end} num_episodes={args.num_episodes}")
    _log(f"configs={[c['name'] for c in configs]}")
    _log(f"output_dir={output_dir}")

    cumulative_records: list[dict] = []
    per_seed_results: list[dict] = []
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            prior = json.load(f)
        cumulative_records = list(prior.get("cumulative_records", []))
        per_seed_results = list(prior.get("per_seed_results", []))
        _log(f"Resuming from {summary_path} ({len(cumulative_records)} prior runs)")

    for seed in range(args.seed_start, args.seed_end + 1):
        seed_entry = {"seed": seed, "configs": {}}
        _log("")
        _log(f"========== Seed {seed} ==========")

        for config in configs:
            eval_output = per_run_dir / f"seed_{seed:03d}_{config['name']}.json"
            if eval_output.exists():
                _log(f"[skip] seed={seed} config={config['name']} (exists)")
                with open(eval_output) as f:
                    result = json.load(f)
            else:
                result = _run_single_inference(
                    repo_dir=repo_dir,
                    python_bin=python_bin,
                    task=args.task,
                    ckpt=args.ckpt,
                    seed=seed,
                    config=config,
                    eval_output=eval_output,
                    num_episodes=args.num_episodes,
                    num_candidates=args.num_candidates,
                    run_suffix=run_suffix,
                )
            seed_entry["configs"][config["name"]] = {
                "mean_normalized_score_x100": result["mean_normalized_score_x100"],
                "std_normalized_score_x100": result["std_normalized_score_x100"],
                "mean_raw_reward": result["mean_raw_reward"],
                "mean_survival_steps": result["mean_survival_steps"],
                "eval_output": str(eval_output),
            }
            record = {
                "seed": seed,
                "config": config["name"],
                "normalized_score_x100": result["mean_normalized_score_x100"],
                "raw_reward": result["mean_raw_reward"],
                "survival_steps": result["mean_survival_steps"],
            }
            if not any(
                r["seed"] == seed and r["config"] == config["name"] for r in cumulative_records
            ):
                cumulative_records.append(record)

        existing_idx = next(
            (i for i, e in enumerate(per_seed_results) if e["seed"] == seed), None
        )
        if existing_idx is None:
            per_seed_results.append(seed_entry)
        else:
            per_seed_results[existing_idx] = seed_entry

        _log(f"--- Per-seed summary (normalized x100) seed={seed} ---")
        for config in configs:
            cfg = seed_entry["configs"][config["name"]]
            _log(
                f"  {config['name']:28s} "
                f"norm={cfg['mean_normalized_score_x100']:6.2f} "
                f"raw={cfg['mean_raw_reward']:8.1f} "
                f"steps={cfg['mean_survival_steps']:6.1f}"
            )

        _log(f"--- Cumulative mean up to seed {seed} (normalized x100) ---")
        cumulative_summary = {}
        for config in configs:
            values = [
                r["normalized_score_x100"]
                for r in cumulative_records
                if r["config"] == config["name"]
            ]
            mean, std = _mean_std(values)
            cumulative_summary[config["name"]] = {
                "mean_normalized_score_x100": mean,
                "std_normalized_score_x100": std,
                "n": len(values),
            }
            _log(_format_row(config["name"], mean, std))

        progress = {
            "completed_seeds": seed - args.seed_start + 1,
            "last_seed": seed,
            "per_seed_latest": seed_entry,
            "cumulative_summary": cumulative_summary,
        }
        with open(output_dir / "progress.json", "w") as f:
            json.dump(progress, f, indent=2)

        partial_payload = {
            "task": args.task,
            "ckpt": args.ckpt,
            "seed_start": args.seed_start,
            "seed_end": args.seed_end,
            "num_episodes": args.num_episodes,
            "num_candidates": args.num_candidates,
            "configs": configs,
            "per_seed_results": per_seed_results,
            "cumulative_records": cumulative_records,
            "output_dir": str(output_dir),
        }
        with open(summary_path, "w") as f:
            json.dump(partial_payload, f, indent=2)

    final_summary = {"configs": {}}
    _log("")
    _log("========== Final summary (all seeds) ==========")
    for config in configs:
        values = [
            r["normalized_score_x100"]
            for r in cumulative_records
            if r["config"] == config["name"]
        ]
        raw_values = [
            r["raw_reward"] for r in cumulative_records if r["config"] == config["name"]
        ]
        step_values = [
            r["survival_steps"] for r in cumulative_records if r["config"] == config["name"]
        ]
        mean_n, std_n = _mean_std(values)
        mean_r, std_r = _mean_std(raw_values)
        mean_s, std_s = _mean_std(step_values)
        final_summary["configs"][config["name"]] = {
            "mean_normalized_score_x100": mean_n,
            "std_normalized_score_x100": std_n,
            "mean_raw_reward": mean_r,
            "std_raw_reward": std_r,
            "mean_survival_steps": mean_s,
            "std_survival_steps": std_s,
            "n_seeds": len(values),
        }
        _log(_format_row(config["name"], mean_n, std_n))
        _log(
            f"{'':28s} raw={mean_r:8.1f} ± {std_r:5.1f}  "
            f"steps={mean_s:6.1f} ± {std_s:5.1f}"
        )

    best = max(
        final_summary["configs"].items(),
        key=lambda item: item[1]["mean_normalized_score_x100"],
    )
    _log(f"Best by mean normalized score: {best[0]} ({best[1]['mean_normalized_score_x100']:.2f})")

    payload = {
        "task": args.task,
        "ckpt": args.ckpt,
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "num_episodes": args.num_episodes,
        "num_candidates": args.num_candidates,
        "configs": configs,
        "per_seed_results": per_seed_results,
        "cumulative_records": cumulative_records,
        "final_summary": final_summary,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(payload, f, indent=2)
    _log(f"Saved {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
