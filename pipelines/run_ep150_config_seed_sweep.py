"""Run one ep150 config across many seeds (1 episode each) with interim mean/SE logging."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_hopper_trajectory import (  # noqa: E402
    _log,
    benchmark_sample,
    build_agent,
    rollout_with_trajectory,
)
from diffuser_d4rl_mujoco import _load_checkpoints  # noqa: E402
from utils import set_episode_seed

import d4rl  # noqa: F401
import gym

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from d4rl_render_utils import is_offline_d4rl_env, make_sim_eval_env


def _append_log(log_path: Path, msg: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")
    print(msg, flush=True)


def _compute_stats(rewards: list[float], norms: list[float]) -> dict:
    n = len(rewards)
    if n == 0:
        return {
            "n": 0,
            "mean_reward": float("nan"),
            "std_reward": float("nan"),
            "se_reward": float("nan"),
            "mean_norm_x100": float("nan"),
            "std_norm_x100": float("nan"),
            "se_norm_x100": float("nan"),
        }
    r = np.asarray(rewards, dtype=np.float64)
    z = np.asarray(norms, dtype=np.float64)
    std_r = float(r.std(ddof=0))
    std_z = float(z.std(ddof=0))
    return {
        "n": n,
        "mean_reward": float(r.mean()),
        "std_reward": std_r,
        "se_reward": std_r / math.sqrt(n),
        "mean_norm_x100": float(z.mean()),
        "std_norm_x100": std_z,
        "se_norm_x100": std_z / math.sqrt(n),
    }


def _format_stats(label: str, stats: dict) -> str:
    return (
        f"[{label}] n={stats['n']} "
        f"reward={stats['mean_reward']:.2f} ± {stats['se_reward']:.2f} "
        f"(std={stats['std_reward']:.2f}) "
        f"norm_x100={stats['mean_norm_x100']:.2f} ± {stats['se_norm_x100']:.2f} "
        f"(std={stats['std_norm_x100']:.2f})"
    )


def _build_args(args_cli):
    base = OmegaConf.load(Path(__file__).resolve().parent / args_cli.config)
    task_cfg = OmegaConf.load(
        Path(__file__).resolve().parent.parent / f"configs/diffuser/mujoco/task/{args_cli.task}.yaml"
    )
    args = OmegaConf.merge(base, {"task": task_cfg, "mode": "inference", "ckpt": args_cli.ckpt})
    if args_cli.guidance_mode is not None:
        args.guidance_mode = args_cli.guidance_mode
    if args_cli.optimization_guidance_scale is not None:
        args.optimization_guidance_scale = args_cli.optimization_guidance_scale
    if args_cli.w_cg is not None:
        args.task.w_cg = args_cli.w_cg
    args.sim_env_name = args_cli.sim_env_name
    if args_cli.solver is not None:
        args.solver = args_cli.solver
    if args_cli.sampling_steps is not None:
        args.sampling_steps = args_cli.sampling_steps
    if args_cli.temperature is not None:
        args.temperature = args_cli.temperature
    if args_cli.optimization_guidance_last_steps is not None:
        args.optimization_guidance_last_steps = args_cli.optimization_guidance_last_steps
    else:
        args.optimization_guidance_last_steps = 10
    args.noise_schedule = args_cli.noise_schedule
    args.ddim_eta = args_cli.ddim_eta
    args.optimization_guidance_alpha_sigma_scale = bool(
        getattr(args_cli, "optimization_guidance_alpha_sigma_scale", False)
    )
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/diffuser/mujoco/mujoco.yaml")
    parser.add_argument("--task", default="hopper-medium-v2")
    parser.add_argument("--ckpt", default="latest")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interim-dir", default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=149)
    parser.add_argument("--stats-interval", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--sim-env-name", default="hopper-medium-v2")
    parser.add_argument("--guidance_mode", default=None, choices=["standard", "optimization", "hybrid"])
    parser.add_argument("--optimization_guidance_scale", type=float, default=None)
    parser.add_argument("--w_cg", type=float, default=None)
    parser.add_argument("--solver", default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--optimization-guidance-last-steps", type=int, default=None)
    parser.add_argument("--noise-schedule", default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument(
        "--optimization-guidance-alpha-sigma-scale",
        action="store_true",
        help="Scale optimization shift by sigma**2/alpha at each reverse step.",
    )
    args_cli = parser.parse_args()

    log_path = Path(args_cli.log_file)
    out_path = Path(args_cli.output)
    interim_dir = Path(args_cli.interim_dir) if args_cli.interim_dir else out_path.parent / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if log_path.exists():
        log_path.unlink()

    args = _build_args(args_cli)
    set_episode_seed(args_cli.seed_start)

    save_path = f"results/{args.pipeline_name}/{args.task.env_name}/"
    env = gym.make(args.task.env_name)
    dataset = D4RLMuJoCoDataset(
        env.get_dataset(),
        horizon=args.task.horizon,
        terminal_penalty=args.terminal_penalty,
        discount=args.discount,
    )
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    agent = build_agent(args, obs_dim, act_dim)
    _load_checkpoints(agent, save_path, args.ckpt)
    agent.eval()
    normalizer = dataset.get_normalizer()

    use_sim_fallback = is_offline_d4rl_env(env)
    env.close()

    env_eval, sim_name = make_sim_eval_env(
        args.task.env_name,
        sim_env_name=args.sim_env_name,
        render=False,
        ignore_termination=False,
    )

    _append_log(
        log_path,
        "=== ep150 seed sweep start ===\n"
        f"run_name={args_cli.run_name}\n"
        f"task={args.task.env_name} ckpt={args_cli.ckpt} sim={sim_name}\n"
        f"seeds={args_cli.seed_start}..{args_cli.seed_end} (1 episode each)\n"
        f"stats_interval={args_cli.stats_interval}\n"
        f"solver={args.solver} temp={args.temperature} w_cg={args.task.w_cg} "
        f"guidance_mode={args.guidance_mode} opt_scale={args.optimization_guidance_scale} "
        f"opt_last={args.optimization_guidance_last_steps} ddim_eta={args.ddim_eta} "
        f"alpha_sigma_scale={getattr(args, 'optimization_guidance_alpha_sigma_scale', False)}\n"
        f"device={args.device} offline_fallback={use_sim_fallback}",
    )

    sample_s = benchmark_sample(agent, normalizer, args, obs_dim, act_dim, env_eval)
    _append_log(log_path, f"[setup] one sample call took {sample_s:.2f}s")

    score_env = gym.make(args.task.env_name)
    episodes = []
    rewards: list[float] = []
    norms: list[float] = []
    run_start = time.perf_counter()

    for seed in range(args_cli.seed_start, args_cli.seed_end + 1):
        set_episode_seed(seed)
        ep_result = rollout_with_trajectory(
            env_eval,
            agent,
            normalizer,
            args,
            obs_dim,
            act_dim,
            max_steps=args_cli.max_steps,
            episode_idx=seed,
            log_interval=args_cli.max_steps + 1,
            expected_sample_s=sample_s,
            env_seed=seed,
        )
        normalized = float(score_env.get_normalized_score(ep_result["total_reward"]))
        ep_result.update(
            {
                "seed": seed,
                "episode": seed,
                "normalized_score": normalized,
                "normalized_score_x100": normalized * 100.0,
            }
        )
        episodes.append(ep_result)
        rewards.append(float(ep_result["total_reward"]))
        norms.append(float(ep_result["normalized_score_x100"]))

        _append_log(
            log_path,
            f"[seed {seed}] reward={ep_result['total_reward']:.1f} "
            f"norm_x100={ep_result['normalized_score_x100']:.2f} "
            f"survival={ep_result['survival_steps']} "
            f"class={ep_result['classification']} "
            f"fall_step={ep_result['first_fall_step']}",
        )

        completed = seed - args_cli.seed_start + 1
        if completed % args_cli.stats_interval == 0:
            stats = _compute_stats(rewards, norms)
            _append_log(log_path, _format_stats(f"interim n={completed}", stats))
            interim_payload = {
                "run_name": args_cli.run_name,
                "seeds_completed": completed,
                "seed_start": args_cli.seed_start,
                "seed_end": args_cli.seed_end,
                **stats,
                "episodes": episodes,
            }
            interim_path = interim_dir / f"{args_cli.run_name}_seed{completed:03d}.json"
            with open(interim_path, "w", encoding="utf-8") as f:
                json.dump(interim_payload, f, indent=2)
            _append_log(log_path, f"[interim] wrote {interim_path}")

    score_env.close()
    env_eval.close()

    final_stats = _compute_stats(rewards, norms)
    class_counts: dict[str, int] = {}
    for ep in episodes:
        cls = ep["classification"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    summary = {
        "run_name": args_cli.run_name,
        "task": args.task.env_name,
        "sim_env": sim_name,
        "ckpt": args_cli.ckpt,
        "seed_start": args_cli.seed_start,
        "seed_end": args_cli.seed_end,
        "num_seeds": len(episodes),
        "episodes_per_seed": 1,
        "guidance_mode": args.guidance_mode,
        "optimization_guidance_scale": float(args.optimization_guidance_scale),
        "optimization_guidance_last_steps": int(args.optimization_guidance_last_steps),
        "w_cg": float(args.task.w_cg),
        "solver": str(args.solver),
        "sampling_steps": int(args.sampling_steps),
        "temperature": float(args.temperature),
        "noise_schedule": str(args.noise_schedule),
        "ddim_eta": float(args.ddim_eta),
        "optimization_guidance_alpha_sigma_scale": bool(
            getattr(args, "optimization_guidance_alpha_sigma_scale", False)
        ),
        "device": str(args.device),
        "max_steps": args_cli.max_steps,
        "stats_interval": args_cli.stats_interval,
        "class_counts": class_counts,
        "mean_survival_steps": float(np.mean([e["survival_steps"] for e in episodes])),
        "std_survival_steps": float(np.std([e["survival_steps"] for e in episodes], ddof=0)),
        "mean_reward": final_stats["mean_reward"],
        "std_reward": final_stats["std_reward"],
        "se_reward": final_stats["se_reward"],
        "mean_normalized_score_x100": final_stats["mean_norm_x100"],
        "std_normalized_score_x100": final_stats["std_norm_x100"],
        "se_normalized_score_x100": final_stats["se_norm_x100"],
        "wall_time_min": (time.perf_counter() - run_start) / 60.0,
        "episodes": episodes,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _append_log(log_path, _format_stats("final", final_stats))
    _append_log(
        log_path,
        f"[eval] wall_time={(time.perf_counter() - run_start)/60:.1f} min class_counts={class_counts}",
    )
    _append_log(log_path, f"[eval] saved {out_path}")
    _append_log(log_path, "=== ep150 seed sweep complete ===")


if __name__ == "__main__":
    main()
