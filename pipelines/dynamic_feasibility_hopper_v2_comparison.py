"""Dynamic feasibility on hopper-v2: compare guidance methods with 64 candidates.

For each of ``num_seeds`` trials:
  1. Randomly sample one initial condition (dataset episode start).
  2. For each guidance config, reset RNG to the same comparison seed and sample
     ``num_candidates`` diffusion plans (trajA) with identical initial state.
  3. Open-loop rollout each candidate's actions in native hopper-v2 (trajB).
  4. Report per-candidate normalized L2 gaps and rewards; aggregate mean/std over 64.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import d4rl  # noqa: F401
import gym
import numpy as np
import torch
from omegaconf import OmegaConf

from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.diffusion.guidance import validate_guidance_config
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import JannerUNet1d

from d4rl_render_utils import env_reset, env_step, make_sim_eval_env, resolve_ckpt_stem
from guidance_comparison_eval import (
    MONTE_CARLO_CONFIG,
    STANDARD_CONFIG,
    build_configs,
)
from utils import set_seed

os.environ.setdefault("MUJOCO_GL", "egl")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _episode_starts(terminals: np.ndarray, timeouts: np.ndarray) -> np.ndarray:
    done = np.asarray(terminals, dtype=bool) | np.asarray(timeouts, dtype=bool)
    ends = np.where(done)[0]
    return np.concatenate([[0], ends[:-1] + 1]).astype(np.int64)


def _load_task_settings(repo_dir: Path, task: str) -> dict:
    task_yaml = repo_dir / "configs" / "diffuser" / "mujoco" / "task" / f"{task}.yaml"
    if not task_yaml.exists():
        raise FileNotFoundError(f"Missing task config: {task_yaml}")
    return OmegaConf.to_container(OmegaConf.load(task_yaml), resolve=True)


def _get_mujoco_state(env) -> tuple[np.ndarray, np.ndarray]:
    unwrapped = env.unwrapped
    qpos = np.array(unwrapped.data.qpos, dtype=np.float64).copy()
    qvel = np.array(unwrapped.data.qvel, dtype=np.float64).copy()
    return qpos, qvel


def _set_mujoco_state(env, qpos: np.ndarray, qvel: np.ndarray) -> None:
    env.unwrapped.set_state(
        np.asarray(qpos, dtype=np.float64),
        np.asarray(qvel, dtype=np.float64),
    )


def _current_obs(env) -> np.ndarray:
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "_get_obs"):
        return np.asarray(unwrapped._get_obs(), dtype=np.float32)
    return env_reset(env)


def _obs_std_vector(normalizer, obs_dim: int) -> np.ndarray:
    std = np.asarray(normalizer.std, dtype=np.float64).reshape(-1)
    if std.size < obs_dim:
        raise ValueError(f"Normalizer std has {std.size} dims, expected {obs_dim}")
    std = std[:obs_dim].copy()
    std[std == 0] = 1.0
    return std


def _feasibility_gaps(
    planned_obs: np.ndarray,
    rollout_obs: np.ndarray,
    obs_std: np.ndarray,
    *,
    store_per_step: bool,
) -> dict[str, float | list[float]]:
    delta = planned_obs - rollout_obs
    per_step = np.linalg.norm(delta, axis=1)
    per_step_norm = np.linalg.norm(delta / obs_std, axis=1)
    future = per_step[1:] if per_step.size > 1 else per_step[:0]
    future_norm = per_step_norm[1:] if per_step_norm.size > 1 else per_step_norm[:0]
    out: dict[str, float | list[float]] = {
        "mean_l2_all": float(per_step.mean()),
        "std_l2_all": float(per_step.std(ddof=0)),
        "mean_l2_future": float(future.mean()) if future.size else 0.0,
        "std_l2_future": float(future.std(ddof=0)) if future.size else 0.0,
        "max_l2": float(per_step.max()),
        "final_l2": float(per_step[-1]),
        "mean_l2_norm_all": float(per_step_norm.mean()),
        "std_l2_norm_all": float(per_step_norm.std(ddof=0)),
        "mean_l2_norm_future": float(future_norm.mean()) if future_norm.size else 0.0,
        "std_l2_norm_future": float(future_norm.std(ddof=0)) if future_norm.size else 0.0,
        "max_l2_norm": float(per_step_norm.max()),
        "final_l2_norm": float(per_step_norm[-1]),
    }
    if store_per_step:
        out["per_step_l2"] = per_step.astype(float).tolist()
        out["per_step_l2_norm"] = per_step_norm.astype(float).tolist()
    return out


DEFAULT_CONFIGS = [
    MONTE_CARLO_CONFIG["name"],
    STANDARD_CONFIG["name"],
    "optimization_scale_0p1",
    "optimization_scale_0p3",
    "optimization_scale_0p05",
]

EP150_FEASIBILITY_CONFIGS = [
    {
        "name": "monte_carlo_w_cg0",
        "guidance_mode": "standard",
        "optimization_guidance_scale": 0.0,
        "w_cg": 0.0,
        "solver": "ddpm",
        "temperature": 1.0,
        "sampling_steps": 20,
    },
    {
        "name": "standard_wcg0p3",
        "guidance_mode": "standard",
        "optimization_guidance_scale": 0.0,
        "w_cg": 0.3,
        "solver": "ddpm",
        "temperature": 0.5,
        "sampling_steps": 20,
    },
    {
        "name": "standard_wcg0p5_temp1",
        "guidance_mode": "standard",
        "optimization_guidance_scale": 0.0,
        "w_cg": 0.5,
        "solver": "ddpm",
        "temperature": 1.0,
        "sampling_steps": 20,
    },
    {
        "name": "opt_scale01_optlast10",
        "guidance_mode": "optimization",
        "optimization_guidance_scale": 0.1,
        "w_cg": 0.0,
        "solver": "ddim",
        "temperature": 1.0,
        "sampling_steps": 20,
        "optimization_guidance_last_steps": 10,
        "ddim_eta": 1.0,
    },
    {
        "name": "opt_scale01_optlast5",
        "guidance_mode": "optimization",
        "optimization_guidance_scale": 0.1,
        "w_cg": 0.0,
        "solver": "ddim",
        "temperature": 1.0,
        "sampling_steps": 20,
        "optimization_guidance_last_steps": 5,
        "ddim_eta": 1.0,
    },

]

EP150_PLUS_OPT0_FEASIBILITY_CONFIGS = EP150_FEASIBILITY_CONFIGS + [
    {
        "name": "opt_scale0_optlast10",
        "guidance_mode": "optimization",
        "optimization_guidance_scale": 0.0,
        "w_cg": 0.0,
        "solver": "ddim",
        "temperature": 1.0,
        "sampling_steps": 20,
        "optimization_guidance_last_steps": 10,
        "ddim_eta": 1.0,
    },
]


def _scale_label(scale: float) -> str:
    return str(scale).replace(".", "p")


def _standard_repo_config(w_cg: float) -> dict:
    return {
        "name": "standard_repo",
        "guidance_mode": "standard",
        "optimization_guidance_scale": 0.0,
        "w_cg": float(w_cg),
        "solver": "ddpm",
        "temperature": 0.5,
        "sampling_steps": 20,
    }


def _monte_carlo_config() -> dict:
    return {
        "name": "monte_carlo_w_cg0",
        "guidance_mode": "standard",
        "optimization_guidance_scale": 0.0,
        "w_cg": 0.0,
        "solver": "ddpm",
        "temperature": 1.0,
        "sampling_steps": 20,
    }


def _hybrid_opt_config(
    opt_scale: float,
    *,
    w_cg: float = 0.0,
    opt_last: int = 20,
    name: str | None = None,
) -> dict:
    label = _scale_label(opt_scale)
    wcg_label = _scale_label(w_cg)
    return {
        "name": name or f"hybrid_wcg{wcg_label}_opt{label}_optlast{opt_last}",
        "guidance_mode": "hybrid",
        "optimization_guidance_scale": float(opt_scale),
        "w_cg": float(w_cg),
        "solver": "ddim",
        "temperature": 1.0,
        "sampling_steps": 20,
        "optimization_guidance_last_steps": int(opt_last),
        "ddim_eta": 1.0,
        "optimization_guidance_alpha_sigma_scale": True,
    }


EP150_STD_VS_OPT_PRESETS: dict[str, dict] = {
    "hopper-medium-v2": {
        "opt_scale": 0.9,
        "w_cg": 0.0,
        "opt_last": 20,
        "hybrid_name": "hybrid_wcg0_opt0p9_optlast20",
    },
    "halfcheetah-medium-v2": {
        "opt_scale": 0.00003,
        "w_cg": 0.0,
        "opt_last": 20,
        "hybrid_name": "hybrid_wcg0_opt0p00003_optlast20",
    },
    "walker2d-medium-v2": {
        "opt_scale": 0.05,
        "w_cg": 0.0,
        "opt_last": 20,
        "hybrid_name": "hybrid_wcg0_opt0p05_optlast20",
    },
}


def _hybrid_config_from_task_settings(task_settings: dict) -> dict | None:
    best = task_settings.get("best_hybrid")
    if not isinstance(best, dict):
        return None
    required = ("optimization_guidance_scale",)
    if not all(key in best for key in required):
        return None
    return best


def build_ep150_std_vs_opt_configs(
    task: str,
    task_settings: dict,
    *,
    opt_scale: float | None = None,
    w_cg: float | None = None,
    opt_last: int = 20,
    hybrid_name: str | None = None,
) -> list[dict]:
    task_best = _hybrid_config_from_task_settings(task_settings)
    preset = EP150_STD_VS_OPT_PRESETS.get(task)
    if task_best is None and preset is None and opt_scale is None:
        raise ValueError(
            f"No ep150_std_vs_opt preset for task={task!r}; pass --opt-scale explicitly."
        )

    if opt_scale is not None:
        resolved_opt_scale = float(opt_scale)
    elif task_best is not None:
        resolved_opt_scale = float(task_best["optimization_guidance_scale"])
    else:
        resolved_opt_scale = float(preset["opt_scale"])

    if w_cg is not None:
        resolved_w_cg = float(w_cg)
    elif task_best is not None:
        resolved_w_cg = float(task_best.get("w_cg", 0.0))
    elif preset is not None and "w_cg" in preset:
        resolved_w_cg = float(preset["w_cg"])
    else:
        resolved_w_cg = 0.0

    if task_best is not None:
        resolved_opt_last = int(task_best.get("optimization_guidance_last_steps", opt_last))
    elif preset is not None and opt_last == 20:
        resolved_opt_last = int(preset["opt_last"])
    else:
        resolved_opt_last = int(opt_last)

    resolved_hybrid_name = hybrid_name
    if resolved_hybrid_name is None and task_best is not None:
        resolved_hybrid_name = task_best.get("name")
    if resolved_hybrid_name is None and preset is not None:
        resolved_hybrid_name = preset.get("hybrid_name")

    hybrid_config = None
    if task_best is not None:
        hybrid_config = {
            "name": resolved_hybrid_name or task_best.get("name") or _hybrid_opt_config(
                resolved_opt_scale, w_cg=resolved_w_cg, opt_last=resolved_opt_last
            )["name"],
            "guidance_mode": task_best.get("guidance_mode", "hybrid"),
            "optimization_guidance_scale": resolved_opt_scale,
            "w_cg": resolved_w_cg,
            "solver": task_best.get("solver", "ddim"),
            "temperature": float(task_best.get("temperature", 1.0)),
            "sampling_steps": int(task_best.get("sampling_steps", 20)),
            "optimization_guidance_last_steps": resolved_opt_last,
            "ddim_eta": float(task_best.get("ddim_eta", 1.0)),
            "optimization_guidance_alpha_sigma_scale": bool(
                task_best.get("optimization_guidance_alpha_sigma_scale", True)
            ),
        }
    else:
        hybrid_config = _hybrid_opt_config(
            resolved_opt_scale,
            w_cg=resolved_w_cg,
            opt_last=resolved_opt_last,
            name=resolved_hybrid_name,
        )

    return [
        _monte_carlo_config(),
        _standard_repo_config(float(task_settings["w_cg"])),
        hybrid_config,
    ]


def _resolve_sampling_params(config: dict, args) -> dict:
    return {
        "solver": config.get("solver", args.solver),
        "sample_steps": int(config.get("sampling_steps", args.sampling_steps)),
        "temperature": float(config.get("temperature", args.temperature)),
        "optimization_guidance_last_steps": int(
            config.get("optimization_guidance_last_steps", args.optimization_guidance_last_steps)
        ),
        "ddim_eta": float(config.get("ddim_eta", 0.0)),
    }


def _load_agent(args, save_path: str, obs_dim: int, act_dim: int, horizon: int) -> DiscreteDiffusionSDE:
    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim,
        model_dim=args.model_dim,
        emb_dim=args.model_dim,
        dim_mult=args.dim_mult,
        timestep_emb_type="positional",
        attention=False,
        kernel_size=5,
    )
    nn_classifier = HalfJannerUNet1d(
        horizon,
        obs_dim + act_dim,
        out_dim=1,
        model_dim=args.model_dim,
        emb_dim=args.model_dim,
        dim_mult=args.dim_mult,
        timestep_emb_type="positional",
        kernel_size=3,
    )
    classifier = CumRewClassifier(nn_classifier, device=args.device)

    fix_mask = torch.zeros((horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.0
    loss_weight = torch.ones((horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = args.action_loss_weight

    agent = DiscreteDiffusionSDE(
        nn_diffusion,
        None,
        fix_mask=fix_mask,
        loss_weight=loss_weight,
        classifier=classifier,
        ema_rate=args.ema_rate,
        device=args.device,
        diffusion_steps=args.diffusion_steps,
        training_diffusion_steps=args.training_diffusion_steps,
        predict_noise=args.predict_noise,
        noise_schedule="cosine",
    )

    ckpt_stem = resolve_ckpt_stem(str(args.ckpt))
    agent.load(save_path + f"diffusion_ckpt_{ckpt_stem}.pt")
    agent.classifier.load(save_path + f"classifier_ckpt_{ckpt_stem}.pt")
    agent.eval()
    return agent


def _sample_one_init(
    env,
    raw_dataset: dict,
    rng: np.random.Generator,
) -> dict:
    qpos = raw_dataset["infos/qpos"]
    qvel = raw_dataset["infos/qvel"]
    starts = _episode_starts(raw_dataset["terminals"], raw_dataset["timeouts"])
    start = int(rng.choice(starts))
    env_reset(env)
    _set_mujoco_state(env, qpos[start], qvel[start])
    return {
        "dataset_index": start,
        "qpos": qpos[start].astype(np.float64),
        "qvel": qvel[start].astype(np.float64),
        "obs": _current_obs(env),
    }


def _sample_all_candidate_plans(
    agent: DiscreteDiffusionSDE,
    normalizer,
    obs: np.ndarray,
    prior: torch.Tensor,
    args,
    config: dict,
    obs_dim: int,
    act_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs_norm = torch.tensor(normalizer.normalize(obs[None, :]), device=args.device, dtype=torch.float32)
    prior.zero_()
    prior[:, 0, :obs_dim] = obs_norm

    sampling = _resolve_sampling_params(config, args)
    traj, log = agent.sample(
        prior,
        solver=sampling["solver"],
        n_samples=args.num_candidates,
        sample_steps=sampling["sample_steps"],
        sample_step_schedule=args.sample_step_schedule,
        use_ema=args.use_ema,
        w_cg=config["w_cg"],
        guidance_mode=config["guidance_mode"],
        optimization_guidance_scale=config["optimization_guidance_scale"],
        optimization_guidance_last_steps=sampling["optimization_guidance_last_steps"],
        temperature=sampling["temperature"],
        ddim_eta=sampling["ddim_eta"],
        optimization_guidance_alpha_sigma_scale=bool(
            config.get("optimization_guidance_alpha_sigma_scale", False)
        ),
    )

    plans = traj.view(args.num_candidates, args.horizon, obs_dim + act_dim).detach()
    planned_obs = normalizer.unnormalize(plans[:, :, :obs_dim].cpu().numpy())
    planned_act = np.clip(plans[:, :, obs_dim:].cpu().numpy(), -1.0, 1.0)

    with torch.no_grad():
        t_zero = torch.zeros((args.num_candidates,), dtype=torch.float32, device=args.device)
        reward_traj_a = agent.classifier.logp(plans, t_zero, None).view(-1).cpu().numpy()

    if log.get("log_p") is not None:
        sample_logp = log["log_p"].view(args.num_candidates, -1).sum(-1).detach().cpu().numpy()
    else:
        sample_logp = reward_traj_a.copy()

    return planned_obs.astype(np.float32), planned_act.astype(np.float32), reward_traj_a.astype(np.float64), sample_logp.astype(np.float64)


def _open_loop_rollout(
    env,
    qpos: np.ndarray,
    qvel: np.ndarray,
    actions: np.ndarray,
    horizon: int,
    obs_dim: int,
) -> tuple[np.ndarray, float]:
    env_reset(env)
    _set_mujoco_state(env, qpos, qvel)
    rollout_obs = np.zeros((horizon, obs_dim), dtype=np.float32)
    total_reward = 0.0

    for t in range(horizon):
        rollout_obs[t] = _current_obs(env)[:obs_dim]
        if t >= horizon - 1:
            break
        _, rew, done, _ = env_step(env, actions[t])
        total_reward += float(rew)
        if done:
            rollout_obs[t + 1 :] = rollout_obs[t]
            break

    return rollout_obs, total_reward


def _evaluate_config_on_init(
    env,
    agent: DiscreteDiffusionSDE,
    normalizer,
    obs_std: np.ndarray,
    prior: torch.Tensor,
    init: dict,
    args,
    config: dict,
    obs_dim: int,
    act_dim: int,
    score_env,
) -> dict:
    validate_guidance_config(
        str(config["guidance_mode"]),
        float(config["w_cg"]),
        float(config["optimization_guidance_scale"]),
    )

    set_seed(args.comparison_seed)
    planned_obs, planned_act, reward_traj_a, sample_logp = _sample_all_candidate_plans(
        agent,
        normalizer,
        init["obs"],
        prior,
        args,
        config,
        obs_dim,
        act_dim,
    )

    qpos = init["qpos"]
    qvel = init["qvel"]
    per_candidate = []
    for cand_idx in range(args.num_candidates):
        rollout_obs, reward_traj_b = _open_loop_rollout(
            env,
            qpos,
            qvel,
            planned_act[cand_idx],
            args.horizon,
            obs_dim,
        )
        gaps = _feasibility_gaps(
            planned_obs[cand_idx],
            rollout_obs,
            obs_std,
            store_per_step=False,
        )
        per_candidate.append(
            {
                "candidate_idx": cand_idx,
                "mean_l2_all": float(gaps["mean_l2_all"]),
                "mean_l2_future": float(gaps["mean_l2_future"]),
                "mean_l2_norm_all": float(gaps["mean_l2_norm_all"]),
                "mean_l2_norm_future": float(gaps["mean_l2_norm_future"]),
                "reward_traj_a_classifier": float(reward_traj_a[cand_idx]),
                "reward_traj_a_sample_logp": float(sample_logp[cand_idx]),
                "reward_traj_b_rollout": float(reward_traj_b),
                "normalized_score_traj_b_x100": float(score_env.get_normalized_score(reward_traj_b) * 100.0),
            }
        )

    def collect(key: str) -> list[float]:
        return [row[key] for row in per_candidate]

    sampling = _resolve_sampling_params(config, args)
    summary = {
        "config": config["name"],
        "guidance_mode": config["guidance_mode"],
        "w_cg": float(config["w_cg"]),
        "optimization_guidance_scale": float(config["optimization_guidance_scale"]),
        "solver": sampling["solver"],
        "temperature": sampling["temperature"],
        "sampling_steps": sampling["sample_steps"],
        "optimization_guidance_last_steps": sampling["optimization_guidance_last_steps"],
        "ddim_eta": sampling["ddim_eta"],
        "optimization_guidance_alpha_sigma_scale": bool(
            config.get("optimization_guidance_alpha_sigma_scale", False)
        ),
        "n_candidates": args.num_candidates,
        "mean_l2_all": _mean_std(collect("mean_l2_all")),
        "mean_l2_future": _mean_std(collect("mean_l2_future")),
        "mean_l2_norm_all": _mean_std(collect("mean_l2_norm_all")),
        "mean_l2_norm_future": _mean_std(collect("mean_l2_norm_future")),
        "reward_traj_a_classifier": _mean_std(collect("reward_traj_a_classifier")),
        "reward_traj_a_sample_logp": _mean_std(collect("reward_traj_a_sample_logp")),
        "reward_traj_b_rollout": _mean_std(collect("reward_traj_b_rollout")),
        "normalized_score_traj_b_x100": _mean_std(collect("normalized_score_traj_b_x100")),
        "per_candidate": per_candidate,
    }
    return summary


def _print_candidate_summary(label: str, mean: float, std: float) -> None:
    _log(f"  {label:34s} {mean:10.4f} ± {std:8.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--task", default="hopper-medium-v2")
    parser.add_argument("--ckpt", default="latest")
    parser.add_argument("--sim-env-name", default="hopper-medium-v2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed-start", type=int, default=0, help="Base seed; trial i uses seed_start + i.")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--dim-mult", default=None)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--sampling-steps", type=int, default=100)
    parser.add_argument("--training-diffusion-steps", type=int, default=20)
    parser.add_argument("--sample-step-schedule", default="uniform")
    parser.add_argument("--optimization-guidance-last-steps", type=int, default=None)
    parser.add_argument("--optimization-scale", type=float, default=0.1)
    parser.add_argument("--solver", default="ddim")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--predict-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ema-rate", type=float, default=0.9999)
    parser.add_argument("--action-loss-weight", type=float, default=10.0)
    parser.add_argument("--terminal-penalty", type=float, default=-100.0)
    parser.add_argument("--discount", type=float, default=0.997)
    parser.add_argument(
        "--config-preset",
        default="default",
        choices=["default", "ep150", "ep150_plus_opt0", "ep150_std_vs_opt"],
        help="Guidance config set: default (legacy), ep150, ep150 + opt scale=0, or ep150 standard vs hybrid opt.",
    )
    parser.add_argument(
        "--opt-scale",
        type=float,
        default=None,
        help="Hybrid opt_scale for ep150_std_vs_opt preset (overrides task default).",
    )
    parser.add_argument(
        "--opt-w-cg",
        type=float,
        default=None,
        help="Hybrid w_cg for ep150_std_vs_opt preset (default: task preset or opt_scale).",
    )
    parser.add_argument(
        "--configs-json",
        default=None,
        help="Optional JSON file with a list of guidance config dicts (overrides preset).",
    )
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    task_settings = _load_task_settings(repo_dir, args.task)
    if args.horizon is None:
        args.horizon = int(task_settings["horizon"])
    if args.dim_mult is None:
        args.dim_mult = task_settings["dim_mult"]
    elif isinstance(args.dim_mult, str):
        args.dim_mult = [int(x) for x in args.dim_mult.split(",") if x.strip()]
    else:
        args.dim_mult = list(args.dim_mult)

    args.device = args.device if torch.cuda.is_available() else "cpu"
    if args.optimization_guidance_last_steps is None:
        args.optimization_guidance_last_steps = args.sampling_steps // 2

    if args.configs_json:
        with open(args.configs_json, encoding="utf-8") as f:
            configs = json.load(f)
        if not isinstance(configs, list):
            raise ValueError(f"--configs-json must contain a list, got {type(configs)}")
    elif args.config_preset == "ep150":
        configs = EP150_FEASIBILITY_CONFIGS
    elif args.config_preset == "ep150_plus_opt0":
        configs = EP150_PLUS_OPT0_FEASIBILITY_CONFIGS
    elif args.config_preset == "ep150_std_vs_opt":
        configs = build_ep150_std_vs_opt_configs(
            args.task,
            task_settings,
            opt_scale=args.opt_scale,
            w_cg=args.opt_w_cg,
        )
    else:
        configs = build_configs(
            opt_scales=[args.optimization_scale],
            include_monte_carlo=True,
            config_names=DEFAULT_CONFIGS,
        )

    save_path = f"results/diffuser_d4rl_mujoco/{args.task}/"
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = repo_dir / save_path / "dynamic_feasibility" / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json) if args.output_json else output_dir / "summary.json"

    _log("============================================================")
    _log("Dynamic feasibility comparison (open-loop rollout)")
    _log("============================================================")
    _log(f"task={args.task} ckpt={args.ckpt} sim_env={args.sim_env_name}")
    _log(f"device={args.device} cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        _log(f"gpu={torch.cuda.get_device_name(0)}")
    _log(
        f"solver={args.solver} sampling_steps={args.sampling_steps} "
        f"temperature={args.temperature} num_candidates={args.num_candidates}"
    )
    _log(f"config_preset={args.config_preset}")
    _log(f"configs={[c['name'] for c in configs]}")
    _log(f"num_seeds={args.num_seeds} seed_start={args.seed_start}")
    _log(f"output_json={output_json}")

    env_data = gym.make(args.task)
    raw_dataset = env_data.get_dataset()
    env_data.close()
    for key in ("infos/qpos", "infos/qvel"):
        if key not in raw_dataset:
            raise KeyError(f"{key} missing from dataset.")

    dataset = D4RLMuJoCoDataset(
        raw_dataset,
        horizon=args.horizon,
        terminal_penalty=args.terminal_penalty,
        discount=args.discount,
    )
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    normalizer = dataset.get_normalizer()
    obs_std = _obs_std_vector(normalizer, obs_dim)

    agent = _load_agent(args, save_path, obs_dim, act_dim, args.horizon)
    env_eval, sim_name = make_sim_eval_env(
        args.task,
        sim_env_name=args.sim_env_name,
        render=False,
        ignore_termination=False,
    )
    score_env = gym.make(args.task)
    _log(f"rollout sim env: {sim_name}")

    prior = torch.zeros(
        (args.num_candidates, args.horizon, obs_dim + act_dim),
        device=args.device,
    )

    seed_trials = []
    for trial_idx in range(args.num_seeds):
        comparison_seed = args.seed_start + trial_idx
        args.comparison_seed = comparison_seed
        set_seed(comparison_seed)
        rng = np.random.default_rng(comparison_seed)
        init = _sample_one_init(env_eval, raw_dataset, rng)

        _log("")
        _log(f"=== Trial {trial_idx} | comparison_seed={comparison_seed} ===")
        _log(
            f"init dataset_index={init['dataset_index']} "
            f"obs[0:3]={np.round(init['obs'][:3], 4).tolist()}"
        )

        config_results = []
        for config in configs:
            _log(f"-- config: {config['name']}")
            result = _evaluate_config_on_init(
                env_eval,
                agent,
                normalizer,
                obs_std,
                prior,
                init,
                args,
                config,
                obs_dim,
                act_dim,
                score_env,
            )
            config_results.append(result)
            m, s = result["mean_l2_all"]
            _print_candidate_summary("mean_l2_all (64 cand, raw)", m, s)
            m, s = result["mean_l2_norm_all"]
            _print_candidate_summary("mean_l2_norm_all (64 cand)", m, s)
            m, s = result["reward_traj_a_classifier"]
            _print_candidate_summary("reward_traj_a (classifier)", m, s)
            m, s = result["reward_traj_b_rollout"]
            _print_candidate_summary("reward_traj_b (rollout)", m, s)
            m, s = result["normalized_score_traj_b_x100"]
            _print_candidate_summary("norm_score_traj_b x100", m, s)

        seed_trials.append(
            {
                "trial_idx": trial_idx,
                "comparison_seed": comparison_seed,
                "init": {
                    "dataset_index": init["dataset_index"],
                    "qpos": init["qpos"].astype(float).tolist(),
                    "qvel": init["qvel"].astype(float).tolist(),
                    "obs": init["obs"].astype(float).tolist(),
                },
                "configs": config_results,
            }
        )

    env_eval.close()
    score_env.close()

    aggregate = {}
    _log("\n========== Aggregate over seeds (per config, mean of per-seed candidate means) ==========")
    for config in configs:
        name = config["name"]
        cfg_rows = []
        for trial in seed_trials:
            cfg = next(c for c in trial["configs"] if c["config"] == name)
            cfg_rows.append(cfg)

        def agg_metric(key: str) -> tuple[float, float]:
            vals = [row[key][0] for row in cfg_rows]
            return _mean_std(vals)

        aggregate[name] = {
            "n_seeds": args.num_seeds,
            "mean_l2_all": agg_metric("mean_l2_all"),
            "mean_l2_future": agg_metric("mean_l2_future"),
            "mean_l2_norm_all": agg_metric("mean_l2_norm_all"),
            "mean_l2_norm_future": agg_metric("mean_l2_norm_future"),
            "reward_traj_a_classifier": agg_metric("reward_traj_a_classifier"),
            "reward_traj_b_rollout": agg_metric("reward_traj_b_rollout"),
            "normalized_score_traj_b_x100": agg_metric("normalized_score_traj_b_x100"),
            "guidance_mode": config["guidance_mode"],
            "w_cg": float(config["w_cg"]),
            "optimization_guidance_scale": float(config["optimization_guidance_scale"]),
        }
        m_raw, s_raw = aggregate[name]["mean_l2_all"]
        m, s = aggregate[name]["mean_l2_norm_all"]
        m_ra, s_ra = aggregate[name]["reward_traj_a_classifier"]
        m_rb, s_rb = aggregate[name]["reward_traj_b_rollout"]
        _log(
            f"{name:28s} l2_raw={m_raw:7.4f}±{s_raw:6.4f}  l2_norm={m:7.4f}±{s:6.4f}  "
            f"rewA={m_ra:7.2f}±{s_ra:6.2f}  rewB={m_rb:7.2f}±{s_rb:6.2f}"
        )

    payload = {
        "task": args.task,
        "ckpt": args.ckpt,
        "sim_env": sim_name,
        "device": args.device,
        "horizon": args.horizon,
        "num_seeds": args.num_seeds,
        "seed_start": args.seed_start,
        "num_candidates": args.num_candidates,
        "solver": args.solver,
        "sampling_steps": args.sampling_steps,
        "temperature": args.temperature,
        "config_preset": args.config_preset,
        "configs": configs,
        "seed_trials": seed_trials,
        "aggregate": aggregate,
        "method": (
            "For each seed trial: sample one random dataset episode start, then for each "
            "guidance config reset RNG to the same comparison_seed and draw 64 diffusion "
            "plans (trajA). Open-loop rollout each candidate's actions in the native sim env "
            "(trajB). Per-step obs error is ||planned - rollout||_2. mean_l2_all is the raw "
            "L2 gap; mean_l2_norm_all divides obs errors by dataset Gaussian std before L2. "
            "trajA reward is classifier predicted cumulative reward; trajB reward is simulator "
            "rollout sum."
        ),
        "gap_metrics": {
            "mean_l2_all": "Unnormalized mean per-step L2 obs gap.",
            "mean_l2_future": "Unnormalized mean L2 obs gap over future steps (t>=1).",
            "mean_l2_norm_all": "Normalized mean per-step L2 obs gap (error / dataset std).",
            "mean_l2_norm_future": "Normalized mean L2 obs gap over future steps (t>=1).",
        },
    }

    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)
    _log(f"\nSaved {output_json}")


if __name__ == "__main__":
    main()
