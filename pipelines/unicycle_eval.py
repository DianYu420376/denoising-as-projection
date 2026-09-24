"""Evaluate unicycle diffusion: dynamic feasibility and heart-shape tracking.

Dynamic feasibility (horizon 64, fix t=0 obs, unguided diffusion):
  - single unconditional sample per init, open-loop rollout, L2 gap metrics
  - XY plots comparing diffusion plan vs simulator rollout

Heart tracking (MPC replanning with horizon-64 model):
  - execute the first K planned actions (default K=10), then replan
  - executed path vs full heart reference

Heart GD sanity (no diffusion):
  - optimize normalized trajectory with gradient ascent on the same tracking reward
  - pins t=0 observation like fix_mask; should converge xy to the reference window
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

import cleandiffuser.env.unicycle  # noqa: F401
from cleandiffuser.classifier import RuntimeRewardClassifier
from cleandiffuser.dataset.unicycle_dataset import UnicycleDataset, load_unicycle_hdf5
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.diffusion.guidance import validate_guidance_config
from cleandiffuser.nn_diffusion import JannerUNet1d
from d4rl_render_utils import resolve_ckpt_stem
from guidance_comparison_eval import build_configs
from utils import set_seed

DEFAULT_OPT_SCALES = [0.1, 0.3, 0.5]
UNGUIDED_CONFIG = {
    "name": "unguided",
    "guidance_mode": "standard",
    "w_cg": 0.0,
    "optimization_guidance_scale": 0.0,
}
V_BOUNDS = (0.0, 2.0)
W_BOUNDS = (-1.5, 1.5)
XY_LIMITS = (-8.0, 8.0)
PLAN_PLOT_LEGEND_FONTSIZE = 14
PLAN_PLOT_DPI = 200


def _apply_figure_plot_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.default": "regular",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": PLAN_PLOT_LEGEND_FONTSIZE,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 2.0,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.35,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _centered_xy_limits(
    *xy_arrays: np.ndarray,
    pad_frac: float = 0.14,
    min_span: float = 1.5,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis limits centered on plotted trajectories with equal aspect padding."""
    chunks = [np.asarray(arr, dtype=np.float64).reshape(-1, 2) for arr in xy_arrays if arr is not None]
    if not chunks:
        return XY_LIMITS, XY_LIMITS
    pts = np.vstack(chunks)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(float(xmax - xmin), float(ymax - ymin), min_span)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * span * (1.0 + pad_frac)
    return (cx - half, cx + half), (cy - half, cy + half)


def _setup_plan_axes(ax, x_lim: tuple[float, float], y_lim: tuple[float, float]) -> None:
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, which="major", alpha=0.45)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.22, linestyle=":")


def _save_plan_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PLAN_PLOT_DPI)
    if output_path.suffix.lower() == ".png":
        fig.savefig(output_path.with_suffix(".pdf"))
    elif output_path.suffix.lower() == ".pdf":
        fig.savefig(output_path.with_suffix(".png"), dpi=PLAN_PLOT_DPI)
    plt.close(fig)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _episode_starts(episode_ends: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], episode_ends[:-1]]).astype(np.int64)


def _load_task_settings(repo_dir: Path) -> dict:
    task_yaml = repo_dir / "configs" / "diffuser" / "unicycle" / "task" / "unicycle-v0.yaml"
    return OmegaConf.to_container(OmegaConf.load(task_yaml), resolve=True)


def _resolve_save_path(repo_dir: Path, run_suffix: str | None) -> Path:
    base = repo_dir / "results" / "diffuser_unicycle" / "Unicycle-v0"
    return base / run_suffix if run_suffix else base


def _clip_actions(actions: np.ndarray) -> np.ndarray:
    clipped = np.asarray(actions, dtype=np.float32).copy()
    clipped[..., 0] = np.clip(clipped[..., 0], V_BOUNDS[0], V_BOUNDS[1])
    clipped[..., 1] = np.clip(clipped[..., 1], W_BOUNDS[0], W_BOUNDS[1])
    return clipped


def _set_state_from_obs(env: gym.Env, obs: np.ndarray) -> None:
    x, y, cos_t, sin_t = obs[:4]
    theta = float(np.arctan2(sin_t, cos_t))
    env.unwrapped.set_state(x, y, theta)


def _current_obs(env: gym.Env) -> np.ndarray:
    return np.asarray(env.unwrapped._get_obs(), dtype=np.float32)


def _obs_std_vector(normalizer, obs_dim: int) -> np.ndarray:
    std = np.asarray(normalizer.std, dtype=np.float64).reshape(-1)
    std = std[:obs_dim].copy()
    std[std == 0] = 1.0
    return std


def _feasibility_gaps(planned_obs: np.ndarray, rollout_obs: np.ndarray, obs_std: np.ndarray) -> dict:
    delta = planned_obs - rollout_obs
    per_step = np.linalg.norm(delta, axis=1)
    per_step_norm = np.linalg.norm(delta / obs_std, axis=1)
    future = per_step[1:] if per_step.size > 1 else per_step[:0]
    future_norm = per_step_norm[1:] if per_step_norm.size > 1 else per_step_norm[:0]
    return {
        "mean_l2_future": float(future.mean()) if future.size else 0.0,
        "std_l2_future": float(future.std(ddof=0)) if future.size else 0.0,
        "mean_l2_norm_future": float(future_norm.mean()) if future_norm.size else 0.0,
        "std_l2_norm_future": float(future_norm.std(ddof=0)) if future_norm.size else 0.0,
        "max_l2": float(per_step.max()),
        "final_l2": float(per_step[-1]),
    }


def _reset_to_obs(env: gym.Env, obs: np.ndarray) -> None:
    env.reset()
    _set_state_from_obs(env, obs)


def _open_loop_rollout_obs(env: gym.Env, init_obs: np.ndarray, actions: np.ndarray, horizon: int) -> np.ndarray:
    _reset_to_obs(env, init_obs)
    rollout_obs = np.zeros((horizon, init_obs.shape[0]), dtype=np.float32)
    for t in range(horizon):
        rollout_obs[t] = _current_obs(env)
        if t >= horizon - 1:
            break
        _, _, done, _ = env.step(_clip_actions(actions[t]))
        if done:
            rollout_obs[t + 1 :] = rollout_obs[t]
            break
    return rollout_obs


def _obs_xy(obs: np.ndarray) -> np.ndarray:
    return obs[:, :2]


def _scatter_xy_steps(
    ax,
    xy: np.ndarray,
    *,
    color: str,
    marker: str,
    size: float,
    label: str | None = None,
    alpha: float = 0.85,
    zorder: int = 4,
) -> None:
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=color,
        s=size,
        marker=marker,
        alpha=alpha,
        linewidths=0.4,
        edgecolors="white",
        zorder=zorder,
        label=label,
    )


def _plot_feasibility_trajectory(
    planned_xy: np.ndarray,
    rollout_xy: np.ndarray,
    init_xy: np.ndarray,
    traj_idx: int,
    output_path: Path,
    gaps: dict,
    *,
    x_lim: tuple[float, float] = XY_LIMITS,
    y_lim: tuple[float, float] = XY_LIMITS,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(planned_xy[:, 0], planned_xy[:, 1], color="tab:blue", linewidth=1.2, alpha=0.5, label="diffusion plan")
    ax.plot(rollout_xy[:, 0], rollout_xy[:, 1], color="tab:orange", linewidth=1.2, alpha=0.5, linestyle="--", label="rollout")
    _scatter_xy_steps(ax, planned_xy, color="tab:blue", marker="o", size=42, label="plan steps (64)")
    _scatter_xy_steps(ax, rollout_xy, color="tab:orange", marker="s", size=38, label="rollout steps (64)")
    ax.scatter(init_xy[0], init_xy[1], color="black", s=90, marker="*", zorder=7, label="start (fixed init)")
    ax.scatter(planned_xy[-1, 0], planned_xy[-1, 1], color="tab:blue", s=90, marker="X", zorder=6, label="plan end")
    ax.scatter(rollout_xy[-1, 0], rollout_xy[-1, 1], color="tab:orange", s=90, marker="X", zorder=6, label="rollout end")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Dynamic feasibility traj {traj_idx} (DDIM, T=1.0) | "
        f"init=({init_xy[0]:.2f}, {init_xy[1]:.2f}) | "
        f"mean L2 future={gaps['mean_l2_future']:.3f} "
        f"(norm={gaps['mean_l2_norm_future']:.3f})"
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_feasibility_grid(
    trajectories: list[dict],
    output_path: Path,
    *,
    x_lim: tuple[float, float] = XY_LIMITS,
    y_lim: tuple[float, float] = XY_LIMITS,
) -> None:
    n = len(trajectories)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, item in enumerate(trajectories):
        ax = axes[idx // ncols, idx % ncols]
        planned_xy = item["planned_xy"]
        rollout_xy = item["rollout_xy"]
        ax.plot(planned_xy[:, 0], planned_xy[:, 1], color="tab:blue", linewidth=0.8, alpha=0.4)
        ax.plot(rollout_xy[:, 0], rollout_xy[:, 1], color="tab:orange", linewidth=0.8, linestyle="--", alpha=0.4)
        _scatter_xy_steps(ax, planned_xy, color="tab:blue", marker="o", size=14, alpha=0.9, zorder=4)
        _scatter_xy_steps(ax, rollout_xy, color="tab:orange", marker="s", size=12, alpha=0.9, zorder=4)
        init_xy = item["init_xy"]
        ax.scatter(init_xy[0], init_xy[1], color="black", s=36, marker="*", zorder=5)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"#{item['traj_idx']} L2={item['mean_l2_future']:.2f}", fontsize=9)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    fig.suptitle("Unguided diffusion plan vs open-loop rollout", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _make_xy_tracking_reward_fn(
    normalizer,
    target_xy_physical: np.ndarray,
    temperature: float = 5.0,
):
    """Reward = -temperature * mean_t>=1 ||xy_t - target_t||^2 in normalized xy space.

    Timestep 0 is excluded because fix_mask pins the initial observation.
    """
    mean = torch.tensor(normalizer.mean[:2], dtype=torch.float32)
    std = torch.tensor(normalizer.std[:2], dtype=torch.float32)
    target_norm = (torch.tensor(target_xy_physical, dtype=torch.float32) - mean) / std

    def reward_fn(x: torch.Tensor, c=None) -> torch.Tensor:
        xy = x[..., :2]
        n = min(xy.shape[1], target_norm.shape[0])
        if n <= 1:
            return torch.zeros(xy.shape[0], device=xy.device, dtype=xy.dtype)
        err = xy[:, 1:n, :] - target_norm[1:n].to(xy.device)
        dist_sq = (err**2).sum(dim=-1).mean(dim=-1)
        return -float(temperature) * dist_sq

    return reward_fn


def heart_curve(
    num_steps: int,
    scale: float = 4.0,
    center: tuple[float, float] = (0.0, -1.0),
) -> np.ndarray:
    u = np.linspace(0.0, 2.0 * np.pi, num_steps, endpoint=False)
    x = 16.0 * np.sin(u) ** 3
    y = 13.0 * np.cos(u) - 5.0 * np.cos(2.0 * u) - 2.0 * np.cos(3.0 * u) - np.cos(4.0 * u)
    xy = np.stack([x, y], axis=-1).astype(np.float64)
    xy = xy / np.max(np.abs(xy)) * scale
    xy[:, 0] += center[0]
    xy[:, 1] += center[1]
    return xy.astype(np.float32)


def heart_init_pose(
    num_steps: int,
    scale: float = 4.0,
    center: tuple[float, float] = (0.0, -1.0),
) -> tuple[float, float, float]:
    """Robot pose at the heart reference start: position u=0, heading along the curve."""
    xy = heart_curve(num_steps, scale=scale, center=center)
    delta = xy[1] - xy[0]
    theta = float(np.arctan2(delta[1], delta[0]))
    return float(xy[0, 0]), float(xy[0, 1]), theta


def _load_agent(
    args,
    save_path: Path,
    obs_dim: int,
    act_dim: int,
    horizon: int,
    classifier: RuntimeRewardClassifier | None = None,
) -> DiscreteDiffusionSDE:
    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim,
        model_dim=args.model_dim,
        emb_dim=args.model_dim,
        dim_mult=args.dim_mult,
        timestep_emb_type="positional",
        attention=False,
        kernel_size=5,
    )

    fix_mask = torch.zeros((horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.0
    loss_weight = torch.ones((horizon, obs_dim + act_dim))

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
    )

    ckpt_stem = resolve_ckpt_stem(str(args.ckpt))
    if getattr(args, "ckpt_path", None):
        ckpt_path = Path(args.ckpt_path)
    else:
        ckpt_path = save_path / f"diffusion_ckpt_{ckpt_stem}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {ckpt_path}")
    _log(f"Loading diffusion checkpoint: {ckpt_path}")
    agent.load(str(ckpt_path))
    agent.eval()
    return agent


def _sample_plan(
    agent: DiscreteDiffusionSDE,
    normalizer,
    init_obs: np.ndarray,
    prior: torch.Tensor,
    args,
    config: dict,
    obs_dim: int,
    act_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    obs_norm = torch.tensor(normalizer.normalize(init_obs[None, :]), device=args.device, dtype=torch.float32)
    prior.zero_()
    prior[:, 0, :obs_dim] = obs_norm

    traj, _ = agent.sample(
        prior,
        solver=args.solver,
        n_samples=1,
        sample_steps=args.sampling_steps,
        sample_step_schedule=args.sample_step_schedule,
        use_ema=args.use_ema,
        w_cg=config["w_cg"],
        guidance_mode=config["guidance_mode"],
        optimization_guidance_scale=config["optimization_guidance_scale"],
        optimization_guidance_last_steps=args.optimization_guidance_last_steps,
        temperature=args.temperature,
    )

    plan = traj[0].detach().cpu().numpy()
    planned_obs = normalizer.unnormalize(plan[:, :obs_dim])
    planned_act = _clip_actions(plan[:, obs_dim:])
    return planned_obs.astype(np.float32), planned_act, plan.astype(np.float32)


def _tracking_reward_score(
    plan_norm: np.ndarray,
    reward_fn,
    device: str,
) -> float:
    x = torch.tensor(plan_norm[None, ...], device=device, dtype=torch.float32)
    return float(reward_fn(x).reshape(-1)[0].item())


def _trajectory_norm_from_obs_act(
    normalizer,
    obs: np.ndarray,
    act: np.ndarray,
    obs_dim: int,
) -> np.ndarray:
    traj = np.zeros((obs.shape[0], obs_dim + act.shape[-1]), dtype=np.float32)
    traj[:, :obs_dim] = normalizer.normalize(obs.astype(np.float32))
    traj[:, obs_dim:] = act.astype(np.float32)
    return traj


def _candidate_metric_summary(candidates: list[dict]) -> dict:
    plan_rewards = [c["plan_reward"] for c in candidates]
    rollout_rewards = [c["rollout_reward"] for c in candidates]
    feasibility_errors = [c["feasibility_mean_l2_future"] for c in candidates]
    plan_mean, plan_std = _mean_std(plan_rewards)
    rollout_mean, rollout_std = _mean_std(rollout_rewards)
    feas_mean, feas_std = _mean_std(feasibility_errors)
    return {
        "plan_reward": {"mean": plan_mean, "std": plan_std},
        "rollout_reward": {"mean": rollout_mean, "std": rollout_std},
        "feasibility_mean_l2_future": {"mean": feas_mean, "std": feas_std},
    }


def _xy_tracking_errors(
    planned_obs: np.ndarray,
    target_xy_physical: np.ndarray,
) -> dict:
    n = min(planned_obs.shape[0], target_xy_physical.shape[0])
    if n <= 1:
        return {
            "per_step_l2": [],
            "mean_l2_future": 0.0,
            "max_l2_future": 0.0,
            "final_l2": 0.0,
        }
    err = planned_obs[1:n, :2] - target_xy_physical[1:n]
    per_step = np.linalg.norm(err, axis=1)
    return {
        "per_step_l2": per_step.astype(float).tolist(),
        "mean_l2_future": float(per_step.mean()),
        "max_l2_future": float(per_step.max()),
        "final_l2": float(per_step[-1]),
    }


def _optimize_tracking_gd(
    reward_fn,
    init_obs_norm: torch.Tensor,
    *,
    horizon: int,
    obs_dim: int,
    act_dim: int,
    device: str,
    num_steps: int,
    lr: float,
    init_noise_scale: float,
    seed: int,
) -> tuple[torch.Tensor, list[float]]:
    """Gradient ascent on tracking reward; no diffusion model."""
    dim = obs_dim + act_dim
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    x = torch.randn((1, horizon, dim), device=device, generator=gen, dtype=torch.float32)
    x *= float(init_noise_scale)
    x[:, 0, :obs_dim] = init_obs_norm

    fix_mask = torch.zeros((horizon, dim), device=device, dtype=torch.float32)
    fix_mask[0, :obs_dim] = 1.0

    reward_history: list[float] = []
    for step in range(num_steps):
        x = x.detach().requires_grad_(True)
        reward = reward_fn(x)
        loss = -reward.mean()
        grad = torch.autograd.grad(loss, x, retain_graph=False)[0]
        with torch.no_grad():
            grad = grad * (1.0 - fix_mask)
            x = x - float(lr) * grad
            x[:, 0, :obs_dim] = init_obs_norm
            reward_history.append(float(reward_fn(x).reshape(-1)[0].item()))
        if step == 0 or (step + 1) % max(num_steps // 10, 1) == 0 or step + 1 == num_steps:
            _log(f"[heart_gd] step {step + 1}/{num_steps}: reward={reward_history[-1]:.6f}")

    return x.detach(), reward_history


def _plot_heart_gd_sanity(
    planned_xy: np.ndarray,
    reference_xy: np.ndarray,
    init_xy: np.ndarray,
    reward: float,
    mean_xy_error: float,
    max_xy_error: float,
    num_gd_steps: int,
    output_path: Path,
    *,
    x_lim: tuple[float, float] = XY_LIMITS,
    y_lim: tuple[float, float] = XY_LIMITS,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(reference_xy[:, 0], reference_xy[:, 1], "k--", linewidth=2.0, alpha=0.65, label="reference heart")
    ax.plot(planned_xy[:, 0], planned_xy[:, 1], color="tab:blue", linewidth=1.5, label="GD plan (no diffusion)")
    _scatter_xy_steps(ax, planned_xy, color="tab:blue", marker="o", size=36, label="GD plan steps")
    ax.scatter(init_xy[0], init_xy[1], color="black", s=90, marker="*", zorder=7, label="start (fix_mask prior)")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Heart GD sanity — {num_gd_steps} steps | reward={reward:.3f} | "
        f"mean xy err={mean_xy_error:.4e} max={max_xy_error:.4e}"
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_gd_reward_curve(reward_history: list[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(reward_history) + 1), reward_history, color="tab:blue", linewidth=1.5)
    ax.set_xlabel("GD step")
    ax.set_ylabel("tracking reward")
    ax.set_title("Heart GD sanity — reward vs step")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _planned_xy_in_bounds(planned_xy: np.ndarray, xy_min: float, xy_max: float) -> bool:
    """True when every planned (x, y) lies in [xy_min, xy_max]."""
    return bool((planned_xy[:, 0] >= xy_min).all() and (planned_xy[:, 0] <= xy_max).all()
                and (planned_xy[:, 1] >= xy_min).all() and (planned_xy[:, 1] <= xy_max).all())


def _sample_initial_conditions(
    raw_dataset: dict,
    count: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    episode_ends = raw_dataset["episode_ends"].astype(np.int64)
    starts = _episode_starts(episode_ends)
    pick = rng.choice(starts, size=count, replace=count > len(starts))
    observations = raw_dataset["observations"]
    inits = []
    for traj_idx, start in enumerate(pick):
        inits.append(
            {
                "traj_idx": traj_idx,
                "dataset_index": int(start),
                "obs": observations[start].astype(np.float32),
            }
        )
    return inits


def run_dynamic_feasibility(args, repo_dir: Path, plot_dir: Path) -> dict:
    task = _load_task_settings(repo_dir)
    horizon = int(task["horizon"])
    save_path = _resolve_save_path(repo_dir, args.run_suffix)

    dataset_h5path = args.dataset_h5path or str(repo_dir / "results" / "unicycle_offline" / "unicycle_offline.hdf5")
    raw_dataset = load_unicycle_hdf5(dataset_h5path)
    dataset = UnicycleDataset(raw_dataset, horizon=horizon)
    normalizer = dataset.get_normalizer()
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    obs_std = _obs_std_vector(normalizer, obs_dim)

    config = UNGUIDED_CONFIG
    validate_guidance_config(config["guidance_mode"], config["w_cg"], config["optimization_guidance_scale"])

    env = gym.make("Unicycle-v0", max_episode_steps=horizon)
    prior = torch.zeros((1, horizon, obs_dim + act_dim), device=args.device)
    agent = _load_agent(args, save_path, obs_dim, act_dim, horizon, classifier=None)

    inits = _sample_initial_conditions(raw_dataset, args.min_trajectories, args.seed)
    samples: list[dict] = []
    plot_records: list[dict] = []
    rejected_plan_bounds = 0
    n_attempts = 0
    max_attempts = args.feasibility_max_attempts
    plan_xy_min = args.feasibility_plan_xy_min
    plan_xy_max = args.feasibility_plan_xy_max
    filter_plan_xy = plan_xy_min is not None and plan_xy_max is not None

    rng = np.random.default_rng(args.seed + 17)
    episode_ends = raw_dataset["episode_ends"].astype(np.int64)
    starts = _episode_starts(episode_ends)
    observations = raw_dataset["observations"]

    kept_count = 0
    plot_count = 0
    while kept_count < args.min_trajectories:
        n_attempts += 1
        if max_attempts is not None and n_attempts > max_attempts:
            raise RuntimeError(
                f"Only collected {kept_count}/{args.min_trajectories} feasibility trajectories "
                f"with plan xy in [{plan_xy_min}, {plan_xy_max}] after {max_attempts} attempts "
                f"({rejected_plan_bounds} rejected)."
            )

        start = int(rng.choice(starts))
        init_obs = observations[start].astype(np.float32)
        set_seed(args.seed + n_attempts)

        planned_obs, planned_act, _ = _sample_plan(
            agent, normalizer, init_obs, prior, args, config, obs_dim, act_dim
        )
        planned_xy = _obs_xy(planned_obs)

        if filter_plan_xy and not _planned_xy_in_bounds(planned_xy, plan_xy_min, plan_xy_max):
            rejected_plan_bounds += 1
            continue

        rollout_obs = _open_loop_rollout_obs(env, init_obs, planned_act, horizon)
        gaps = _feasibility_gaps(planned_obs, rollout_obs, obs_std)
        rollout_xy = _obs_xy(rollout_obs)
        sample = {
            "traj_idx": kept_count,
            "dataset_index": start,
            "attempt_index": n_attempts,
            **gaps,
        }
        samples.append(sample)

        if plot_count < args.num_feasibility_plots:
            plot_path = plot_dir / f"feasibility_traj_{kept_count:03d}.png"
            init_xy = init_obs[:2]
            _plot_feasibility_trajectory(
                planned_xy, rollout_xy, init_xy, kept_count, plot_path, gaps
            )
            plot_records.append(
                {
                    "traj_idx": kept_count,
                    "init_xy": init_xy,
                    "planned_xy": planned_xy,
                    "rollout_xy": rollout_xy,
                    "mean_l2_future": gaps["mean_l2_future"],
                }
            )
            plot_count += 1

        kept_count += 1
        if kept_count % 10 == 0 or kept_count == args.min_trajectories:
            _log(
                f"[feasibility] kept {kept_count}/{args.min_trajectories} "
                f"(attempts={n_attempts}, rejected_bounds={rejected_plan_bounds})"
            )

    if plot_records:
        _plot_feasibility_grid(plot_records, plot_dir / "feasibility_traj_grid.png")

    env.close()

    m_future, s_future = _mean_std([s["mean_l2_future"] for s in samples])
    m_norm_future, s_norm_future = _mean_std([s["mean_l2_norm_future"] for s in samples])

    return {
        "task": "dynamic_feasibility",
        "horizon": horizon,
        "guidance": config,
        "min_trajectories": args.min_trajectories,
        "num_plots": len(plot_records),
        "plot_dir": str(plot_dir),
        "plan_xy_filter": (
            {"min": plan_xy_min, "max": plan_xy_max} if filter_plan_xy else None
        ),
        "n_attempts": n_attempts,
        "n_rejected_plan_bounds": rejected_plan_bounds,
        "summary": {
            "n_trajectories": len(samples),
            "mean_l2_future": m_future,
            "std_l2_future": s_future,
            "mean_l2_norm_future": m_norm_future,
            "std_l2_norm_future": s_norm_future,
        },
        "per_trajectory": samples,
    }


def _get_unicycle_state(env: gym.Env) -> tuple[float, float, float, int, int]:
    """Save pose plus both unwrapped and Gym TimeLimit step counters."""
    unwrapped = env.unwrapped
    x, y, theta = unwrapped._state
    elapsed = int(unwrapped._elapsed)
    wrapper_steps = int(getattr(env, "_elapsed_steps", elapsed))
    return float(x), float(y), float(theta), elapsed, wrapper_steps


def _restore_unicycle_state(env: gym.Env, state: tuple[float, float, float, int, int]) -> None:
    x, y, theta, elapsed, wrapper_steps = state
    env.unwrapped.set_state(x, y, theta)
    env.unwrapped._elapsed = elapsed
    if hasattr(env, "_elapsed_steps"):
        env._elapsed_steps = wrapper_steps


def _open_loop_rollout_obs_preserve(
    env: gym.Env,
    init_obs: np.ndarray,
    actions: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Open-loop rollout without changing the env state after the probe."""
    saved_state = _get_unicycle_state(env)
    rollout_obs = _open_loop_rollout_obs(env, init_obs, actions, horizon)
    _restore_unicycle_state(env, saved_state)
    return rollout_obs


def _target_window(target_xy: np.ndarray, global_t: int, horizon: int) -> np.ndarray:
    """Reference window for planning at global_t: target_xy[t], target_xy[t+1], ..., over horizon steps."""
    window = target_xy[global_t : global_t + horizon]
    if window.shape[0] < horizon:
        pad = np.repeat(window[-1:], horizon - window.shape[0], axis=0)
        window = np.vstack([window, pad])
    return window.astype(np.float32)


def _executed_xy_window(executed_xy: np.ndarray, global_t: int, n_points: int) -> np.ndarray:
    """Slice the main MPC executed path from a checkpoint for plotting/metrics."""
    end = min(global_t + n_points, len(executed_xy))
    window = executed_xy[global_t:end]
    if window.shape[0] < n_points and window.shape[0] > 0:
        pad = np.repeat(window[-1:], n_points - window.shape[0], axis=0)
        window = np.vstack([window, pad])
    return window.astype(np.float32)


def _plot_heart_planned_vs_reference(
    planned_xy: np.ndarray,
    reference_xy: np.ndarray,
    executed_xy: np.ndarray,
    global_t: int,
    config_name: str,
    output_path: Path,
    *,
    horizon: int,
    current_xy: np.ndarray | None = None,
    x_lim: tuple[float, float] = XY_LIMITS,
    y_lim: tuple[float, float] = XY_LIMITS,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(reference_xy[:, 0], reference_xy[:, 1], "k--", linewidth=2.0, alpha=0.75, label="reference heart")
    ax.plot(
        planned_xy[:, 0],
        planned_xy[:, 1],
        color="tab:blue",
        linewidth=2.0,
        label=f"diffusion plan ({horizon}-step sample)",
    )
    ax.plot(
        executed_xy[:, 0],
        executed_xy[:, 1],
        color="tab:green",
        linewidth=2.0,
        label="MPC executed (main rollout)",
    )
    if current_xy is not None:
        ax.scatter(
            current_xy[0],
            current_xy[1],
            color="black",
            s=90,
            marker="*",
            zorder=7,
            label="MPC state (fix_mask prior)",
        )
    ax.scatter(planned_xy[0, 0], planned_xy[0, 1], color="tab:blue", s=36, marker="o", zorder=5)
    ax.scatter(executed_xy[0, 0], executed_xy[0, 1], color="tab:green", s=36, marker="o", zorder=5)
    ax.scatter(planned_xy[-1, 0], planned_xy[-1, 1], color="tab:blue", s=48, marker="x", zorder=5)
    ax.scatter(executed_xy[-1, 0], executed_xy[-1, 1], color="tab:green", s=48, marker="x", zorder=5)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Heart t={global_t} — plan vs reference vs MPC ({config_name})")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_heart_guided_vs_rollout(
    planned_xy: np.ndarray,
    rollout_xy: np.ndarray,
    executed_xy: np.ndarray,
    global_t: int,
    config_name: str,
    output_path: Path,
    *,
    horizon: int,
    current_xy: np.ndarray | None = None,
    reference_xy: np.ndarray | None = None,
    x_lim: tuple[float, float] = XY_LIMITS,
    y_lim: tuple[float, float] = XY_LIMITS,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    if reference_xy is not None:
        ax.plot(
            reference_xy[:, 0],
            reference_xy[:, 1],
            "k--",
            linewidth=1.5,
            alpha=0.45,
            label="reference heart",
        )
    ax.plot(
        planned_xy[:, 0],
        planned_xy[:, 1],
        color="tab:blue",
        linewidth=2.0,
        label=f"diffusion plan ({horizon}-step sample)",
    )
    ax.plot(
        executed_xy[:, 0],
        executed_xy[:, 1],
        color="tab:green",
        linewidth=2.0,
        label="MPC executed (main rollout)",
    )
    ax.plot(
        rollout_xy[:, 0],
        rollout_xy[:, 1],
        color="tab:orange",
        linewidth=2.0,
        linestyle="--",
        label="open-loop rollout (full planned actions)",
    )
    if current_xy is not None:
        ax.scatter(
            current_xy[0],
            current_xy[1],
            color="black",
            s=90,
            marker="*",
            zorder=7,
            label="MPC state (fix_mask prior)",
        )
    ax.scatter(planned_xy[0, 0], planned_xy[0, 1], color="tab:blue", s=36, marker="o", zorder=5)
    ax.scatter(executed_xy[0, 0], executed_xy[0, 1], color="tab:green", s=36, marker="o", zorder=5)
    ax.scatter(rollout_xy[0, 0], rollout_xy[0, 1], color="tab:orange", s=36, marker="o", zorder=5)
    ax.scatter(executed_xy[-1, 0], executed_xy[-1, 1], color="tab:green", s=48, marker="x", zorder=5)
    ax.scatter(rollout_xy[-1, 0], rollout_xy[-1, 1], color="tab:orange", s=48, marker="x", zorder=5)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Heart t={global_t} — plan vs MPC vs open-loop ({config_name})")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _rollout_heart_tracking(
    env: gym.Env,
    agent: DiscreteDiffusionSDE,
    normalizer,
    prior: torch.Tensor,
    args,
    config: dict,
    target_xy: np.ndarray,
    init_state: tuple[float, float, float],
    obs_dim: int,
    act_dim: int,
    horizon: int,
    total_steps: int,
    checkpoint_steps: set[int] | None = None,
) -> dict:
    agent.classifier = None
    checkpoint_steps = checkpoint_steps or set()

    env.reset(options={"initial_state": init_state})
    executed_xy = [_current_obs(env)[:2].copy()]
    tracking_err = [float(np.linalg.norm(executed_xy[0] - target_xy[0]))]
    snapshots: list[dict] = []

    global_t = 0
    while global_t < total_steps:
        obs = _current_obs(env)
        window = _target_window(target_xy, global_t, horizon)

        reward_fn = _make_xy_tracking_reward_fn(normalizer, window, temperature=args.reward_temperature)
        agent.classifier = RuntimeRewardClassifier(reward_fn, device=args.device)

        planned_obs, planned_act, _ = _sample_plan(
            agent, normalizer, obs, prior, args, config, obs_dim, act_dim
        )

        if global_t in checkpoint_steps:
            rollout_obs = _open_loop_rollout_obs_preserve(env, obs, planned_act, horizon)
            snapshots.append(
                {
                    "global_t": int(global_t),
                    "reference_global_t0": int(global_t),
                    "reference_global_t1": int(min(global_t + horizon, total_steps)),
                    "current_xy": obs[:2].copy(),
                    "planned_xy": _obs_xy(planned_obs),
                    "reference_xy": window,
                    "rollout_xy": _obs_xy(rollout_obs),
                    "plan_start_error": float(np.linalg.norm(planned_obs[0, :2] - obs[:2])),
                }
            )

        n_execute = min(args.heart_mpc_execute_steps, total_steps - global_t)
        for k in range(n_execute):
            obs, _, _done, _ = env.step(planned_act[k])
            xy = obs[:2]
            executed_xy.append(xy.copy())
            target_pt = target_xy[min(global_t + 1, total_steps - 1)]
            tracking_err.append(float(np.linalg.norm(xy - target_pt)))
            global_t += 1

    executed_xy = np.asarray(executed_xy, dtype=np.float32)
    env_steps = max(0, len(executed_xy) - 1)
    return {
        "executed_xy": executed_xy,
        "mean_tracking_error": float(np.mean(tracking_err)) if tracking_err else float("nan"),
        "final_tracking_error": float(tracking_err[-1]) if tracking_err else float("nan"),
        "steps_completed": int(len(tracking_err)),
        "env_steps_completed": int(env_steps),
        "tracking_errors": tracking_err,
        "checkpoints": snapshots,
    }


def _plot_heart_result(
    target_xy: np.ndarray,
    executed_xy: np.ndarray,
    config_name: str,
    output_path: Path,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    *,
    mpc_execute_steps: int = 10,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(target_xy[:, 0], target_xy[:, 1], "k--", linewidth=2.0, alpha=0.7, label="target heart")
    ax.plot(
        executed_xy[:, 0],
        executed_xy[:, 1],
        color="tab:green",
        linewidth=1.8,
        label=f"MPC executed ({mpc_execute_steps}-step blocks, replan)",
    )
    ax.scatter(executed_xy[0, 0], executed_xy[0, 1], s=40, marker="o")
    ax.scatter(executed_xy[-1, 0], executed_xy[-1, 1], s=50, marker="x")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Heart tracking — {config_name}")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_plan_vs_rollout(
    planned_xy: np.ndarray,
    rollout_xy: np.ndarray,
    reference_xy: np.ndarray,
    init_xy: np.ndarray,
    plan_reward: float,
    rollout_reward: float,
    feasibility_l2: float,
    candidate_idx: int,
    rank: int,
    reference_name: str,
    config_name: str,
    output_path: Path,
    *,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
) -> None:
    _apply_figure_plot_style()
    if x_lim is None or y_lim is None:
        x_lim, y_lim = _centered_xy_limits(reference_xy, planned_xy, rollout_xy, init_xy[None, :])

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(
        reference_xy[:, 0],
        reference_xy[:, 1],
        color="0.15",
        linestyle="--",
        linewidth=2.2,
        alpha=0.85,
        zorder=2,
    )
    ax.plot(
        planned_xy[:, 0],
        planned_xy[:, 1],
        color="#2166ac",
        linewidth=2.0,
        alpha=0.95,
        zorder=4,
    )
    ax.plot(
        rollout_xy[:, 0],
        rollout_xy[:, 1],
        color="#d95f02",
        linewidth=2.0,
        linestyle="--",
        alpha=0.95,
        zorder=3,
    )
    ax.scatter(
        init_xy[0],
        init_xy[1],
        color="black",
        s=110,
        marker="*",
        zorder=7,
        edgecolors="white",
        linewidths=0.6,
    )
    _setup_plan_axes(ax, x_lim, y_lim)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    fig.tight_layout()
    _save_plan_figure(fig, output_path)


def _plot_heart_plan_vs_rollout(
    planned_xy: np.ndarray,
    rollout_xy: np.ndarray,
    reference_xy: np.ndarray,
    init_xy: np.ndarray,
    reward: float,
    candidate_idx: int,
    config_name: str,
    output_path: Path,
    *,
    x_lim: tuple[float, float] = XY_LIMITS,
    y_lim: tuple[float, float] = XY_LIMITS,
) -> None:
    _plot_plan_vs_rollout(
        planned_xy,
        rollout_xy,
        reference_xy,
        init_xy,
        reward,
        reward,
        0.0,
        candidate_idx,
        1,
        "heart",
        config_name,
        output_path,
        x_lim=x_lim,
        y_lim=y_lim,
    )


def run_heart_gd_sanity(args, repo_dir: Path, plot_dir: Path | None = None) -> dict:
    """Sanity check: pure GD on tracking reward with no diffusion model."""
    task = _load_task_settings(repo_dir)
    horizon = int(task["horizon"])

    dataset_h5path = args.dataset_h5path or str(repo_dir / "results" / "unicycle_offline" / "unicycle_offline.hdf5")
    raw_dataset = load_unicycle_hdf5(dataset_h5path)
    dataset = UnicycleDataset(raw_dataset, horizon=horizon)
    normalizer = dataset.get_normalizer()
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    ref_dense = heart_curve(
        args.heart_reference_steps,
        scale=args.heart_scale,
        center=(args.heart_cx, args.heart_cy),
    )
    target_xy = ref_dense[:horizon].copy()
    init_x, init_y, init_theta = heart_init_pose(
        args.heart_reference_steps,
        scale=args.heart_scale,
        center=(args.heart_cx, args.heart_cy),
    )
    init_state = (init_x, init_y, init_theta)

    env = gym.make("Unicycle-v0", max_episode_steps=horizon, terminate_on_oob=False)
    env.reset(options={"initial_state": init_state})
    init_obs = _current_obs(env)
    init_obs_norm = torch.tensor(
        normalizer.normalize(init_obs[None, :]),
        device=args.device,
        dtype=torch.float32,
    )[0]

    reward_fn = _make_xy_tracking_reward_fn(normalizer, target_xy, temperature=args.reward_temperature)

    gd_lr = args.gd_lr
    if gd_lr is None:
        gd_lr = (horizon - 1) / (2.0 * args.reward_temperature)

    if plot_dir is None:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = repo_dir / "results" / "unicycle_eval" / "heart_gd" / run_tag / "plots"

    _log(
        f"[heart_gd] no diffusion; reference: first {horizon} of {args.heart_reference_steps}-step heart; "
        f"init=({init_x:.4f}, {init_y:.4f}); gd_steps={args.sampling_steps}; lr={gd_lr}; "
        f"init_noise_scale={args.temperature}"
    )

    plan_norm_t, reward_history = _optimize_tracking_gd(
        reward_fn,
        init_obs_norm,
        horizon=horizon,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=args.device,
        num_steps=args.sampling_steps,
        lr=gd_lr,
        init_noise_scale=args.temperature,
        seed=args.seed,
    )
    plan_norm = plan_norm_t[0].detach().cpu().numpy()
    planned_obs = normalizer.unnormalize(plan_norm[:, :obs_dim])
    planned_act = _clip_actions(plan_norm[:, obs_dim:])
    reward = _tracking_reward_score(plan_norm, reward_fn, args.device)
    xy_errors = _xy_tracking_errors(planned_obs, target_xy)

    rollout_obs = _open_loop_rollout_obs(env, init_obs, planned_act, horizon)
    rollout_xy = _obs_xy(rollout_obs)
    planned_xy = _obs_xy(planned_obs)

    _plot_heart_gd_sanity(
        planned_xy,
        target_xy,
        init_obs[:2],
        reward,
        xy_errors["mean_l2_future"],
        xy_errors["max_l2_future"],
        args.sampling_steps,
        plot_dir / "heart_gd_plan_vs_reference.png",
    )
    _plot_gd_reward_curve(reward_history, plot_dir / "heart_gd_reward_curve.png")

    env.close()
    return {
        "task": "heart_gd",
        "horizon": horizon,
        "heart_reference_steps": args.heart_reference_steps,
        "gd_steps": args.sampling_steps,
        "gd_lr": gd_lr,
        "init_noise_scale": args.temperature,
        "reward_temperature": args.reward_temperature,
        "init_state": {"x": init_x, "y": init_y, "theta": init_theta},
        "target_xy_first64": target_xy.tolist(),
        "reward": reward,
        "reward_history": reward_history,
        "xy_errors": xy_errors,
        "plan_start_error": float(np.linalg.norm(planned_obs[0, :2] - init_obs[:2])),
        "plot_dir": str(plot_dir),
    }


def _save_plan_trajectories(
    traj_dir: Path,
    *,
    reference_name: str,
    config_name: str,
    reference_xy: np.ndarray,
    init_obs: np.ndarray,
    candidates: list[dict],
    metadata: dict,
) -> Path:
    traj_dir.mkdir(parents=True, exist_ok=True)
    n = len(candidates)
    horizon = int(reference_xy.shape[0])
    planned_xy = np.stack([c["planned_xy"] for c in candidates], axis=0)
    rollout_xy = np.stack([c["rollout_xy"] for c in candidates], axis=0)
    planned_obs = np.stack([c["planned_obs"] for c in candidates], axis=0)
    planned_act = np.stack([c["planned_act"] for c in candidates], axis=0)

    npz_path = traj_dir / "candidates.npz"
    np.savez_compressed(
        npz_path,
        reference_xy=reference_xy.astype(np.float32),
        init_xy=init_obs[:2].astype(np.float32),
        init_obs=init_obs.astype(np.float32),
        planned_xy=planned_xy.astype(np.float32),
        rollout_xy=rollout_xy.astype(np.float32),
        planned_obs=planned_obs.astype(np.float32),
        planned_act=planned_act.astype(np.float32),
        candidate_idx=np.asarray([c["candidate_idx"] for c in candidates], dtype=np.int32),
        seed=np.asarray([c["seed"] for c in candidates], dtype=np.int32),
        plan_reward=np.asarray([c["plan_reward"] for c in candidates], dtype=np.float64),
        rollout_reward=np.asarray([c["rollout_reward"] for c in candidates], dtype=np.float64),
        feasibility_mean_l2_future=np.asarray(
            [c["feasibility_mean_l2_future"] for c in candidates], dtype=np.float64
        ),
    )

    ranked = sorted(candidates, key=lambda c: c["plan_reward"], reverse=True)
    meta_payload = {
        **metadata,
        "reference_name": reference_name,
        "config_name": config_name,
        "horizon": horizon,
        "n_candidates": n,
        "npz_path": str(npz_path),
        "ranked_candidates": [
            {
                "rank": rank,
                "candidate_idx": cand["candidate_idx"],
                "seed": cand["seed"],
                "plan_reward": cand["plan_reward"],
                "rollout_reward": cand["rollout_reward"],
                "feasibility_mean_l2_future": cand["feasibility_mean_l2_future"],
                "feasibility_mean_l2_norm_future": cand["feasibility_mean_l2_norm_future"],
                "plan_start_error": cand["plan_start_error"],
            }
            for rank, cand in enumerate(ranked, start=1)
        ],
    }
    meta_path = traj_dir / "candidates.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_payload, f, indent=2)
    _log(f"[trajectories] wrote {npz_path} and {meta_path}")
    return npz_path


def _load_plan_trajectories(traj_dir: Path) -> dict:
    npz_path = traj_dir / "candidates.npz"
    meta_path = traj_dir / "candidates.meta.json"
    if not npz_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Trajectory cache missing under {traj_dir}")
    with open(meta_path) as f:
        meta = json.load(f)
    data = np.load(npz_path)
    candidates: list[dict] = []
    n = int(data["candidate_idx"].shape[0])
    for i in range(n):
        candidates.append(
            {
                "candidate_idx": int(data["candidate_idx"][i]),
                "seed": int(data["seed"][i]),
                "plan_reward": float(data["plan_reward"][i]),
                "rollout_reward": float(data["rollout_reward"][i]),
                "feasibility_mean_l2_future": float(data["feasibility_mean_l2_future"][i]),
                "planned_xy": data["planned_xy"][i],
                "rollout_xy": data["rollout_xy"][i],
                "planned_obs": data["planned_obs"][i],
                "planned_act": data["planned_act"][i],
                "plan_start_error": 0.0,
            }
        )
    return {
        "meta": meta,
        "reference_xy": data["reference_xy"],
        "init_xy": data["init_xy"],
        "init_obs": data["init_obs"],
        "candidates": candidates,
    }


def _write_plan_candidate_log(log_path: Path, result: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"reference={result['reference_name']} config={result['config']['name']}",
        f"horizon={result['horizon']} candidates={result['plan_candidates']}",
        f"init_state=({result['init_state']['x']:.6f}, {result['init_state']['y']:.6f}, "
        f"{result['init_state']['theta']:.6f})",
        "",
        "rank  candidate_idx  seed  plan_reward  rollout_reward  feasibility_l2",
    ]
    for entry in result.get("all_candidates_ranked", []):
        lines.append(
            f"{entry['rank']:>4}  {entry['candidate_idx']:>13}  {entry['seed']:>4}  "
            f"{entry['plan_reward']:>11.6f}  {entry['rollout_reward']:>14.6f}  "
            f"{entry['feasibility_mean_l2_future']:>14.6f}"
        )
    best = result["best_candidate"]
    lines.extend(
        [
            "",
            (
                f"best_candidate_idx={best['candidate_idx']} seed={best['seed']} "
                f"plan_reward={best['plan_reward']:.6f} rollout_reward={best['rollout_reward']:.6f} "
                f"feasibility_l2={best['feasibility_mean_l2_future']:.6f}"
            ),
            "",
            "plotted_top_k:",
        ]
    )
    for plot_entry in result.get("top_candidates_by_plan_reward", []):
        lines.append(
            f"  rank={plot_entry['rank']} candidate_idx={plot_entry['candidate_idx']} "
            f"plan_reward={plot_entry['plan_reward']:.6f} "
            f"rollout_reward={plot_entry['rollout_reward']:.6f} "
            f"feasibility_l2={plot_entry['feasibility_mean_l2_future']:.6f} "
            f"plot={plot_entry['plot_path']}"
        )
    with open(log_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_plan_eval(
    args,
    repo_dir: Path,
    *,
    reference_name: str,
    ref_dense: np.ndarray,
    init_state: tuple[float, float, float],
    task_label: str,
    plot_dir: Path | None = None,
    reference_meta: dict | None = None,
) -> dict:
    """Single-shot guided diffusion plan eval for an arbitrary reference XY curve."""
    task = _load_task_settings(repo_dir)
    horizon = int(task["horizon"])
    save_path = _resolve_save_path(repo_dir, args.run_suffix)

    dataset_h5path = args.dataset_h5path or str(repo_dir / "results" / "unicycle_offline" / "unicycle_offline.hdf5")
    raw_dataset = load_unicycle_hdf5(dataset_h5path)
    dataset = UnicycleDataset(raw_dataset, horizon=horizon)
    normalizer = dataset.get_normalizer()
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    configs = build_configs(
        DEFAULT_OPT_SCALES,
        include_monte_carlo=False,
        config_names=args.heart_config_names,
    )
    if len(configs) != 1:
        raise ValueError(f"{task_label} expects exactly one --heart-config-names entry.")

    target_xy = ref_dense[:horizon].copy()
    init_x, init_y, init_theta = init_state

    env = gym.make("Unicycle-v0", max_episode_steps=horizon, terminate_on_oob=False)
    env.reset(options={"initial_state": init_state})
    init_obs = _current_obs(env)

    reward_fn = _make_xy_tracking_reward_fn(normalizer, target_xy, temperature=args.reward_temperature)
    prior = torch.zeros((1, horizon, obs_dim + act_dim), device=args.device)
    obs_std = _obs_std_vector(normalizer, obs_dim)

    if plot_dir is None:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = repo_dir / "results" / "unicycle_eval" / task_label / run_tag / "plots"

    cfg = configs[0]
    ckpt_display = args.ckpt_path or str(_resolve_save_path(repo_dir, args.run_suffix) / "diffusion_ckpt_latest.pt")
    _log(
        f"[{task_label}] reference={reference_name}; checkpoint={ckpt_display}; "
        f"target: first {horizon} of {ref_dense.shape[0]}-step curve; "
        f"init=({init_x:.4f}, {init_y:.4f}); candidates={args.heart_plan_candidates}; "
        f"config={cfg['name']}"
    )

    validate_guidance_config(cfg["guidance_mode"], cfg["w_cg"], cfg["optimization_guidance_scale"])
    agent = _load_agent(args, save_path, obs_dim, act_dim, horizon, classifier=None)
    unguided = cfg["w_cg"] == 0.0 and cfg["optimization_guidance_scale"] == 0.0
    if unguided:
        agent.classifier = None
        _log(f"[{task_label}] unguided sampling; prior pins t=0 obs to init state")
    else:
        agent.classifier = RuntimeRewardClassifier(reward_fn, device=args.device)

    candidates: list[dict] = []
    for i in range(args.heart_plan_candidates):
        set_seed(args.seed + i)
        planned_obs, planned_act, plan_norm = _sample_plan(
            agent, normalizer, init_obs, prior, args, cfg, obs_dim, act_dim
        )
        rollout_obs = _open_loop_rollout_obs(env, init_obs, planned_act, horizon)
        plan_reward = _tracking_reward_score(plan_norm, reward_fn, args.device)
        rollout_norm = _trajectory_norm_from_obs_act(normalizer, rollout_obs, planned_act, obs_dim)
        rollout_reward = _tracking_reward_score(rollout_norm, reward_fn, args.device)
        feasibility = _feasibility_gaps(planned_obs, rollout_obs, obs_std)
        candidates.append(
            {
                "candidate_idx": i,
                "seed": args.seed + i,
                "plan_reward": plan_reward,
                "rollout_reward": rollout_reward,
                "feasibility_mean_l2_future": feasibility["mean_l2_future"],
                "feasibility_mean_l2_norm_future": feasibility["mean_l2_norm_future"],
                "planned_obs": planned_obs,
                "planned_act": planned_act,
                "planned_xy": _obs_xy(planned_obs),
                "rollout_xy": _obs_xy(rollout_obs),
                "plan_start_error": float(np.linalg.norm(planned_obs[0, :2] - init_obs[:2])),
            }
        )
        c = candidates[-1]
        _log(
            f"[{task_label}] candidate {i}: plan_reward={c['plan_reward']:.4f} "
            f"rollout_reward={c['rollout_reward']:.4f} "
            f"feasibility_l2={c['feasibility_mean_l2_future']:.4f}"
        )

    metric_summary = _candidate_metric_summary(candidates)
    _log(
        f"[{task_label}] candidate stats (n={len(candidates)}): "
        f"plan_reward={metric_summary['plan_reward']['mean']:.4f}"
        f"±{metric_summary['plan_reward']['std']:.4f}; "
        f"rollout_reward={metric_summary['rollout_reward']['mean']:.4f}"
        f"±{metric_summary['rollout_reward']['std']:.4f}; "
        f"feasibility_l2={metric_summary['feasibility_mean_l2_future']['mean']:.4f}"
        f"±{metric_summary['feasibility_mean_l2_future']['std']:.4f}"
    )

    ranked = sorted(candidates, key=lambda c: c["plan_reward"], reverse=True)
    top_k = max(1, int(args.plan_plot_top_k))
    top_plots: list[dict] = []
    for rank, cand in enumerate(ranked[:top_k], start=1):
        plot_path = plot_dir / (
            f"plan_{reference_name}_{cfg['name']}_rank{rank:02d}_cand{cand['candidate_idx']}.png"
        )
        _plot_plan_vs_rollout(
            cand["planned_xy"],
            cand["rollout_xy"],
            target_xy,
            init_obs[:2],
            cand["plan_reward"],
            cand["rollout_reward"],
            cand["feasibility_mean_l2_future"],
            cand["candidate_idx"],
            rank,
            reference_name,
            cfg["name"],
            plot_path,
        )
        top_plots.append(
            {
                "rank": rank,
                "candidate_idx": cand["candidate_idx"],
                "seed": cand["seed"],
                "plan_reward": cand["plan_reward"],
                "rollout_reward": cand["rollout_reward"],
                "feasibility_mean_l2_future": cand["feasibility_mean_l2_future"],
                "plot_path": str(plot_path),
            }
        )

    traj_dir = plot_dir.parent / "trajectories"
    traj_metadata = {
        "task": task_label,
        "reference_name": reference_name,
        "config": cfg,
        "reward_temperature": args.reward_temperature,
        "sampling_steps": args.sampling_steps,
        "solver": args.solver,
        "temperature": args.temperature,
        "seed": args.seed,
        "ckpt": args.ckpt,
        "ckpt_path": getattr(args, "ckpt_path", None) or "",
    }
    _save_plan_trajectories(
        traj_dir,
        reference_name=reference_name,
        config_name=cfg["name"],
        reference_xy=target_xy,
        init_obs=init_obs,
        candidates=candidates,
        metadata=traj_metadata,
    )

    best = ranked[0]
    env.close()
    all_candidates_ranked = [
        {
            "rank": rank,
            "candidate_idx": cand["candidate_idx"],
            "seed": cand["seed"],
            "plan_reward": cand["plan_reward"],
            "rollout_reward": cand["rollout_reward"],
            "feasibility_mean_l2_future": cand["feasibility_mean_l2_future"],
        }
        for rank, cand in enumerate(ranked, start=1)
    ]
    result = {
        "task": task_label,
        "reference_name": reference_name,
        "horizon": horizon,
        "reference_steps": int(ref_dense.shape[0]),
        "plan_candidates": args.heart_plan_candidates,
        "plan_plot_top_k": top_k,
        "init_state": {"x": init_x, "y": init_y, "theta": init_theta},
        "target_xy_first_horizon": target_xy.tolist(),
        "config": cfg,
        "candidate_metrics_summary": metric_summary,
        "best_candidate": {
            "rank": 1,
            "candidate_idx": best["candidate_idx"],
            "seed": best["seed"],
            "plan_reward": best["plan_reward"],
            "rollout_reward": best["rollout_reward"],
            "feasibility_mean_l2_future": best["feasibility_mean_l2_future"],
            "plan_start_error": best["plan_start_error"],
        },
        "top_candidates_by_plan_reward": top_plots,
        "all_candidates": [
            {
                "candidate_idx": c["candidate_idx"],
                "seed": c["seed"],
                "plan_reward": c["plan_reward"],
                "rollout_reward": c["rollout_reward"],
                "feasibility_mean_l2_future": c["feasibility_mean_l2_future"],
            }
            for c in candidates
        ],
        "all_candidates_ranked": all_candidates_ranked,
        "trajectory_dir": str(traj_dir),
        "plot_dir": str(plot_dir),
    }
    if reference_meta is not None:
        result["reference_meta"] = reference_meta

    _write_plan_candidate_log(plot_dir.parent / "candidate_index.log", result)
    return result


def run_heart_plan(args, repo_dir: Path, plot_dir: Path | None = None) -> dict:
    """Single-shot guided diffusion: sample K trajectories, pick best reward, plot plan vs rollout."""
    ref_dense = heart_curve(
        args.heart_reference_steps,
        scale=args.heart_scale,
        center=(args.heart_cx, args.heart_cy),
    )
    init_x, init_y, init_theta = heart_init_pose(
        args.heart_reference_steps,
        scale=args.heart_scale,
        center=(args.heart_cx, args.heart_cy),
    )
    if plot_dir is None:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = repo_dir / "results" / "unicycle_eval" / "heart_plan" / run_tag / "plots"
    return run_plan_eval(
        args,
        repo_dir,
        reference_name="heart",
        ref_dense=ref_dense,
        init_state=(init_x, init_y, init_theta),
        task_label="heart_plan",
        plot_dir=plot_dir,
        reference_meta={
            "family": "heart",
            "heart_scale": args.heart_scale,
            "heart_center": [args.heart_cx, args.heart_cy],
        },
    )


def replot_plan_eval_from_cache(
    args,
    output_dir: Path,
    *,
    reference_name: str,
    config_name: str,
) -> dict:
    """Regenerate publication-style plots from saved candidate trajectories."""
    traj_dir = output_dir / "reference_plan" / "trajectories"
    plot_dir = output_dir / "reference_plan" / "plots"
    cached = _load_plan_trajectories(traj_dir)
    meta = cached["meta"]
    if meta.get("reference_name") != reference_name or meta.get("config_name") != config_name:
        raise ValueError(
            f"Cache mismatch in {traj_dir}: "
            f"expected {reference_name}/{config_name}, got "
            f"{meta.get('reference_name')}/{meta.get('config_name')}"
        )

    ranked = sorted(cached["candidates"], key=lambda c: c["plan_reward"], reverse=True)
    top_k = max(1, int(args.plan_plot_top_k))
    top_plots: list[dict] = []
    for rank, cand in enumerate(ranked[:top_k], start=1):
        plot_path = plot_dir / (
            f"plan_{reference_name}_{config_name}_rank{rank:02d}_cand{cand['candidate_idx']}.png"
        )
        _plot_plan_vs_rollout(
            cand["planned_xy"],
            cand["rollout_xy"],
            cached["reference_xy"],
            cached["init_xy"],
            cand["plan_reward"],
            cand["rollout_reward"],
            cand["feasibility_mean_l2_future"],
            cand["candidate_idx"],
            rank,
            reference_name,
            config_name,
            plot_path,
        )
        top_plots.append(
            {
                "rank": rank,
                "candidate_idx": cand["candidate_idx"],
                "seed": cand["seed"],
                "plan_reward": cand["plan_reward"],
                "rollout_reward": cand["rollout_reward"],
                "feasibility_mean_l2_future": cand["feasibility_mean_l2_future"],
                "plot_path": str(plot_path),
            }
        )

    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        with open(summary_path) as f:
            payload = json.load(f)
        block = payload.get("reference_plan", payload.get("heart_plan", {}))
    else:
        block = {}

    block["top_candidates_by_plan_reward"] = top_plots
    block["plot_dir"] = str(plot_dir)
    block["replot_from_cache"] = True
    if summary_path.is_file():
        payload["reference_plan"] = block
        with open(summary_path, "w") as f:
            json.dump(payload, f, indent=2)

    _log(f"[replot] regenerated {len(top_plots)} plots in {plot_dir}")
    return block


def run_reference_plan(args, repo_dir: Path, plot_dir: Path | None = None) -> dict:
    from reference_trajectories import get_reference_trajectory

    if getattr(args, "plot_only_from_cache", False):
        if not args.output_dir:
            raise ValueError("--plot-only-from-cache requires --output-dir pointing at a completed run.")
        output_dir = Path(args.output_dir)
        configs = build_configs(
            DEFAULT_OPT_SCALES,
            include_monte_carlo=False,
            config_names=args.heart_config_names,
        )
        if len(configs) != 1:
            raise ValueError("reference_plan replot expects exactly one --heart-config-names entry.")
        return replot_plan_eval_from_cache(
            args,
            output_dir,
            reference_name=args.reference_name,
            config_name=configs[0]["name"],
        )

    ref_traj = get_reference_trajectory(args.reference_name, num_steps=args.reference_steps)
    init_x, init_y, init_theta = ref_traj.init_pose
    return run_plan_eval(
        args,
        repo_dir,
        reference_name=ref_traj.name,
        ref_dense=ref_traj.xy,
        init_state=(init_x, init_y, init_theta),
        task_label="reference_plan",
        plot_dir=plot_dir,
        reference_meta={
            "family": ref_traj.family,
            "description": ref_traj.description,
            **({"seed": ref_traj.seed} if ref_traj.seed is not None else {}),
        },
    )


def run_heart_tracking(args, repo_dir: Path, plot_dir: Path | None = None) -> dict:
    task = _load_task_settings(repo_dir)
    horizon = int(task["horizon"])
    save_path = _resolve_save_path(repo_dir, args.run_suffix)

    dataset_h5path = args.dataset_h5path or str(repo_dir / "results" / "unicycle_offline" / "unicycle_offline.hdf5")
    raw_dataset = load_unicycle_hdf5(dataset_h5path)
    dataset = UnicycleDataset(raw_dataset, horizon=horizon)
    normalizer = dataset.get_normalizer()
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    configs = build_configs(
        DEFAULT_OPT_SCALES,
        include_monte_carlo=False,
        config_names=args.heart_config_names,
    )
    target_xy = heart_curve(args.tracking_steps, scale=args.heart_scale, center=(args.heart_cx, args.heart_cy))
    if args.init_from_heart_start:
        init_x, init_y, init_theta = heart_init_pose(
            args.tracking_steps,
            scale=args.heart_scale,
            center=(args.heart_cx, args.heart_cy),
        )
        init_state = (init_x, init_y, init_theta)
        _log(
            f"[heart] init from heart start: x={init_x:.4f} y={init_y:.4f} theta={init_theta:.4f} "
            f"(ref start={target_xy[0].tolist()})"
        )
    else:
        init_state = (args.init_x, args.init_y, args.init_theta)
    checkpoint_steps = {s for s in args.heart_checkpoint_steps if s < args.tracking_steps}
    _log(
        f"[heart] reference trajectory: {args.tracking_steps} steps per heart loop "
        f"(u in [0, 2pi)); planning horizon={horizon}; "
        f"window at t is target[t:t+{horizon}]"
    )

    env = gym.make(
        "Unicycle-v0",
        max_episode_steps=args.tracking_steps + horizon,
        terminate_on_oob=False,
    )
    prior = torch.zeros((1, horizon, obs_dim + act_dim), device=args.device)

    if plot_dir is None:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = repo_dir / "results" / "unicycle_eval" / "heart_tracking" / run_tag / "plots"
    checkpoint_plot_dir = plot_dir / "checkpoints"

    per_config = []
    for cfg in configs:
        validate_guidance_config(cfg["guidance_mode"], cfg["w_cg"], cfg["optimization_guidance_scale"])
        agent = _load_agent(args, save_path, obs_dim, act_dim, horizon, classifier=None)
        rollout = _rollout_heart_tracking(
            env,
            agent,
            normalizer,
            prior,
            args,
            cfg,
            target_xy,
            init_state,
            obs_dim,
            act_dim,
            horizon,
            args.tracking_steps,
            checkpoint_steps=checkpoint_steps if (
                args.heart_plot_planned_reference or args.heart_plot_guided_rollout
            ) else None,
        )
        _plot_heart_result(
            target_xy,
            rollout["executed_xy"],
            cfg["name"],
            plot_dir / f"heart_executed_{cfg['name']}.png",
            x_lim=XY_LIMITS,
            y_lim=XY_LIMITS,
            mpc_execute_steps=args.heart_mpc_execute_steps,
        )

        checkpoint_summaries = []
        for snap in rollout["checkpoints"]:
            step = snap["global_t"]
            executed_seg = _executed_xy_window(
                rollout["executed_xy"], step, len(snap["planned_xy"])
            )
            if args.heart_plot_planned_reference:
                _plot_heart_planned_vs_reference(
                    snap["planned_xy"],
                    snap["reference_xy"],
                    executed_seg,
                    step,
                    cfg["name"],
                    checkpoint_plot_dir / f"heart_{cfg['name']}_t{step:03d}_planned_vs_reference.png",
                    horizon=horizon,
                    current_xy=snap.get("current_xy"),
                )
            if args.heart_plot_guided_rollout:
                _plot_heart_guided_vs_rollout(
                    snap["planned_xy"],
                    snap["rollout_xy"],
                    executed_seg,
                    step,
                    cfg["name"],
                    checkpoint_plot_dir / f"heart_{cfg['name']}_t{step:03d}_guided_vs_rollout.png",
                    horizon=horizon,
                    current_xy=snap.get("current_xy"),
                    reference_xy=snap["reference_xy"] if args.heart_plot_reference_on_rollout else None,
                )
            n_exec = min(len(executed_seg), len(snap["planned_xy"]))
            checkpoint_summaries.append(
                {
                    "global_t": step,
                    "plan_start_error": float(snap.get("plan_start_error", 0.0)),
                    "mean_plan_ref_dist": float(
                        np.linalg.norm(snap["planned_xy"] - snap["reference_xy"], axis=1).mean()
                    ),
                    "mean_plan_rollout_dist": float(
                        np.linalg.norm(snap["planned_xy"] - snap["rollout_xy"], axis=1).mean()
                    ),
                    "mean_plan_executed_dist": float(
                        np.linalg.norm(
                            snap["planned_xy"][:n_exec] - executed_seg[:n_exec],
                            axis=1,
                        ).mean()
                    ),
                    "mean_executed_ref_dist": float(
                        np.linalg.norm(
                            executed_seg[: min(n_exec, len(snap["reference_xy"]))]
                            - snap["reference_xy"][: min(n_exec, len(snap["reference_xy"]))],
                            axis=1,
                        ).mean()
                    ),
                }
            )

        per_config.append(
            {
                "name": cfg["name"],
                "guidance_mode": cfg["guidance_mode"],
                "w_cg": cfg["w_cg"],
                "optimization_guidance_scale": cfg["optimization_guidance_scale"],
                "mean_tracking_error": rollout["mean_tracking_error"],
                "final_tracking_error": rollout["final_tracking_error"],
                "steps_completed": rollout["steps_completed"],
                "env_steps_completed": rollout["env_steps_completed"],
                "checkpoint_summaries": checkpoint_summaries,
            }
        )
        _log(
            f"[heart] config={cfg['name']} mean_err={rollout['mean_tracking_error']:.3f} "
            f"env_steps={rollout['env_steps_completed']}/{args.tracking_steps} "
            f"checkpoints={len(rollout['checkpoints'])}"
        )
        if rollout["env_steps_completed"] < args.tracking_steps:
            _log(
                f"[heart] WARNING: expected {args.tracking_steps} env steps, "
                f"got {rollout['env_steps_completed']}"
            )

    env.close()
    return {
        "task": "heart_tracking",
        "tracking_steps": args.tracking_steps,
        "heart_mpc_execute_steps": args.heart_mpc_execute_steps,
        "sampler": {
            "solver": args.solver,
            "sample_step_schedule": args.sample_step_schedule,
            "sampling_steps": args.sampling_steps,
            "temperature": args.temperature,
            "use_ema": args.use_ema,
        },
        "init_state": {
            "x": init_state[0],
            "y": init_state[1],
            "theta": init_state[2],
            "from_heart_start": args.init_from_heart_start,
        },
        "heart_reference_start": target_xy[0].tolist(),
        "heart_scale": args.heart_scale,
        "heart_center": [args.heart_cx, args.heart_cy],
        "checkpoint_steps": sorted(checkpoint_steps),
        "heart_plot_planned_reference": args.heart_plot_planned_reference,
        "heart_plot_guided_rollout": args.heart_plot_guided_rollout,
        "configs": per_config,
        "plot_dir": str(plot_dir),
        "checkpoint_plot_dir": str(checkpoint_plot_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-task",
        choices=["feasibility", "heart", "heart_plan", "reference_plan", "heart_gd", "both"],
        default="feasibility",
    )
    parser.add_argument("--repo-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--dataset-h5path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--ckpt", default="latest")
    parser.add_argument(
        "--ckpt-path",
        default="",
        help="Optional explicit path to diffusion_ckpt_*.pt (overrides --ckpt and --run-suffix).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-trajectories", type=int, default=30)
    parser.add_argument("--num-feasibility-plots", type=int, default=12)
    parser.add_argument(
        "--feasibility-plan-xy-min",
        type=float,
        default=None,
        help="If set with --feasibility-plan-xy-max, only keep plans whose x,y stay in bounds.",
    )
    parser.add_argument("--feasibility-plan-xy-max", type=float, default=None)
    parser.add_argument(
        "--feasibility-max-attempts",
        type=int,
        default=None,
        help="Max diffusion samples when filtering plans by xy bounds (default: 50 * min-trajectories).",
    )
    parser.add_argument("--tracking-steps", type=int, default=256)
    parser.add_argument("--heart-scale", type=float, default=4.0)
    parser.add_argument("--heart-cx", type=float, default=0.0)
    parser.add_argument("--heart-cy", type=float, default=-1.0)
    parser.add_argument("--init-x", type=float, default=-4.0)
    parser.add_argument("--init-y", type=float, default=-3.0)
    parser.add_argument("--init-theta", type=float, default=0.0)
    parser.add_argument(
        "--init-from-heart-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Place the robot at the heart reference start (u=0) with heading along the curve.",
    )
    parser.add_argument("--reward-temperature", type=float, default=5.0)
    parser.add_argument(
        "--gd-lr",
        type=float,
        default=None,
        help="Learning rate for heart_gd. Default: (horizon-1)/(2*reward_temperature) for one-step exact xy fit.",
    )
    parser.add_argument(
        "--heart-reference-steps",
        type=int,
        default=256,
        help="Heart curve discretization; heart_plan uses the first planning-horizon points.",
    )
    parser.add_argument(
        "--heart-plan-candidates",
        type=int,
        default=5,
        help="Number of diffusion samples for plan eval; rank/plot by plan_reward.",
    )
    parser.add_argument(
        "--plan-plot-top-k",
        type=int,
        default=1,
        help="Plot top-K candidates ranked by plan_reward (default 1 for heart_plan).",
    )
    parser.add_argument(
        "--plot-only-from-cache",
        action="store_true",
        help="Regenerate plots from saved trajectories under --output-dir (no GPU sampling).",
    )
    parser.add_argument(
        "--reference-name",
        default="",
        help="Named reference trajectory for reference_plan (e.g. circle, sinusoid_freq1).",
    )
    parser.add_argument(
        "--reference-steps",
        type=int,
        default=64,
        help="Reference curve discretization; default 64 = full path within planning horizon.",
    )
    parser.add_argument(
        "--heart-config-names",
        nargs="+",
        default=None,
        help="Heart tracking guidance configs to run (e.g. standard_w_cg0p3). Default: all standard configs.",
    )
    parser.add_argument(
        "--heart-mpc-execute-steps",
        type=int,
        default=10,
        help="Execute this many planned actions before replanning during heart MPC.",
    )
    parser.add_argument(
        "--heart-checkpoint-steps",
        nargs="+",
        type=int,
        default=[0, 64, 128, 192, 240],
        help="Replan timesteps at which to snapshot guided plan / rollout plots (align with --heart-mpc-execute-steps).",
    )
    parser.add_argument(
        "--heart-plot-planned-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="At checkpoint steps, plot guided diffusion XY plan vs reference heart window.",
    )
    parser.add_argument(
        "--heart-plot-guided-rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="At checkpoint steps, plot guided diffusion plan vs open-loop rollout of its actions.",
    )
    parser.add_argument(
        "--heart-plot-reference-on-rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay faint reference heart on guided-vs-rollout checkpoint plots.",
    )
    parser.add_argument("--solver", default="ddim", help="Use ddim for deterministic reverse steps (eta=0).")
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--sample-step-schedule", default="uniform")
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument(
        "--training-diffusion-steps",
        type=int,
        default=20,
        help="Checkpoint training discretization (default 20 for full64 unicycle).",
    )
    parser.add_argument("--predict-noise", action="store_true")
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--dim-mult", nargs="+", type=int, default=[1, 2, 2])
    parser.add_argument("--ema-rate", type=float, default=0.9999)
    parser.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use EMA diffusion weights at sampling time.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Initial noise scale for x_T. Use 1.0 with DDIM; values <1 shrink trajectories.",
    )
    parser.add_argument("--optimization-guidance-last-steps", type=int, default=10)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    if args.feasibility_max_attempts is None:
        args.feasibility_max_attempts = max(args.min_trajectories * 50, args.min_trajectories + 100)

    repo_dir = Path(args.repo_dir)
    sys.path.insert(0, str(repo_dir / "pipelines"))
    sys.path.insert(0, str(repo_dir))

    set_seed(args.seed)
    os.environ.setdefault("MUJOCO_GL", "disable")

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) if args.output_dir else repo_dir / "results" / "unicycle_eval" / run_tag
    output_root.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "seed": args.seed,
        "ckpt": args.ckpt,
        "run_suffix": args.run_suffix,
        "device": args.device,
    }

    if args.eval_task in ("feasibility", "both"):
        feasibility_plot_dir = output_root / "feasibility" / "plots"
        payload["dynamic_feasibility"] = run_dynamic_feasibility(args, repo_dir, feasibility_plot_dir)

    if args.eval_task in ("heart", "both"):
        heart_plot_dir = output_root / "heart" / "plots"
        payload["heart_tracking"] = run_heart_tracking(args, repo_dir, heart_plot_dir)

    if args.eval_task == "heart_plan":
        heart_plan_plot_dir = output_root / "heart_plan" / "plots"
        payload["heart_plan"] = run_heart_plan(args, repo_dir, heart_plan_plot_dir)

    if args.eval_task == "reference_plan":
        if not args.reference_name:
            raise ValueError("reference_plan requires --reference-name")
        reference_plan_plot_dir = output_root / "reference_plan" / "plots"
        payload["reference_plan"] = run_reference_plan(args, repo_dir, reference_plan_plot_dir)

    if args.eval_task == "heart_gd":
        heart_gd_plot_dir = output_root / "heart_gd" / "plots"
        payload["heart_gd"] = run_heart_gd_sanity(args, repo_dir, heart_gd_plot_dir)

    summary_path = output_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)
    _log(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
