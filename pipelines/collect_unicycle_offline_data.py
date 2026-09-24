#!/usr/bin/env python3
"""Collect smooth random unicycle trajectories and plot offline samples.

Dynamics (dt = eta):
    x_{t+1}     = x_t + dt * v_t * cos(theta_t)
    y_{t+1}     = y_t + dt * v_t * sin(theta_t)
    theta_{t+1} = theta_t + dt * w_t

Default dt=0.1 s with horizon 64 gives a 6.4 s segment; at v~1 m/s typical
path length is ~4-8 m inside a 16x16 m box.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gym
import h5py
import matplotlib.pyplot as plt
import numpy as np

import cleandiffuser.env.unicycle  # noqa: F401  # register Unicycle-v0


def sample_controls(
    horizon: int,
    rng: np.random.Generator,
    v_bounds: tuple[float, float],
    w_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Diverse (v, w) with moderate smoothness — not overly filtered."""
    t = np.arange(horizon, dtype=np.float64)
    v = rng.normal(0.0, 0.35, horizon)
    w = rng.normal(0.0, 0.45, horizon)

    n_v_modes = int(rng.integers(3, 7))
    n_w_modes = int(rng.integers(3, 7))
    for _ in range(n_v_modes):
        amp = rng.uniform(0.08, 0.55)
        freq = rng.uniform(0.12, 0.65)
        phase = rng.uniform(0.0, 2 * np.pi)
        v += amp * np.sin(2 * np.pi * freq * t / horizon + phase)
    for _ in range(n_w_modes):
        amp = rng.uniform(0.1, 0.65)
        freq = rng.uniform(0.15, 0.75)
        phase = rng.uniform(0.0, 2 * np.pi)
        w += amp * np.sin(2 * np.pi * freq * t / horizon + phase)

    v += rng.uniform(0.35, 1.5)
    w += rng.uniform(-0.35, 0.35)

    # Light smoothing only — keeps variation between neighboring steps.
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    v = np.convolve(v, kernel, mode="same")
    w = np.convolve(w, kernel, mode="same")

    v += rng.normal(0.0, 0.12, horizon)
    w += rng.normal(0.0, 0.16, horizon)

    # Occasional sharper local changes.
    n_jumps = int(rng.integers(2, 6))
    for _ in range(n_jumps):
        idx = int(rng.integers(1, horizon - 1))
        v[idx : idx + 2] += rng.uniform(-0.35, 0.35)
        w[idx : idx + 2] += rng.uniform(-0.5, 0.5)

    v = np.clip(v, *v_bounds)
    w = np.clip(w, *w_bounds)
    return v.astype(np.float32), w.astype(np.float32)


def rollout_episode(
    env: gym.Env,
    rng: np.random.Generator,
    horizon: int,
    v_bounds: tuple[float, float],
    w_bounds: tuple[float, float],
) -> dict[str, np.ndarray]:
    obs_list: list[np.ndarray] = []
    next_obs_list: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminals: list[bool] = []
    timeouts: list[bool] = []

    v_traj, w_traj = sample_controls(horizon, rng, v_bounds, w_bounds)
    seed_val = int(rng.integers(0, 2**31 - 1))
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    if hasattr(unwrapped, "seed"):
        unwrapped.seed(seed_val)
    obs = env.reset()
    obs_list.append(np.asarray(obs, dtype=np.float32).copy())

    for t in range(horizon):
        action = np.array([v_traj[t], w_traj[t]], dtype=np.float32)
        next_obs, reward, done, info = env.step(action)
        actions.append(action)
        rewards.append(float(reward))
        terminated = bool(info.get("terminated", done))
        truncated = bool(info.get("truncated", False))
        terminals.append(terminated)
        timeouts.append(truncated and not terminated)
        next_obs_list.append(np.asarray(next_obs, dtype=np.float32).copy())
        obs_list.append(np.asarray(next_obs, dtype=np.float32).copy())
        obs = next_obs
        if done:
            break

    actual_len = len(actions)
    return {
        "observations": np.stack(obs_list[: actual_len + 1], axis=0),
        "actions": np.stack(actions, axis=0),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminals": np.asarray(terminals, dtype=bool),
        "timeouts": np.asarray(timeouts, dtype=bool),
        "next_observations": np.stack(next_obs_list, axis=0),
        "length": actual_len,
    }


def collect_dataset(
    num_episodes: int,
    horizon: int,
    dt: float,
    seed: int,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    v_bounds: tuple[float, float],
    w_bounds: tuple[float, float],
    *,
    require_full_horizon: bool = False,
    target_full_episodes: int | None = None,
    max_attempts: int | None = None,
) -> dict[str, np.ndarray]:
    """Collect offline rollouts.

    If ``require_full_horizon`` is set, only keep episodes that complete all
    ``horizon`` steps without early out-of-bounds termination.
    When ``target_full_episodes`` is set, keep sampling until that many valid
    episodes are collected (ignores ``num_episodes`` as a keep count).
    """
    rng = np.random.default_rng(seed)
    env = gym.make(
        "Unicycle-v0",
        dt=dt,
        x_lim=x_lim,
        y_lim=y_lim,
        v_bounds=v_bounds,
        w_bounds=w_bounds,
        max_episode_steps=horizon,
    )

    obs_chunks, act_chunks, rew_chunks = [], [], []
    next_obs_chunks, term_chunks, timeout_chunks = [], [], []
    episode_ends: list[int] = []
    cursor = 0

    kept = 0
    attempts = 0
    skipped = 0
    attempt_limit = max_attempts
    if require_full_horizon and target_full_episodes is not None and attempt_limit is None:
        attempt_limit = max(target_full_episodes * 4, target_full_episodes + 1000)

    while True:
        if target_full_episodes is None:
            if attempts >= num_episodes:
                break
        else:
            if kept >= target_full_episodes:
                break
            if attempt_limit is not None and attempts >= attempt_limit:
                raise RuntimeError(
                    f"Stopped after {attempts} attempts with only {kept}/{target_full_episodes} "
                    "full-horizon episodes."
                )

        ep_data = rollout_episode(env, rng, horizon, v_bounds, w_bounds)
        attempts += 1
        n = ep_data["length"]
        if n == 0:
            skipped += 1
            continue

        if require_full_horizon:
            is_full = n == horizon and bool(ep_data["timeouts"][-1]) and not bool(ep_data["terminals"][-1])
            if not is_full:
                skipped += 1
                continue

        obs = ep_data["observations"]
        obs_chunks.append(obs[:-1])
        next_obs_chunks.append(ep_data["next_observations"])
        act_chunks.append(ep_data["actions"])
        rew_chunks.append(ep_data["rewards"])
        term_chunks.append(ep_data["terminals"])
        timeout_chunks.append(ep_data["timeouts"])

        cursor += n
        episode_ends.append(cursor)
        kept += 1

        if kept % 500 == 0:
            print(f"  kept {kept} episodes ({attempts} attempts, {skipped} skipped)", flush=True)

    env.close()
    print(
        f"  collection done: kept={kept} attempts={attempts} skipped={skipped}",
        flush=True,
    )

    return {
        "observations": np.concatenate(obs_chunks, axis=0),
        "actions": np.concatenate(act_chunks, axis=0),
        "rewards": np.concatenate(rew_chunks, axis=0),
        "next_observations": np.concatenate(next_obs_chunks, axis=0),
        "terminals": np.concatenate(term_chunks, axis=0),
        "timeouts": np.concatenate(timeout_chunks, axis=0),
        "episode_ends": np.asarray(episode_ends, dtype=np.int64),
    }


def save_hdf5(dataset: dict[str, np.ndarray], path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key in (
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "terminals",
            "timeouts",
            "episode_ends",
        ):
            f.create_dataset(key, data=dataset[key], compression="gzip")
        f.attrs["metadata_json"] = json.dumps(metadata)


def _plot_ten_trajectories(
    dataset_path: Path,
    episode_ends: np.ndarray,
    plot_idx: int,
    start_ep: int,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    ep_start = 0
    ep_id = 0
    drawn = 0
    while ep_id < start_ep + 10 and ep_start < len(episode_ends):
        ep_end = int(episode_ends[ep_start])
        ep_begin = 0 if ep_start == 0 else int(episode_ends[ep_start - 1])
        if ep_id >= start_ep:
            with h5py.File(dataset_path, "r") as f:
                obs_ep = f["observations"][ep_begin:ep_end]
                last_next = f["next_observations"][ep_end - 1 : ep_end]
            xy = np.vstack([obs_ep[:, :2], last_next[:, :2]])
            ax.plot(xy[:, 0], xy[:, 1], color=colors[drawn], linewidth=1.8, alpha=0.9)
            ax.scatter(xy[0, 0], xy[0, 1], color=colors[drawn], s=28, marker="o")
            ax.scatter(xy[-1, 0], xy[-1, 1], color=colors[drawn], s=36, marker="x")
            drawn += 1
        ep_start += 1
        ep_id += 1

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Unicycle offline samples (plot {plot_idx + 1}: 10 trajectories)")
    fig.tight_layout()
    out = output_dir / f"trajectories_plot_{plot_idx + 1:02d}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


def make_plots(
    dataset_path: Path,
    episode_ends: np.ndarray,
    num_plots: int,
    trajectories_per_plot: int,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_episodes = len(episode_ends)
    max_start = max(0, n_episodes - trajectories_per_plot)
    if num_plots == 1:
        starts = [0]
    else:
        starts = np.linspace(0, max_start, num=num_plots, dtype=int)

    for plot_idx, start_ep in enumerate(starts):
        _plot_ten_trajectories(
            dataset_path,
            episode_ends,
            plot_idx,
            int(start_ep),
            x_lim,
            y_lim,
            output_dir,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/unicycle_offline")
    parser.add_argument("--num-episodes", type=int, default=4000)
    parser.add_argument(
        "--target-full-episodes",
        type=int,
        default=None,
        help="If set with --require-full-horizon, collect until this many valid episodes.",
    )
    parser.add_argument(
        "--require-full-horizon",
        action="store_true",
        help="Drop early-terminated (out-of-bounds) episodes; keep only full horizon rollouts.",
    )
    parser.add_argument("--max-attempts", type=int, default=None, help="Cap rollout attempts when using --target-full-episodes.")
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.1, help="Discrete step eta (seconds).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-plots", type=int, default=12)
    parser.add_argument("--x-min", type=float, default=-8.0)
    parser.add_argument("--x-max", type=float, default=8.0)
    parser.add_argument("--y-min", type=float, default=-8.0)
    parser.add_argument("--y-max", type=float, default=8.0)
    parser.add_argument("--v-min", type=float, default=0.0)
    parser.add_argument("--v-max", type=float, default=2.0)
    parser.add_argument("--w-min", type=float, default=-1.5)
    parser.add_argument("--w-max", type=float, default=1.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    x_lim = (args.x_min, args.x_max)
    y_lim = (args.y_min, args.y_max)
    v_bounds = (args.v_min, args.v_max)
    w_bounds = (args.w_min, args.w_max)

    metadata = {
        "env": "Unicycle-v0",
        "dt": args.dt,
        "horizon": args.horizon,
        "target_full_episodes": args.target_full_episodes,
        "require_full_horizon": args.require_full_horizon,
        "x_lim": x_lim,
        "y_lim": y_lim,
        "v_bounds": v_bounds,
        "w_bounds": w_bounds,
        "observation_format": "[x, y, cos(theta), sin(theta)]",
        "action_format": "[v, w]",
        "dynamics": "x+=dt*v*cos(theta), y+=dt*v*sin(theta), theta+=dt*w",
    }

    print("Collecting unicycle offline dataset...")
    if args.require_full_horizon and args.target_full_episodes:
        print(
            f"  target_full_episodes={args.target_full_episodes} "
            f"horizon={args.horizon} dt={args.dt} (drop early OOB)"
        )
    else:
        print(f"  episodes={args.num_episodes} horizon={args.horizon} dt={args.dt}")
    dataset = collect_dataset(
        num_episodes=args.num_episodes,
        horizon=args.horizon,
        dt=args.dt,
        seed=args.seed,
        x_lim=x_lim,
        y_lim=y_lim,
        v_bounds=v_bounds,
        w_bounds=w_bounds,
        require_full_horizon=args.require_full_horizon,
        target_full_episodes=args.target_full_episodes,
        max_attempts=args.max_attempts,
    )
    metadata["num_episodes"] = len(dataset["episode_ends"])

    h5_path = output_dir / "unicycle_offline.hdf5"
    save_hdf5(dataset, h5_path, metadata)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {h5_path}")
    print(f"  transitions={dataset['observations'].shape[0]}")
    print(f"  episodes={len(dataset['episode_ends'])}")

    plot_dir = output_dir / "plots"
    make_plots(
        h5_path,
        dataset["episode_ends"],
        num_plots=args.num_plots,
        trajectories_per_plot=10,
        x_lim=x_lim,
        y_lim=y_lim,
        output_dir=plot_dir,
    )
    print(f"Saved {args.num_plots} plots to {plot_dir}")


if __name__ == "__main__":
    main()
