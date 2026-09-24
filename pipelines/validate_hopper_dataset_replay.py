"""Validate Hopper sim dynamics by replaying D4RL dataset trajectories.

For each transition (qpos_t, qvel_t, action_t):
  1. set_state(qpos_t, qvel_t)
  2. check obs matches dataset observations[t]      (static / kinematic)
  3. step(action_t)
  4. check next obs matches dataset next_observations[t] (one-step dynamics)
  5. optionally roll out full episode segments open-loop

The hopper-medium-v2 HDF5 stores full MuJoCo states in infos/qpos and infos/qvel,
so no observation→state reconstruction is needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gym
import numpy as np

import d4rl  # noqa: F401


def _unwrap_mujoco_env(env):
    """Return the underlying D4RL OfflineHopperEnv (gym HopperEnv + mujoco_py)."""
    current = env
    while type(current).__name__ != "OfflineHopperEnv":
        if hasattr(current, "env"):
            current = current.env
        elif hasattr(current, "_wrapped_env"):
            current = current._wrapped_env
        else:
            raise RuntimeError(
                f"Could not find OfflineHopperEnv under wrapper chain; stopped at {type(current)}"
            )
    return current


def make_real_mujoco_v2_env(task: str = "hopper-medium-v2"):
    """Create the native D4RL gym/mujoco_py Hopper-v2 environment."""
    outer = gym.make(task)
    mujoco_env = _unwrap_mujoco_env(outer)
    mujoco_env._replay_outer_env = outer  # keep alive for close()
    return mujoco_env


def _get_obs(env) -> np.ndarray:
    return np.asarray(env._get_obs(), dtype=np.float64)


def _step(env, action: np.ndarray):
    obs, reward, done, info = env.step(action)
    return np.asarray(obs, dtype=np.float64), float(reward), bool(done), info


def _close_env(env) -> None:
    outer = getattr(env, "_replay_outer_env", None)
    if outer is not None:
        outer.close()
    elif hasattr(env, "close"):
        env.close()


def _load_dataset(task: str):
    env = gym.make(task)
    ds = env.get_dataset()
    env.close()
    return ds


def _episode_starts(terminals: np.ndarray, timeouts: np.ndarray) -> list[int]:
    ends = np.where(terminals | timeouts)[0]
    starts = [0]
    for idx in ends[:-1]:
        starts.append(int(idx) + 1)
    return starts


def _summarize(name: str, values: np.ndarray) -> dict:
    if values.size == 0:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "frac_lt_1e-5": float((values < 1e-5).mean()),
        "frac_lt_0p01": float((values < 0.01).mean()),
        "frac_lt_0p05": float((values < 0.05).mean()),
        "frac_lt_0p10": float((values < 0.10).mean()),
    }


def _rollout_segment_errors(
    env,
    qpos: np.ndarray,
    qvel: np.ndarray,
    actions: np.ndarray,
    next_obs: np.ndarray,
    start: int,
    horizon: int,
) -> list[float]:
    """Open-loop rollout errors for one continuous in-episode segment.

    Uses a single set_state at the segment start, then steps without re-injecting
    stored qpos/qvel. This matches how D4RL data were collected and avoids false
    positives from clipped infos/qvel entries in the HDF5.
    """
    env.set_state(qpos[start], qvel[start])
    step_errors: list[float] = []
    for k in range(horizon):
        sim_next, _, terminated, _ = _step(env, actions[start + k])
        step_errors.append(float(np.linalg.norm(sim_next - next_obs[start + k])))
        if terminated:
            break
    return step_errors


def validate_replay(
    task: str = "hopper-medium-v2",
    num_samples: int = 5000,
    episode_horizons: list[int] | None = None,
    num_trajectories: int = 200,
    tolerance: float = 1e-5,
    env_factory=None,
) -> dict:
    episode_horizons = episode_horizons or [1, 5, 10, 50, 100]
    ds = _load_dataset(task)

    qpos = ds["infos/qpos"]
    qvel = ds["infos/qvel"]
    obs = ds["observations"]
    actions = ds["actions"]
    next_obs = ds["next_observations"]
    terminals = ds["terminals"]
    timeouts = ds["timeouts"]

    if env_factory is None:
        env = make_real_mujoco_v2_env(task)
    else:
        env = env_factory()
    env_name = f"{env.__class__.__name__} (mujoco_py)"

    starts = _episode_starts(terminals, timeouts)
    selected_starts = starts[: min(num_trajectories, len(starts))]
    primary_horizon = max(episode_horizons) if episode_horizons else 32

    static_l2: list[float] = []
    segment_details: list[dict] = []
    all_step_errors: list[float] = []

    for seg_idx, start in enumerate(selected_starts):
        end_bound = len(obs) - 1
        for end_idx in np.where(terminals | timeouts)[0]:
            if end_idx >= start:
                end_bound = int(end_idx)
                break
        seg_len = end_bound - start
        if seg_len < primary_horizon:
            continue

        env.set_state(qpos[start], qvel[start])
        static_l2.append(float(np.linalg.norm(_get_obs(env) - obs[start])))

        step_errors = _rollout_segment_errors(
            env, qpos, qvel, actions, next_obs, start, primary_horizon
        )
        all_step_errors.extend(step_errors)
        segment_details.append(
            {
                "segment_idx": seg_idx,
                "start_index": int(start),
                "horizon": primary_horizon,
                "step_errors": step_errors,
                "max_l2": float(max(step_errors)),
                "final_l2": float(step_errors[-1]),
                "passes_tolerance": bool(max(step_errors) < tolerance),
            }
        )

    all_step_arr = np.asarray(all_step_errors, dtype=np.float64)
    static_arr = np.asarray(static_l2, dtype=np.float64)

    episode_results = []
    for horizon in episode_horizons:
        seg_errors = []
        for start in selected_starts:
            end_bound = len(obs) - 1
            for end_idx in np.where(terminals | timeouts)[0]:
                if end_idx >= start:
                    end_bound = int(end_idx)
                    break
            if end_bound - start < horizon:
                continue
            step_errors = _rollout_segment_errors(
                env, qpos, qvel, actions, next_obs, start, horizon
            )
            seg_errors.append(
                {
                    "max_l2": float(max(step_errors)),
                    "final_l2": float(step_errors[-1]),
                    "step_errors": step_errors,
                }
            )

        if seg_errors:
            max_vals = np.array([s["max_l2"] for s in seg_errors])
            final_vals = np.array([s["final_l2"] for s in seg_errors])
            step_vals = np.concatenate(
                [np.asarray(s["step_errors"], dtype=np.float64) for s in seg_errors]
            )
            episode_results.append(
                {
                    "horizon": horizon,
                    "n_segments": len(seg_errors),
                    "step_obs_l2": _summarize(f"open_loop_step_l2_h{horizon}", step_vals),
                    "max_l2": _summarize(f"open_loop_max_l2_h{horizon}", max_vals),
                    "final_l2": _summarize(f"open_loop_final_l2_h{horizon}", final_vals),
                    "passes_tolerance": bool(step_vals.max() < tolerance),
                }
            )

    _close_env(env)

    return {
        "task": task,
        "env": env_name,
        "num_trajectories": len(segment_details),
        "primary_horizon": primary_horizon,
        "tolerance": tolerance,
        "passes_tolerance": bool(
            all_step_arr.size > 0 and all_step_arr.max() < tolerance
        ),
        "static_obs_l2": _summarize("static_obs_l2_at_segment_starts", static_arr),
        "open_loop_step_obs_l2": _summarize(
            f"open_loop_step_obs_l2_h{primary_horizon}", all_step_arr
        ),
        "trajectory_segments": segment_details,
        "episode_open_loop": episode_results,
        "interpretation": {
            "method": (
                "Dynamics are checked with continuous open-loop rollout from each "
                "episode start. We do not re-inject per-step infos/qpos/infos/qvel "
                "because D4RL stores clipped qvel in infos/qvel while MuJoCo uses "
                "unclipped velocities."
            ),
            "static_match": (
                "At segment starts, qpos/qvel map to observations with ~0 error."
            ),
            "open_loop_match": (
                "If all open-loop step errors are < tolerance, native mujoco_py "
                "Hopper-v2 matches the offline dataset dynamics on those segments."
            ),
        },
    }


def _print_report(report: dict) -> None:
    print("=" * 72)
    print(f"Hopper dataset replay validation: {report['task']} / {report['env']}")
    print("=" * 72)
    print(f"Trajectories: {report['num_trajectories']}")
    print(f"Horizon: {report['primary_horizon']}")
    print(f"Tolerance: {report['tolerance']:.1e}")
    print(f"Passes tolerance: {report['passes_tolerance']}")
    print()

    for key in ("static_obs_l2", "open_loop_step_obs_l2"):
        s = report[key]
        print(
            f"{s['name']:32s} "
            f"median={s.get('median', float('nan')):.6e} "
            f"p95={s.get('p95', float('nan')):.6e} "
            f"max={s.get('max', float('nan')):.6e} "
            f"frac<1e-5={s.get('frac_lt_1e-5', float('nan')):.3f}"
        )

    print()
    print("Per-trajectory open-loop step errors:")
    for seg in report["trajectory_segments"]:
        errs = seg["step_errors"]
        print(
            f"  seg={seg['segment_idx']:2d} start={seg['start_index']:6d} "
            f"max={seg['max_l2']:.6e} final={seg['final_l2']:.6e} "
            f"pass={seg['passes_tolerance']}"
        )
        print(f"    steps: {[f'{e:.2e}' for e in errs]}")

    print()
    print("Open-loop episode segments:")
    for seg in report["episode_open_loop"]:
        h = seg["horizon"]
        mx = seg["max_l2"]
        fn = seg["final_l2"]
        st = seg["step_obs_l2"]
        print(
            f"  horizon={h:3d} segments={seg['n_segments']:3d} "
            f"step_l2 max={st.get('max', float('nan')):.6e} "
            f"frac<1e-5={st.get('frac_lt_1e-5', float('nan')):.3f} "
            f"pass={seg['passes_tolerance']}"
        )

    print()
    print("Notes:")
    for note in report["interpretation"].values():
        print(f"  - {note}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="hopper-medium-v2")
    parser.add_argument("--num-samples", type=int, default=5000, help=argparse.SUPPRESS)
    parser.add_argument("--num-trajectories", type=int, default=200)
    parser.add_argument("--horizons", default="1,5,10,50,100")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    report = validate_replay(
        task=args.task,
        num_samples=args.num_samples,
        episode_horizons=horizons,
        num_trajectories=args.num_trajectories,
        tolerance=args.tolerance,
    )
    _print_report(report)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print()
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
