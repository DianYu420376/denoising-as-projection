"""Helpers for D4RL MuJoCo rollout rendering on headless clusters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import gym
import numpy as np

try:
    from d4rl.offline_env import OfflineEnv
except Exception:  # pragma: no cover - d4rl may be partially unavailable
    OfflineEnv = tuple()

GymEnv = Union[gym.Env, object]

# Standalone OpenAI Gym v2 MuJoCo envs (mujoco_py) used only when the D4RL task
# env cannot be instantiated for rollout on this machine.
D4RL_MUJOCO_V2_GYM_FALLBACK = {
    "halfcheetah": "HalfCheetah-v2",
    "walker2d": "Walker2d-v2",
    "hopper": "Hopper-v2",
    "ant": "Ant-v2",
}

# D4RL v2 offline tasks whose rollouts should use matching mujoco_py dynamics
# (the task env itself, e.g. halfcheetah-medium-v2), not Gymnasium v4.
D4RL_MUJOCO_NATIVE_GYM_TASKS = {
    "hopper",
    "halfcheetah",
    "walker2d",
}


def parse_mujoco_agent(env_name: str) -> str:
    for agent in D4RL_MUJOCO_V2_GYM_FALLBACK:
        if env_name.startswith(agent):
            return agent
    raise ValueError(f"Cannot infer MuJoCo agent from task name: {env_name}")


def is_offline_d4rl_env(env: GymEnv) -> bool:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    if OfflineEnv and isinstance(unwrapped, OfflineEnv):
        return True
    return "Offline" in type(unwrapped).__name__


def resolve_ckpt_stem(ckpt: str) -> str:
    if ckpt in ("latest", "newest"):
        return "latest"
    if ckpt.isdigit():
        return ckpt
    return ckpt


def setup_headless_rendering():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", cuda_device.split(",")[0])


def gym_v2_sim_env_name(d4rl_task_name: str) -> str:
    """Return standalone OpenAI Gym v2 MuJoCo env for a D4RL v2 offline task."""
    agent = parse_mujoco_agent(d4rl_task_name)
    if agent in D4RL_MUJOCO_V2_GYM_FALLBACK:
        return D4RL_MUJOCO_V2_GYM_FALLBACK[agent]
    raise ValueError(f"No Gym v2 sim env mapping for task: {d4rl_task_name}")


def make_sim_eval_env(
    d4rl_task_name: str,
    sim_env_name: Optional[str] = None,
    render: bool = False,
    render_width: int = 480,
    render_height: int = 480,
    ignore_termination: bool = False,
) -> Tuple[GymEnv, str]:
    """Create a physics env for rollout/render.

    Prefer D4RL v2 task envs (mujoco_py) that match the offline dataset dynamics.
    If those are unavailable, fall back to standalone Gym v2 envs (HalfCheetah-v2, etc.).
    """
    if sim_env_name:
        return _make_env(
            sim_env_name, render, render_width, render_height, ignore_termination=ignore_termination
        ), sim_env_name

    probe = gym.make(d4rl_task_name)
    if not is_offline_d4rl_env(probe):
        probe.close()
        return _make_env(
            d4rl_task_name, render, render_width, render_height, ignore_termination=ignore_termination
        ), d4rl_task_name

    probe.close()
    agent = parse_mujoco_agent(d4rl_task_name)
    if agent in D4RL_MUJOCO_NATIVE_GYM_TASKS:
        sim_name = gym_v2_sim_env_name(d4rl_task_name)
        print(
            f"[render] Task `{d4rl_task_name}` is offline-only; "
            f"using Gym v2 sim env `{sim_name}` for rollout."
        )
        return _make_env(
            sim_name,
            render,
            render_width,
            render_height,
            backend="gym",
            ignore_termination=ignore_termination,
        ), sim_name

    sim_name = D4RL_MUJOCO_V2_GYM_FALLBACK[agent]
    print(
        f"[render] Task `{d4rl_task_name}` is offline-only; "
        f"using Gym v2 sim env `{sim_name}` for rollout."
    )
    return _make_env(
        sim_name,
        render,
        render_width,
        render_height,
        backend="gym",
        ignore_termination=ignore_termination,
    ), sim_name


class ContinuousLocomotionWrapper:
    """Keep locomotion rollouts running without episode cuts on fall/timeout."""

    def __init__(self, env: GymEnv):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return obs, reward, False, False, info
        obs, reward, done, info = out
        return obs, reward, False, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()


def _make_env(
    env_name: str,
    render: bool,
    render_width: int,
    render_height: int,
    backend: str = "gym",
    ignore_termination: bool = False,
) -> GymEnv:
    if backend == "gymnasium":
        if env_name in ("Hopper-v2-compat", "hopper-v2-compat"):
            from hopper_v2_compat import make_hopper_v2_compat_env

            env = make_hopper_v2_compat_env(
                render=render,
                render_width=render_width,
                render_height=render_height,
            )
        else:
            import gymnasium as gymnasium

            render_mode = "rgb_array" if render else None
            env = gymnasium.make(
                env_name,
                render_mode=render_mode,
                width=render_width,
                height=render_height,
            )
        if ignore_termination:
            env = ContinuousLocomotionWrapper(env)
        return env

    env = gym.make(env_name)
    if ignore_termination:
        env = ContinuousLocomotionWrapper(env)
    if render and hasattr(env, "metadata"):
        env.metadata.setdefault("render.modes", [])
    return env


def env_reset(env: GymEnv, seed: Optional[int] = None) -> np.ndarray:
    """Reset env and return initial observation.

    When ``seed`` is set, uses ``env.reset(seed=seed)`` when supported, otherwise
    ``env.seed(seed)`` followed by ``env.reset()`` for legacy Gym/mujoco_py envs.
    """
    if seed is not None:
        try:
            out = env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
            out = env.reset()
    else:
        out = env.reset()
    if isinstance(out, tuple):
        return np.asarray(out[0], dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def env_step(env: GymEnv, action: np.ndarray):
    action = np.asarray(action, dtype=np.float32)
    if action.ndim > 1:
        action = action[0]
    out = env.step(action)
    if len(out) == 5:
        obs, rew, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return np.asarray(obs, dtype=np.float32), float(rew), done, info
    obs, rew, done, info = out
    return np.asarray(obs, dtype=np.float32), float(rew), bool(done), info


def _frame_to_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.dtype == np.uint8:
        return frame
    if frame.max() <= 1.0:
        return (255.0 * frame).astype(np.uint8)
    return frame.astype(np.uint8)


def capture_frame(env: GymEnv, width: int, height: int) -> np.ndarray:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env

    if hasattr(unwrapped, "model") and hasattr(unwrapped, "data"):
        import mujoco

        cache_attr = "_cd_mujoco_renderer"
        renderer = getattr(unwrapped, cache_attr, None)
        if renderer is None:
            renderer = mujoco.Renderer(unwrapped.model, height=height, width=width)
            setattr(unwrapped, cache_attr, renderer)
        renderer.update_scene(unwrapped.data, camera="track")
        return _frame_to_uint8(renderer.render())

    if hasattr(env, "render"):
        try:
            frame = env.render()
        except TypeError:
            frame = env.render(mode="rgb_array")
        if frame is not None:
            return _frame_to_uint8(frame)

    if hasattr(unwrapped, "sim") and hasattr(unwrapped.sim, "render"):
        return _frame_to_uint8(unwrapped.sim.render(width=width, height=height))

    raise RuntimeError("Environment does not support rgb_array rendering.")


def make_video_writer(output_path: Path, fps: int):
    import imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(output_path), fps=fps, macro_block_size=1)


def default_video_path(video_dir: str, pipeline_name: str, task_name: str, episode_idx: int) -> Path:
    return Path(video_dir) / pipeline_name / task_name / f"episode_{episode_idx:03d}.mp4"
