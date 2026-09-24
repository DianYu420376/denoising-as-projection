"""Named reference XY trajectories for unicycle plan/tracking eval."""

from __future__ import annotations

from dataclasses import dataclass

import gym
import numpy as np

import cleandiffuser.env.unicycle  # noqa: F401

# Conservative region where most offline data lives (see full64_6k stats).
DATA_XY_MARGIN = 0.75
DATA_XY_LIMIT = 6.0 - DATA_XY_MARGIN  # stay inside ~[-5.25, 5.25]

DT = 0.1
V_BOUNDS = (0.0, 2.0)
W_BOUNDS = (-1.5, 1.5)
# Reference curves are discretized over the planning horizon (full path in 64 steps).
DEFAULT_REFERENCE_STEPS = 64


@dataclass(frozen=True)
class ReferenceTrajectory:
    name: str
    family: str
    description: str
    xy: np.ndarray  # (num_steps, 2)
    init_pose: tuple[float, float, float]  # x, y, theta
    seed: int | None = None  # RNG seed for random_smooth (exact reproduction)

    def to_dict(self) -> dict:
        out = {
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "num_steps": int(self.xy.shape[0]),
            "init_pose": {
                "x": self.init_pose[0],
                "y": self.init_pose[1],
                "theta": self.init_pose[2],
            },
            "xy": self.xy.astype(float).tolist(),
            "xy_min": self.xy.min(axis=0).astype(float).tolist(),
            "xy_max": self.xy.max(axis=0).astype(float).tolist(),
        }
        if self.seed is not None:
            out["seed"] = int(self.seed)
        return out


def init_pose_from_xy(xy: np.ndarray) -> tuple[float, float, float]:
    delta = xy[1] - xy[0]
    theta = float(np.arctan2(delta[1], delta[0]))
    return float(xy[0, 0]), float(xy[0, 1]), theta


def circle_curve(
    num_steps: int,
    *,
    center: tuple[float, float] = (0.0, -0.5),
    radius: float = 2.2,
) -> np.ndarray:
    u = np.linspace(0.0, 2.0 * np.pi, num_steps, endpoint=False)
    xy = np.stack(
        [center[0] + radius * np.cos(u), center[1] + radius * np.sin(u)],
        axis=-1,
    )
    return xy.astype(np.float32)


def heart_curve(
    num_steps: int,
    *,
    scale: float = 1.0,
    center: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Parametric heart; scale sets half-extent (~scale=1.0 → ~2 m × 2 m bbox)."""
    u = np.linspace(0.0, 2.0 * np.pi, num_steps, endpoint=False)
    x = 16.0 * np.sin(u) ** 3
    y = 13.0 * np.cos(u) - 5.0 * np.cos(2.0 * u) - 2.0 * np.cos(3.0 * u) - np.cos(4.0 * u)
    xy = np.stack([x, y], axis=-1).astype(np.float64)
    xy = xy / np.max(np.abs(xy)) * scale
    xy[:, 0] += center[0]
    xy[:, 1] += center[1]
    return xy.astype(np.float32)


def half_heart_curve(
    num_steps: int,
    *,
    y_start: float = -1.0,
    y_end: float = 2.0,
    center: tuple[float, float] = (0.0, 0.0),
    scale_x: float = 2.0,
) -> np.ndarray:
    """Right half of a heart in 64 steps with y mapped from y_start to y_end."""
    u = np.linspace(np.pi, 2.0 * np.pi, num_steps, endpoint=False)
    x = 16.0 * np.sin(u) ** 3
    y = 13.0 * np.cos(u) - 5.0 * np.cos(2.0 * u) - 2.0 * np.cos(3.0 * u) - np.cos(4.0 * u)
    xy = np.stack([x, y], axis=-1).astype(np.float64)
    xy = xy / np.max(np.abs(xy))
    y0, y1 = float(xy[0, 1]), float(xy[-1, 1])
    if abs(y1 - y0) < 1e-8:
        raise ValueError("Degenerate half-heart y range")
    xy[:, 1] = y_start + (y_end - y_start) * (xy[:, 1] - y0) / (y1 - y0)
    x_mid = 0.5 * (xy[:, 0].max() + xy[:, 0].min())
    xy[:, 0] = (xy[:, 0] - x_mid) * scale_x + center[0]
    xy[:, 1] += center[1]
    return xy.astype(np.float32)


def sinusoid_curve(
    num_steps: int,
    *,
    length: float = 5.0,
    amplitude: float = 1.2,
    num_cycles: float = 1.0,
    center: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Parametric sinusoid: x advances uniformly, y oscillates."""
    t = np.linspace(0.0, 1.0, num_steps, endpoint=False)
    x = center[0] + length * (t - 0.5)
    y = center[1] + amplitude * np.sin(2.0 * np.pi * num_cycles * t)
    return np.stack([x, y], axis=-1).astype(np.float32)


def _smooth_controls(
    horizon: int,
    rng: np.random.Generator,
    smoothness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Random (v, w) with smoothness in ~[0, 1]: 0=rough, 1=very smooth."""
    t = np.arange(horizon, dtype=np.float64)
    v = rng.normal(0.0, 0.25, horizon)
    w = rng.normal(0.0, 0.35, horizon)

    n_modes = max(2, int(8 * (1.0 - 0.75 * smoothness)))
    for _ in range(n_modes):
        amp_v = rng.uniform(0.05, 0.35 * (1.0 - 0.5 * smoothness))
        amp_w = rng.uniform(0.08, 0.45 * (1.0 - 0.5 * smoothness))
        freq = rng.uniform(0.1, 0.55)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        v += amp_v * np.sin(2.0 * np.pi * freq * t / horizon + phase)
        w += amp_w * np.sin(2.0 * np.pi * freq * t / horizon + phase)

    v += rng.uniform(0.45, 1.1)
    w += rng.uniform(-0.2, 0.2)

    kernel_width = int(3 + round(10 * smoothness))
    if kernel_width % 2 == 0:
        kernel_width += 1
    kernel = np.ones(kernel_width, dtype=np.float64)
    kernel /= kernel.sum()
    v = np.convolve(v, kernel, mode="same")
    w = np.convolve(w, kernel, mode="same")

    noise_scale = 0.18 * (1.0 - smoothness)
    v += rng.normal(0.0, noise_scale, horizon)
    w += rng.normal(0.0, noise_scale * 1.2, horizon)

    n_jumps = max(0, int(6 * (1.0 - smoothness)))
    for _ in range(n_jumps):
        idx = int(rng.integers(1, horizon - 1))
        v[idx : idx + 2] += rng.uniform(-0.25, 0.25)
        w[idx : idx + 2] += rng.uniform(-0.35, 0.35)

    v = np.clip(v, *V_BOUNDS)
    w = np.clip(w, *W_BOUNDS)
    return v.astype(np.float32), w.astype(np.float32)


def random_smooth_curve(
    num_steps: int,
    *,
    smoothness: float,
    seed: int,
    init_state: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Roll out unicycle dynamics with random controls at a given smoothness level."""
    rng = np.random.default_rng(seed)
    env = gym.make(
        "Unicycle-v0",
        dt=DT,
        x_lim=(-8.0, 8.0),
        y_lim=(-8.0, 8.0),
        v_bounds=V_BOUNDS,
        w_bounds=W_BOUNDS,
        max_episode_steps=num_steps,
        terminate_on_oob=False,
    )
    env.reset(options={"initial_state": init_state})
    v_traj, w_traj = _smooth_controls(num_steps, rng, smoothness)

    xy = [env.unwrapped._state[:2].copy()]
    for t in range(num_steps - 1):
        env.step(np.array([v_traj[t], w_traj[t]], dtype=np.float32))
        xy.append(env.unwrapped._state[:2].copy())
    env.close()
    return np.asarray(xy, dtype=np.float32)


def validate_trajectory(
    xy: np.ndarray,
    *,
    xy_limit: float = DATA_XY_LIMIT,
    max_step: float = 0.22,
) -> dict:
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    in_bounds = bool((mins >= -xy_limit).all() and (maxs <= xy_limit).all())
    return {
        "in_bounds": in_bounds,
        "xy_min": mins.astype(float).tolist(),
        "xy_max": maxs.astype(float).tolist(),
        "max_segment_length": float(seg.max()) if seg.size else 0.0,
        "mean_segment_length": float(seg.mean()) if seg.size else 0.0,
        "step_length_ok": bool(seg.max() <= max_step) if seg.size else True,
    }


# Fixed (smoothness, seed) pairs — seed alone fully reproduces the rolled-out XY path.
RANDOM_SMOOTH_SPECS: list[tuple[str, float, int]] = [
    ("random_smooth_1", 0.05, 1001),
    ("random_smooth_2", 0.20, 1002),
    ("random_smooth_3", 0.35, 1003),
    ("random_smooth_4", 0.50, 1004),
    ("random_smooth_5", 0.70, 1015),
    ("random_smooth_6", 0.90, 1006),
    ("random_smooth_7", 0.15, 2008),
    ("random_smooth_8", 0.40, 2018),
    ("random_smooth_9", 0.55, 2009),
    ("random_smooth_10", 0.75, 2010),
    ("random_smooth_11", 0.85, 2011),
]


def _make_random_smooth(
    name: str,
    smoothness: float,
    seed: int,
    num_steps: int,
) -> ReferenceTrajectory:
    xy = random_smooth_curve(
        num_steps,
        smoothness=smoothness,
        seed=seed,
    )
    stats = validate_trajectory(xy)
    if not stats["in_bounds"] or not stats["step_length_ok"]:
        raise ValueError(
            f"Random trajectory {name} (seed={seed}, smoothness={smoothness}) "
            f"failed validation: {stats}"
        )
    return ReferenceTrajectory(
        name=name,
        family="random_smooth",
        description=(
            f"Rolled-out random controls (smoothness={smoothness:.2f}, seed={seed})"
        ),
        xy=xy,
        init_pose=init_pose_from_xy(xy),
        seed=seed,
    )


def build_reference_catalog(num_steps: int = DEFAULT_REFERENCE_STEPS) -> list[ReferenceTrajectory]:
    """Build circle, half-heart, and seed-fixed random trajectories (64 steps each)."""
    trajectories: list[ReferenceTrajectory] = []

    circle_xy = circle_curve(num_steps, center=(0.0, -0.5), radius=2.0)
    stats = validate_trajectory(circle_xy)
    if not stats["in_bounds"] or not stats["step_length_ok"]:
        raise ValueError(f"Circle failed validation: {stats}")
    trajectories.append(
        ReferenceTrajectory(
            name="circle",
            family="circle",
            description="Full circle in 64 steps, r=2.0 m centered at (0, -0.5)",
            xy=circle_xy,
            init_pose=init_pose_from_xy(circle_xy),
        )
    )

    half_heart_xy = half_heart_curve(num_steps, y_start=-1.0, y_end=2.0, center=(0.0, 0.0), scale_x=2.0)
    stats = validate_trajectory(half_heart_xy)
    if not stats["in_bounds"] or not stats["step_length_ok"]:
        raise ValueError(f"Half heart failed validation: {stats}")
    trajectories.append(
        ReferenceTrajectory(
            name="half_heart",
            family="heart",
            description="Half heart in 64 steps, y from -1 to 2, scale_x=2.0",
            xy=half_heart_xy,
            init_pose=init_pose_from_xy(half_heart_xy),
        )
    )

    for name, smoothness, seed in RANDOM_SMOOTH_SPECS:
        trajectories.append(_make_random_smooth(name, smoothness, seed, num_steps))

    return trajectories


def catalog_names() -> list[str]:
    return ["circle", "half_heart", *[spec[0] for spec in RANDOM_SMOOTH_SPECS]]


def get_reference_trajectory(name: str, num_steps: int = DEFAULT_REFERENCE_STEPS) -> ReferenceTrajectory:
    catalog = {traj.name: traj for traj in build_reference_catalog(num_steps)}
    if name not in catalog:
        raise KeyError(
            f"Unknown reference trajectory {name!r}. "
            f"Known: {sorted(catalog.keys())}"
        )
    return catalog[name]
