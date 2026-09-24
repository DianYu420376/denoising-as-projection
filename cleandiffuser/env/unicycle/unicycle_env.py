"""Minimal unicycle (differential-drive) dynamics in Gym.

Continuous-time model discretized with step size ``dt`` (η in the notes):

    x_{t+1}     = x_t + dt * v_t * cos(theta_t)
    y_{t+1}     = y_t + dt * v_t * sin(theta_t)
    theta_{t+1} = theta_t + dt * w_t

Observation: [x, y, cos(theta), sin(theta)]
Action:      [v, w]  (linear and angular velocity)
"""

from __future__ import annotations

import math

import gym
import numpy as np
from gym import spaces


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


class UnicycleEnv(gym.Env):
    """Simple 2D unicycle for offline trajectory generation."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 0.1,
        x_lim: tuple[float, float] = (-8.0, 8.0),
        y_lim: tuple[float, float] = (-8.0, 8.0),
        v_bounds: tuple[float, float] = (0.0, 2.0),
        w_bounds: tuple[float, float] = (-1.5, 1.5),
        max_episode_steps: int = 64,
        terminate_on_oob: bool = True,
    ):
        super().__init__()
        self.dt = float(dt)
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.v_bounds = v_bounds
        self.w_bounds = w_bounds
        self.max_episode_steps = int(max_episode_steps)
        self.terminate_on_oob = bool(terminate_on_oob)

        x_lo, x_hi = x_lim
        y_lo, y_hi = y_lim
        self.observation_space = spaces.Box(
            low=np.array([x_lo, y_lo, -1.0, -1.0], dtype=np.float32),
            high=np.array([x_hi, y_hi, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array(v_bounds[:1] + w_bounds[:1], dtype=np.float32),
            high=np.array(v_bounds[1:] + w_bounds[1:], dtype=np.float32),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng()
        self._state = np.zeros(3, dtype=np.float64)
        self._elapsed = 0

    def seed(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    @property
    def theta(self) -> float:
        return float(self._state[2])

    def set_state(self, x: float, y: float, theta: float) -> None:
        self._state[:] = (x, y, _wrap_angle(theta))
        self._elapsed = 0

    def _get_obs(self) -> np.ndarray:
        x, y, theta = self._state
        return np.array([x, y, math.cos(theta), math.sin(theta)], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed(seed)
        opts = options or {}

        if "initial_state" in opts:
            x, y, theta = opts["initial_state"]
            self.set_state(x, y, theta)
        else:
            margin = 0.5
            x = self._rng.uniform(self.x_lim[0] + margin, self.x_lim[1] - margin)
            y = self._rng.uniform(self.y_lim[0] + margin, self.y_lim[1] - margin)
            theta = self._rng.uniform(-np.pi, np.pi)
            self.set_state(x, y, theta)

        return self._get_obs()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        v = float(np.clip(action[0], *self.v_bounds))
        w = float(np.clip(action[1], *self.w_bounds))

        x, y, theta = self._state
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        x_next = x + self.dt * v * cos_t
        y_next = y + self.dt * v * sin_t
        theta_next = _wrap_angle(theta + self.dt * w)

        self._state[:] = (x_next, y_next, theta_next)
        self._elapsed += 1

        in_bounds = (
            self.x_lim[0] <= x_next <= self.x_lim[1]
            and self.y_lim[0] <= y_next <= self.y_lim[1]
        )
        terminated = (not in_bounds) if self.terminate_on_oob else False
        truncated = self._elapsed >= self.max_episode_steps
        done = terminated or truncated

        info = {"in_bounds": in_bounds, "terminated": terminated, "truncated": truncated}
        return self._get_obs(), 0.0, done, info
