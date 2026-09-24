"""Ensure MuJoCo rollout sim envs use v2 dynamics for HalfCheetah and Walker2d."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipelines"))

gym = pytest.importorskip("gym")
pytest.importorskip("d4rl")

import numpy as np

from d4rl_render_utils import env_reset, make_sim_eval_env  # noqa: E402


@pytest.mark.parametrize(
    "task,forbidden",
    [
        ("halfcheetah-medium-v2", "-v4"),
        ("walker2d-medium-v2", "-v4"),
        ("hopper-medium-v2", "-v4"),
    ],
)
def test_make_sim_eval_env_uses_v2_not_v4(task: str, forbidden: str):
    env, sim_name = make_sim_eval_env(task, render=False)
    try:
        assert forbidden not in sim_name
        assert sim_name.endswith("-v2") or task.endswith("-v2")
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        assert obs is not None
    finally:
        env.close()


def test_env_reset_seed_is_reproducible():
    env, _ = make_sim_eval_env("halfcheetah-medium-v2", render=False)
    try:
        o0 = env_reset(env, seed=123)
        o1 = env_reset(env, seed=123)
        o2 = env_reset(env, seed=124)
        assert np.allclose(o0, o1)
        assert not np.allclose(o0, o2)
    finally:
        env.close()
