"""Hopper env aligned with OpenAI Gym / D4RL hopper-v2 control logic.

D4RL ``hopper-medium-v2`` data was collected with ``gym.envs.mujoco.HopperEnv``
(mujoco_py + MuJoCo 2.1). On modern Gymnasium + MuJoCo 3.x we cannot load the
original XML (``coordinate="global"`` was removed). Gymnasium ships a converted
``hopper.xml`` that is the official migration of the same model.

This module uses that converted XML but applies the **legacy Gym hopper** reward,
termination, observation, and reset rules exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

_BUNDLED_HOPPER_XML = (
    Path(__file__).resolve().parent.parent / "assets/mujoco/gymnasium_hopper_v3_compat.xml"
)


def _resolve_hopper_xml() -> str:
    if _BUNDLED_HOPPER_XML.exists():
        return str(_BUNDLED_HOPPER_XML)
    import gymnasium

    return str(
        Path(gymnasium.__file__).resolve().parent / "envs/mujoco/assets/hopper.xml"
    )


class HopperV2CompatEnv(MujocoEnv, utils.EzPickle):
    """Legacy Gym Hopper control logic on Gymnasium's migrated hopper model."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
        "render_fps": 125,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        width: int = 480,
        height: int = 480,
        xml_path: str | Path | None = None,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, render_mode, width, height, xml_path, **kwargs)
        model_path = str(xml_path or _resolve_hopper_xml())
        observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            model_path,
            4,
            observation_space=observation_space,
            render_mode=render_mode,
            width=width,
            height=height,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def _get_obs(self) -> np.ndarray:
        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        """D4RL/Gym hopper observation: qpos[1:] plus clipped qvel."""
        return np.concatenate(
            [self.data.qpos.flat[1:], np.clip(self.data.qvel.flat, -10, 10)]
        )

    @staticmethod
    def observation_from_qpos_qvel(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64)
        qvel = np.asarray(qvel, dtype=np.float64)
        return np.concatenate([qpos.flat[1:], np.clip(qvel.flat, -10, 10)])

    def step(self, action):
        posbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        posafter, height, ang = self.data.qpos[0:3]

        reward = (posafter - posbefore) / self.dt
        reward += 1.0
        reward -= 1e-3 * np.square(action).sum()

        state = self.state_vector()
        terminated = not (
            np.isfinite(state).all()
            and (np.abs(state[2:]) < 100).all()
            and (height > 0.7)
            and (abs(ang) < 0.2)
        )

        observation = self._get_obs()
        info = {
            "x_position": float(posafter),
            "x_velocity": float((posafter - posbefore) / self.dt),
        }

        if self.render_mode == "human":
            self.render()

        return observation, float(reward), bool(terminated), False, info

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        return self._get_obs()


def make_hopper_v2_compat_env(
    render: bool = False,
    render_width: int = 480,
    render_height: int = 480,
) -> HopperV2CompatEnv:
    render_mode = "rgb_array" if render else None
    return HopperV2CompatEnv(
        render_mode=render_mode,
        width=render_width,
        height=render_height,
    )
