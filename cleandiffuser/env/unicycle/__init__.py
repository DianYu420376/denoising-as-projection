from gym.envs.registration import register

from cleandiffuser.env.unicycle.unicycle_env import UnicycleEnv

register(
    id="Unicycle-v0",
    entry_point="cleandiffuser.env.unicycle.unicycle_env:UnicycleEnv",
    max_episode_steps=64,
)

__all__ = ["UnicycleEnv"]
