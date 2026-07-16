"""
envs.py — Shared Atari environment builders used by train.py and play.py.

  CnnPolicy -> pixel frames (PongNoFrameskip-v4), 84x84 grayscale, 4-frame stack
  MlpPolicy -> 128-byte RAM state (ALE/Pong-v5, obs_type="ram"), normalized to [0, 1]
"""
import ale_py  # noqa: F401  (importing registers the ALE envs with gymnasium)
import gymnasium as gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


class NormalizeRam(gym.ObservationWrapper):
    """Scales the 128-byte Atari RAM vector from [0, 255] to [0, 1] float32."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return observation.astype(np.float32) / 255.0


def pixel_env_id(base_env_name: str) -> str:
    return f"{base_env_name}NoFrameskip-v4"


def ram_env_label(base_env_name: str) -> str:
    """Display label only - the actual env is built via gym.make(obs_type='ram') below,
    since modern ale-py no longer registers separate '-ram' env ids."""
    return f"ALE/{base_env_name}-v5 (obs_type=ram)"


def make_ram_env_fn(base_env_name: str, render_mode: str | None = None):
    """Returns a zero-arg factory building one Monitor-wrapped, normalized RAM env."""

    def _make():
        e = gym.make(
            f"ALE/{base_env_name}-v5", obs_type="ram",
            frameskip=4, repeat_action_probability=0.0, render_mode=render_mode,
        )
        e = NormalizeRam(e)
        return Monitor(e)

    return _make


def build_vec_env(base_env_name: str, policy: str, seed: int, n_envs: int = 1, render_mode: str | None = None):
    """Builds the vectorized, policy-appropriate training/play environment."""
    if policy == "CnnPolicy":
        env_kwargs = {"render_mode": render_mode} if render_mode else None
        env = make_atari_env(pixel_env_id(base_env_name), n_envs=n_envs, seed=seed, env_kwargs=env_kwargs)
        return VecFrameStack(env, n_stack=4)

    return DummyVecEnv([make_ram_env_fn(base_env_name, render_mode) for _ in range(n_envs)])


def build_eval_env(base_env_name: str, policy: str, seed: int):
    """Builds an eval env wrapped identically to how DQN wraps its training env
    (e.g. VecTransposeImage for CnnPolicy), so observations match layout exactly."""
    env = build_vec_env(base_env_name, policy, seed=seed + 1000, n_envs=1)
    return BaseAlgorithm._wrap_env(env, verbose=0)
