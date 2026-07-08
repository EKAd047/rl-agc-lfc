"""
Single-Area Load Frequency Control (LFC) environment.
Gymnasium-compatible environment wrapping the swing-equation /
turbine / governor dynamics for a single control area.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class LFCEnv(gym.Env):
    """Single-area LFC environment (to be implemented)."""

    def __init__(self):
        super().__init__()
        # TODO: define observation_space and action_space
        pass

    def reset(self, seed=None, options=None):
        # TODO: reset state, sample a random step-load disturbance
        pass

    def step(self, action):
        # TODO: integrate one timestep of the ODEs, compute reward
        pass
