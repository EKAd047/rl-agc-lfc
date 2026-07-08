"""
Single-Area Load Frequency Control (LFC) environment.
Gymnasium-compatible environment wrapping the swing-equation /
turbine / governor dynamics for a single control area.
"""

import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces


class LFCEnv(gym.Env):
    """Single-area LFC environment."""

    def __init__(self, config_path="configs/lfc_params.yaml", dt=0.01, episode_seconds=20.0):
        super().__init__()

        # --- Load system parameters from the config file ---
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        sysp = config["system"]
        self.H = sysp["H_nominal"]      # inertia constant (s) -- can be overridden for inertia sweeps
        self.D = sysp["D"]              # damping coefficient
        self.T_g = sysp["T_governor"]   # governor time constant (s)
        self.T_t = sysp["T_turbine"]    # turbine time constant (s)
        self.R = sysp["R"]              # droop

        dist = config["disturbance"]
        self.load_step_min = dist["load_step_min"]
        self.load_step_max = dist["load_step_max"]

        # --- Simulation timing ---
        self.dt = dt                              # integration timestep (s)
        self.episode_seconds = episode_seconds     # total episode duration (s)
        self.max_steps = int(episode_seconds / dt)
        self.current_step = 0

        # --- State: [delta_f, P_m, P_g] ---
        self.state = np.zeros(3, dtype=np.float32)
        self.integral_error = 0.0  # accumulated frequency error, gives the agent integral-like memory
        self.P_L = 0.0  # load disturbance, fixed for the episode, set in reset()

        # --- Observation space: [delta_f, P_m, P_g, integral_error] ---
        high = np.array([1.0, 1.0, 1.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # --- Action space: single continuous control signal P_ref ---
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)

    def set_inertia(self, H):
        """Allows changing inertia externally, e.g. for reduced-inertia experiments."""
        self.H = H

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset state to equilibrium (no deviation)
        self.state = np.zeros(3, dtype=np.float32)
        self.integral_error = 0.0
        self.current_step = 0

        # Sample a random step-load disturbance for this episode
        self.P_L = self.np_random.uniform(self.load_step_min, self.load_step_max)

        observation = np.append(self.state, self.integral_error).astype(np.float32)
        info = {"P_L": self.P_L}
        return observation, info

    def step(self, action):
        delta_f, P_m, P_g = self.state
        P_ref = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))

        # --- ODEs (Euler integration) ---
        d_delta_f = (1.0 / (2.0 * self.H)) * (P_m - self.P_L - self.D * delta_f)
        d_P_m = (1.0 / self.T_t) * (-P_m + P_g)
        d_P_g = (1.0 / self.T_g) * (-P_g - (1.0 / self.R) * delta_f + P_ref)

        delta_f += self.dt * d_delta_f
        P_m += self.dt * d_P_m
        P_g += self.dt * d_P_g

        self.state = np.array([delta_f, P_m, P_g], dtype=np.float32)
        self.integral_error += delta_f * self.dt
        self.current_step += 1

        # --- Reward: penalize frequency deviation and control effort ---
        control_effort_weight = 0.1
        reward = float(-(100.0 * delta_f ** 2 + control_effort_weight * P_ref ** 2))

        # --- Episode termination ---
        terminated = False  # no early-termination condition for now
        truncated = self.current_step >= self.max_steps

        observation = np.append(self.state, self.integral_error).astype(np.float32)
        info = {"P_L": self.P_L, "P_ref": P_ref}

        return observation, reward, terminated, truncated, info