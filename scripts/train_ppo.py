"""
Train a PPO agent on the single-area LFC environment.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from environments.lfc_env import LFCEnv


def main():
    env = LFCEnv()

    # Sanity check: confirms the environment follows the Gymnasium API correctly
    check_env(env, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )

    total_timesteps = 200_000
    model.learn(total_timesteps=total_timesteps)

    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_lfc")
    print("Training complete. Model saved to models/ppo_lfc.zip")


if __name__ == "__main__":
    main()