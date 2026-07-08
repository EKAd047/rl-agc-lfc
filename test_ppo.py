"""
Quick sanity check: run the trained PPO agent for one episode and
confirm delta_f gets driven back toward zero, similar to the PI test.
"""

from stable_baselines3 import PPO
from environments.lfc_env import LFCEnv

env = LFCEnv()
model = PPO.load("models/ppo_lfc")

obs, info = env.reset()

print(f"Initial observation: {obs}")
print(f"Sampled load disturbance P_L: {info['P_L']:.4f}")
print()

for i in range(2000):  # 3 seconds at dt=0.01
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if i % 20 == 0 or i == 299:
        print(f"Step {i+1:3d} | delta_f={obs[0]:+.5f}  P_ref={action[0]:+.5f}  reward={reward:+.6f}")

    if terminated or truncated:
        break
