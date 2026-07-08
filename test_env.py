"""
Sanity check: run the LFCEnv with the PI controller active, and confirm
that delta_f is driven back toward zero after the load disturbance.
"""

from environments.lfc_env import LFCEnv
from controllers.pi_controller import PIController

env = LFCEnv()
obs, info = env.reset()

# beta = 1/R + D, the standard single-area frequency bias factor
beta = (1.0 / env.R) + env.D
pi = PIController(kp=0.05, ki=0.5, beta=beta)
pi.reset()

print(f"Initial observation: {obs}")
print(f"Sampled load disturbance P_L: {info['P_L']:.4f}")
print(f"beta (frequency bias factor): {beta:.4f}")
print()

for i in range(300):  # 3 seconds at dt=0.01
    delta_f = obs[0]
    P_ref = pi.compute(delta_f, env.dt)
    obs, reward, terminated, truncated, info = env.step([P_ref])

    if i % 20 == 0 or i == 299:  # print every 20th step to keep it readable
        print(f"Step {i+1:3d} | delta_f={obs[0]:+.5f}  P_ref={P_ref:+.5f}  reward={reward:+.6f}")

    if terminated or truncated:
        break
