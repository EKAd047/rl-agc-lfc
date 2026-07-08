# Reinforcement Learning for Single-Area Load Frequency Control

Course project comparing a PPO-trained AGC controller against a
conventional PI controller on a single-area Load Frequency Control (LFC)
environment, across nominal and reduced-inertia scenarios.

## Structure
- `environments/` — Gymnasium-compatible LFC environment (swing equation + turbine + governor dynamics)
- `controllers/` — PI baseline controller
- `scripts/` — training and evaluation scripts
- `configs/` — system parameters (inertia, damping, disturbance ranges)
- `results/` — output plots and metrics (not tracked in git except via .gitkeep)

## Status
🚧 Under development.
