# Gravity Compensation Demo

Small MuJoCo + Lite-SDK2 demos for:

- streaming robot joint state into MuJoCo
- estimating gravity compensation torque
- sending either direct torque commands or PD targets back to the robot

## Setup

Requires Python 3.11+ and `uv`.

```bash
uv sync
```

## Run

Real robot torque demo:

```bash
uv run python run_real_torque.py
```

Real robot PD demo:

```bash
uv run python run_real_pd.py
```

State visualization:

```bash
uv run python visualize_real.py
```

Mujoco-only simulation demo:

```bash
uv run python run_sim.py
```
