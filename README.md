# Lite-Gravity-Compensation

Gravity-compensation runner for the Berkeley Humanoid Lite. Talks to
[`bar_ros2`](https://github.com/Berkeley-Humanoids/bar_ros2)'s
`RemotePolicyController` over **raw CycloneDDS** — no `rclpy`, no
colcon sourcing, no `--system-site-packages`.

Per tick:

1. Drain `/lite/joint_states` into a 14-joint snapshot.
2. Mirror the snapshot into MuJoCo; `mj_forward` + `mj_jacSubtreeCom`
   compute gravity-cancelling generalized forces.
3. Build a `bar_msgs/MITCommand` and publish on
   `/remote_policy_controller/command`. The in-process
   `bar::RemotePolicyController` writes the five MIT command
   interfaces per joint.

MuJoCo here is a dynamics model only — no stepping.

## Modes

| Script | Per joint |
|---|---|
| `run_ros2_torque.py` | `effort = mj_qfrc_applied` (clipped), `K=0`, `D=TORQUE_DAMPING` |
| `run_ros2_pd.py` | `position = q + gravity/Kp` (clipped), `K=PD_POSITION_KP`, `D=PD_VELOCITY_KD` |
| `run_mujoco.py` | Pure MuJoCo viewer demo, no DDS |

Both ROS 2 runners publish in `LITE_ARM_JOINTS` order
(`bar_bringup_lite/config/lite_hardware.yaml` `arm_joints`).
`RemotePolicyController` rejects joint-order mismatches.

## File layout

| File | Role |
|---|---|
| `bar_dds.py` | IdlStruct mirrors of `Time`/`Header`/`MITCommand`/`JointState`, ROS↔DDS topic+type mangling, `DdsContext` |
| `gravity.py` | MuJoCo helpers, joint list, tuning constants |
| `runner.py` | Shared `GravityRunner` base — drain → MuJoCo → publish loop |
| `run_ros2_torque.py` | Torque-mode subclass + main |
| `run_ros2_pd.py` | PD-mode subclass + main |
| `run_mujoco.py` | Standalone MuJoCo demo |

## How this bypasses rclpy

ROS 2 messages are CDR-serialized DDS types. Three name conventions get
the wire endpoints to meet:

- **Topic prefix.** `/foo/bar` → DDS topic `rt/foo/bar`.
- **Type-name namespace.** `pkg/msg/Name` → DDS type
  `pkg::msg::dds_::Name_`.
- **QoS.** RELIABLE + KEEP_LAST + VOLATILE on both sides.

`bar_dds.py` encodes all three. CycloneDDS-python on this side
interoperates with either `rmw_cyclonedds_cpp` or `rmw_fastrtps_cpp` on
the bringup — both speak RTPS-over-UDP with CDR. No
`RMW_IMPLEMENTATION` env override needed.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

(Or `uv sync && source .venv/bin/activate` if you prefer — the env is
plain pip.)

## Run

```bash
# Terminal A — bringup (real or sim):
ros2 launch bar_bringup_lite real.launch.py     # or mujoco.launch.py

# Drive the FSM into REMOTE. Either via gamepad (X → L1+A →
# wait for standby → R1+A) or by hand:
ros2 control switch_controllers --deactivate zero_torque_controller \
    --activate damping_controller
ros2 control switch_controllers --deactivate damping_controller \
    --activate standby_controller
# wait for /standby_controller/state.is_finished == true
ros2 control switch_controllers --deactivate standby_controller \
    --activate remote_policy_controller

# Terminal B — runner (in this venv):
python run_ros2_torque.py   # or run_ros2_pd.py
```

`ROS_DOMAIN_ID` defaults to 0 on both sides; set it explicitly if
you've moved off 0.

## Safe stop

Ctrl-C publishes one final passive command (`K=0`, `D=TORQUE_DAMPING`,
`effort=0`) — actuator coasts under damping. For a real stop, drive
the FSM back to DAMPING from the gamepad. The runner exiting just
releases the topic; `RemotePolicyController`'s stale-command fallback
(100 ms) catches it as fault recovery, not normal shutdown.

## Pure-MuJoCo demo

```bash
python run_mujoco.py
```

Opens the `lite_dummy` MJCF in MuJoCo's passive viewer with gravity
compensation applied in-process. No DDS, no robot.
