"""Gravity-compensation torque mode against bar_ros2 via raw DDS.

Per joint: ``effort = mj_qfrc_applied`` (clipped), ``K=0``,
``D=TORQUE_DAMPING``. The actuator's MIT formula reduces to
``τ = −D·qdot + effort``, i.e. gravity-cancelling torque plus a
passive joint-space viscous brake.

Self-contained example: the drain → mirror-into-MuJoCo → publish loop
lives here in full (no shared base class) so the script reads top to
bottom. ``gravity.py`` provides only the MuJoCo model + gravity math.
"""

import os
import time

import lite_sdk2
import mujoco
import numpy as np
from lite_sdk2 import Header, JointState, MITCommand
from lite_sdk2 import Time as DdsTime
from loop_rate_limiters import RateLimiter

from gravity import (
    COMMAND_HZ,
    LITE_ARM_JOINTS,
    STATUS_HZ,
    TORQUE_DAMPING,
    TORQUE_MAX_TORQUE,
    apply_viscous_damping,
    build_joint_info,
    build_published_joint_mapping,
    compensate_gravity,
    load_model_path,
    world_child_subtree_ids,
)

JOINT_STATE_TOPIC = "/lite/joint_states"
COMMAND_TOPIC = "/remote_policy_controller/command"


def now_time() -> DdsTime:
    t = time.time_ns()
    return DdsTime(sec=int(t // 1_000_000_000), nanosec=int(t % 1_000_000_000))


def make_command(
    joint_names: list[str],
    position: list[float],
    velocity: list[float],
    effort: list[float],
    stiffness: list[float],
    damping: list[float],
) -> MITCommand:
    return MITCommand(
        header=Header(stamp=now_time()),
        joint_names=list(joint_names),
        position=position,
        velocity=velocity,
        effort=effort,
        stiffness=stiffness,
        damping=damping,
    )


class TorqueController:
    """Gravity-compensation torque-mode controller.

    Each tick: drain ``/lite/joint_states``, mirror the snapshot into
    MuJoCo, compute ``qfrc_applied`` for gravity cancellation, and publish
    the per-joint torque (plus a viscous brake) as a ``MITCommand``.
    """

    def __init__(self, domain_id: int) -> None:
        self.model_path = load_model_path()
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        joint_info = build_joint_info(self.model)
        self.subtree_ids = world_child_subtree_ids(self.model)
        if not self.subtree_ids:
            raise RuntimeError("MuJoCo model has no world-rooted subtrees for gravity comp.")

        self.joint_names = list(LITE_ARM_JOINTS)
        self.mapping = build_published_joint_mapping(self.joint_names, joint_info)
        self.dof_ids = np.array([info.dof_index for _, info in self.mapping], dtype=int)
        self.joint_index = {name: i for i, name in enumerate(self.joint_names)}
        self.latest_position = np.full(len(self.joint_names), np.nan)
        self.latest_velocity = np.zeros(len(self.joint_names))
        self.joint_state_seen = False
        self.last_torques = np.zeros(len(self.joint_names))

        # Topic + QoS come from the lite_sdk2 registry (JointState: reliable
        # keep-last 10; MITCommand: reliable keep-last 4) — matching the bringup.
        lite_sdk2.initialize(domain_id=domain_id)
        self.sub = lite_sdk2.subscriber(JointState)
        self.sub.initialize()
        self.pub = lite_sdk2.publisher(MITCommand)
        self.pub.initialize()

        self.last_status_t = 0.0
        self.status_period = 1.0 / STATUS_HZ

        print(
            f"DDS domain={domain_id}  mode=torque  joints={len(self.joint_names)}  "
            f"rate={COMMAND_HZ:.0f} Hz  model={self.model_path}"
        )

    def drain_joint_state(self) -> None:
        for sample in self.sub.read_batch(max_samples=64):
            for i, name in enumerate(sample.name):
                idx = self.joint_index.get(name)
                if idx is None:
                    continue
                if i < len(sample.position):
                    self.latest_position[idx] = sample.position[i]
                if i < len(sample.velocity):
                    self.latest_velocity[idx] = sample.velocity[i]
            if not self.joint_state_seen and not np.any(np.isnan(self.latest_position)):
                self.joint_state_seen = True
                print(f"First complete {JOINT_STATE_TOPIC} received.")

    def refresh_mujoco_state(self) -> None:
        # Joints outside our published set (e.g. neck) stay at zero — they
        # still contribute to the gravity model's mass distribution, but
        # zero is the best neutral guess without a live reading.
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        for pub_idx, info in self.mapping:
            self.data.qpos[info.qpos_index] = self.latest_position[pub_idx]
            self.data.qvel[info.dof_index] = self.latest_velocity[pub_idx]
        mujoco.mj_forward(self.model, self.data)
        compensate_gravity(self.model, self.data, self.subtree_ids)

    def build_command(self) -> MITCommand:
        if self.dof_ids.size:
            apply_viscous_damping(self.data, self.dof_ids, TORQUE_DAMPING)
        n = len(self.joint_names)
        effort = [0.0] * n
        for pub_idx, info in self.mapping:
            torque = float(self.data.qfrc_applied[info.dof_index])
            torque = float(np.clip(torque, -TORQUE_MAX_TORQUE, TORQUE_MAX_TORQUE))
            effort[pub_idx] = torque
            self.last_torques[pub_idx] = torque
        return make_command(
            joint_names=self.joint_names,
            position=[0.0] * n,
            velocity=[0.0] * n,
            effort=effort,
            stiffness=[0.0] * n,
            damping=[TORQUE_DAMPING] * n,
        )

    def print_status(self) -> None:
        torques = self.last_torques
        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"torque mean={float(np.mean(np.abs(torques))):.3f}  max={float(np.max(np.abs(torques))):.3f}"
        )

    def passive_command(self) -> MITCommand:
        """Safe shutdown: zero stiffness, small damping, coast under gravity."""
        n = len(self.joint_names)
        return make_command(
            joint_names=self.joint_names,
            position=[0.0] * n,
            velocity=[0.0] * n,
            effort=[0.0] * n,
            stiffness=[0.0] * n,
            damping=[TORQUE_DAMPING] * n,
        )

    def run(self) -> None:
        rate = RateLimiter(frequency=COMMAND_HZ, warn=False)
        try:
            while True:
                self.drain_joint_state()
                if self.joint_state_seen:
                    self.refresh_mujoco_state()
                    self.pub.write(self.build_command())
                    now = time.monotonic()
                    if now - self.last_status_t >= self.status_period:
                        self.last_status_t = now
                        self.print_status()
                rate.sleep()
        except KeyboardInterrupt:
            print(
                "\nKeyboardInterrupt — publishing passive shutdown command. "
                "Drive the FSM to DAMPING (gamepad X) for a real stop."
            )
            self.pub.write(self.passive_command())


def main() -> None:
    domain_id = int(os.environ.get("ROS_DOMAIN_ID", "0"))
    TorqueController(domain_id=domain_id).run()


if __name__ == "__main__":
    main()
