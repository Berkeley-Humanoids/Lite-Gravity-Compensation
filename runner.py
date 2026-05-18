"""Shared runner base for gravity-compensation modes.

Each tick:
    1. Drain ``/lite/joint_states`` into a per-joint snapshot.
    2. Mirror the snapshot into MuJoCo, compute ``qfrc_applied`` for
       gravity cancellation.
    3. Subclass ``build_command()`` turns that into a ``MITCommand``.
    4. Publish on ``/remote_policy_controller/command``.
"""

import time

import mujoco
import numpy as np
from loop_rate_limiters import RateLimiter

from bar_dds import DdsContext, Header, JointState, MITCommand, reliable_keep_last
from bar_dds import Time as DdsTime
from gravity import (
    COMMAND_HZ,
    LITE_ARM_JOINTS,
    STATUS_HZ,
    TORQUE_DAMPING,
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


class GravityRunner:
    """Common loop. Subclasses implement ``build_command`` + ``shutdown_command``."""

    label: str = "gravity"

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

        self.dds = DdsContext(domain_id=domain_id)
        self.reader = self.dds.reader(JOINT_STATE_TOPIC, JointState, reliable_keep_last(10))
        self.writer = self.dds.writer(COMMAND_TOPIC, MITCommand, reliable_keep_last(4))

        self.last_status_t = 0.0
        self.status_period = 1.0 / STATUS_HZ

        print(
            f"DDS domain={domain_id}  mode={self.label}  joints={len(self.joint_names)}  "
            f"rate={COMMAND_HZ:.0f} Hz  model={self.model_path}"
        )

    def drain_joint_state(self) -> None:
        for sample in self.reader.take(N=64):
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

    def tick(self) -> None:
        self.drain_joint_state()
        if not self.joint_state_seen:
            return
        self.refresh_mujoco_state()
        self.writer.write(self.build_command())
        self.maybe_status()

    def maybe_status(self) -> None:
        now = time.monotonic()
        if now - self.last_status_t < self.status_period:
            return
        self.last_status_t = now
        self.print_status()

    def run(self) -> None:
        rate = RateLimiter(frequency=COMMAND_HZ, warn=False)
        try:
            while True:
                self.tick()
                rate.sleep()
        except KeyboardInterrupt:
            print(
                "\nKeyboardInterrupt — publishing passive shutdown command. "
                "Drive the FSM to DAMPING (gamepad X) for a real stop."
            )
            self.writer.write(self.passive_command())

    def passive_command(self) -> MITCommand:
        """Universal safe shutdown: zero stiffness, small damping, coast.

        Works for both modes regardless of whether we ever saw a joint
        state — no dependency on ``latest_position``.
        """
        n = len(self.joint_names)
        return make_command(
            joint_names=self.joint_names,
            position=[0.0] * n,
            velocity=[0.0] * n,
            effort=[0.0] * n,
            stiffness=[0.0] * n,
            damping=[TORQUE_DAMPING] * n,
        )

    # ---- subclass surface ----------------------------------------------

    def build_command(self) -> MITCommand:
        raise NotImplementedError

    def print_status(self) -> None:
        pass
