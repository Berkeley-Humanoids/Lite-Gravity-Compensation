"""Gravity-compensation PD mode against bar_ros2 via raw DDS.

Per joint: ``position = q + gravity/Kp`` (clipped), ``K=PD_POSITION_KP``,
``D=PD_VELOCITY_KD``. The actuator's MIT formula
``τ = K·(pos_cmd − q) + D·(vel_cmd − qdot) + effort`` reduces to
``K · gravity/K = gravity`` at the joint, while keeping the actuator's
local PD loop on the wire for high-frequency disturbance rejection.
"""

import os
import time

import numpy as np

from lite_sdk2 import MITCommand
from gravity import PD_MAX_POSITION_OFFSET, PD_MAX_TORQUE, PD_POSITION_KP, PD_VELOCITY_KD
from runner import GravityRunner, make_command


class PDRunner(GravityRunner):
    label = "pd"

    def __init__(self, domain_id: int) -> None:
        super().__init__(domain_id)
        self.last_torques = np.zeros(len(self.joint_names))
        self.last_offsets = np.zeros(len(self.joint_names))

    def build_command(self) -> MITCommand:
        n = len(self.joint_names)
        position = [0.0] * n
        for pub_idx, info in self.mapping:
            torque = float(np.clip(self.data.qfrc_applied[info.dof_index], -PD_MAX_TORQUE, PD_MAX_TORQUE))
            offset = float(np.clip(torque / PD_POSITION_KP, -PD_MAX_POSITION_OFFSET, PD_MAX_POSITION_OFFSET))
            q = float(self.latest_position[pub_idx])
            target = float(np.clip(q + offset, info.lower, info.upper))
            position[pub_idx] = target
            self.last_torques[pub_idx] = torque
            self.last_offsets[pub_idx] = target - q
        return make_command(
            joint_names=self.joint_names,
            position=position,
            velocity=[0.0] * n,
            effort=[0.0] * n,
            stiffness=[PD_POSITION_KP] * n,
            damping=[PD_VELOCITY_KD] * n,
        )

    def print_status(self) -> None:
        t, o = self.last_torques, self.last_offsets
        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"gravity mean={float(np.mean(np.abs(t))):.3f}  max={float(np.max(np.abs(t))):.3f}  "
            f"offset mean={float(np.mean(np.abs(o))):.4f}  max={float(np.max(np.abs(o))):.4f}"
        )


def main() -> None:
    domain_id = int(os.environ.get("ROS_DOMAIN_ID", "0"))
    PDRunner(domain_id=domain_id).run()


if __name__ == "__main__":
    main()
