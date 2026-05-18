"""Gravity-compensation torque mode against bar_ros2 via raw DDS.

Per joint: ``effort = mj_qfrc_applied`` (clipped), ``K=0``,
``D=TORQUE_DAMPING``. The actuator's MIT formula reduces to
``τ = −D·qdot + effort``, i.e. gravity-cancelling torque plus a
passive joint-space viscous brake.
"""

import os
import time

import numpy as np

from bar_dds import MITCommand
from gravity import TORQUE_DAMPING, TORQUE_MAX_TORQUE, apply_viscous_damping
from runner import GravityRunner, make_command


class TorqueRunner(GravityRunner):
    label = "torque"

    def __init__(self, domain_id: int) -> None:
        super().__init__(domain_id)
        self.last_torques = np.zeros(len(self.joint_names))

    def build_command(self) -> MITCommand:
        if self.dof_ids.size:
            apply_viscous_damping(self.data, self.dof_ids, TORQUE_DAMPING)
        n = len(self.joint_names)
        effort = [0.0] * n
        for pub_idx, info in self.mapping:
            t = float(self.data.qfrc_applied[info.dof_index])
            t = float(np.clip(t, -TORQUE_MAX_TORQUE, TORQUE_MAX_TORQUE))
            effort[pub_idx] = t
            self.last_torques[pub_idx] = t
        return make_command(
            joint_names=self.joint_names,
            position=[0.0] * n,
            velocity=[0.0] * n,
            effort=effort,
            stiffness=[0.0] * n,
            damping=[TORQUE_DAMPING] * n,
        )

    def print_status(self) -> None:
        t = self.last_torques
        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"torque mean={float(np.mean(np.abs(t))):.3f}  max={float(np.max(np.abs(t))):.3f}"
        )


def main() -> None:
    domain_id = int(os.environ.get("ROS_DOMAIN_ID", "0"))
    TorqueRunner(domain_id=domain_id).run()


if __name__ == "__main__":
    main()
