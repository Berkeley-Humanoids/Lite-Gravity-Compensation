from typing import Sequence

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

from robot_descriptions import load_asset


robot_xml = load_asset("robots/lite_dummy/mjcf/lite.xml")

_COMPLIANT_DAMPING = 0.2


def compensate_gravity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    subtree_ids: Sequence[int],
    qfrc_applied: np.ndarray | None = None,
) -> None:
    """Compute forces to counteract gravity for the given subtrees."""
    qfrc_applied = data.qfrc_applied if qfrc_applied is None else qfrc_applied
    qfrc_applied[:] = 0.0
    jac = np.empty((3, model.nv))
    for subtree_id in subtree_ids:
        total_mass = model.body_subtreemass[subtree_id]
        mujoco.mj_jacSubtreeCom(model, data, jac, subtree_id)
        qfrc_applied[:] -= model.opt.gravity * total_mass @ jac


def apply_viscous_damping(
    data: mujoco.MjData,
    dof_ids: np.ndarray,
    damping: float,
    qfrc_applied: np.ndarray | None = None,
) -> None:
    """Add light joint damping for compliant motion."""
    qfrc_applied = data.qfrc_applied if qfrc_applied is None else qfrc_applied
    qfrc_applied[dof_ids] -= damping * data.qvel[dof_ids]


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)

    # Compensate the full upper-body subtree rooted at the chest.
    subtree_ids = [model.body("chest").id]
    upper_body_dof_ids = np.array(
        [
            model.jnt_dofadr[joint_id]
            for joint_id in range(model.njnt)
            if model.body(model.jnt_bodyid[joint_id]).name != "world"
        ]
    )

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        mujoco.mj_forward(model, data)

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Compliant mode: no target tracking, only gravity compensation.
            data.ctrl[:] = 0.0
            compensate_gravity(model, data, subtree_ids)
            apply_viscous_damping(data, upper_body_dof_ids, _COMPLIANT_DAMPING)
            mujoco.mj_step(model, data)

            viewer.sync()
            rate.sleep()
