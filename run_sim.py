import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

from common import MUJOCO_DEMO_DAMPING, MUJOCO_DEMO_HZ, apply_viscous_damping, compensate_gravity, load_model_asset


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(load_model_asset())
    data = mujoco.MjData(model)

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

        rate = RateLimiter(frequency=MUJOCO_DEMO_HZ, warn=False)
        while viewer.is_running():
            data.ctrl[:] = 0.0
            compensate_gravity(model, data, subtree_ids)
            apply_viscous_damping(data, upper_body_dof_ids, MUJOCO_DEMO_DAMPING)
            mujoco.mj_step(model, data)

            viewer.sync()
            rate.sleep()
