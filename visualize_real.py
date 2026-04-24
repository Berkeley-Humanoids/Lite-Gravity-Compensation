import time

import lite_sdk2
import mujoco
import mujoco.viewer
import numpy as np
from lite_sdk2 import Configuration, LowState
from loop_rate_limiters import RateLimiter

from common import (
    DOMAIN_ID,
    JointInfo,
    LOWSTATE_TOPIC,
    VISUALIZER_HZ,
    VISUALIZER_POSE_BODY,
    VISUALIZER_PRINT_HZ,
    VISUALIZER_READ_TIMEOUT,
    build_joint_info,
    build_mapping,
    load_model_asset,
    resolve_configuration,
)


def _print_pose(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, mapped_count: int) -> None:
    quat = np.empty(4)
    xpos = np.array(data.body(body_name).xpos, copy=True)
    xmat = np.array(data.body(body_name).xmat, copy=True)
    mujoco.mju_mat2Quat(quat, xmat)
    timestamp = time.strftime("%H:%M:%S")
    xyz = ", ".join(f"{value:+.3f}" for value in xpos)
    wxyz = ", ".join(f"{value:+.3f}" for value in quat)
    print(f"[{timestamp}] mapped_joints={mapped_count} {body_name}_pos=[{xyz}] {body_name}_quat_wxyz=[{wxyz}]")


def main() -> None:
    model_path = load_model_asset()
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    joint_info_by_name, model_joint_names = build_joint_info(model)
    body_names = {model.body(body_id).name for body_id in range(model.nbody)}
    if VISUALIZER_POSE_BODY not in body_names:
        raise ValueError(f"Body {VISUALIZER_POSE_BODY!r} does not exist in the default model.")

    lite_sdk2.initialize(DOMAIN_ID)
    subscriber = lite_sdk2.subscriber(LowState, topic=LOWSTATE_TOPIC, domain_id=DOMAIN_ID)
    subscriber.initialize()

    previous_joint_positions = np.array(data.qpos, copy=True)
    print_period = 1.0 / VISUALIZER_PRINT_HZ
    next_print_time = 0.0
    active_mapping: list[tuple[int, JointInfo]] = []
    active_mapping_key: tuple[Configuration, int] | None = None
    warned_about_timeout = False
    warned_about_none_configuration = False

    print(
        f"Listening for low-level state on ROS topic {LOWSTATE_TOPIC!r} in DDS domain {DOMAIN_ID} "
        f"with MuJoCo model {model_path!r}."
    )

    try:
        with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            rate = RateLimiter(frequency=VISUALIZER_HZ, warn=False)
            while viewer.is_running():
                state = subscriber.read(timeout=VISUALIZER_READ_TIMEOUT)
                if state is None:
                    if not warned_about_timeout:
                        print(
                            f"No low-state sample received before the {VISUALIZER_READ_TIMEOUT:.3f}s timeout on topic "
                            f"{LOWSTATE_TOPIC!r}; waiting for the next sample."
                        )
                        warned_about_timeout = True
                    viewer.sync()
                    rate.sleep()
                    continue

                warned_about_timeout = False
                configuration = resolve_configuration(getattr(state, "configuration", None))
                if configuration is None:
                    if not warned_about_none_configuration:
                        print(
                            f"Received low-state sample with configuration=NONE on topic {LOWSTATE_TOPIC!r}; "
                            "waiting for the robot bridge to publish an active layout."
                        )
                        warned_about_none_configuration = True
                    viewer.sync()
                    rate.sleep()
                    continue

                warned_about_none_configuration = False
                actuator_count = len(state.actuator_states)
                mapping_key = (configuration, actuator_count)

                if mapping_key != active_mapping_key:
                    active_mapping, ignored_joints = build_mapping(
                        configuration=configuration,
                        actuator_count=actuator_count,
                        model_joint_names=model_joint_names,
                        joint_info_by_name=joint_info_by_name,
                    )
                    active_mapping_key = mapping_key
                    print(
                        f"Mapped {len(active_mapping)} actuator positions from configuration "
                        f"{configuration.name} ({actuator_count} actuators)."
                    )
                    if ignored_joints:
                        print("Ignoring joints not present in the selected MuJoCo model:", ", ".join(ignored_joints))

                for actuator_index, joint_info in active_mapping:
                    data.qpos[joint_info.qpos_index] = state.actuator_states[actuator_index].position

                dt = max(rate.period, 1e-6)
                data.qvel[:] = (data.qpos - previous_joint_positions) / dt
                previous_joint_positions[:] = data.qpos
                data.ctrl[:] = 0.0
                data.qfrc_applied[:] = 0.0
                mujoco.mj_forward(model, data)

                now = time.monotonic()
                if print_period and now >= next_print_time:
                    _print_pose(model, data, VISUALIZER_POSE_BODY, mapped_count=len(active_mapping))
                    next_print_time = now + print_period

                viewer.sync()
                rate.sleep()
    finally:
        subscriber.close()


if __name__ == "__main__":
    main()
