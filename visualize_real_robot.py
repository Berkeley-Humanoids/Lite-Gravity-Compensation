from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from lite_sdk2 import LowStateSubscriber, initialize_channel_factory
from lite_sdk2.dds.configuration import LowLevelConfiguration
from lite_sdk2.dds.low_state import DEFAULT_LOWSTATE_TOPIC
from loop_rate_limiters import RateLimiter
from robot_descriptions import load_asset


MODEL_ASSETS = {
    "lite_dummy": "robots/lite_dummy/mjcf/scene.xml",
    "lite": "robots/lite/mjcf/lite.xml",
}
DEFAULT_MODEL = "lite_dummy"
DEFAULT_POSE_BODY = "head"


def _load_model_asset(model_name: str) -> str:
    try:
        return str(load_asset(MODEL_ASSETS[model_name]))
    except KeyError as exc:
        raise ValueError(f"Unknown model {model_name!r}. Choose from {sorted(MODEL_ASSETS)}.") from exc


def _is_arm_joint(joint_name: str) -> bool:
    return any(token in joint_name for token in ("shoulder", "elbow", "wrist"))


def _is_leg_joint(joint_name: str) -> bool:
    return any(token in joint_name for token in ("hip", "knee", "ankle"))


def _joint_order_for_configuration(
    configuration: LowLevelConfiguration,
    model_joint_names: list[str],
) -> list[str]:
    if configuration is LowLevelConfiguration.FULL_BODY_WITH_FINGERS:
        return list(model_joint_names)
    if configuration is LowLevelConfiguration.FULL_BODY:
        return [name for name in model_joint_names if "finger" not in name]
    if configuration is LowLevelConfiguration.ARMS_AND_LEGS:
        return [name for name in model_joint_names if _is_arm_joint(name) or _is_leg_joint(name)]
    if configuration is LowLevelConfiguration.BIMANUAL_ARMS:
        return [name for name in model_joint_names if _is_arm_joint(name)]
    if configuration is LowLevelConfiguration.LEFT_ARM:
        return [name for name in model_joint_names if name.startswith("left_") and _is_arm_joint(name)]
    if configuration is LowLevelConfiguration.RIGHT_ARM:
        return [name for name in model_joint_names if name.startswith("right_") and _is_arm_joint(name)]
    raise ValueError(f"Unsupported low-level configuration: {configuration!r}")


def _resolve_configuration(
    config_value: int | None,
    override: str | None,
) -> LowLevelConfiguration:
    if override is not None:
        return LowLevelConfiguration[override]
    if config_value is None:
        raise ValueError(
            "The incoming low-state sample did not include a configuration value. "
            "Pass --configuration to choose the actuator ordering explicitly."
        )
    return LowLevelConfiguration(config_value)


def _build_model_joint_lookup(model: mujoco.MjModel) -> tuple[dict[str, int], list[str]]:
    name_to_qpos_index: dict[str, int] = {}
    ordered_joint_names: list[str] = []
    for joint_id in range(model.njnt):
        joint_name = model.joint(joint_id).name
        ordered_joint_names.append(joint_name)
        name_to_qpos_index[joint_name] = model.jnt_qposadr[joint_id]
    return name_to_qpos_index, ordered_joint_names


def _build_index_mapping(
    configuration: LowLevelConfiguration,
    actuator_count: int,
    model_joint_names: list[str],
    qpos_indices: dict[str, int],
) -> tuple[list[tuple[int, int]], list[str]]:
    ordered_robot_joints = _joint_order_for_configuration(configuration, model_joint_names)
    if actuator_count != len(ordered_robot_joints):
        raise ValueError(
            f"Configuration {configuration.name} expects {len(ordered_robot_joints)} actuators, "
            f"but the low-state sample contains {actuator_count}."
        )

    mapping: list[tuple[int, int]] = []
    ignored_joints: list[str] = []
    for actuator_index, joint_name in enumerate(ordered_robot_joints):
        qpos_index = qpos_indices.get(joint_name)
        if qpos_index is None:
            ignored_joints.append(joint_name)
            continue
        mapping.append((actuator_index, qpos_index))
    return mapping, ignored_joints


def _print_pose(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, mapped_count: int) -> None:
    quat = np.empty(4)
    xpos = np.array(data.body(body_name).xpos, copy=True)
    xmat = np.array(data.body(body_name).xmat, copy=True)
    mujoco.mju_mat2Quat(quat, xmat)
    timestamp = time.strftime("%H:%M:%S")
    xyz = ", ".join(f"{value:+.3f}" for value in xpos)
    wxyz = ", ".join(f"{value:+.3f}" for value in quat)
    print(f"[{timestamp}] mapped_joints={mapped_count} {body_name}_pos=[{xyz}] {body_name}_quat_wxyz=[{wxyz}]")


def _make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize live Lite low-level actuator state in MuJoCo using the same "
            "LowStateSubscriber flow as Lite-SDK2-Python/examples/lowlevel/print_joint_state.py."
        )
    )
    parser.add_argument("--model", choices=sorted(MODEL_ASSETS), default=DEFAULT_MODEL)
    parser.add_argument("--pose-body", default=DEFAULT_POSE_BODY)
    parser.add_argument("--viewer-hz", type=float, default=120.0)
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--joint-scale", type=float, default=1.0)
    parser.add_argument("--domain-id", type=int, default=0, help="CycloneDDS domain ID to join.")
    parser.add_argument("--topic", default=DEFAULT_LOWSTATE_TOPIC, help="ROS topic name for LowState samples.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Read timeout in seconds for each low-state sample.",
    )
    parser.add_argument(
        "--configuration",
        choices=[config.name for config in LowLevelConfiguration],
        help="Override the actuator ordering instead of using the configuration value from the DDS sample.",
    )
    parser.add_argument(
        "--list-model-joints",
        action="store_true",
        help="Print the MuJoCo model joint names and exit.",
    )
    return parser


def main() -> None:
    args = _make_arg_parser().parse_args()
    model = mujoco.MjModel.from_xml_path(_load_model_asset(args.model))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    qpos_indices, model_joint_names = _build_model_joint_lookup(model)
    if args.list_model_joints:
        for name in model_joint_names:
            print(name)
        return

    if args.pose_body not in {model.body(i).name for i in range(model.nbody)}:
        raise ValueError(f"Body {args.pose_body!r} does not exist in the {args.model!r} model.")

    initialize_channel_factory(args.domain_id)
    subscriber = LowStateSubscriber(topic=args.topic, domain_id=args.domain_id)
    subscriber.initialize()

    previous_joint_positions = np.array(data.qpos, copy=True)
    print_period = 0.0 if args.print_hz <= 0 else 1.0 / args.print_hz
    next_print_time = 0.0
    active_mapping: list[tuple[int, int]] = []
    active_mapping_key: tuple[LowLevelConfiguration, int] | None = None
    warned_about_timeout = False

    print(
        f"Listening for low-level state on ROS topic {args.topic!r} in DDS domain {args.domain_id} "
        f"with MuJoCo model {args.model!r}."
    )

    try:
        with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            rate = RateLimiter(frequency=args.viewer_hz, warn=False)
            while viewer.is_running():
                state = subscriber.read(timeout=args.timeout)
                if state is None:
                    if not warned_about_timeout:
                        print(
                            f"No low-state sample received before the {args.timeout:.3f}s timeout on topic "
                            f"{args.topic!r}; waiting for the next sample."
                        )
                        warned_about_timeout = True
                    viewer.sync()
                    rate.sleep()
                    continue
                warned_about_timeout = False

                configuration = _resolve_configuration(
                    config_value=getattr(state, "configuration", None),
                    override=args.configuration,
                )
                actuator_count = len(state.actuator_states)
                mapping_key = (configuration, actuator_count)

                if mapping_key != active_mapping_key:
                    active_mapping, ignored_joints = _build_index_mapping(
                        configuration=configuration,
                        actuator_count=actuator_count,
                        model_joint_names=model_joint_names,
                        qpos_indices=qpos_indices,
                    )
                    active_mapping_key = mapping_key
                    print(
                        f"Mapped {len(active_mapping)} actuator positions from configuration "
                        f"{configuration.name} ({actuator_count} actuators)."
                    )
                    if ignored_joints:
                        print("Ignoring joints not present in the selected MuJoCo model:", ", ".join(ignored_joints))

                for actuator_index, qpos_index in active_mapping:
                    data.qpos[qpos_index] = args.joint_scale * state.actuator_states[actuator_index].position

                dt = max(rate.period, 1e-6)
                data.qvel[:] = (data.qpos - previous_joint_positions) / dt
                previous_joint_positions[:] = data.qpos
                data.ctrl[:] = 0.0
                data.qfrc_applied[:] = 0.0
                mujoco.mj_forward(model, data)

                now = time.monotonic()
                if print_period and now >= next_print_time:
                    _print_pose(model, data, args.pose_body, mapped_count=len(active_mapping))
                    next_print_time = now + print_period

                viewer.sync()
                rate.sleep()
    finally:
        subscriber.close()


if __name__ == "__main__":
    main()
