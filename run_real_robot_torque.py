from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import mujoco
import numpy as np
from lite_sdk2 import LowCommandPublisher, LowStateSubscriber, initialize_channel_factory
from lite_sdk2.dds.actuator_command import make_zero_actuator_commands
from lite_sdk2.dds.configuration import LowLevelConfiguration
from lite_sdk2.dds.low_command import LowCommand
from lite_sdk2.dds.low_state import DEFAULT_LOWSTATE_TOPIC
from loop_rate_limiters import RateLimiter
from robot_descriptions import load_asset

try:
    from lite_sdk2.dds.low_command import DEFAULT_LOWCOMMAND_TOPIC
except ImportError:
    DEFAULT_LOWCOMMAND_TOPIC = "/lowcommand"


MODEL_ASSETS = {
    "lite_dummy": "robots/lite_dummy/mjcf/lite.xml",
}
DEFAULT_MODEL = "lite_dummy"
DEFAULT_CONTROL_MODE = 1
DISABLED_CONTROL_MODE = 0
EXIT_DAMPING_KP = 0.0
EXIT_DAMPING_KD = 6.0
NONE_CONFIGURATION = getattr(LowLevelConfiguration, "NONE", None)


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
    if NONE_CONFIGURATION is not None and configuration is NONE_CONFIGURATION:
        return []
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
) -> LowLevelConfiguration | None:
    if override is not None:
        configuration = LowLevelConfiguration[override]
    else:
        if config_value is None:
            raise ValueError(
                "The incoming low-state sample did not include a configuration value. "
                "Pass --configuration to choose the actuator ordering explicitly."
            )
        configuration = LowLevelConfiguration(config_value)

    if NONE_CONFIGURATION is not None and configuration is NONE_CONFIGURATION:
        return None
    return configuration


def _configuration_choices() -> list[str]:
    return [
        configuration.name
        for configuration in LowLevelConfiguration
        if NONE_CONFIGURATION is None or configuration is not NONE_CONFIGURATION
    ]


def _build_model_joint_lookup(model: mujoco.MjModel) -> tuple[dict[str, tuple[int, int]], list[str]]:
    name_to_indices: dict[str, tuple[int, int]] = {}
    ordered_joint_names: list[str] = []
    for joint_id in range(model.njnt):
        joint_name = model.joint(joint_id).name
        ordered_joint_names.append(joint_name)
        name_to_indices[joint_name] = (model.jnt_qposadr[joint_id], model.jnt_dofadr[joint_id])
    return name_to_indices, ordered_joint_names


def _build_index_mapping(
    configuration: LowLevelConfiguration,
    actuator_count: int,
    model_joint_names: list[str],
    model_joint_indices: dict[str, tuple[int, int]],
) -> tuple[list[tuple[int, int, int]], list[str]]:
    ordered_robot_joints = _joint_order_for_configuration(configuration, model_joint_names)
    if actuator_count != len(ordered_robot_joints):
        raise ValueError(
            f"Configuration {configuration.name} expects {len(ordered_robot_joints)} actuators, "
            f"but the low-state sample contains {actuator_count}."
        )

    mapping: list[tuple[int, int, int]] = []
    ignored_joints: list[str] = []
    for actuator_index, joint_name in enumerate(ordered_robot_joints):
        indices = model_joint_indices.get(joint_name)
        if indices is None:
            ignored_joints.append(joint_name)
            continue
        qpos_index, dof_index = indices
        mapping.append((actuator_index, qpos_index, dof_index))
    return mapping, ignored_joints


def _world_child_subtree_ids(model: mujoco.MjModel) -> list[int]:
    return [body_id for body_id in range(1, model.nbody) if model.body_parentid[body_id] == 0]


def compensate_gravity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    subtree_ids: Sequence[int],
    qfrc_applied: np.ndarray | None = None,
) -> None:
    qfrc_applied = data.qfrc_applied if qfrc_applied is None else qfrc_applied
    qfrc_applied[:] = 0.0
    jac = np.empty((3, model.nv))
    for subtree_id in subtree_ids:
        total_mass = model.body_subtreemass[subtree_id]
        mujoco.mj_jacSubtreeCom(model, data, jac, subtree_id)
        qfrc_applied[:] -= (model.opt.gravity * total_mass) @ jac


def apply_viscous_damping(
    data: mujoco.MjData,
    dof_ids: np.ndarray,
    damping: float,
    qfrc_applied: np.ndarray | None = None,
) -> None:
    qfrc_applied = data.qfrc_applied if qfrc_applied is None else qfrc_applied
    qfrc_applied[dof_ids] -= damping * data.qvel[dof_ids]


def _build_zero_command(
    configuration: LowLevelConfiguration,
    actuator_count: int,
    mode: int,
) -> LowCommand:
    return _build_uniform_command(
        configuration=configuration,
        actuator_count=actuator_count,
        mode=mode,
    )


def _build_uniform_command(
    configuration: LowLevelConfiguration,
    actuator_count: int,
    mode: int,
    *,
    kp: float = 0.0,
    kd: float = 0.0,
) -> LowCommand:
    return LowCommand(
        configuration=configuration,
        actuator_commands=make_zero_actuator_commands(actuator_count, mode=mode, kp=kp, kd=kd),
    )


def _print_status(
    configuration: LowLevelConfiguration,
    mapped_count: int,
    torques: np.ndarray,
) -> None:
    timestamp = time.strftime("%H:%M:%S")
    if torques.size == 0:
        print(f"[{timestamp}] configuration={configuration.name} mapped_joints={mapped_count} torque=unavailable")
        return

    mean_abs_torque = float(np.mean(np.abs(torques)))
    max_abs_torque = float(np.max(np.abs(torques)))
    print(
        f"[{timestamp}] configuration={configuration.name} mapped_joints={mapped_count} "
        f"mean_abs_torque={mean_abs_torque:.3f} max_abs_torque={max_abs_torque:.3f}"
    )


def _make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Subscribe to Lite low-level state, estimate gravity compensation torques in MuJoCo, "
            "and publish LowCommand messages back to the real robot."
        )
    )
    parser.add_argument("--model", choices=sorted(MODEL_ASSETS), default=DEFAULT_MODEL)
    parser.add_argument("--command-hz", type=float, default=200.0)
    parser.add_argument("--status-hz", type=float, default=2.0)
    parser.add_argument("--damping", type=float, default=0.5, help="Viscous joint damping added on top of gravity compensation.")
    parser.add_argument("--kp", type=float, default=0.0, help="Optional position gain around the startup pose.")
    parser.add_argument("--kd", type=float, default=0.0, help="Optional velocity gain around the startup pose.")
    parser.add_argument(
        "--max-torque",
        type=float,
        default=10.0,
        help="Clamp each actuator torque command to +/- this limit. Set <= 0 to disable clamping.",
    )
    parser.add_argument("--mode", type=int, default=DEFAULT_CONTROL_MODE, help="Actuator control mode written into each LowCommand.")
    parser.add_argument("--domain-id", type=int, default=0, help="CycloneDDS domain ID to join.")
    parser.add_argument("--state-topic", default=DEFAULT_LOWSTATE_TOPIC, help="ROS topic name for LowState samples.")
    parser.add_argument("--command-topic", default=DEFAULT_LOWCOMMAND_TOPIC, help="ROS topic name for LowCommand samples.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.1,
        help="Read timeout in seconds for each low-state sample.",
    )
    parser.add_argument(
        "--write-timeout",
        type=float,
        default=0.05,
        help="Write timeout in seconds for each low-command sample.",
    )
    parser.add_argument(
        "--configuration",
        choices=_configuration_choices(),
        help="Override the actuator ordering instead of using the configuration value from the DDS sample.",
    )
    parser.add_argument(
        "--list-model-joints",
        action="store_true",
        help="Print the MuJoCo model joint names and exit.",
    )
    parser.add_argument(
        "--zero-on-timeout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish a zero-torque command when the state stream times out.",
    )
    parser.add_argument(
        "--zero-on-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish a short burst of zero commands before shutting down.",
    )
    parser.add_argument(
        "--zero-command-count",
        type=int,
        default=10,
        help="Number of zero-command samples to publish during timeout recovery or shutdown.",
    )
    return parser


def main() -> None:
    args = _make_arg_parser().parse_args()
    model = mujoco.MjModel.from_xml_path(_load_model_asset(args.model))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    model_joint_indices, model_joint_names = _build_model_joint_lookup(model)
    if args.list_model_joints:
        for name in model_joint_names:
            print(name)
        return

    subtree_ids = _world_child_subtree_ids(model)
    if not subtree_ids:
        raise RuntimeError("Could not find any world-rooted body subtrees for gravity compensation.")

    initialize_channel_factory(args.domain_id)
    subscriber = LowStateSubscriber(topic=args.state_topic, domain_id=args.domain_id)
    publisher = LowCommandPublisher(topic=args.command_topic, domain_id=args.domain_id)
    subscriber.initialize()
    publisher.initialize()

    status_period = 0.0 if args.status_hz <= 0 else 1.0 / args.status_hz
    next_status_time = 0.0
    active_mapping: list[tuple[int, int, int]] = []
    active_mapping_key: tuple[LowLevelConfiguration, int] | None = None
    active_dof_ids = np.array([], dtype=int)
    active_configuration: LowLevelConfiguration | None = None
    actuator_count = 0
    hold_positions = np.array([], dtype=float)
    warned_about_timeout = False
    warned_about_write_failure = False
    warned_about_decode_failure = False
    warned_about_none_configuration = False
    shutdown_command: LowCommand | None = None

    print(
        f"Listening for low-level state on ROS topic {args.state_topic!r} and publishing gravity-compensated "
        f"commands to {args.command_topic!r} in DDS domain {args.domain_id} with MuJoCo model {args.model!r}."
    )

    try:
        rate = RateLimiter(frequency=args.command_hz, warn=False)
        while True:
            try:
                state = subscriber.read(timeout=args.timeout)
            except ValueError as exc:
                if not warned_about_decode_failure:
                    print(f"Ignoring malformed low-state sample on topic {args.state_topic!r}: {exc}")
                    warned_about_decode_failure = True
                rate.sleep()
                continue

            warned_about_decode_failure = False
            if state is None:
                if not warned_about_timeout:
                    print(
                        f"No low-state sample received before the {args.timeout:.3f}s timeout on topic "
                        f"{args.state_topic!r}; publishing zero torque while waiting."
                    )
                    warned_about_timeout = True

                if args.zero_on_timeout and active_configuration is not None and actuator_count > 0:
                    zero_command = _build_zero_command(active_configuration, actuator_count, args.mode)
                    for _ in range(max(args.zero_command_count, 1)):
                        publisher.write(zero_command, timeout=args.write_timeout)

                rate.sleep()
                continue

            warned_about_timeout = False
            configuration = _resolve_configuration(
                config_value=getattr(state, "configuration", None),
                override=args.configuration,
            )
            if configuration is None:
                if not warned_about_none_configuration:
                    print(
                        f"Received low-state sample with configuration=NONE on topic {args.state_topic!r}; "
                        "waiting for the robot bridge to publish an active layout."
                    )
                    warned_about_none_configuration = True
                rate.sleep()
                continue

            warned_about_none_configuration = False
            actuator_count = len(state.actuator_states)
            mapping_key = (configuration, actuator_count)

            if mapping_key != active_mapping_key:
                active_mapping, ignored_joints = _build_index_mapping(
                    configuration=configuration,
                    actuator_count=actuator_count,
                    model_joint_names=model_joint_names,
                    model_joint_indices=model_joint_indices,
                )
                active_mapping_key = mapping_key
                active_configuration = configuration
                active_dof_ids = np.array([dof_index for _, _, dof_index in active_mapping], dtype=int)
                hold_positions = np.array([actuator.position for actuator in state.actuator_states], dtype=float)

                print(
                    f"Mapped {len(active_mapping)} actuator states from configuration "
                    f"{configuration.name} ({actuator_count} actuators)."
                )
                if ignored_joints:
                    print("Ignoring joints not present in the selected MuJoCo model:", ", ".join(ignored_joints))

            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            for actuator_index, qpos_index, dof_index in active_mapping:
                actuator_state = state.actuator_states[actuator_index]
                data.qpos[qpos_index] = actuator_state.position
                data.qvel[dof_index] = actuator_state.velocity

            mujoco.mj_forward(model, data)
            compensate_gravity(model, data, subtree_ids)
            if args.damping != 0.0 and active_dof_ids.size:
                apply_viscous_damping(data, active_dof_ids, args.damping)

            commanded_torques = np.zeros(len(active_mapping), dtype=float)
            command = LowCommand(
                configuration=configuration,
                actuator_commands=make_zero_actuator_commands(
                    actuator_count,
                    mode=args.mode,
                    kp=args.kp,
                    kd=args.kd,
                ),
            )
            for mapped_index, (actuator_index, _, dof_index) in enumerate(active_mapping):
                torque = data.qfrc_applied[dof_index]
                if args.max_torque > 0:
                    torque = float(np.clip(torque, -args.max_torque, args.max_torque))

                actuator_command = command.actuator_commands[actuator_index]
                actuator_command.position = float(hold_positions[actuator_index]) if args.kp != 0.0 else 0.0
                actuator_command.velocity = 0.0
                actuator_command.torque = torque
                commanded_torques[mapped_index] = torque

            if not publisher.write(command, timeout=args.write_timeout):
                if not warned_about_write_failure:
                    print(
                        f"Failed to publish LowCommand to topic {args.command_topic!r} within "
                        f"{args.write_timeout:.3f}s; continuing."
                    )
                    warned_about_write_failure = True
            else:
                warned_about_write_failure = False

            now = time.monotonic()
            if status_period and now >= next_status_time and active_configuration is not None:
                _print_status(active_configuration, mapped_count=len(active_mapping), torques=commanded_torques)
                next_status_time = now + status_period

            rate.sleep()
    except KeyboardInterrupt:
        if active_configuration is None or actuator_count <= 0:
            print("Keyboard interrupt received before any active actuator layout was available; exiting.")
        else:
            damping_command = _build_uniform_command(
                active_configuration,
                actuator_count,
                args.mode,
                kp=EXIT_DAMPING_KP,
                kd=EXIT_DAMPING_KD,
            )
            shutdown_command = _build_zero_command(active_configuration, actuator_count, DISABLED_CONTROL_MODE)
            print(
                "Keyboard interrupt received. Switching all actuators to damping mode "
                f"(kp={EXIT_DAMPING_KP:g}, kd={EXIT_DAMPING_KD:g}, torque=0). "
                "Press Ctrl+C again to disable actuators and exit."
            )

            try:
                damping_rate = RateLimiter(frequency=args.command_hz, warn=False)
                while True:
                    if not publisher.write(damping_command, timeout=args.write_timeout):
                        if not warned_about_write_failure:
                            print(
                                f"Failed to publish LowCommand to topic {args.command_topic!r} within "
                                f"{args.write_timeout:.3f}s while in damping mode; continuing."
                            )
                            warned_about_write_failure = True
                    else:
                        warned_about_write_failure = False
                    damping_rate.sleep()
            except KeyboardInterrupt:
                print("Second keyboard interrupt received. Disabling actuators and exiting.")
    finally:
        if shutdown_command is not None:
            for _ in range(max(args.zero_command_count, 1)):
                publisher.write(shutdown_command, timeout=args.write_timeout)
        elif args.zero_on_exit and active_configuration is not None and actuator_count > 0:
            zero_command = _build_zero_command(active_configuration, actuator_count, args.mode)
            for _ in range(max(args.zero_command_count, 1)):
                publisher.write(zero_command, timeout=args.write_timeout)
        publisher.close()
        subscriber.close()


if __name__ == "__main__":
    main()
