import time

import mujoco
import numpy as np
from lite_sdk2 import LowCommandPublisher, LowStateSubscriber, initialize_channel_factory
from lite_sdk2.dds.configuration import LowLevelConfiguration
from lite_sdk2.dds.low_command import LowCommand
from lite_sdk2.dds.low_state import DEFAULT_LOWSTATE_TOPIC
from loop_rate_limiters import RateLimiter

from common import (
    COMMAND_HZ,
    DEFAULT_CONTROL_MODE,
    DEFAULT_LOWCOMMAND_TOPIC,
    DISABLED_CONTROL_MODE,
    DOMAIN_ID,
    EXIT_DAMPING_KD,
    EXIT_DAMPING_KP,
    JointInfo,
    PD_MAX_POSITION_OFFSET,
    PD_MAX_TORQUE,
    PD_POSITION_KP,
    PD_TORQUE_SCALE,
    PD_VELOCITY_KD,
    READ_TIMEOUT,
    STATUS_HZ,
    WRITE_TIMEOUT,
    ZERO_COMMAND_COUNT,
    build_command,
    build_joint_info,
    build_mapping,
    compensate_gravity,
    load_model_asset,
    publish_command_burst,
    resolve_configuration,
    world_child_subtree_ids,
)


def _print_status(
    configuration: LowLevelConfiguration,
    mapped_count: int,
    gravity_torques: np.ndarray,
    position_offsets: np.ndarray,
) -> None:
    timestamp = time.strftime("%H:%M:%S")
    if gravity_torques.size == 0:
        print(f"[{timestamp}] configuration={configuration.name} mapped_joints={mapped_count} command=unavailable")
        return

    mean_abs_torque = float(np.mean(np.abs(gravity_torques)))
    max_abs_torque = float(np.max(np.abs(gravity_torques)))
    mean_abs_offset = float(np.mean(np.abs(position_offsets)))
    max_abs_offset = float(np.max(np.abs(position_offsets)))
    print(
        f"[{timestamp}] configuration={configuration.name} mapped_joints={mapped_count} "
        f"mean_abs_gravity_torque={mean_abs_torque:.3f} max_abs_gravity_torque={max_abs_torque:.3f} "
        f"mean_abs_pos_offset={mean_abs_offset:.4f} max_abs_pos_offset={max_abs_offset:.4f}"
    )


def main() -> None:
    model_path = load_model_asset()
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    joint_info_by_name, model_joint_names = build_joint_info(model)
    subtree_ids = world_child_subtree_ids(model)
    if not subtree_ids:
        raise RuntimeError("Could not find any world-rooted body subtrees for gravity compensation.")

    initialize_channel_factory(DOMAIN_ID)
    subscriber = LowStateSubscriber(topic=DEFAULT_LOWSTATE_TOPIC, domain_id=DOMAIN_ID)
    publisher = LowCommandPublisher(topic=DEFAULT_LOWCOMMAND_TOPIC, domain_id=DOMAIN_ID)
    subscriber.initialize()
    publisher.initialize()

    status_period = 1.0 / STATUS_HZ
    next_status_time = 0.0
    active_mapping: list[tuple[int, JointInfo]] = []
    active_mapping_key: tuple[LowLevelConfiguration, int] | None = None
    active_configuration: LowLevelConfiguration | None = None
    actuator_count = 0
    warned_about_timeout = False
    warned_about_write_failure = False
    warned_about_decode_failure = False
    warned_about_none_configuration = False
    shutdown_command: LowCommand | None = None

    print(
        f"Listening for low-level state on ROS topic {DEFAULT_LOWSTATE_TOPIC!r} and publishing PD-based gravity "
        f"compensation commands to {DEFAULT_LOWCOMMAND_TOPIC!r} in DDS domain {DOMAIN_ID} with MuJoCo model "
        f"{model_path!r}."
    )

    try:
        rate = RateLimiter(frequency=COMMAND_HZ, warn=False)
        while True:
            try:
                state = subscriber.read(timeout=READ_TIMEOUT)
            except ValueError as exc:
                if not warned_about_decode_failure:
                    print(f"Ignoring malformed low-state sample on topic {DEFAULT_LOWSTATE_TOPIC!r}: {exc}")
                    warned_about_decode_failure = True
                rate.sleep()
                continue

            warned_about_decode_failure = False
            if state is None:
                if not warned_about_timeout:
                    print(
                        f"No low-state sample received before the {READ_TIMEOUT:.3f}s timeout on topic "
                        f"{DEFAULT_LOWSTATE_TOPIC!r}; publishing zero command while waiting."
                    )
                    warned_about_timeout = True

                if active_configuration is not None and actuator_count > 0:
                    publish_command_burst(
                        publisher,
                        build_command(active_configuration, actuator_count, DEFAULT_CONTROL_MODE),
                        ZERO_COMMAND_COUNT,
                        WRITE_TIMEOUT,
                    )

                rate.sleep()
                continue

            warned_about_timeout = False
            configuration = resolve_configuration(getattr(state, "configuration", None))
            if configuration is None:
                if not warned_about_none_configuration:
                    print(
                        f"Received low-state sample with configuration=NONE on topic {DEFAULT_LOWSTATE_TOPIC!r}; "
                        "waiting for the robot bridge to publish an active layout."
                    )
                    warned_about_none_configuration = True
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
                active_configuration = configuration

                print(
                    f"Mapped {len(active_mapping)} actuator states from configuration "
                    f"{configuration.name} ({actuator_count} actuators)."
                )
                if ignored_joints:
                    print("Ignoring joints not present in the selected MuJoCo model:", ", ".join(ignored_joints))

            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            for actuator_index, joint_info in active_mapping:
                actuator_state = state.actuator_states[actuator_index]
                data.qpos[joint_info.qpos_index] = actuator_state.position
                data.qvel[joint_info.dof_index] = actuator_state.velocity

            mujoco.mj_forward(model, data)
            compensate_gravity(model, data, subtree_ids)

            gravity_torques = np.zeros(len(active_mapping), dtype=float)
            position_offsets = np.zeros(len(active_mapping), dtype=float)
            command = build_command(configuration, actuator_count, DEFAULT_CONTROL_MODE)
            for mapped_index, (actuator_index, joint_info) in enumerate(active_mapping):
                actuator_state = state.actuator_states[actuator_index]
                gravity_torque = PD_TORQUE_SCALE * data.qfrc_applied[joint_info.dof_index]
                gravity_torque = float(np.clip(gravity_torque, -PD_MAX_TORQUE, PD_MAX_TORQUE))

                position_offset = gravity_torque / PD_POSITION_KP
                position_offset = float(np.clip(position_offset, -PD_MAX_POSITION_OFFSET, PD_MAX_POSITION_OFFSET))

                position_target = actuator_state.position + position_offset
                position_target = float(np.clip(position_target, joint_info.lower, joint_info.upper))

                actuator_command = command.actuator_commands[actuator_index]
                actuator_command.position = position_target
                actuator_command.velocity = 0.0
                actuator_command.torque = 0.0
                actuator_command.kp = PD_POSITION_KP
                actuator_command.kd = PD_VELOCITY_KD

                gravity_torques[mapped_index] = gravity_torque
                position_offsets[mapped_index] = position_target - actuator_state.position

            if not publisher.write(command, timeout=WRITE_TIMEOUT):
                if not warned_about_write_failure:
                    print(
                        f"Failed to publish LowCommand to topic {DEFAULT_LOWCOMMAND_TOPIC!r} within "
                        f"{WRITE_TIMEOUT:.3f}s; continuing."
                    )
                    warned_about_write_failure = True
            else:
                warned_about_write_failure = False

            now = time.monotonic()
            if status_period and now >= next_status_time and active_configuration is not None:
                _print_status(
                    active_configuration,
                    mapped_count=len(active_mapping),
                    gravity_torques=gravity_torques,
                    position_offsets=position_offsets,
                )
                next_status_time = now + status_period

            rate.sleep()
    except KeyboardInterrupt:
        if active_configuration is None or actuator_count <= 0:
            print("Keyboard interrupt received before any active actuator layout was available; exiting.")
        else:
            damping_command = build_command(
                active_configuration,
                actuator_count,
                DEFAULT_CONTROL_MODE,
                kp=EXIT_DAMPING_KP,
                kd=EXIT_DAMPING_KD,
            )
            shutdown_command = build_command(active_configuration, actuator_count, DISABLED_CONTROL_MODE)
            print(
                "Keyboard interrupt received. Switching all actuators to damping mode "
                f"(kp={EXIT_DAMPING_KP:g}, kd={EXIT_DAMPING_KD:g}, torque=0). "
                "Press Ctrl+C again to disable actuators and exit."
            )

            try:
                damping_rate = RateLimiter(frequency=COMMAND_HZ, warn=False)
                while True:
                    if not publisher.write(damping_command, timeout=WRITE_TIMEOUT):
                        if not warned_about_write_failure:
                            print(
                                f"Failed to publish LowCommand to topic {DEFAULT_LOWCOMMAND_TOPIC!r} within "
                                f"{WRITE_TIMEOUT:.3f}s while in damping mode; continuing."
                            )
                            warned_about_write_failure = True
                    else:
                        warned_about_write_failure = False
                    damping_rate.sleep()
            except KeyboardInterrupt:
                print("Second keyboard interrupt received. Disabling actuators and exiting.")
    finally:
        if shutdown_command is not None:
            publish_command_burst(publisher, shutdown_command, ZERO_COMMAND_COUNT, WRITE_TIMEOUT)
        elif active_configuration is not None and actuator_count > 0:
            zero_command = build_command(active_configuration, actuator_count, DEFAULT_CONTROL_MODE)
            publish_command_burst(publisher, zero_command, ZERO_COMMAND_COUNT, WRITE_TIMEOUT)
        publisher.close()
        subscriber.close()


if __name__ == "__main__":
    main()
