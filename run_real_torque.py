import time

import lite_sdk2
import mujoco
import numpy as np
from lite_sdk2 import Configuration, LowCommand, LowState
from loop_rate_limiters import RateLimiter

from common import (
    COMMAND_HZ,
    DEFAULT_CONTROL_MODE,
    DISABLED_CONTROL_MODE,
    DOMAIN_ID,
    EXIT_DAMPING_KD,
    EXIT_DAMPING_KP,
    JointInfo,
    LOWCOMMAND_TOPIC,
    LOWSTATE_TOPIC,
    READ_TIMEOUT,
    STARTUP_MATCH_TIMEOUT,
    STATUS_HZ,
    TORQUE_DAMPING,
    TORQUE_MAX_TORQUE,
    ZERO_COMMAND_COUNT,
    apply_viscous_damping,
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
    configuration: Configuration,
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


def main() -> None:
    model_path = load_model_asset()
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    joint_info_by_name, model_joint_names = build_joint_info(model)
    subtree_ids = world_child_subtree_ids(model)
    if not subtree_ids:
        raise RuntimeError("Could not find any world-rooted body subtrees for gravity compensation.")

    lite_sdk2.initialize(DOMAIN_ID)
    subscriber = lite_sdk2.subscriber(LowState, topic=LOWSTATE_TOPIC, domain_id=DOMAIN_ID)
    publisher = lite_sdk2.publisher(LowCommand, topic=LOWCOMMAND_TOPIC, domain_id=DOMAIN_ID)
    subscriber.initialize()
    publisher.initialize()

    if not publisher.wait_for_reader(STARTUP_MATCH_TIMEOUT):
        print(
            f"No LowCommand reader matched within {STARTUP_MATCH_TIMEOUT:.2f}s on {LOWCOMMAND_TOPIC!r}; "
            "publishing anyway."
        )

    status_period = 1.0 / STATUS_HZ
    next_status_time = 0.0
    active_mapping: list[tuple[int, JointInfo]] = []
    active_mapping_key: tuple[Configuration, int] | None = None
    active_dof_ids = np.array([], dtype=int)
    active_configuration: Configuration | None = None
    actuator_count = 0
    warned_about_timeout = False
    warned_about_decode_failure = False
    warned_about_none_configuration = False
    shutdown_command: LowCommand | None = None

    print(
        f"Listening for low-level state on ROS topic {LOWSTATE_TOPIC!r} and publishing gravity-compensated "
        f"commands to {LOWCOMMAND_TOPIC!r} in DDS domain {DOMAIN_ID} with MuJoCo model {model_path!r}."
    )

    try:
        rate = RateLimiter(frequency=COMMAND_HZ, warn=False)
        while True:
            try:
                state = subscriber.read(timeout=READ_TIMEOUT)
            except ValueError as exc:
                if not warned_about_decode_failure:
                    print(f"Ignoring malformed low-state sample on topic {LOWSTATE_TOPIC!r}: {exc}")
                    warned_about_decode_failure = True
                rate.sleep()
                continue

            warned_about_decode_failure = False
            if state is None:
                if not warned_about_timeout:
                    print(
                        f"No low-state sample received before the {READ_TIMEOUT:.3f}s timeout on topic "
                        f"{LOWSTATE_TOPIC!r}; publishing zero torque while waiting."
                    )
                    warned_about_timeout = True

                if active_configuration is not None and actuator_count > 0:
                    publish_command_burst(
                        publisher,
                        build_command(active_configuration, actuator_count, DEFAULT_CONTROL_MODE),
                        ZERO_COMMAND_COUNT,
                    )

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
                active_dof_ids = np.array([joint_info.dof_index for _, joint_info in active_mapping], dtype=int)

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
            if active_dof_ids.size:
                apply_viscous_damping(data, active_dof_ids, TORQUE_DAMPING)

            commanded_torques = np.zeros(len(active_mapping), dtype=float)
            command = build_command(configuration, actuator_count, DEFAULT_CONTROL_MODE)
            for mapped_index, (actuator_index, joint_info) in enumerate(active_mapping):
                torque = data.qfrc_applied[joint_info.dof_index]
                torque = float(np.clip(torque, -TORQUE_MAX_TORQUE, TORQUE_MAX_TORQUE))

                actuator_command = command.actuator_commands[actuator_index]
                actuator_command.position = 0.0
                actuator_command.velocity = 0.0
                actuator_command.torque = torque
                commanded_torques[mapped_index] = torque

            publisher.write(command)

            now = time.monotonic()
            if status_period and now >= next_status_time and active_configuration is not None:
                _print_status(active_configuration, mapped_count=len(active_mapping), torques=commanded_torques)
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
                    publisher.write(damping_command)
                    damping_rate.sleep()
            except KeyboardInterrupt:
                print("Second keyboard interrupt received. Disabling actuators and exiting.")
    finally:
        if shutdown_command is not None:
            publish_command_burst(publisher, shutdown_command, ZERO_COMMAND_COUNT)
        elif active_configuration is not None and actuator_count > 0:
            zero_command = build_command(active_configuration, actuator_count, DEFAULT_CONTROL_MODE)
            publish_command_burst(publisher, zero_command, ZERO_COMMAND_COUNT)
        publisher.close()
        subscriber.close()


if __name__ == "__main__":
    main()
