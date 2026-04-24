from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np
from lite_sdk2 import Configuration, LowCommand, zero_actuator_commands
from lite_sdk2.topics import LOWCOMMAND, LOWSTATE
from robot_descriptions import load_asset


LOWCOMMAND_TOPIC = LOWCOMMAND
LOWSTATE_TOPIC = LOWSTATE


_MODEL_ASSETS = {
    "lite_dummy": "robots/lite_dummy/mjcf/lite.xml",
}
DEFAULT_MODEL = "lite_dummy"

DOMAIN_ID = 0
DEFAULT_CONTROL_MODE = 1
DISABLED_CONTROL_MODE = 0

COMMAND_HZ = 200.0
STATUS_HZ = 2.0
READ_TIMEOUT = 0.1
STARTUP_MATCH_TIMEOUT = 2.0
ZERO_COMMAND_COUNT = 10

EXIT_DAMPING_KP = 0.0
EXIT_DAMPING_KD = 6.0

TORQUE_DAMPING = 0.5
TORQUE_MAX_TORQUE = 5.0

PD_POSITION_KP = 10.0
PD_VELOCITY_KD = 0.2
PD_MAX_POSITION_OFFSET = 0.5
PD_TORQUE_SCALE = 1.0
PD_MAX_TORQUE = 10.0

VISUALIZER_POSE_BODY = "head"
VISUALIZER_HZ = 120.0
VISUALIZER_PRINT_HZ = 2.0
VISUALIZER_READ_TIMEOUT = 1.0

MUJOCO_DEMO_DAMPING = 0.5
MUJOCO_DEMO_HZ = 200.0

_ARM_JOINT_TOKENS = ("shoulder", "elbow", "wrist")
_LEG_JOINT_TOKENS = ("hip", "knee", "ankle")


@dataclass(frozen=True, slots=True)
class JointInfo:
    qpos_index: int
    dof_index: int
    lower: float = -np.inf
    upper: float = np.inf


def load_model_asset(model_name: str = DEFAULT_MODEL) -> str:
    try:
        return str(load_asset(_MODEL_ASSETS[model_name]))
    except KeyError as exc:
        raise ValueError(f"Unknown model {model_name!r}. Choose from {sorted(_MODEL_ASSETS)}.") from exc


def _joint_order_for_configuration(
    configuration: Configuration,
    model_joint_names: list[str],
) -> list[str]:
    if configuration is Configuration.NONE:
        return []
    if configuration is Configuration.FULL_BODY_WITH_FINGERS:
        return list(model_joint_names)
    if configuration is Configuration.FULL_BODY:
        return [name for name in model_joint_names if "finger" not in name]
    if configuration is Configuration.ARMS_AND_LEGS:
        return [
            name
            for name in model_joint_names
            if any(token in name for token in _ARM_JOINT_TOKENS + _LEG_JOINT_TOKENS)
        ]
    if configuration is Configuration.BIMANUAL_ARMS:
        return [name for name in model_joint_names if any(token in name for token in _ARM_JOINT_TOKENS)]
    if configuration is Configuration.LEFT_ARM:
        return [
            name
            for name in model_joint_names
            if name.startswith("left_") and any(token in name for token in _ARM_JOINT_TOKENS)
        ]
    if configuration is Configuration.RIGHT_ARM:
        return [
            name
            for name in model_joint_names
            if name.startswith("right_") and any(token in name for token in _ARM_JOINT_TOKENS)
        ]
    raise ValueError(f"Unsupported low-level configuration: {configuration!r}")


def resolve_configuration(config_value: int | None) -> Configuration | None:
    if config_value is None:
        raise ValueError(
            "The incoming low-state sample did not include a configuration value. "
            "The robot bridge must publish an active actuator layout."
        )

    configuration = Configuration(config_value)
    if configuration is Configuration.NONE:
        return None
    return configuration


def build_joint_info(model: mujoco.MjModel) -> tuple[dict[str, JointInfo], list[str]]:
    joint_info_by_name: dict[str, JointInfo] = {}
    joint_names: list[str] = []
    for joint_id in range(model.njnt):
        joint_name = model.joint(joint_id).name
        joint_names.append(joint_name)

        lower = -np.inf
        upper = np.inf
        if model.jnt_limited[joint_id]:
            lower = float(model.jnt_range[joint_id][0])
            upper = float(model.jnt_range[joint_id][1])

        joint_info_by_name[joint_name] = JointInfo(
            qpos_index=int(model.jnt_qposadr[joint_id]),
            dof_index=int(model.jnt_dofadr[joint_id]),
            lower=lower,
            upper=upper,
        )
    return joint_info_by_name, joint_names


def build_mapping(
    configuration: Configuration,
    actuator_count: int,
    model_joint_names: list[str],
    joint_info_by_name: dict[str, JointInfo],
) -> tuple[list[tuple[int, JointInfo]], list[str]]:
    ordered_robot_joints = _joint_order_for_configuration(configuration, model_joint_names)
    if actuator_count != len(ordered_robot_joints):
        raise ValueError(
            f"Configuration {configuration.name} expects {len(ordered_robot_joints)} actuators, "
            f"but the low-state sample contains {actuator_count}."
        )

    mapping: list[tuple[int, JointInfo]] = []
    ignored_joints: list[str] = []
    for actuator_index, joint_name in enumerate(ordered_robot_joints):
        joint_info = joint_info_by_name.get(joint_name)
        if joint_info is None:
            ignored_joints.append(joint_name)
            continue
        mapping.append((actuator_index, joint_info))
    return mapping, ignored_joints


def world_child_subtree_ids(model: mujoco.MjModel) -> list[int]:
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
        mujoco.mj_jacSubtreeCom(model, data, jac, subtree_id)
        qfrc_applied[:] -= (model.opt.gravity * model.body_subtreemass[subtree_id]) @ jac


def apply_viscous_damping(
    data: mujoco.MjData,
    dof_ids: np.ndarray,
    damping: float,
    qfrc_applied: np.ndarray | None = None,
) -> None:
    qfrc_applied = data.qfrc_applied if qfrc_applied is None else qfrc_applied
    qfrc_applied[dof_ids] -= damping * data.qvel[dof_ids]


def build_command(
    configuration: Configuration,
    actuator_count: int,
    mode: int,
    *,
    kp: float = 0.0,
    kd: float = 0.0,
) -> LowCommand:
    return LowCommand(
        configuration=configuration,
        actuator_commands=zero_actuator_commands(actuator_count, mode=mode, kp=kp, kd=kd),
    )


def publish_command_burst(publisher: object, command: LowCommand, repeat_count: int) -> None:
    for _ in range(max(repeat_count, 1)):
        publisher.write(command)
