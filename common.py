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

COMMAND_HZ = 200.0          # control frequency in Hz
STATUS_HZ = 2.0             # status report frequency in Hz
READ_TIMEOUT = 0.1          # timeout for reading a low-state sample in seconds
STARTUP_MATCH_TIMEOUT = 2.0
ZERO_COMMAND_COUNT = 10     # number of zero commands to publish when no low-state sample is available

EXIT_DAMPING_KP = 0.0       # position gain used while shutting down (no position holding)
EXIT_DAMPING_KD = 6.0       # velocity gain used while shutting down (bleeds off motion)

TORQUE_DAMPING = 0.5        # joint-space viscous damping for the torque demo, in N*m/(rad/s)
TORQUE_MAX_TORQUE = 5.0     # per-joint torque clamp sent to the real robot, in N*m

PD_POSITION_KP = 10.0       # position gain used by the PD demo, in N*m/rad
PD_VELOCITY_KD = 0.2        # velocity gain used by the PD demo, in N*m/(rad/s)
PD_MAX_POSITION_OFFSET = 0.5  # cap on how far the position target may shift from the measurement
PD_TORQUE_SCALE = 1.0       # multiplier applied to the gravity torque before converting to a position offset
PD_MAX_TORQUE = 10.0        # cap used when converting torque -> position offset, in N*m

VISUALIZER_POSE_BODY = "head"
VISUALIZER_HZ = 120.0
VISUALIZER_PRINT_HZ = 2.0
VISUALIZER_READ_TIMEOUT = 1.0

MUJOCO_DEMO_DAMPING = 0.5
MUJOCO_DEMO_HZ = 200.0

COLLISION_SAFETY_MARGIN = 0.02  # safety margin for collision detection in meters
COLLISION_STIFFNESS = 400.0     # virtual repulsive spring stiffness, in N/m
COLLISION_DAMPING = 5.0         # damping along the contact normal, in N/(m/s); only resists approach
COLLISION_MAX_FORCE = 30.0      # cap on the per-pair repulsive force in N

_ARM_JOINT_TOKENS = ("shoulder", "elbow", "wrist")
_LEG_JOINT_TOKENS = ("hip", "knee", "ankle")

# Geom-name groups used to build the default collision-pair spec for the lite_dummy
# model, in the same shape mink.CollisionAvoidanceLimit consumes (a list of (group_a,
# group_b) tuples, where each group is a list of geom names).
LITE_LEFT_ARM_COLLISION_GEOMS: tuple[str, ...] = (
    "left_shoulder_yaw_collision",
    "left_wrist_yaw_collision",
    "left_hand_finger_collision",
    "left_hand_palm_collision",
)
LITE_RIGHT_ARM_COLLISION_GEOMS: tuple[str, ...] = (
    "right_shoulder_yaw_collision",
    "right_wrist_yaw_collision",
    "right_hand_finger_collision",
    "right_hand_palm_collision",
)
LITE_TORSO_COLLISION_GEOMS: tuple[str, ...] = (
    "chest_collision",
    "chest_collar_left_collision",
    "chest_collar_right_collision",
    "head_collision",
)
DEFAULT_COLLISION_GEOM_PAIRS: tuple[tuple[Sequence[str], Sequence[str]], ...] = (
    (LITE_LEFT_ARM_COLLISION_GEOMS, LITE_RIGHT_ARM_COLLISION_GEOMS),
    (LITE_LEFT_ARM_COLLISION_GEOMS, LITE_TORSO_COLLISION_GEOMS),
    (LITE_RIGHT_ARM_COLLISION_GEOMS, LITE_TORSO_COLLISION_GEOMS),
)


@dataclass(frozen=True, slots=True)
class JointInfo:
    """MuJoCo addressing info for one joint, plus its position limits."""

    qpos_index: int
    dof_index: int
    lower: float = -np.inf
    upper: float = np.inf


@dataclass(frozen=True, slots=True)
class CollisionPair:
    """One geom-pair to monitor for self-collision, with the bodies they belong to."""

    geom1: int
    geom2: int
    body1: int
    body2: int


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
    """Overwrite qfrc_applied with joint torques that cancel gravity for each subtree.

    For every subtree root, the weight (m * g) is mapped into joint space using the
    Jacobian of the subtree center of mass. Subtracting this gives the torque each
    joint must produce to hold the limb up against gravity.
    """
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
    """Add a `-damping * qvel` torque on the listed DoFs (bleeds off velocity)."""
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


def _resolve_geom_id(model: mujoco.MjModel, geom_name: str) -> int:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        raise ValueError(f"Geom {geom_name!r} not found in the MuJoCo model.")
    return geom_id


def build_collision_pairs(
    model: mujoco.MjModel,
    geom_pairs: Sequence[tuple[Sequence[str], Sequence[str]]] = DEFAULT_COLLISION_GEOM_PAIRS,
) -> list[CollisionPair]:
    """Resolve a mink-style geom-pair spec into a flat list of CollisionPair entries.

    `geom_pairs` follows the same shape as `mink.CollisionAvoidanceLimit.geom_pairs`:
    each entry is a tuple (group_a, group_b) where each group is a list of geom
    names. The Cartesian product across the two groups becomes the watched pairs.
    Pairs on the same body, or pairs that resolve to the same geom, are skipped.
    Duplicates across entries are deduplicated. Run this once at startup; the
    returned list is cheap to iterate every tick.
    """
    seen: set[tuple[int, int]] = set()
    pairs: list[CollisionPair] = []
    for group_a, group_b in geom_pairs:
        ids_a = [_resolve_geom_id(model, name) for name in group_a]
        ids_b = [_resolve_geom_id(model, name) for name in group_b]
        for geom_a in ids_a:
            body_a = int(model.geom_bodyid[geom_a])
            for geom_b in ids_b:
                if geom_a == geom_b:
                    continue
                body_b = int(model.geom_bodyid[geom_b])
                if body_a == body_b:
                    continue
                key = (geom_a, geom_b) if geom_a < geom_b else (geom_b, geom_a)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(CollisionPair(geom_a, geom_b, body_a, body_b))
    return pairs


def apply_collision_repulsion(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pairs: Sequence[CollisionPair],
    *,
    margin: float = COLLISION_SAFETY_MARGIN,
    stiffness: float = COLLISION_STIFFNESS,
    damping: float = COLLISION_DAMPING,
    max_force: float = COLLISION_MAX_FORCE,
    qfrc_applied: np.ndarray | None = None,
) -> None:
    """Add a virtual repulsive-spring torque to qfrc_applied for every near-collision pair.

    For each pair we ask MuJoCo for the signed distance and the closest points on
    the two geoms (`mj_geomDistance` populates `fromto`: p1 on geom1, p2 on geom2).
    When the distance is below `margin` we synthesize a force along the contact
    normal n = (p1 - p2)/|p1 - p2|:

        F = clip(stiffness * (margin - distance) + damping_term, max_force) * n

    The `damping_term` only kicks in when the two contact points are approaching
    each other along n (computed via the Jacobians and qvel). The force is then
    mapped to joint torques on BOTH bodies through their contact-point Jacobians:

        tau += J1^T F   (push body1 along +n)
        tau -= J2^T F   (push body2 along -n)

    The bilateral mapping is what gives the demo its "push the other arm away"
    behavior. For a body fixed to the world (e.g. the chest) the Jacobian is
    zero, so only the moving limb feels torque; this falls out of the math
    rather than needing a special case.
    """
    if not pairs:
        return
    qfrc_applied = data.qfrc_applied if qfrc_applied is None else qfrc_applied
    fromto = np.zeros(6)
    jac1 = np.zeros((3, model.nv))
    jac2 = np.zeros((3, model.nv))
    for pair in pairs:
        # mj_geomDistance returns mjMAXVAL (large) when the gap exceeds `margin`,
        # so this also acts as a cheap broad-phase filter.
        distance = mujoco.mj_geomDistance(model, data, pair.geom1, pair.geom2, margin, fromto)
        if distance >= margin:
            continue
        delta = fromto[0:3] - fromto[3:6]
        norm = float(np.linalg.norm(delta))
        if norm < 1e-9:
            # Closest points coincide (deep penetration); contact normal is
            # ill-defined here, so skip rather than apply a random direction.
            continue
        direction = delta / norm
        penetration = max(margin - distance, 0.0)
        force_mag = stiffness * penetration
        mujoco.mj_jac(model, data, jac1, None, fromto[0:3], pair.body1)
        mujoco.mj_jac(model, data, jac2, None, fromto[3:6], pair.body2)
        if damping > 0.0:
            # Relative velocity of p1 w.r.t. p2, projected onto the normal.
            # Negative = bodies approaching; we only damp on approach so we
            # never accelerate a separating contact.
            v_rel_normal = float((jac1 - jac2) @ data.qvel @ direction)
            if v_rel_normal < 0.0:
                force_mag += -damping * v_rel_normal
        force_mag = min(force_mag, max_force)
        force = force_mag * direction
        qfrc_applied[:] += jac1.T @ force
        qfrc_applied[:] -= jac2.T @ force
