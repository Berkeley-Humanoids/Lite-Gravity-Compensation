"""MuJoCo gravity-compensation helpers and shared constants.

The MuJoCo model is used purely as a kinematic+dynamics model: we set
``qpos`` / ``qvel`` from live robot state, run a forward pass, and read
``qfrc_applied``. No simulation stepping against this model.

The published joint list (``LITE_ARM_JOINTS``) must match the order
``RemotePolicyController`` claims at launch time — see
``bar_bringup_lite/config/lite_hardware.yaml`` ``arm_joints``.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np
from robot_descriptions import load_asset

LITE_DUMMY_MJCF = "robots/lite_dummy/mjcf/lite.xml"

LITE_ARM_JOINTS: tuple[str, ...] = (
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow_pitch",
    "left_wrist_yaw",
    "left_wrist_roll",
    "left_wrist_pitch",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow_pitch",
    "right_wrist_yaw",
    "right_wrist_roll",
    "right_wrist_pitch",
)

# Publisher tick rate. The downstream RemotePolicyController's stale-command
# timeout defaults to 100 ms, so anything from ~50 Hz upward is fine.
COMMAND_HZ = 200.0
STATUS_HZ = 2.0

# Torque-mode tuning (publishes effort = mj_qfrc_applied, K=0, D=TORQUE_DAMPING).
TORQUE_DAMPING = 0.5
TORQUE_MAX_TORQUE = 5.0

# PD-mode tuning (publishes position = q + gravity/Kp, K=PD_POSITION_KP, D=PD_VELOCITY_KD).
PD_POSITION_KP = 10.0
PD_VELOCITY_KD = 0.2
PD_MAX_POSITION_OFFSET = 0.5
PD_MAX_TORQUE = 10.0

# Pure-MuJoCo demo (run_mujoco.py).
MUJOCO_DEMO_DAMPING = 0.5
MUJOCO_DEMO_HZ = 200.0


@dataclass(frozen=True, slots=True)
class JointInfo:
    qpos_index: int
    dof_index: int
    lower: float = -np.inf
    upper: float = np.inf


def load_model_path() -> str:
    """Return the filesystem path to the Lite dummy MuJoCo (MJCF) model."""
    return str(load_asset(LITE_DUMMY_MJCF))


def build_joint_info(model: mujoco.MjModel) -> dict[str, JointInfo]:
    """Map each joint name to its qpos/dof indices and position limits.

    Args:
      model: Loaded MuJoCo model.

    Returns:
      Dict keyed by joint name; position limits are in radians (``±inf``
      when the joint is unlimited).
    """
    info: dict[str, JointInfo] = {}
    for joint_id in range(model.njnt):
        name = model.joint(joint_id).name
        if model.jnt_limited[joint_id]:
            lower = float(model.jnt_range[joint_id][0])
            upper = float(model.jnt_range[joint_id][1])
        else:
            lower, upper = -np.inf, np.inf
        info[name] = JointInfo(
            qpos_index=int(model.jnt_qposadr[joint_id]),
            dof_index=int(model.jnt_dofadr[joint_id]),
            lower=lower,
            upper=upper,
        )
    return info


def build_published_joint_mapping(
    published_joint_names: Sequence[str],
    joint_info_by_name: dict[str, JointInfo],
) -> list[tuple[int, JointInfo]]:
    """Map each published joint to its MuJoCo ``JointInfo``.

    Raises if any joint is missing from the model — undefined gravity
    torque for a published joint is a setup bug, not a soft warning.
    """
    mapping: list[tuple[int, JointInfo]] = []
    missing: list[str] = []
    for i, name in enumerate(published_joint_names):
        info = joint_info_by_name.get(name)
        if info is None:
            missing.append(name)
        else:
            mapping.append((i, info))
    if missing:
        raise ValueError(
            f"Published joints missing from the MuJoCo model: {missing}. "
            "Fix LITE_ARM_JOINTS or load a model that covers them."
        )
    return mapping


def world_child_subtree_ids(model: mujoco.MjModel) -> list[int]:
    """Return body ids whose parent is the world body.

    These are the roots of each kinematic subtree attached to the world;
    ``compensate_gravity`` sums the gravity force over each one.
    """
    return [b for b in range(1, model.nbody) if model.body_parentid[b] == 0]


def compensate_gravity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    subtree_ids: Sequence[int],
) -> None:
    """Write gravity-cancelling generalized forces into ``data.qfrc_applied``."""
    data.qfrc_applied[:] = 0.0
    jac = np.empty((3, model.nv))
    for sid in subtree_ids:
        mujoco.mj_jacSubtreeCom(model, data, jac, sid)
        data.qfrc_applied[:] -= (model.opt.gravity * model.body_subtreemass[sid]) @ jac


def apply_viscous_damping(
    data: mujoco.MjData,
    dof_ids: np.ndarray,
    damping: float,
) -> None:
    data.qfrc_applied[dof_ids] -= damping * data.qvel[dof_ids]
