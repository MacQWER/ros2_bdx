from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import numpy as np


JOINT_NAMES = [
    "Left_Hip_Yaw",
    "Right_Hip_Yaw",
    "Left_Hip_Roll",
    "Right_Hip_Roll",
    "Left_Hip_Pitch",
    "Right_Hip_Pitch",
    "Left_Knee",
    "Right_Knee",
    "Left_Ankle",
    "Right_Ankle",
]

OBS_DIM = 39
ACTION_DIM = 10
ACTION_SCALE = 0.5
ACTION_CLIP = 1.0

DEFAULT_JOINT_POS = np.zeros(ACTION_DIM, dtype=np.float32)
DEFAULT_JOINT_LOWER_LIMITS = np.array(
    [
        -0.2617993877991494,
        -0.2617993877991494,
        -0.2617993877991494,
        -0.3490658503988659,
        -1.0471975511965976,
        -0.6981317007977318,
        -0.9599310885968813,
        -1.3089969389957472,
        -0.785,
        -1.2217304763960306,
    ],
    dtype=np.float32,
)
DEFAULT_JOINT_UPPER_LIMITS = np.array(
    [
        0.2617993877991494,
        0.2617993877991494,
        0.3490658503988659,
        0.2617993877991494,
        0.6981317007977318,
        1.0471975511965976,
        1.3089969389957472,
        0.9599310885968813,
        1.2217304763960306,
        0.785,
    ],
    dtype=np.float32,
)
DEFAULT_KP = np.array([80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 40.0, 40.0], dtype=np.float32)
DEFAULT_KD = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 2.0, 2.0], dtype=np.float32)
DEFAULT_EFFORT_LIMITS = np.array(
    [42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 11.9, 11.9],
    dtype=np.float32,
)


class PolicyInterfaceError(ValueError):
    """Raised when policy input/output data violates the deployment contract."""


@dataclass(frozen=True)
class PolicyTarget:
    raw_action: np.ndarray
    clipped_action: np.ndarray
    requested_target_joint_pos: np.ndarray
    target_joint_pos: np.ndarray
    target_was_clipped: bool


def as_float32_vector(values: Sequence[float] | np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.shape != (size,):
        raise PolicyInterfaceError(f"{name} must have shape ({size},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise PolicyInterfaceError(f"{name} contains NaN or Inf")
    return vector


def quaternion_xyzw_to_rotation_matrix(quaternion_xyzw: Sequence[float] | np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(-1)
    if q.shape != (4,):
        raise PolicyInterfaceError(f"orientation quaternion must have shape (4,), got {q.shape}")
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm <= 0.0:
        raise PolicyInterfaceError("orientation quaternion has zero or invalid norm")

    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def projected_gravity_from_quaternion_xyzw(quaternion_xyzw: Sequence[float] | np.ndarray) -> np.ndarray:
    rotation_world_from_imu = quaternion_xyzw_to_rotation_matrix(quaternion_xyzw)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return (rotation_world_from_imu.T @ gravity_world).astype(np.float32)


def roll_pitch_from_quaternion_xyzw(quaternion_xyzw: Sequence[float] | np.ndarray) -> tuple[float, float]:
    rotation_world_from_imu = quaternion_xyzw_to_rotation_matrix(quaternion_xyzw)
    roll = math.atan2(rotation_world_from_imu[2, 1], rotation_world_from_imu[2, 2])
    pitch = math.asin(float(np.clip(-rotation_world_from_imu[2, 0], -1.0, 1.0)))
    return roll, pitch


def build_observation(
    imu_ang_vel_rad_s: Sequence[float] | np.ndarray,
    imu_projected_gravity: Sequence[float] | np.ndarray,
    joint_pos_rad: Sequence[float] | np.ndarray,
    joint_vel_rad_s: Sequence[float] | np.ndarray,
    last_action: Sequence[float] | np.ndarray,
    command: Sequence[float] | np.ndarray,
    default_joint_pos: Sequence[float] | np.ndarray = DEFAULT_JOINT_POS,
) -> np.ndarray:
    imu_ang_vel = as_float32_vector(imu_ang_vel_rad_s, 3, "imu_ang_vel_rad_s") * np.float32(0.2)
    projected_gravity = as_float32_vector(imu_projected_gravity, 3, "imu_projected_gravity")
    joint_pos = as_float32_vector(joint_pos_rad, ACTION_DIM, "joint_pos_rad")
    joint_vel = as_float32_vector(joint_vel_rad_s, ACTION_DIM, "joint_vel_rad_s") * np.float32(0.05)
    previous_action = as_float32_vector(last_action, ACTION_DIM, "last_action")
    cmd = as_float32_vector(command, 3, "command")
    default_pos = as_float32_vector(default_joint_pos, ACTION_DIM, "default_joint_pos")

    obs = np.concatenate(
        [imu_ang_vel, projected_gravity, joint_pos - default_pos, joint_vel, previous_action, cmd],
        dtype=np.float32,
    )
    if obs.shape != (OBS_DIM,):
        raise PolicyInterfaceError(f"observation must have shape ({OBS_DIM},), got {obs.shape}")
    return obs.reshape(1, OBS_DIM)


def reorder_joint_state(
    names: Sequence[str],
    positions: Sequence[float],
    velocities: Sequence[float],
    joint_names: Sequence[str] = JOINT_NAMES,
) -> tuple[np.ndarray, np.ndarray]:
    if len(positions) != len(names):
        raise PolicyInterfaceError("JointState.position length must match JointState.name length")
    if len(velocities) != len(names):
        raise PolicyInterfaceError("JointState.velocity length must match JointState.name length")

    index_by_name = {name: index for index, name in enumerate(names)}
    missing = [name for name in joint_names if name not in index_by_name]
    if missing:
        raise PolicyInterfaceError(f"JointState missing required joints: {', '.join(missing)}")

    ordered_pos = np.array([positions[index_by_name[name]] for name in joint_names], dtype=np.float32)
    ordered_vel = np.array([velocities[index_by_name[name]] for name in joint_names], dtype=np.float32)
    if not np.all(np.isfinite(ordered_pos)) or not np.all(np.isfinite(ordered_vel)):
        raise PolicyInterfaceError("JointState contains NaN or Inf")
    return ordered_pos, ordered_vel


def clip_command(
    command: Sequence[float] | np.ndarray,
    command_lower_limits: Sequence[float] | np.ndarray,
    command_upper_limits: Sequence[float] | np.ndarray,
) -> np.ndarray:
    cmd = as_float32_vector(command, 3, "command")
    lower = as_float32_vector(command_lower_limits, 3, "command_lower_limits")
    upper = as_float32_vector(command_upper_limits, 3, "command_upper_limits")
    if np.any(lower > upper):
        raise PolicyInterfaceError("command lower limits must be <= upper limits")
    return np.clip(cmd, lower, upper).astype(np.float32)


def compute_policy_target(
    action: Sequence[float] | np.ndarray,
    default_joint_pos: Sequence[float] | np.ndarray = DEFAULT_JOINT_POS,
    joint_lower_limits: Sequence[float] | np.ndarray = DEFAULT_JOINT_LOWER_LIMITS,
    joint_upper_limits: Sequence[float] | np.ndarray = DEFAULT_JOINT_UPPER_LIMITS,
    action_clip: float = ACTION_CLIP,
    action_scale: float = ACTION_SCALE,
) -> PolicyTarget:
    raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
    if raw_action.shape != (ACTION_DIM,):
        raise PolicyInterfaceError(f"policy action must have shape ({ACTION_DIM},), got {raw_action.shape}")
    if not np.all(np.isfinite(raw_action)):
        raise PolicyInterfaceError("policy action contains NaN or Inf")
    if action_clip <= 0.0:
        raise PolicyInterfaceError("action_clip must be positive")

    default_pos = as_float32_vector(default_joint_pos, ACTION_DIM, "default_joint_pos")
    lower = as_float32_vector(joint_lower_limits, ACTION_DIM, "joint_lower_limits")
    upper = as_float32_vector(joint_upper_limits, ACTION_DIM, "joint_upper_limits")
    if np.any(lower > upper):
        raise PolicyInterfaceError("joint lower limits must be <= upper limits")

    clipped_action = np.clip(raw_action, -action_clip, action_clip).astype(np.float32)
    requested_target = (default_pos + clipped_action * np.float32(action_scale)).astype(np.float32)
    target = np.clip(requested_target, lower, upper).astype(np.float32)
    target_was_clipped = bool(np.any(np.not_equal(requested_target, target)))
    return PolicyTarget(raw_action, clipped_action, requested_target, target, target_was_clipped)


def compute_pd_torque(
    target_joint_pos: Sequence[float] | np.ndarray,
    joint_pos: Sequence[float] | np.ndarray,
    joint_vel: Sequence[float] | np.ndarray,
    kp: Sequence[float] | np.ndarray = DEFAULT_KP,
    kd: Sequence[float] | np.ndarray = DEFAULT_KD,
    effort_limits: Sequence[float] | np.ndarray = DEFAULT_EFFORT_LIMITS,
) -> np.ndarray:
    target = as_float32_vector(target_joint_pos, ACTION_DIM, "target_joint_pos")
    pos = as_float32_vector(joint_pos, ACTION_DIM, "joint_pos")
    vel = as_float32_vector(joint_vel, ACTION_DIM, "joint_vel")
    kp_values = as_float32_vector(kp, ACTION_DIM, "kp")
    kd_values = as_float32_vector(kd, ACTION_DIM, "kd")
    limits = as_float32_vector(effort_limits, ACTION_DIM, "effort_limits")
    if np.any(limits < 0.0):
        raise PolicyInterfaceError("effort limits must be non-negative")

    torque = kp_values * (target - pos) - kd_values * vel
    return np.clip(torque, -limits, limits).astype(np.float32)


def finite_or_none(values: np.ndarray | None) -> bool:
    return values is None or bool(np.all(np.isfinite(values)))
