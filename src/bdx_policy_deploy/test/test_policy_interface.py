import math

import numpy as np
import pytest

from bdx_policy_deploy.policy_interface import (
    ACTION_DIM,
    JOINT_NAMES,
    OBS_DIM,
    PolicyInterfaceError,
    build_observation,
    clip_command,
    compute_pd_torque,
    compute_policy_target,
    projected_gravity_from_quaternion_xyzw,
    reorder_joint_state,
    roll_pitch_from_quaternion_xyzw,
)
from bdx_policy_deploy.resource_paths import resolve_resource_path


def test_build_observation_matches_sim2sim_layout() -> None:
    imu_ang_vel = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    joint_pos = np.arange(ACTION_DIM, dtype=np.float32) * 0.1
    joint_vel = np.arange(ACTION_DIM, dtype=np.float32) * 2.0
    last_action = np.full(ACTION_DIM, 0.25, dtype=np.float32)
    command = np.array([0.1, -0.05, 0.2], dtype=np.float32)

    obs = build_observation(imu_ang_vel, projected_gravity, joint_pos, joint_vel, last_action, command)

    assert obs.shape == (1, OBS_DIM)
    assert obs.dtype == np.float32
    np.testing.assert_allclose(obs[0, 0:3], imu_ang_vel * 0.2)
    np.testing.assert_allclose(obs[0, 3:6], projected_gravity)
    np.testing.assert_allclose(obs[0, 6:16], joint_pos)
    np.testing.assert_allclose(obs[0, 16:26], joint_vel * 0.05)
    np.testing.assert_allclose(obs[0, 26:36], last_action)
    np.testing.assert_allclose(obs[0, 36:39], command)


def test_projected_gravity_identity_orientation_is_down() -> None:
    projected_gravity = projected_gravity_from_quaternion_xyzw([0.0, 0.0, 0.0, 1.0])
    roll, pitch = roll_pitch_from_quaternion_xyzw([0.0, 0.0, 0.0, 1.0])

    np.testing.assert_allclose(projected_gravity, [0.0, 0.0, -1.0], atol=1e-6)
    assert roll == pytest.approx(0.0)
    assert pitch == pytest.approx(0.0)


def test_projected_gravity_roll_orientation() -> None:
    half_angle = math.pi / 4.0
    quaternion = [math.sin(half_angle), 0.0, 0.0, math.cos(half_angle)]

    projected_gravity = projected_gravity_from_quaternion_xyzw(quaternion)
    roll, pitch = roll_pitch_from_quaternion_xyzw(quaternion)

    np.testing.assert_allclose(projected_gravity, [0.0, -1.0, 0.0], atol=1e-6)
    assert roll == pytest.approx(math.pi / 2.0)
    assert pitch == pytest.approx(0.0)


def test_reorder_joint_state_uses_joint_names_not_message_order() -> None:
    shuffled_names = list(reversed(JOINT_NAMES))
    position_by_name = {name: float(index) for index, name in enumerate(JOINT_NAMES)}
    velocity_by_name = {name: float(index + 10) for index, name in enumerate(JOINT_NAMES)}

    positions = [position_by_name[name] for name in shuffled_names]
    velocities = [velocity_by_name[name] for name in shuffled_names]

    ordered_pos, ordered_vel = reorder_joint_state(shuffled_names, positions, velocities)

    np.testing.assert_allclose(ordered_pos, np.arange(ACTION_DIM, dtype=np.float32))
    np.testing.assert_allclose(ordered_vel, np.arange(10, 10 + ACTION_DIM, dtype=np.float32))


def test_reorder_joint_state_rejects_missing_velocity() -> None:
    with pytest.raises(PolicyInterfaceError):
        reorder_joint_state(JOINT_NAMES, [0.0] * ACTION_DIM, [])


def test_compute_policy_target_clips_action_and_joint_limits() -> None:
    action = np.array([2.0, -2.0] + [0.2] * 8, dtype=np.float32)
    lower = np.full(ACTION_DIM, -0.25, dtype=np.float32)
    upper = np.full(ACTION_DIM, 0.25, dtype=np.float32)

    target = compute_policy_target(action, joint_lower_limits=lower, joint_upper_limits=upper, action_clip=1.0)

    np.testing.assert_allclose(target.clipped_action[:2], [1.0, -1.0])
    np.testing.assert_allclose(target.requested_target_joint_pos[:2], [0.5, -0.5])
    np.testing.assert_allclose(target.target_joint_pos[:2], [0.25, -0.25])
    assert target.target_was_clipped


def test_compute_policy_target_zero_action_clip_disables_action_clipping() -> None:
    action = np.array([2.0, -2.0] + [0.2] * 8, dtype=np.float32)
    lower = np.full(ACTION_DIM, -10.0, dtype=np.float32)
    upper = np.full(ACTION_DIM, 10.0, dtype=np.float32)

    target = compute_policy_target(action, joint_lower_limits=lower, joint_upper_limits=upper, action_clip=0.0)

    np.testing.assert_allclose(target.clipped_action, action)
    np.testing.assert_allclose(target.requested_target_joint_pos[:2], [1.0, -1.0])
    assert not target.target_was_clipped


def test_compute_policy_target_rejects_negative_action_clip() -> None:
    with pytest.raises(PolicyInterfaceError, match="non-negative"):
        compute_policy_target(np.zeros(ACTION_DIM, dtype=np.float32), action_clip=-1.0)


def test_compute_pd_torque_matches_sim2sim_formula_and_limits() -> None:
    target = np.full(ACTION_DIM, 1.0, dtype=np.float32)
    position = np.zeros(ACTION_DIM, dtype=np.float32)
    velocity = np.ones(ACTION_DIM, dtype=np.float32)
    kp = np.full(ACTION_DIM, 10.0, dtype=np.float32)
    kd = np.full(ACTION_DIM, 2.0, dtype=np.float32)
    effort_limits = np.full(ACTION_DIM, 5.0, dtype=np.float32)

    torque = compute_pd_torque(target, position, velocity, kp, kd, effort_limits)

    np.testing.assert_allclose(torque, np.full(ACTION_DIM, 5.0, dtype=np.float32))


def test_clip_command_uses_configured_limits() -> None:
    command = clip_command([0.4, -0.4, 1.0], [-0.1, -0.05, -0.2], [0.1, 0.05, 0.2])

    np.testing.assert_allclose(command, [0.1, -0.05, 0.2])


def test_resolve_resource_path_keeps_absolute_paths() -> None:
    assert resolve_resource_path("/tmp/example").as_posix() == "/tmp/example"
