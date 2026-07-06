import json
import math

import pytest
from sensor_msgs.msg import JointState

from bdx_policy_deploy.joint_pose_socket_bridge_node import (
    build_joint_pose_payload_json,
    extract_joint_positions_rad,
)
from bdx_policy_deploy.policy_interface import ACTION_DIM, JOINT_NAMES


def test_extract_joint_positions_reorders_by_name() -> None:
    msg = JointState()
    msg.name = list(reversed(JOINT_NAMES))
    msg.position = list(reversed([float(index) for index in range(ACTION_DIM)]))

    positions = extract_joint_positions_rad(msg)

    assert positions.tolist() == [float(index) for index in range(ACTION_DIM)]


def test_payload_json_uses_rad_and_optional_degrees() -> None:
    msg = JointState()
    msg.header.stamp.sec = 12
    msg.header.stamp.nanosec = 34
    msg.name = list(JOINT_NAMES)
    msg.position = [math.pi / 2.0] + [0.0] * (ACTION_DIM - 1)

    payload = json.loads(build_joint_pose_payload_json(msg, sequence=7, include_degrees=True))

    assert payload["type"] == "bdx_joint_pose"
    assert payload["seq"] == 7
    assert payload["stamp"] == {"sec": 12, "nanosec": 34}
    assert payload["joint_names"] == JOINT_NAMES
    assert payload["position_rad"][0] == pytest.approx(math.pi / 2.0)
    assert payload["position_deg"][0] == pytest.approx(90.0)


def test_extract_joint_positions_rejects_missing_joint() -> None:
    msg = JointState()
    msg.name = list(JOINT_NAMES[:-1])
    msg.position = [0.0] * (ACTION_DIM - 1)

    with pytest.raises(ValueError, match="missing required joints"):
        extract_joint_positions_rad(msg)
