from __future__ import annotations

import json
import math
import socket
from collections.abc import Sequence

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from bdx_policy_deploy.policy_interface import ACTION_DIM, JOINT_NAMES


def extract_joint_positions_rad(msg: JointState, joint_names: Sequence[str] = JOINT_NAMES) -> np.ndarray:
    if len(msg.position) == 0:
        raise ValueError("JointState.position is empty")

    if msg.name:
        if len(msg.position) != len(msg.name):
            raise ValueError("JointState.name and JointState.position lengths differ")
        if len(set(msg.name)) != len(msg.name):
            raise ValueError("JointState.name contains duplicate entries")

        index_by_name = {name: index for index, name in enumerate(msg.name)}
        missing = [name for name in joint_names if name not in index_by_name]
        if missing:
            raise ValueError(f"JointState missing required joints: {', '.join(missing)}")
        positions = np.array([msg.position[index_by_name[name]] for name in joint_names], dtype=np.float32)
    else:
        positions = np.asarray(msg.position, dtype=np.float32).reshape(-1)
        if positions.shape != (len(joint_names),):
            raise ValueError(f"JointState.position must have {len(joint_names)} entries, got {positions.shape[0]}")

    if not np.all(np.isfinite(positions)):
        raise ValueError("JointState.position contains NaN or Inf")
    return positions


def build_joint_pose_payload_json(
    msg: JointState,
    joint_names: Sequence[str] = JOINT_NAMES,
    sequence: int = 0,
    include_degrees: bool = False,
) -> str:
    positions_rad = extract_joint_positions_rad(msg, joint_names)
    payload: dict[str, object] = {
        "type": "bdx_joint_pose",
        "seq": int(sequence),
        "stamp": {
            "sec": int(msg.header.stamp.sec),
            "nanosec": int(msg.header.stamp.nanosec),
        },
        "joint_names": list(joint_names),
        "position_rad": [float(value) for value in positions_rad],
    }
    if include_degrees:
        payload["position_deg"] = [math.degrees(float(value)) for value in positions_rad]
    return json.dumps(payload, separators=(",", ":"))


class JointPoseSocketBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("bdx_joint_pose_socket_bridge_node")
        self._declare_parameters()
        self._load_parameters()

        self.socket_handle: socket.socket | None = None
        self.sequence = 0
        self.last_send_ns: int | None = None
        self.last_connect_attempt_ns: int | None = None

        self.target_sub = self.create_subscription(
            JointState,
            self.target_joint_state_topic,
            self._on_target_joint_state,
            10,
        )

        self.get_logger().info(
            "Joint pose socket bridge started: %s -> %s://%s:%d"
            % (self.target_joint_state_topic, self.socket_protocol, self.remote_host, self.remote_port)
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("target_joint_state_topic", "/bdx_policy/target_joint_states")
        self.declare_parameter("remote_host", "192.168.31.202")
        self.declare_parameter("remote_port", 2333)
        self.declare_parameter("socket_protocol", "udp")
        self.declare_parameter("connect_timeout_s", 0.2)
        self.declare_parameter("reconnect_period_s", 1.0)
        self.declare_parameter("send_rate_limit_hz", 0.0)
        self.declare_parameter("include_degrees", False)
        self.declare_parameter("joint_names", JOINT_NAMES)

    def _load_parameters(self) -> None:
        self.target_joint_state_topic = str(self.get_parameter("target_joint_state_topic").value)
        self.remote_host = str(self.get_parameter("remote_host").value)
        self.remote_port = int(self.get_parameter("remote_port").value)
        self.socket_protocol = str(self.get_parameter("socket_protocol").value).lower()
        self.connect_timeout_s = float(self.get_parameter("connect_timeout_s").value)
        self.reconnect_period_s = float(self.get_parameter("reconnect_period_s").value)
        self.send_rate_limit_hz = float(self.get_parameter("send_rate_limit_hz").value)
        self.include_degrees = bool(self.get_parameter("include_degrees").value)
        self.joint_names = [str(name) for name in self.get_parameter("joint_names").value]

        if len(self.joint_names) != ACTION_DIM:
            raise ValueError(f"joint_names must have {ACTION_DIM} entries")
        if not self.remote_host:
            raise ValueError("remote_host must not be empty")
        if not 0 < self.remote_port <= 65535:
            raise ValueError("remote_port must be in 1..65535")
        if self.socket_protocol not in ("udp", "tcp"):
            raise ValueError("socket_protocol must be 'udp' or 'tcp'")
        if self.connect_timeout_s <= 0.0:
            raise ValueError("connect_timeout_s must be positive")
        if self.reconnect_period_s <= 0.0:
            raise ValueError("reconnect_period_s must be positive")
        if self.send_rate_limit_hz < 0.0:
            raise ValueError("send_rate_limit_hz must be non-negative")

    def _on_target_joint_state(self, msg: JointState) -> None:
        if not self._rate_limit_allows_send():
            return

        try:
            payload = build_joint_pose_payload_json(
                msg,
                self.joint_names,
                sequence=self.sequence,
                include_degrees=self.include_degrees,
            )
        except ValueError as exc:
            self.get_logger().warning(f"Ignoring invalid target JointState: {exc}", throttle_duration_sec=1.0)
            return

        encoded = (payload + "\n").encode("utf-8")
        if self._send(encoded):
            self.sequence += 1
            self.last_send_ns = self.get_clock().now().nanoseconds

    def _rate_limit_allows_send(self) -> bool:
        if self.send_rate_limit_hz <= 0.0 or self.last_send_ns is None:
            return True
        min_period_ns = int(1e9 / self.send_rate_limit_hz)
        return self.get_clock().now().nanoseconds - self.last_send_ns >= min_period_ns

    def _send(self, payload: bytes) -> bool:
        try:
            if self.socket_protocol == "udp":
                self._ensure_udp_socket()
                assert self.socket_handle is not None
                self.socket_handle.sendto(payload, (self.remote_host, self.remote_port))
            else:
                if not self._ensure_tcp_socket():
                    return False
                assert self.socket_handle is not None
                self.socket_handle.sendall(payload)
            return True
        except OSError as exc:
            self._close_socket()
            self.get_logger().warning(
                "Failed to send joint pose socket payload to %s:%d: %s"
                % (self.remote_host, self.remote_port, exc),
                throttle_duration_sec=1.0,
            )
            return False

    def _ensure_udp_socket(self) -> None:
        if self.socket_handle is None:
            self.socket_handle = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _ensure_tcp_socket(self) -> bool:
        if self.socket_handle is not None:
            return True

        now_ns = self.get_clock().now().nanoseconds
        if (
            self.last_connect_attempt_ns is not None
            and now_ns - self.last_connect_attempt_ns < int(self.reconnect_period_s * 1e9)
        ):
            return False
        self.last_connect_attempt_ns = now_ns

        try:
            self.socket_handle = socket.create_connection(
                (self.remote_host, self.remote_port),
                timeout=self.connect_timeout_s,
            )
            self.socket_handle.settimeout(self.connect_timeout_s)
            return True
        except OSError as exc:
            self._close_socket()
            self.get_logger().warning(
                "Failed to connect joint pose socket to %s:%d: %s" % (self.remote_host, self.remote_port, exc),
                throttle_duration_sec=1.0,
            )
            return False

    def _close_socket(self) -> None:
        if self.socket_handle is None:
            return
        try:
            self.socket_handle.close()
        finally:
            self.socket_handle = None

    def destroy_node(self) -> None:
        self._close_socket()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: JointPoseSocketBridgeNode | None = None
    try:
        node = JointPoseSocketBridgeNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
