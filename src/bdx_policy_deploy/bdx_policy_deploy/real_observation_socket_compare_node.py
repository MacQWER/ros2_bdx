from __future__ import annotations

import json
import socket
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from bdx_policy_deploy.policy_interface import ACTION_DIM, OBS_DIM, as_float32_vector


OBSERVATION_BLOCKS = (
    ("imu_ang_vel", 0, 3),
    ("projected_gravity", 3, 6),
    ("joint_pos", 6, 16),
    ("joint_vel", 16, 26),
    ("last_action", 26, 36),
    ("command", 36, 39),
)


def parse_real_observation_payload(data: bytes) -> np.ndarray:
    stripped = data.strip()
    if len(data) == OBS_DIM * 4 and stripped[:1] not in (b"{", b"["):
        obs = np.frombuffer(data, dtype="<f4").astype(np.float32)
        return as_float32_vector(obs, OBS_DIM, "real_observation")

    try:
        payload = json.loads(stripped.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("real observation packet must be JSON or 39 little-endian float32 values") from exc

    values = _extract_observation_values(payload)
    return as_float32_vector(values, OBS_DIM, "real_observation")


def _extract_observation_values(payload: Any) -> Any:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("real observation JSON must be a list or object")

    for key in ("obs", "observation", "policy_obs"):
        if key in payload:
            return payload[key]

    return [
        *_component(payload, ("imu_ang_vel", "imu_ang_vel_rad_s"), 3),
        *_component(payload, ("projected_gravity", "imu_projected_gravity"), 3),
        *_component(payload, ("joint_pos", "joint_pos_rad"), ACTION_DIM),
        *_component(payload, ("joint_vel", "joint_vel_rad_s"), ACTION_DIM),
        *_component(payload, ("last_action", "action"), ACTION_DIM),
        *_component(payload, ("command", "cmd"), 3),
    ]


def _component(payload: dict[str, Any], keys: tuple[str, ...], size: int) -> list[float]:
    for key in keys:
        if key in payload:
            return as_float32_vector(payload[key], size, key).astype(float).tolist()
    raise ValueError(f"real observation JSON missing component: {keys[0]}")


def summarize_observation_difference(real_obs: np.ndarray, sim_obs: np.ndarray) -> dict[str, Any]:
    real = as_float32_vector(real_obs, OBS_DIM, "real_observation")
    sim = as_float32_vector(sim_obs, OBS_DIM, "sim_observation")
    error = (real - sim).astype(np.float32)
    summary: dict[str, Any] = {
        "max_abs": float(np.max(np.abs(error))),
        "rms": float(np.sqrt(np.mean(np.square(error)))),
        "blocks": {},
    }
    blocks: dict[str, Any] = {}
    for name, start, end in OBSERVATION_BLOCKS:
        block = error[start:end]
        blocks[name] = {
            "max_abs": float(np.max(np.abs(block))),
            "rms": float(np.sqrt(np.mean(np.square(block)))),
        }
    summary["blocks"] = blocks
    return summary


class RealObservationSocketCompareNode(Node):
    def __init__(self) -> None:
        super().__init__("bdx_real_observation_socket_compare_node")
        self._declare_parameters()
        self._load_parameters()

        self.socket_handle = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket_handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket_handle.bind((self.bind_host, self.bind_port))
        self.socket_handle.setblocking(False)

        self.real_obs: np.ndarray | None = None
        self.sim_obs: np.ndarray | None = None
        self.last_real_obs_ns: int | None = None
        self.last_sim_obs_ns: int | None = None
        self.real_sequence = 0
        self.policy_mode = self.initial_policy_mode

        self.sim_sub = self.create_subscription(Float32MultiArray, self.sim_observation_topic, self._on_sim_obs, 10)
        self.mode_sub = self.create_subscription(String, self.policy_mode_topic, self._on_policy_mode, 10)
        self.real_pub = self.create_publisher(Float32MultiArray, self.real_observation_topic, 10)
        self.error_pub = self.create_publisher(Float32MultiArray, self.observation_error_topic, 10)
        self.summary_pub = self.create_publisher(String, self.comparison_topic, 10)
        self.poll_timer = self.create_timer(1.0 / self.poll_rate_hz, self._poll_socket)

        remote = self.remote_host if self.remote_host else "any"
        self.get_logger().info(
            "Real observation compare node started: udp://%s:%d, remote=%s, sim_topic=%s, compare_mode=%s"
            % (
                self.bind_host,
                self.bind_port,
                remote,
                self.sim_observation_topic,
                self.compare_only_in_policy_mode or "any",
            )
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("bind_host", "0.0.0.0")
        self.declare_parameter("bind_port", 2333)
        self.declare_parameter("remote_host", "192.168.31.202")
        self.declare_parameter("poll_rate_hz", 200.0)
        self.declare_parameter("max_packets_per_tick", 10)
        self.declare_parameter("stale_timeout_s", 0.5)
        self.declare_parameter("compare_only_in_policy_mode", "zero_action")
        self.declare_parameter("initial_policy_mode", "disabled")
        self.declare_parameter("sim_observation_topic", "/bdx_policy/debug/observation")
        self.declare_parameter("policy_mode_topic", "/bdx_policy/mode")
        self.declare_parameter("real_observation_topic", "/bdx_policy/debug/real_observation")
        self.declare_parameter("observation_error_topic", "/bdx_policy/debug/real_minus_sim_observation")
        self.declare_parameter("comparison_topic", "/bdx_policy/debug/observation_compare")

    def _load_parameters(self) -> None:
        self.bind_host = str(self.get_parameter("bind_host").value)
        self.bind_port = int(self.get_parameter("bind_port").value)
        self.remote_host = str(self.get_parameter("remote_host").value)
        self.poll_rate_hz = float(self.get_parameter("poll_rate_hz").value)
        self.max_packets_per_tick = int(self.get_parameter("max_packets_per_tick").value)
        self.stale_timeout_s = float(self.get_parameter("stale_timeout_s").value)
        self.compare_only_in_policy_mode = str(self.get_parameter("compare_only_in_policy_mode").value)
        self.initial_policy_mode = str(self.get_parameter("initial_policy_mode").value)
        self.sim_observation_topic = str(self.get_parameter("sim_observation_topic").value)
        self.policy_mode_topic = str(self.get_parameter("policy_mode_topic").value)
        self.real_observation_topic = str(self.get_parameter("real_observation_topic").value)
        self.observation_error_topic = str(self.get_parameter("observation_error_topic").value)
        self.comparison_topic = str(self.get_parameter("comparison_topic").value)

        if not 0 < self.bind_port <= 65535:
            raise ValueError("bind_port must be in 1..65535")
        if self.poll_rate_hz <= 0.0:
            raise ValueError("poll_rate_hz must be positive")
        if self.max_packets_per_tick <= 0:
            raise ValueError("max_packets_per_tick must be positive")
        if self.stale_timeout_s <= 0.0:
            raise ValueError("stale_timeout_s must be positive")

    def _poll_socket(self) -> None:
        for _ in range(self.max_packets_per_tick):
            try:
                data, address = self.socket_handle.recvfrom(65535)
            except BlockingIOError:
                return
            except OSError as exc:
                self.get_logger().warning(f"Failed to receive real observation packet: {exc}", throttle_duration_sec=1.0)
                return

            if self.remote_host and address[0] != self.remote_host:
                self.get_logger().warning(
                    "Ignoring real observation packet from unexpected host %s" % address[0],
                    throttle_duration_sec=1.0,
                )
                continue

            try:
                obs = parse_real_observation_payload(data)
            except ValueError as exc:
                self.get_logger().warning(f"Ignoring invalid real observation packet: {exc}", throttle_duration_sec=1.0)
                continue

            self.real_obs = obs
            self.last_real_obs_ns = self.get_clock().now().nanoseconds
            self.real_sequence += 1
            self.real_pub.publish(Float32MultiArray(data=[float(value) for value in obs]))
            self._publish_comparison_if_ready()

    def _on_sim_obs(self, msg: Float32MultiArray) -> None:
        try:
            self.sim_obs = as_float32_vector(msg.data, OBS_DIM, "sim_observation")
        except ValueError as exc:
            self.get_logger().warning(f"Ignoring invalid sim observation: {exc}", throttle_duration_sec=1.0)
            return
        self.last_sim_obs_ns = self.get_clock().now().nanoseconds
        self._publish_comparison_if_ready()

    def _on_policy_mode(self, msg: String) -> None:
        self.policy_mode = msg.data.strip()

    def _publish_comparison_if_ready(self) -> None:
        if self.real_obs is None or self.sim_obs is None:
            return
        if self.compare_only_in_policy_mode and self.policy_mode != self.compare_only_in_policy_mode:
            return
        now_ns = self.get_clock().now().nanoseconds
        if self.last_real_obs_ns is None or (now_ns - self.last_real_obs_ns) * 1e-9 > self.stale_timeout_s:
            return
        if self.last_sim_obs_ns is None or (now_ns - self.last_sim_obs_ns) * 1e-9 > self.stale_timeout_s:
            return

        error = (self.real_obs - self.sim_obs).astype(np.float32)
        summary = summarize_observation_difference(self.real_obs, self.sim_obs)
        summary["seq"] = self.real_sequence
        summary["mode"] = self.policy_mode
        summary["real_age_s"] = (now_ns - self.last_real_obs_ns) * 1e-9
        summary["sim_age_s"] = (now_ns - self.last_sim_obs_ns) * 1e-9

        self.error_pub.publish(Float32MultiArray(data=[float(value) for value in error]))
        self.summary_pub.publish(String(data=json.dumps(summary, separators=(",", ":"))))

    def destroy_node(self) -> None:
        self.socket_handle.close()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: RealObservationSocketCompareNode | None = None
    try:
        node = RealObservationSocketCompareNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
