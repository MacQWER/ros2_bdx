from __future__ import annotations

import json
import math
from typing import Any

from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import Twist
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


VALID_POLICY_MODES = ("disabled", "zero_action", "policy")


def yaw_from_quaternion_wxyz(quaternion_wxyz: np.ndarray) -> float:
    w, x, y, z = quaternion_wxyz.astype(float)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0 or not math.isfinite(norm):
        return 0.0
    w /= norm
    x /= norm
    y /= norm
    z /= norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class PygameHeadingCommandNode(Node):
    def __init__(self) -> None:
        super().__init__("bdx_pygame_heading_command_node")
        self._declare_parameters()
        self._load_parameters()

        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "pygame is required by bdx_pygame_heading_command_node. Install it in the "
                "ROS 2 Python environment: /usr/bin/python3 -m pip install --user pygame --break-system-packages"
            ) from exc

        self.pygame = pygame
        pygame.init()
        pygame.display.set_caption("BDX Heading Command")
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)

        self.command_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.mode_pub = self.create_publisher(String, self.policy_mode_topic, 10)
        self.obs_sub = self.create_subscription(Float32MultiArray, self.observation_topic, self._on_observation, 10)
        self.real_obs_sub = self.create_subscription(
            Float32MultiArray,
            self.real_observation_topic,
            self._on_real_observation,
            10,
        )
        self.obs_error_sub = self.create_subscription(
            Float32MultiArray,
            self.observation_error_topic,
            self._on_observation_error,
            10,
        )
        self.obs_compare_sub = self.create_subscription(
            String,
            self.observation_compare_topic,
            self._on_observation_compare,
            10,
        )
        self.base_sub = self.create_subscription(Float32MultiArray, self.base_state_topic, self._on_base_state, 10)
        self.diag_sub = self.create_subscription(DiagnosticArray, self.diagnostics_topic, self._on_diagnostics, 10)
        self.command_timer = self.create_timer(1.0 / self.publish_rate_hz, self._publish_command)

        self.linear_x = float(np.clip(self.initial_linear_x, self.linear_x_min, self.linear_x_max))
        self.linear_y = float(np.clip(self.initial_linear_y, self.linear_y_min, self.linear_y_max))
        self.target_heading = self.initial_heading_rad
        self.policy_mode = self.initial_policy_mode
        self.current_yaw = 0.0
        self.current_base = np.zeros(14, dtype=np.float32)
        self.last_observation: np.ndarray | None = None
        self.last_real_observation: np.ndarray | None = None
        self.last_observation_error: np.ndarray | None = None
        self.last_observation_compare: dict[str, Any] | None = None
        self.last_diag_message = "waiting"
        self.last_yaw_rate = 0.0

        self.get_logger().info(
            "Pygame heading command node started: cmd_topic=%s, heading_kp=%.3f"
            % (self.cmd_vel_topic, self.heading_kp)
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("policy_mode_topic", "/bdx_policy/mode")
        self.declare_parameter("observation_topic", "/bdx_policy/debug/observation")
        self.declare_parameter("real_observation_topic", "/bdx_policy/debug/real_observation")
        self.declare_parameter("observation_error_topic", "/bdx_policy/debug/real_minus_sim_observation")
        self.declare_parameter("observation_compare_topic", "/bdx_policy/debug/observation_compare")
        self.declare_parameter("base_state_topic", "/bdx_mujoco/debug/base_state")
        self.declare_parameter("diagnostics_topic", "/bdx_policy/diagnostics")
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("window_width", 1000)
        self.declare_parameter("window_height", 720)

        self.declare_parameter("initial_linear_x", 0.0)
        self.declare_parameter("initial_linear_y", 0.0)
        self.declare_parameter("initial_heading_deg", 0.0)
        self.declare_parameter("initial_policy_mode", "disabled")
        self.declare_parameter("linear_x_step", 0.2)
        self.declare_parameter("linear_y_step", 0.3)
        self.declare_parameter("heading_step_deg", 30.0)
        self.declare_parameter("heading_kp", 1.5)

        self.declare_parameter("linear_x_min", -0.4)
        self.declare_parameter("linear_x_max", 0.7)
        self.declare_parameter("linear_y_min", -0.4)
        self.declare_parameter("linear_y_max", 0.4)
        self.declare_parameter("max_yaw_rate", 1.0)

    def _load_parameters(self) -> None:
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.policy_mode_topic = str(self.get_parameter("policy_mode_topic").value)
        self.observation_topic = str(self.get_parameter("observation_topic").value)
        self.real_observation_topic = str(self.get_parameter("real_observation_topic").value)
        self.observation_error_topic = str(self.get_parameter("observation_error_topic").value)
        self.observation_compare_topic = str(self.get_parameter("observation_compare_topic").value)
        self.base_state_topic = str(self.get_parameter("base_state_topic").value)
        self.diagnostics_topic = str(self.get_parameter("diagnostics_topic").value)
        self.publish_rate_hz = self._positive_float_parameter("publish_rate_hz")
        self.window_width = int(self.get_parameter("window_width").value)
        self.window_height = int(self.get_parameter("window_height").value)

        self.initial_linear_x = float(self.get_parameter("initial_linear_x").value)
        self.initial_linear_y = float(self.get_parameter("initial_linear_y").value)
        self.initial_heading_rad = math.radians(float(self.get_parameter("initial_heading_deg").value))
        self.initial_policy_mode = str(self.get_parameter("initial_policy_mode").value)
        self.linear_x_step = float(self.get_parameter("linear_x_step").value)
        self.linear_y_step = float(self.get_parameter("linear_y_step").value)
        self.heading_step_rad = math.radians(float(self.get_parameter("heading_step_deg").value))
        self.heading_kp = float(self.get_parameter("heading_kp").value)

        self.linear_x_min = float(self.get_parameter("linear_x_min").value)
        self.linear_x_max = float(self.get_parameter("linear_x_max").value)
        self.linear_y_min = float(self.get_parameter("linear_y_min").value)
        self.linear_y_max = float(self.get_parameter("linear_y_max").value)
        self.max_yaw_rate = float(self.get_parameter("max_yaw_rate").value)

        if self.window_width <= 0 or self.window_height <= 0:
            raise ValueError("window_width and window_height must be positive")
        if self.linear_x_min > self.linear_x_max or self.linear_y_min > self.linear_y_max:
            raise ValueError("linear command min must be <= max")
        if self.max_yaw_rate <= 0.0:
            raise ValueError("max_yaw_rate must be positive")
        if self.linear_x_step <= 0.0 or self.linear_y_step <= 0.0:
            raise ValueError("linear_x_step and linear_y_step must be positive")
        if self.initial_policy_mode not in VALID_POLICY_MODES:
            raise ValueError(f"initial_policy_mode must be one of {VALID_POLICY_MODES}")

    def _positive_float_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
        return value

    def _on_observation(self, msg: Float32MultiArray) -> None:
        obs = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if obs.shape == (39,) and np.all(np.isfinite(obs)):
            self.last_observation = obs

    def _on_real_observation(self, msg: Float32MultiArray) -> None:
        obs = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if obs.shape == (39,) and np.all(np.isfinite(obs)):
            self.last_real_observation = obs

    def _on_observation_error(self, msg: Float32MultiArray) -> None:
        error = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if error.shape == (39,) and np.all(np.isfinite(error)):
            self.last_observation_error = error

    def _on_observation_compare(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict):
            self.last_observation_compare = payload

    def _on_base_state(self, msg: Float32MultiArray) -> None:
        data = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if data.shape[0] >= 8 and np.all(np.isfinite(data[:8])):
            self.current_base = np.zeros(14, dtype=np.float32)
            self.current_base[: min(data.shape[0], 14)] = data[:14]
            self.current_yaw = yaw_from_quaternion_wxyz(data[4:8])

    def _on_diagnostics(self, msg: DiagnosticArray) -> None:
        if msg.status:
            self.last_diag_message = msg.status[0].message

    def _heading_error(self) -> float:
        return wrap_to_pi(self.target_heading - self.current_yaw)

    def _command_yaw_rate(self) -> float:
        yaw_rate = self.heading_kp * self._heading_error()
        return float(np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate))

    def _publish_command(self) -> None:
        self.last_yaw_rate = self._command_yaw_rate()
        msg = Twist()
        msg.linear.x = float(self.linear_x)
        msg.linear.y = float(self.linear_y)
        msg.angular.z = float(self.last_yaw_rate)
        self.command_pub.publish(msg)
        self._publish_mode()

    def _publish_mode(self) -> None:
        msg = String()
        msg.data = self.policy_mode
        self.mode_pub.publish(msg)

    def _handle_keydown(self, key: int, modifiers: int) -> None:
        pygame = self.pygame
        speed = 3.0 if modifiers & pygame.KMOD_SHIFT else 1.0
        linear_x_step = self.linear_x_step * speed
        linear_y_step = self.linear_y_step * speed
        heading_step = self.heading_step_rad * speed

        if key in (pygame.K_UP, pygame.K_w):
            self.linear_x = float(np.clip(self.linear_x + linear_x_step, self.linear_x_min, self.linear_x_max))
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.linear_x = float(np.clip(self.linear_x - linear_x_step, self.linear_x_min, self.linear_x_max))
        elif key in (pygame.K_a,):
            self.linear_y = float(np.clip(self.linear_y + linear_y_step, self.linear_y_min, self.linear_y_max))
        elif key in (pygame.K_d,):
            self.linear_y = float(np.clip(self.linear_y - linear_y_step, self.linear_y_min, self.linear_y_max))
        elif key in (pygame.K_LEFT, pygame.K_q):
            self.target_heading = wrap_to_pi(self.target_heading + heading_step)
        elif key in (pygame.K_RIGHT, pygame.K_e):
            self.target_heading = wrap_to_pi(self.target_heading - heading_step)
        elif key == pygame.K_1:
            self.policy_mode = "disabled"
            self._publish_mode()
        elif key == pygame.K_2:
            self.policy_mode = "zero_action"
            self._publish_mode()
        elif key == pygame.K_3:
            self.policy_mode = "policy"
            self._publish_mode()
        elif key == pygame.K_r:
            self.target_heading = self.current_yaw
        elif key == pygame.K_SPACE:
            self.linear_x = 0.0
            self.linear_y = 0.0
            self.target_heading = self.current_yaw

    def spin_ui(self) -> None:
        pygame = self.pygame
        running = True
        while rclpy.ok() and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    else:
                        self._handle_keydown(event.key, pygame.key.get_mods())

            rclpy.spin_once(self, timeout_sec=0.0)
            self._draw()
            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()

    def _draw_text(self, text: str, x: int, y: int, color: tuple[int, int, int] = (230, 230, 230)) -> int:
        surface = self.font.render(text, True, color)
        self.screen.blit(surface, (x, y))
        return y + 26

    def _draw_small_text(self, text: str, x: int, y: int, color: tuple[int, int, int] = (210, 210, 210)) -> int:
        surface = self.small_font.render(text, True, color)
        self.screen.blit(surface, (x, y))
        return y + 21

    def _draw_vector_values(
        self,
        label: str,
        values: np.ndarray,
        x: int,
        y: int,
        color: tuple[int, int, int] = (210, 210, 210),
        max_chars: int = 66,
    ) -> int:
        line = f"{label}: " + " ".join(f"{value:+.3f}" for value in values)
        while line:
            chunk = line[:max_chars]
            line = line[max_chars:]
            y = self._draw_small_text(chunk, x, y, color)
        return y

    def _comparison_text_for_block(self, block_name: str) -> str | None:
        if self.last_observation_compare is None:
            return None
        blocks = self.last_observation_compare.get("blocks")
        if not isinstance(blocks, dict):
            return None
        block = blocks.get(block_name)
        if not isinstance(block, dict):
            return None
        max_abs = block.get("max_abs")
        rms = block.get("rms")
        if not isinstance(max_abs, (float, int)) or not isinstance(rms, (float, int)):
            return None
        return f"err max/rms: {float(max_abs):.4f} / {float(rms):.4f}"

    def _draw(self) -> None:
        pygame = self.pygame
        self.screen.fill((18, 21, 24))
        ok_color = (108, 219, 141) if self.last_diag_message == "ok" else (245, 196, 90)

        yaw_deg = math.degrees(self.current_yaw)
        target_deg = math.degrees(self.target_heading)
        error_deg = math.degrees(self._heading_error())

        y = 18
        y = self._draw_text("BDX Heading Command", 24, y, (255, 255, 255))
        y = self._draw_text(f"diag: {self.last_diag_message}", 24, y, ok_color)
        y = self._draw_text(f"mode: {self.policy_mode}", 24, y, (168, 204, 255))
        y = self._draw_text(f"vx: {self.linear_x:+.2f}  vy: {self.linear_y:+.2f}  yaw_rate_cmd: {self.last_yaw_rate:+.2f}", 24, y)
        y = self._draw_text(f"heading target/current/error: {target_deg:+.1f} / {yaw_deg:+.1f} / {error_deg:+.1f} deg", 24, y)
        y = self._draw_text(
            f"base xyz: {self.current_base[1]:+.2f}, {self.current_base[2]:+.2f}, {self.current_base[3]:+.2f}",
            24,
            y,
        )

        y += 10
        y = self._draw_small_text(
            "Controls: 1 disabled, 2 zero action, 3 policy, W/S vx, A/D vy, Q/E heading, Space stop, Esc quit",
            24,
            y,
        )

        pygame.draw.line(self.screen, (70, 75, 82), (24, y + 8), (self.window_width - 24, y + 8), 1)
        y += 28

        if self.last_observation is None:
            self._draw_text("obs: waiting", 24, y, (245, 196, 90))
            return

        compare_text = "real obs: waiting"
        compare_color = (245, 196, 90)
        if self.last_real_observation is not None:
            compare_text = "real obs: received"
            compare_color = (108, 219, 141)
        if self.last_observation_compare is not None:
            max_abs = self.last_observation_compare.get("max_abs")
            rms = self.last_observation_compare.get("rms")
            if isinstance(max_abs, (float, int)) and isinstance(rms, (float, int)):
                compare_text = f"real-vs-sim obs: max_abs={float(max_abs):.4f} rms={float(rms):.4f}"
                compare_color = (108, 219, 141) if float(max_abs) < 0.05 else (245, 196, 90)
        y = self._draw_text(compare_text, 24, y, compare_color)
        y += 4

        obs = self.last_observation
        real_obs = self.last_real_observation
        error_obs = self.last_observation_error
        groups: list[tuple[str, str, slice]] = [
            ("imu_ang_vel_scaled", "imu_ang_vel", slice(0, 3)),
            ("projected_gravity", "projected_gravity", slice(3, 6)),
            ("joint_pos_minus_default", "joint_pos", slice(6, 16)),
            ("joint_vel_scaled", "joint_vel", slice(16, 26)),
            ("last_action", "last_action", slice(26, 36)),
            ("policy_command", "command", slice(36, 39)),
        ]

        left_x = 24
        right_x = 520
        left_y = y
        right_y = y
        for index, (name, block_name, obs_slice) in enumerate(groups):
            x = left_x if index < 3 else right_x
            draw_y = left_y if index < 3 else right_y
            draw_y = self._draw_text(name, x, draw_y, (168, 204, 255))
            draw_y = self._draw_vector_values("sim ", obs[obs_slice], x, draw_y)
            if real_obs is not None:
                draw_y = self._draw_vector_values("real", real_obs[obs_slice], x, draw_y, (170, 235, 180))
            if error_obs is not None:
                comparison_text = self._comparison_text_for_block(block_name)
                if comparison_text is not None:
                    draw_y = self._draw_small_text(comparison_text, x, draw_y, (245, 196, 90))
            draw_y += 10
            if index < 3:
                left_y = draw_y
            else:
                right_y = draw_y


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: PygameHeadingCommandNode | None = None
    try:
        node = PygameHeadingCommandNode()
        node.spin_ui()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
