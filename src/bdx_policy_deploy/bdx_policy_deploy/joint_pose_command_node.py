from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from bdx_policy_deploy.policy_interface import (
    ACTION_DIM,
    DEFAULT_JOINT_LOWER_LIMITS,
    DEFAULT_JOINT_POS,
    DEFAULT_JOINT_UPPER_LIMITS,
    JOINT_NAMES,
    PolicyInterfaceError,
    reorder_joint_state,
)


class JointPoseCommandNode(Node):
    def __init__(self) -> None:
        super().__init__("bdx_joint_pose_command_node")
        self._declare_parameters()
        self._load_parameters()

        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "pygame is required by bdx_joint_pose_command_node. Install it in the "
                "ROS 2 Python environment: /usr/bin/python3 -m pip install --user pygame --break-system-packages"
            ) from exc

        self.pygame = pygame
        pygame.init()
        pygame.display.set_caption("BDX Joint Pose Command")
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)

        self.target_joint_pos = self.initial_joint_pos.copy()
        self.current_joint_pos: np.ndarray | None = None
        self.current_joint_vel: np.ndarray | None = None
        self.selected_joint_index = 0
        self.active_drag_index: int | None = None
        self.slider_left = 310
        self.slider_right_margin = 34
        self.slider_top = 190
        self.slider_row_height = 54

        self.target_pub = self.create_publisher(JointState, self.target_joint_state_topic, 10)
        self.mode_pub = self.create_publisher(String, self.policy_mode_topic, 10)
        self.joint_sub = self.create_subscription(JointState, self.joint_state_topic, self._on_joint_state, 10)
        self.publish_timer = self.create_timer(1.0 / self.publish_rate_hz, self._publish_target)

        self.get_logger().info(
            "Joint pose command node started: target_topic=%s, publish_rate=%.1f Hz"
            % (self.target_joint_state_topic, self.publish_rate_hz)
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("target_joint_state_topic", "/bdx_policy/target_joint_states")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("policy_mode_topic", "/bdx_policy/mode")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("window_width", 1040)
        self.declare_parameter("window_height", 720)
        self.declare_parameter("joint_names", JOINT_NAMES)
        self.declare_parameter("initial_joint_pos", DEFAULT_JOINT_POS.tolist())
        self.declare_parameter("joint_lower_limits", DEFAULT_JOINT_LOWER_LIMITS.tolist())
        self.declare_parameter("joint_upper_limits", DEFAULT_JOINT_UPPER_LIMITS.tolist())
        self.declare_parameter("step_deg", 1.0)
        self.declare_parameter("large_step_multiplier", 5.0)
        self.declare_parameter("publish_policy_mode", True)
        self.declare_parameter("policy_mode", "disabled")

    def _load_parameters(self) -> None:
        self.target_joint_state_topic = str(self.get_parameter("target_joint_state_topic").value)
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.policy_mode_topic = str(self.get_parameter("policy_mode_topic").value)
        self.publish_rate_hz = self._positive_float_parameter("publish_rate_hz")
        self.window_width = int(self.get_parameter("window_width").value)
        self.window_height = int(self.get_parameter("window_height").value)
        self.joint_names = [str(name) for name in self.get_parameter("joint_names").value]
        self.initial_joint_pos = self._vector_parameter("initial_joint_pos", ACTION_DIM)
        self.joint_lower_limits = self._vector_parameter("joint_lower_limits", ACTION_DIM)
        self.joint_upper_limits = self._vector_parameter("joint_upper_limits", ACTION_DIM)
        self.step_rad = math.radians(float(self.get_parameter("step_deg").value))
        self.large_step_multiplier = float(self.get_parameter("large_step_multiplier").value)
        self.publish_policy_mode = bool(self.get_parameter("publish_policy_mode").value)
        self.policy_mode = str(self.get_parameter("policy_mode").value)

        if len(self.joint_names) != ACTION_DIM:
            raise ValueError(f"joint_names must have {ACTION_DIM} entries")
        if self.window_width <= 0 or self.window_height <= 0:
            raise ValueError("window_width and window_height must be positive")
        if self.step_rad <= 0.0:
            raise ValueError("step_deg must be positive")
        if self.large_step_multiplier <= 0.0:
            raise ValueError("large_step_multiplier must be positive")
        if np.any(self.joint_lower_limits > self.joint_upper_limits):
            raise ValueError("joint_lower_limits must be <= joint_upper_limits")

    def _positive_float_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
        return value

    def _vector_parameter(self, name: str, size: int) -> np.ndarray:
        vector = np.asarray(self.get_parameter(name).value, dtype=np.float32).reshape(-1)
        if vector.shape != (size,):
            raise ValueError(f"{name} must have {size} entries, got {vector.shape[0]}")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} contains NaN or Inf")
        return vector

    def _on_joint_state(self, msg: JointState) -> None:
        try:
            self.current_joint_pos, self.current_joint_vel = reorder_joint_state(
                msg.name,
                msg.position,
                msg.velocity,
                self.joint_names,
            )
        except PolicyInterfaceError as exc:
            self.get_logger().warning(f"Ignoring invalid JointState: {exc}", throttle_duration_sec=1.0)

    def _publish_target(self) -> None:
        if self.publish_policy_mode:
            mode_msg = String()
            mode_msg.data = self.policy_mode
            self.mode_pub.publish(mode_msg)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_names)
        msg.position = self.target_joint_pos.astype(float).tolist()
        msg.velocity = []
        msg.effort = []
        self.target_pub.publish(msg)

    def _handle_keydown(self, key: int, modifiers: int) -> None:
        pygame = self.pygame
        multiplier = self.large_step_multiplier if modifiers & pygame.KMOD_SHIFT else 1.0
        step = self.step_rad * multiplier

        if pygame.K_1 <= key <= pygame.K_9:
            self.selected_joint_index = int(key - pygame.K_1)
        elif key == pygame.K_0:
            self.selected_joint_index = 9
        elif key in (pygame.K_UP, pygame.K_w):
            self.selected_joint_index = (self.selected_joint_index - 1) % ACTION_DIM
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.selected_joint_index = (self.selected_joint_index + 1) % ACTION_DIM
        elif key in (pygame.K_LEFT, pygame.K_a):
            self._nudge_selected(-step)
        elif key in (pygame.K_RIGHT, pygame.K_d):
            self._nudge_selected(step)
        elif key == pygame.K_HOME:
            self._reset_selected()
        elif key in (pygame.K_BACKSPACE, pygame.K_r):
            self._reset_all()

    def _handle_mouse_down(self, pos: tuple[int, int]) -> None:
        index = self._slider_hit_index(pos)
        if index is None:
            return
        self.active_drag_index = index
        self.selected_joint_index = index
        self._set_slider_from_x(index, pos[0])

    def _handle_mouse_motion(self, pos: tuple[int, int]) -> None:
        if self.active_drag_index is None:
            return
        self._set_slider_from_x(self.active_drag_index, pos[0])

    def _handle_mouse_up(self) -> None:
        self.active_drag_index = None

    def _slider_hit_index(self, pos: tuple[int, int]) -> int | None:
        x, y = pos
        left, right = self._slider_bounds()
        for index in range(ACTION_DIM):
            center_y = self._slider_y(index)
            if left - 18 <= x <= right + 18 and center_y - 18 <= y <= center_y + 18:
                return index
        return None

    def _slider_bounds(self) -> tuple[int, int]:
        right = max(self.slider_left + 120, self.window_width - self.slider_right_margin)
        return self.slider_left, right

    def _slider_y(self, index: int) -> int:
        return self.slider_top + index * self.slider_row_height

    def _target_x(self, index: int) -> int:
        return self._value_to_x(index, float(self.target_joint_pos[index]))

    def _current_x(self, index: int) -> int | None:
        if self.current_joint_pos is None:
            return None
        return self._value_to_x(index, float(self.current_joint_pos[index]))

    def _value_to_x(self, index: int, value: float) -> int:
        left, right = self._slider_bounds()
        lower = float(self.joint_lower_limits[index])
        upper = float(self.joint_upper_limits[index])
        if upper <= lower:
            return left
        ratio = (float(np.clip(value, lower, upper)) - lower) / (upper - lower)
        return int(round(left + ratio * (right - left)))

    def _x_to_value(self, index: int, x: int) -> float:
        left, right = self._slider_bounds()
        lower = float(self.joint_lower_limits[index])
        upper = float(self.joint_upper_limits[index])
        ratio = float(np.clip((x - left) / max(1, right - left), 0.0, 1.0))
        return lower + ratio * (upper - lower)

    def _set_slider_from_x(self, index: int, x: int) -> None:
        self.target_joint_pos[index] = np.float32(self._x_to_value(index, x))

    def _nudge_selected(self, delta: float) -> None:
        index = self.selected_joint_index
        self.target_joint_pos[index] = float(
            np.clip(
                self.target_joint_pos[index] + delta,
                self.joint_lower_limits[index],
                self.joint_upper_limits[index],
            )
        )

    def _reset_selected(self) -> None:
        index = self.selected_joint_index
        self.target_joint_pos[index] = float(
            np.clip(self.initial_joint_pos[index], self.joint_lower_limits[index], self.joint_upper_limits[index])
        )

    def _reset_all(self) -> None:
        self.target_joint_pos = np.clip(
            self.initial_joint_pos,
            self.joint_lower_limits,
            self.joint_upper_limits,
        ).astype(np.float32)

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
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_mouse_down(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    self._handle_mouse_motion(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self._handle_mouse_up()

            rclpy.spin_once(self, timeout_sec=0.0)
            self._draw()
            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()

    def _draw_text(self, text: str, x: int, y: int, color: tuple[int, int, int] = (230, 230, 230)) -> int:
        surface = self.font.render(text, True, color)
        self.screen.blit(surface, (x, y))
        return y + 25

    def _draw_small_text(self, text: str, x: int, y: int, color: tuple[int, int, int] = (210, 210, 210)) -> int:
        surface = self.small_font.render(text, True, color)
        self.screen.blit(surface, (x, y))
        return y + 20

    def _draw(self) -> None:
        pygame = self.pygame
        self.screen.fill((17, 20, 24))
        y = 18
        y = self._draw_text("BDX Joint Pose Command", 24, y, (255, 255, 255))
        y = self._draw_text("Base is frozen by MuJoCo at z=0.33 in the tuning launch.", 24, y, (168, 204, 255))
        y = self._draw_small_text(
            "Drag sliders to set joint targets. Keyboard: 1-0 select, A/D fine adjust, Shift larger step, Home reset joint, R reset all, Esc quit",
            24,
            y,
        )
        y += 14

        step_deg = math.degrees(self.step_rad)
        y = self._draw_small_text(f"step: {step_deg:.2f} deg, Shift: {step_deg * self.large_step_multiplier:.2f} deg", 24, y)
        y = self._draw_small_text("yellow knob = target, blue tick = measured joint position", 24, y)
        y += 18

        current = self.current_joint_pos
        for index, name in enumerate(self.joint_names):
            self._draw_slider(index)
            target = float(self.target_joint_pos[index])
            measured = float(current[index]) if current is not None else float("nan")
            error = target - measured if current is not None else float("nan")
            selected = index == self.selected_joint_index
            color = (255, 230, 140) if selected else (220, 220, 220)
            y = self._slider_y(index)
            self._draw_small_text(f"{index + 1 if index < 9 else 0:>2} {name}", 24, y - 24, color)
            detail = (
                f"t {math.degrees(target):+6.1f}  "
                f"c {math.degrees(measured):+6.1f}  "
                f"e {math.degrees(error):+6.1f} deg"
            )
            self._draw_small_text(detail, 24, y - 4, color)

    def _draw_slider(self, index: int) -> None:
        pygame = self.pygame
        left, right = self._slider_bounds()
        y = self._slider_y(index)
        selected = index == self.selected_joint_index

        track_color = (80, 88, 98) if not selected else (124, 116, 84)
        pygame.draw.rect(self.screen, track_color, pygame.Rect(left, y - 4, right - left, 8), border_radius=4)

        zero_x = self._value_to_x(index, 0.0)
        if left <= zero_x <= right:
            pygame.draw.line(self.screen, (150, 150, 150), (zero_x, y - 11), (zero_x, y + 11), 2)

        current_x = self._current_x(index)
        if current_x is not None:
            pygame.draw.line(self.screen, (92, 181, 255), (current_x, y - 16), (current_x, y + 16), 4)

        target_x = self._target_x(index)
        pygame.draw.circle(self.screen, (255, 217, 102), (target_x, y), 10)
        pygame.draw.circle(self.screen, (32, 32, 32), (target_x, y), 10, 2)

        lower = math.degrees(float(self.joint_lower_limits[index]))
        upper = math.degrees(float(self.joint_upper_limits[index]))
        limit_text = f"{lower:+.0f} deg"
        upper_text = f"{upper:+.0f} deg"
        self._draw_small_text(limit_text, left, y + 10, (150, 150, 150))
        upper_surface = self.small_font.render(upper_text, True, (150, 150, 150))
        self.screen.blit(upper_surface, (right - upper_surface.get_width(), y + 10))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: JointPoseCommandNode | None = None
    try:
        node = JointPoseCommandNode()
        node.spin_ui()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
