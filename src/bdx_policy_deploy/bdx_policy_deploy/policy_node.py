from __future__ import annotations

from typing import Any

import numpy as np

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Float32MultiArray, Float64MultiArray, String

from bdx_policy_deploy.policy_interface import (
    ACTION_DIM,
    ACTION_SCALE,
    DEFAULT_EFFORT_LIMITS,
    DEFAULT_JOINT_LOWER_LIMITS,
    DEFAULT_JOINT_POS,
    DEFAULT_JOINT_UPPER_LIMITS,
    DEFAULT_KD,
    DEFAULT_KP,
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


VALID_POLICY_MODES = ("disabled", "zero_action", "policy")


class OnnxPolicy:
    def __init__(self, model_path: str) -> None:
        path = resolve_resource_path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX policy not found: {path}")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required by bdx_policy_deploy. Install it in the Python "
                "environment used by ROS 2, for example: /usr/bin/python3 -m pip install --user onnxruntime"
            ) from exc

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        if len(input_shape) != 2 or input_shape[-1] not in (OBS_DIM, "obs_dim", None):
            raise RuntimeError(f"Unexpected ONNX input shape {input_shape}; expected [1, {OBS_DIM}]")
        if len(output_shape) != 2 or output_shape[-1] not in (ACTION_DIM, "action_dim", None):
            raise RuntimeError(f"Unexpected ONNX output shape {output_shape}; expected [1, {ACTION_DIM}]")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        action = self.session.run([self.output_name], {self.input_name: obs.astype(np.float32)})[0]
        return np.asarray(action, dtype=np.float32).reshape(-1)


class BdxPolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("bdx_policy_node")
        self._declare_parameters()
        self._load_parameters()

        self.policy = OnnxPolicy(self.policy_path)

        self.joint_pos: np.ndarray | None = None
        self.joint_vel: np.ndarray | None = None
        self.imu_ang_vel: np.ndarray | None = None
        self.imu_quat_xyzw: np.ndarray | None = None
        self.command = np.zeros(3, dtype=np.float32)
        self.policy_mode = self.initial_policy_mode
        self.last_raw_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.last_target_joint_pos = self.default_joint_pos.copy()
        self.last_target_was_clipped = False
        self.last_torque = np.zeros(ACTION_DIM, dtype=np.float32)
        self.last_observation: np.ndarray | None = None
        self.safety_enabled = not self.require_enable
        self.fault_reasons: list[str] = ["waiting_for_sensors"]

        self.last_joint_stamp: Time | None = None
        self.last_imu_stamp: Time | None = None
        self.last_cmd_stamp: Time | None = None
        self.start_time = self.get_clock().now()
        self.imu_message_count = 0

        self.joint_sub = self.create_subscription(JointState, self.joint_state_topic, self._on_joint_state, 10)
        self.imu_sub = self.create_subscription(Imu, self.imu_topic, self._on_imu, 10)
        self.cmd_sub = self.create_subscription(Twist, self.cmd_vel_topic, self._on_cmd_vel, 10)
        self.mode_sub = self.create_subscription(String, self.policy_mode_topic, self._on_policy_mode, 10)
        self.enable_sub = None
        if self.enable_topic:
            self.enable_sub = self.create_subscription(Bool, self.enable_topic, self._on_enable, 10)

        self.target_pub = self.create_publisher(JointState, self.target_joint_state_topic, 10)
        self.torque_pub = self.create_publisher(Float64MultiArray, self.torque_command_topic, 10)
        self.diag_pub = self.create_publisher(DiagnosticArray, self.diagnostics_topic, 10)
        self.observation_pub = self.create_publisher(Float32MultiArray, self.observation_topic, 10)
        self.raw_action_pub = self.create_publisher(Float32MultiArray, self.raw_action_topic, 10)
        self.action_pub = self.create_publisher(Float32MultiArray, self.action_topic, 10)
        self.target_debug_pub = self.create_publisher(Float32MultiArray, self.target_debug_topic, 10)
        self.torque_debug_pub = self.create_publisher(Float32MultiArray, self.torque_debug_topic, 10)

        self.policy_timer = None
        if self.policy_trigger_mode == "timer":
            self.policy_timer = self.create_timer(1.0 / self.policy_rate_hz, self._policy_tick)
        self.control_timer = None
        if self.actuator_mode == "torque_pd":
            self.control_timer = self.create_timer(1.0 / self.control_rate_hz, self._control_tick)

        self.get_logger().info(
            "BDX policy node started: dry_run=%s, actuator_mode=%s, trigger=%s, policy_rate=%.1f Hz"
            % (self.dry_run, self.actuator_mode, self.policy_trigger_mode, self.policy_rate_hz)
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("policy_path", "")
        self.declare_parameter("dry_run", True)
        self.declare_parameter("actuator_mode", "position")
        self.declare_parameter("policy_trigger_mode", "timer")
        self.declare_parameter("policy_decimation", 4)
        self.declare_parameter("policy_rate_hz", 50.0)
        self.declare_parameter("control_rate_hz", 200.0)
        self.declare_parameter("action_clip", 0.0)
        self.declare_parameter("action_scale", float(ACTION_SCALE))
        self.declare_parameter("cmd_timeout_s", 0.2)
        self.declare_parameter("sensor_timeout_s", 0.05)
        self.declare_parameter("max_abs_roll_pitch_rad", 0.7)
        self.declare_parameter("max_joint_velocity_rad_s", 25.0)
        self.declare_parameter("torque_ramp_time_s", 2.0)
        self.declare_parameter("publish_debug", True)
        self.declare_parameter("require_enable", False)
        self.declare_parameter("fault_on_target_limit", False)
        self.declare_parameter("fault_on_cmd_timeout", True)
        self.declare_parameter("zero_command_on_cmd_timeout", True)
        self.declare_parameter("initial_policy_mode", "policy")

        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("policy_mode_topic", "/bdx_policy/mode")
        self.declare_parameter("enable_topic", "")
        self.declare_parameter("target_joint_state_topic", "/bdx_policy/target_joint_states")
        self.declare_parameter("torque_command_topic", "/bdx_policy/torque_cmd")
        self.declare_parameter("diagnostics_topic", "/bdx_policy/diagnostics")
        self.declare_parameter("observation_topic", "/bdx_policy/debug/observation")
        self.declare_parameter("raw_action_topic", "/bdx_policy/debug/raw_action")
        self.declare_parameter("action_topic", "/bdx_policy/debug/action")
        self.declare_parameter("target_debug_topic", "/bdx_policy/debug/target_joint_pos")
        self.declare_parameter("torque_debug_topic", "/bdx_policy/debug/torque")

        self.declare_parameter("joint_names", JOINT_NAMES)
        self.declare_parameter("default_joint_pos", DEFAULT_JOINT_POS.tolist())
        self.declare_parameter("joint_lower_limits", DEFAULT_JOINT_LOWER_LIMITS.tolist())
        self.declare_parameter("joint_upper_limits", DEFAULT_JOINT_UPPER_LIMITS.tolist())
        self.declare_parameter("safe_joint_lower_limits", DEFAULT_JOINT_LOWER_LIMITS.tolist())
        self.declare_parameter("safe_joint_upper_limits", DEFAULT_JOINT_UPPER_LIMITS.tolist())
        self.declare_parameter("kp", DEFAULT_KP.tolist())
        self.declare_parameter("kd", DEFAULT_KD.tolist())
        self.declare_parameter("effort_limits", DEFAULT_EFFORT_LIMITS.tolist())
        self.declare_parameter("command_lower_limits", [-0.4, -0.4, -1.0])
        self.declare_parameter("command_upper_limits", [0.7, 0.4, 1.0])

    def _load_parameters(self) -> None:
        self.policy_path = str(self.get_parameter("policy_path").value)
        if not self.policy_path:
            raise ValueError("policy_path parameter is required")

        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.actuator_mode = str(self.get_parameter("actuator_mode").value)
        if self.actuator_mode not in ("position", "torque_pd"):
            raise ValueError("actuator_mode must be 'position' or 'torque_pd'")

        self.policy_trigger_mode = str(self.get_parameter("policy_trigger_mode").value)
        if self.policy_trigger_mode not in ("timer", "imu_decimation"):
            raise ValueError("policy_trigger_mode must be 'timer' or 'imu_decimation'")
        self.policy_decimation = int(self.get_parameter("policy_decimation").value)
        if self.policy_decimation <= 0:
            raise ValueError("policy_decimation must be positive")
        self.policy_rate_hz = self._positive_float_parameter("policy_rate_hz")
        self.control_rate_hz = self._positive_float_parameter("control_rate_hz")
        self.action_clip = self._non_negative_float_parameter("action_clip")
        self.action_scale = self._positive_float_parameter("action_scale")
        self.cmd_timeout_s = self._positive_float_parameter("cmd_timeout_s")
        self.sensor_timeout_s = self._positive_float_parameter("sensor_timeout_s")
        self.max_abs_roll_pitch_rad = self._positive_float_parameter("max_abs_roll_pitch_rad")
        self.max_joint_velocity_rad_s = self._positive_float_parameter("max_joint_velocity_rad_s")
        self.torque_ramp_time_s = float(self.get_parameter("torque_ramp_time_s").value)
        if self.torque_ramp_time_s < 0.0:
            raise ValueError("torque_ramp_time_s must be non-negative")
        self.publish_debug = bool(self.get_parameter("publish_debug").value)
        self.require_enable = bool(self.get_parameter("require_enable").value)
        self.fault_on_target_limit = bool(self.get_parameter("fault_on_target_limit").value)
        self.fault_on_cmd_timeout = bool(self.get_parameter("fault_on_cmd_timeout").value)
        self.zero_command_on_cmd_timeout = bool(self.get_parameter("zero_command_on_cmd_timeout").value)
        self.initial_policy_mode = str(self.get_parameter("initial_policy_mode").value)
        if self.initial_policy_mode not in VALID_POLICY_MODES:
            raise ValueError(f"initial_policy_mode must be one of {VALID_POLICY_MODES}")

        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.policy_mode_topic = str(self.get_parameter("policy_mode_topic").value)
        self.enable_topic = str(self.get_parameter("enable_topic").value)
        self.target_joint_state_topic = str(self.get_parameter("target_joint_state_topic").value)
        self.torque_command_topic = str(self.get_parameter("torque_command_topic").value)
        self.diagnostics_topic = str(self.get_parameter("diagnostics_topic").value)
        self.observation_topic = str(self.get_parameter("observation_topic").value)
        self.raw_action_topic = str(self.get_parameter("raw_action_topic").value)
        self.action_topic = str(self.get_parameter("action_topic").value)
        self.target_debug_topic = str(self.get_parameter("target_debug_topic").value)
        self.torque_debug_topic = str(self.get_parameter("torque_debug_topic").value)

        self.joint_names = [str(name) for name in self.get_parameter("joint_names").value]
        if self.joint_names != JOINT_NAMES:
            self.get_logger().warning("joint_names parameter differs from the sim2sim training order")
        self.default_joint_pos = self._vector_parameter("default_joint_pos", ACTION_DIM)
        self.joint_lower_limits = self._vector_parameter("joint_lower_limits", ACTION_DIM)
        self.joint_upper_limits = self._vector_parameter("joint_upper_limits", ACTION_DIM)
        self.safe_joint_lower_limits = self._vector_parameter("safe_joint_lower_limits", ACTION_DIM)
        self.safe_joint_upper_limits = self._vector_parameter("safe_joint_upper_limits", ACTION_DIM)
        self.kp = self._vector_parameter("kp", ACTION_DIM)
        self.kd = self._vector_parameter("kd", ACTION_DIM)
        self.effort_limits = self._vector_parameter("effort_limits", ACTION_DIM)
        self.command_lower_limits = self._vector_parameter("command_lower_limits", 3)
        self.command_upper_limits = self._vector_parameter("command_upper_limits", 3)

        if np.any(self.joint_lower_limits > self.joint_upper_limits):
            raise ValueError("joint_lower_limits must be <= joint_upper_limits")
        if np.any(self.safe_joint_lower_limits > self.safe_joint_upper_limits):
            raise ValueError("safe_joint_lower_limits must be <= safe_joint_upper_limits")

    def _positive_float_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
        return value

    def _non_negative_float_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return value

    def _vector_parameter(self, name: str, size: int) -> np.ndarray:
        value = np.asarray(self.get_parameter(name).value, dtype=np.float32).reshape(-1)
        if value.shape != (size,):
            raise ValueError(f"{name} must have {size} entries, got {value.shape[0]}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains NaN or Inf")
        return value

    def _on_joint_state(self, msg: JointState) -> None:
        try:
            if len(msg.velocity) != len(msg.name):
                raise PolicyInterfaceError("JointState velocity array must be populated for all required joints")
            self.joint_pos, self.joint_vel = reorder_joint_state(
                msg.name,
                msg.position,
                msg.velocity,
                self.joint_names,
            )
            self.last_joint_stamp = self._message_time_or_now(msg.header.stamp)
        except PolicyInterfaceError as exc:
            self.get_logger().warning(f"Ignoring invalid JointState: {exc}", throttle_duration_sec=1.0)

    def _on_imu(self, msg: Imu) -> None:
        quat = np.array(
            [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            dtype=np.float32,
        )
        ang_vel = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(quat)) or not np.all(np.isfinite(ang_vel)):
            self.get_logger().warning("Ignoring invalid IMU with NaN or Inf", throttle_duration_sec=1.0)
            return
        self.imu_quat_xyzw = quat
        self.imu_ang_vel = ang_vel
        self.last_imu_stamp = self._message_time_or_now(msg.header.stamp)
        if self.policy_trigger_mode == "imu_decimation":
            self.imu_message_count += 1
            if self.imu_message_count % self.policy_decimation == 0:
                self._policy_tick()

    def _on_cmd_vel(self, msg: Twist) -> None:
        command = np.array([msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32)
        try:
            self.command = clip_command(command, self.command_lower_limits, self.command_upper_limits)
            self.last_cmd_stamp = self.get_clock().now()
        except PolicyInterfaceError as exc:
            self.get_logger().warning(f"Ignoring invalid cmd_vel: {exc}", throttle_duration_sec=1.0)

    def _on_policy_mode(self, msg: String) -> None:
        mode = msg.data.strip()
        if mode not in VALID_POLICY_MODES:
            self.get_logger().warning(
                f"Ignoring invalid policy mode '{msg.data}'. Expected one of {VALID_POLICY_MODES}",
                throttle_duration_sec=1.0,
            )
            return
        if mode != self.policy_mode:
            self.get_logger().info(f"Policy mode changed: {self.policy_mode} -> {mode}")
            self.policy_mode = mode
            if mode == "zero_action":
                self.last_raw_action = np.zeros(ACTION_DIM, dtype=np.float32)
                self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
                self.last_target_joint_pos = self.default_joint_pos.copy()
                self.last_target_was_clipped = False

    def _on_enable(self, msg: Bool) -> None:
        self.safety_enabled = bool(msg.data)

    def _message_time_or_now(self, stamp: Any) -> Time:
        if stamp.sec == 0 and stamp.nanosec == 0:
            return self.get_clock().now()
        return Time.from_msg(stamp)

    def _policy_tick(self) -> None:
        now = self.get_clock().now()
        if self.policy_mode == "disabled":
            self.fault_reasons = ["mode_disabled"]
            self._publish_diagnostics(now, self.fault_reasons)
            return

        if (
            self.zero_command_on_cmd_timeout
            and (self.last_cmd_stamp is None or self._age_s(now, self.last_cmd_stamp) > self.cmd_timeout_s)
        ):
            self.command = np.zeros(3, dtype=np.float32)

        reasons = self._policy_input_faults(now)
        if reasons:
            self.fault_reasons = reasons
            self._publish_diagnostics(now, reasons)
            return

        assert self.joint_pos is not None
        assert self.joint_vel is not None
        assert self.imu_ang_vel is not None
        assert self.imu_quat_xyzw is not None

        try:
            projected_gravity = projected_gravity_from_quaternion_xyzw(self.imu_quat_xyzw)
            obs = build_observation(
                self.imu_ang_vel,
                projected_gravity,
                self.joint_pos,
                self.joint_vel,
                self.last_action,
                self.command,
                self.default_joint_pos,
            )
            if self.policy_mode == "zero_action":
                raw_action = np.zeros(ACTION_DIM, dtype=np.float32)
            else:
                raw_action = self.policy(obs)
            target = compute_policy_target(
                raw_action,
                default_joint_pos=self.default_joint_pos,
                joint_lower_limits=self.joint_lower_limits,
                joint_upper_limits=self.joint_upper_limits,
                action_clip=self.action_clip,
                action_scale=self.action_scale,
            )
            self.last_observation = obs.reshape(-1)
            self.last_raw_action = target.raw_action
            self.last_action = target.clipped_action
            self.last_target_joint_pos = target.target_joint_pos
            self.last_target_was_clipped = target.target_was_clipped
            self.last_torque = compute_pd_torque(
                self.last_target_joint_pos,
                self.joint_pos,
                self.joint_vel,
                self.kp,
                self.kd,
                self.effort_limits,
            )
        except (PolicyInterfaceError, RuntimeError) as exc:
            self.fault_reasons = [str(exc)]
            self._publish_diagnostics(now, self.fault_reasons)
            return

        self.fault_reasons = self._safety_faults(now, include_output_checks=True)
        self._publish_debug(target.raw_action)
        self._publish_diagnostics(now, self.fault_reasons)

        if self.fault_reasons:
            return
        if self.actuator_mode == "position":
            self._publish_position_target(now)

    def _control_tick(self) -> None:
        now = self.get_clock().now()
        reasons = self._safety_faults(now, include_output_checks=True)
        if reasons:
            self.fault_reasons = reasons
            self._publish_diagnostics(now, reasons)
            return
        self._publish_torque_command()

    def _policy_input_faults(self, now: Time) -> list[str]:
        reasons: list[str] = []
        if self.joint_pos is None or self.joint_vel is None or self.last_joint_stamp is None:
            reasons.append("missing_joint_state")
        elif self._age_s(now, self.last_joint_stamp) > self.sensor_timeout_s:
            reasons.append("stale_joint_state")
        if self.imu_ang_vel is None or self.imu_quat_xyzw is None or self.last_imu_stamp is None:
            reasons.append("missing_imu")
        elif self._age_s(now, self.last_imu_stamp) > self.sensor_timeout_s:
            reasons.append("stale_imu")
        for name, value in (
            ("joint_pos", self.joint_pos),
            ("joint_vel", self.joint_vel),
            ("imu_ang_vel", self.imu_ang_vel),
            ("imu_quat", self.imu_quat_xyzw),
        ):
            if value is not None and not np.all(np.isfinite(value)):
                reasons.append(f"{name}_nan_or_inf")
        return reasons

    def _safety_faults(
        self,
        now: Time,
        include_output_checks: bool = False,
        include_dry_run: bool = True,
    ) -> list[str]:
        reasons: list[str] = []
        if include_dry_run and self.dry_run:
            reasons.append("dry_run")
        if self.require_enable and not self.safety_enabled:
            reasons.append("policy_enable_false")
        reasons.extend(self._policy_input_faults(now))
        if self.fault_on_cmd_timeout:
            if self.last_cmd_stamp is None:
                reasons.append("missing_cmd_vel")
            elif self._age_s(now, self.last_cmd_stamp) > self.cmd_timeout_s:
                reasons.append("stale_cmd_vel")

        if self.joint_pos is not None:
            if np.any(self.joint_pos < self.safe_joint_lower_limits) or np.any(self.joint_pos > self.safe_joint_upper_limits):
                reasons.append("joint_position_outside_safe_range")
        if self.joint_vel is not None and np.any(np.abs(self.joint_vel) > self.max_joint_velocity_rad_s):
            reasons.append("joint_velocity_limit")
        if self.imu_quat_xyzw is not None:
            try:
                roll, pitch = roll_pitch_from_quaternion_xyzw(self.imu_quat_xyzw)
                if abs(roll) > self.max_abs_roll_pitch_rad or abs(pitch) > self.max_abs_roll_pitch_rad:
                    reasons.append("roll_pitch_limit")
            except PolicyInterfaceError as exc:
                reasons.append(str(exc))

        arrays = [
            ("command", self.command),
            ("joint_pos", self.joint_pos),
            ("joint_vel", self.joint_vel),
            ("imu_ang_vel", self.imu_ang_vel),
            ("imu_quat", self.imu_quat_xyzw),
        ]
        if include_output_checks:
            arrays.extend(
                [
                    ("observation", self.last_observation),
                    ("raw_action", self.last_raw_action),
                    ("action", self.last_action),
                    ("target_joint_pos", self.last_target_joint_pos),
                    ("torque", self.last_torque),
                ]
            )
            if self.last_observation is None:
                reasons.append("policy_output_not_ready")
            if self.fault_on_target_limit and self.last_target_was_clipped:
                reasons.append("requested_target_position_limit")
            if np.any(self.last_target_joint_pos < self.joint_lower_limits) or np.any(
                self.last_target_joint_pos > self.joint_upper_limits
            ):
                reasons.append("target_joint_position_limit")
        for name, value in arrays:
            if value is not None and not np.all(np.isfinite(value)):
                reasons.append(f"{name}_nan_or_inf")

        return reasons

    def _age_s(self, now: Time, stamp: Time) -> float:
        return (now.nanoseconds - stamp.nanoseconds) * 1e-9

    def _ramp_scale(self) -> float:
        if self.torque_ramp_time_s <= 0.0:
            return 1.0
        elapsed = (self.get_clock().now().nanoseconds - self.start_time.nanoseconds) * 1e-9
        return float(np.clip(elapsed / self.torque_ramp_time_s, 0.0, 1.0))

    def _publish_position_target(self, now: Time) -> None:
        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = list(self.joint_names)
        msg.position = self.last_target_joint_pos.astype(float).tolist()
        msg.velocity = []
        msg.effort = []
        self.target_pub.publish(msg)

    def _publish_torque_command(self) -> None:
        torque = (self.last_torque * np.float32(self._ramp_scale())).astype(np.float64)
        msg = Float64MultiArray()
        msg.data = torque.tolist()
        self.torque_pub.publish(msg)

    def _publish_debug(self, raw_action: np.ndarray) -> None:
        if not self.publish_debug:
            return
        if self.last_observation is not None:
            self.observation_pub.publish(Float32MultiArray(data=self.last_observation.astype(np.float32).tolist()))
        self.raw_action_pub.publish(Float32MultiArray(data=np.asarray(raw_action, dtype=np.float32).tolist()))
        self.action_pub.publish(Float32MultiArray(data=self.last_action.astype(np.float32).tolist()))
        self.target_debug_pub.publish(Float32MultiArray(data=self.last_target_joint_pos.astype(np.float32).tolist()))
        self.torque_debug_pub.publish(Float32MultiArray(data=self.last_torque.astype(np.float32).tolist()))

    def _publish_diagnostics(self, now: Time, reasons: list[str]) -> None:
        status = DiagnosticStatus()
        status.name = "bdx_policy_node"
        status.hardware_id = "bdx"
        if reasons:
            command_blocked = any(reason not in ("dry_run", "mode_disabled") for reason in reasons) or self.dry_run
            status.level = DiagnosticStatus.WARN if command_blocked else DiagnosticStatus.OK
            status.message = ",".join(reasons)
        else:
            status.level = DiagnosticStatus.OK
            status.message = "ok"

        values = {
            "dry_run": str(self.dry_run),
            "actuator_mode": self.actuator_mode,
            "policy_trigger_mode": self.policy_trigger_mode,
            "policy_mode": self.policy_mode,
            "safety_enabled": str(self.safety_enabled),
            "target_was_clipped": str(self.last_target_was_clipped),
            "command": np.array2string(self.command, precision=4),
            "last_action_norm": f"{np.linalg.norm(self.last_action):.6f}",
            "target_norm": f"{np.linalg.norm(self.last_target_joint_pos):.6f}",
            "torque_norm": f"{np.linalg.norm(self.last_torque):.6f}",
        }
        status.values = [KeyValue(key=key, value=value) for key, value in values.items()]
        diag = DiagnosticArray()
        diag.header.stamp = now.to_msg()
        diag.status = [status]
        self.diag_pub.publish(diag)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: BdxPolicyNode | None = None
    try:
        node = BdxPolicyNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
