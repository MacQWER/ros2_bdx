from __future__ import annotations

from typing import Any

from geometry_msgs.msg import Twist
import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32MultiArray, Float64MultiArray, String

from bdx_policy_deploy.policy_interface import (
    ACTION_DIM,
    DEFAULT_EFFORT_LIMITS,
    DEFAULT_JOINT_POS,
    DEFAULT_KD,
    DEFAULT_KP,
    JOINT_NAMES,
    as_float32_vector,
)
from bdx_policy_deploy.resource_paths import resolve_resource_path


VALID_POLICY_MODES = ("disabled", "zero_action", "policy")


class MuJoCoBodyNode(Node):
    """Use MuJoCo as the robot body around the ROS policy node."""

    def __init__(self) -> None:
        super().__init__("bdx_mujoco_body_node")
        self._declare_parameters()
        self._load_parameters()

        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                "mujoco is required by bdx_mujoco_body_node. Install it in the Python "
                "environment used by ROS 2: /usr/bin/python3 -m pip install --user mujoco --break-system-packages"
            ) from exc

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.sim_dt

        self.joint_ids = [self._name_to_joint_id(name) for name in self.joint_names]
        self.actuator_ids = [self._name_to_actuator_id(f"{name}_servo") for name in self.joint_names]
        self.imu_sensor_id = self._name_to_sensor_id("imu_ang_vel")
        self.imu_site_id = self._name_to_site_id("imu")
        self.com_body_id = self._name_to_body_id(self.com_body_name)
        self.free_qpos_addr = self._free_joint_qpos_addr()
        self.free_qvel_addr = self._free_joint_qvel_addr()

        self._prepare_com_offset()
        self._set_com_z_offset(self.com_z_offset)
        self._configure_actuators()
        self._configure_contact_properties()
        self._configure_robot_visual_alpha()
        self._reset_robot()
        self.frozen_base_qpos: np.ndarray | None = None
        if self.policy_mode == "disabled":
            self._capture_base_freeze()
        self.viewer_handle = None
        if self.viewer:
            from mujoco import viewer as mujoco_viewer

            self.viewer_handle = mujoco_viewer.launch_passive(self.model, self.data)
            self.viewer_handle.cam.lookat[:] = [0.0, 0.0, 0.3]
            self.viewer_handle.cam.distance = 1.8
            self.viewer_handle.cam.azimuth = 30
            self.viewer_handle.cam.elevation = -20

        self.target_joint_pos = self.default_joint_pos.copy()
        self.applied_torque = np.zeros(ACTION_DIM, dtype=np.float32)
        self.last_target_stamp = self.get_clock().now()
        self.last_torque_stamp = self.get_clock().now()
        self.step_count = 0
        self.publish_decimation = max(1, int(round((1.0 / self.publish_rate_hz) / self.sim_dt)))
        self.command_decimation = max(1, int(round((1.0 / self.command_rate_hz) / self.sim_dt)))

        self.joint_pub = self.create_publisher(JointState, self.joint_state_topic, 10)
        self.imu_pub = self.create_publisher(Imu, self.imu_topic, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.state_debug_pub = self.create_publisher(Float32MultiArray, self.sim_state_topic, 10)
        self.base_state_debug_pub = self.create_publisher(Float32MultiArray, self.base_state_topic, 10)

        self.target_sub = self.create_subscription(JointState, self.target_joint_state_topic, self._on_target, 10)
        self.torque_sub = self.create_subscription(Float64MultiArray, self.torque_command_topic, self._on_torque, 10)
        self.mode_sub = self.create_subscription(String, self.policy_mode_topic, self._on_policy_mode, 10)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.sim_timer = self.create_timer(self.sim_dt, self._sim_tick)

        self.get_logger().info(
            "MuJoCo body node started: xml=%s, control_mode=%s, sim_dt=%.4f, "
            "viewer=%s, com_z_offset=%.4f m"
            % (self.xml_path, self.control_mode, self.sim_dt, self.viewer, self.com_z_offset)
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("xml_path", "package://bdx_policy_deploy/assets/mujoco/xmls/scene.xml")
        self.declare_parameter("viewer", False)
        self.declare_parameter("control_mode", "position")
        self.declare_parameter("sim_dt", 0.005)
        self.declare_parameter("publish_rate_hz", 200.0)
        self.declare_parameter("base_height", 0.33)
        self.declare_parameter("reset_on_fall", False)
        self.declare_parameter("fall_height", 0.18)
        self.declare_parameter("target_timeout_s", 0.2)
        self.declare_parameter("disabled_base_height", 0.24)
        self.declare_parameter("foot_friction", 0.6)
        self.declare_parameter("floor_friction", 0.6)
        self.declare_parameter("torsional_friction", 0.005)
        self.declare_parameter("rolling_friction", 0.0001)
        self.declare_parameter("initial_policy_mode", "policy")
        self.declare_parameter("show_imu_visual", False)
        self.declare_parameter("imu_axis_length", 0.08)
        self.declare_parameter("imu_axis_radius", 0.004)
        self.declare_parameter("imu_marker_radius", 0.018)
        self.declare_parameter("show_com_visual", False)
        self.declare_parameter("com_body_name", "base_link")
        self.declare_parameter("com_marker_radius", 0.025)
        self.declare_parameter("com_z_offset", 0.0)
        self.declare_parameter("robot_model_alpha", 1.0)

        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("policy_mode_topic", "/bdx_policy/mode")
        self.declare_parameter("target_joint_state_topic", "/bdx_policy/target_joint_states")
        self.declare_parameter("torque_command_topic", "/bdx_policy/torque_cmd")
        self.declare_parameter("sim_state_topic", "/bdx_mujoco/debug/state")
        self.declare_parameter("base_state_topic", "/bdx_mujoco/debug/base_state")

        self.declare_parameter("publish_cmd_vel", True)
        self.declare_parameter("command_rate_hz", 20.0)
        self.declare_parameter("command", [0.0, 0.0, 0.0])
        self.declare_parameter("command_x", 0.0)
        self.declare_parameter("command_y", 0.0)
        self.declare_parameter("command_yaw", 0.0)

        self.declare_parameter("joint_names", JOINT_NAMES)
        self.declare_parameter("default_joint_pos", DEFAULT_JOINT_POS.tolist())
        self.declare_parameter("kp", DEFAULT_KP.tolist())
        self.declare_parameter("kd", DEFAULT_KD.tolist())
        self.declare_parameter("effort_limits", DEFAULT_EFFORT_LIMITS.tolist())
        self.declare_parameter("armature_default", 0.02)
        self.declare_parameter("armature_ankle", 0.0042)

    def _load_parameters(self) -> None:
        self.xml_path = resolve_resource_path(str(self.get_parameter("xml_path").value))
        if not self.xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {self.xml_path}")

        self.viewer = bool(self.get_parameter("viewer").value)
        self.control_mode = str(self.get_parameter("control_mode").value)
        if self.control_mode not in ("position", "torque"):
            raise ValueError("control_mode must be 'position' or 'torque'")
        self.sim_dt = self._positive_float_parameter("sim_dt")
        self.publish_rate_hz = self._positive_float_parameter("publish_rate_hz")
        self.base_height = float(self.get_parameter("base_height").value)
        self.reset_on_fall = bool(self.get_parameter("reset_on_fall").value)
        self.fall_height = float(self.get_parameter("fall_height").value)
        self.target_timeout_s = self._positive_float_parameter("target_timeout_s")
        self.disabled_base_height = self._positive_float_parameter("disabled_base_height")
        self.foot_friction = self._nonnegative_float_parameter("foot_friction")
        self.floor_friction = self._nonnegative_float_parameter("floor_friction")
        self.torsional_friction = self._nonnegative_float_parameter("torsional_friction")
        self.rolling_friction = self._nonnegative_float_parameter("rolling_friction")
        self.policy_mode = str(self.get_parameter("initial_policy_mode").value)
        if self.policy_mode not in VALID_POLICY_MODES:
            raise ValueError(f"initial_policy_mode must be one of {VALID_POLICY_MODES}")
        self.show_imu_visual = bool(self.get_parameter("show_imu_visual").value)
        self.imu_axis_length = self._positive_float_parameter("imu_axis_length")
        self.imu_axis_radius = self._positive_float_parameter("imu_axis_radius")
        self.imu_marker_radius = self._positive_float_parameter("imu_marker_radius")
        self.show_com_visual = bool(self.get_parameter("show_com_visual").value)
        self.com_body_name = str(self.get_parameter("com_body_name").value)
        self.com_marker_radius = self._positive_float_parameter("com_marker_radius")
        self.com_z_offset = self._finite_float_parameter("com_z_offset")
        self.robot_model_alpha = self._unit_float_parameter("robot_model_alpha")

        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.policy_mode_topic = str(self.get_parameter("policy_mode_topic").value)
        self.target_joint_state_topic = str(self.get_parameter("target_joint_state_topic").value)
        self.torque_command_topic = str(self.get_parameter("torque_command_topic").value)
        self.sim_state_topic = str(self.get_parameter("sim_state_topic").value)
        self.base_state_topic = str(self.get_parameter("base_state_topic").value)

        self.publish_cmd_vel = bool(self.get_parameter("publish_cmd_vel").value)
        self.command_rate_hz = self._positive_float_parameter("command_rate_hz")
        self.command = as_float32_vector(self.get_parameter("command").value, 3, "command")
        self.command = np.array(
            [
                float(self.get_parameter("command_x").value),
                float(self.get_parameter("command_y").value),
                float(self.get_parameter("command_yaw").value),
            ],
            dtype=np.float32,
        )

        self.joint_names = [str(name) for name in self.get_parameter("joint_names").value]
        self.default_joint_pos = as_float32_vector(self.get_parameter("default_joint_pos").value, ACTION_DIM, "default_joint_pos")
        self.kp = as_float32_vector(self.get_parameter("kp").value, ACTION_DIM, "kp")
        self.kd = as_float32_vector(self.get_parameter("kd").value, ACTION_DIM, "kd")
        self.effort_limits = as_float32_vector(self.get_parameter("effort_limits").value, ACTION_DIM, "effort_limits")
        self.armature_default = self._nonnegative_float_parameter("armature_default")
        self.armature_ankle = self._nonnegative_float_parameter("armature_ankle")

    def _positive_float_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
        return value

    def _nonnegative_float_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return value

    def _unit_float_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0")
        return value

    def _finite_float_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def _name_to_joint_id(self, joint_name: str) -> int:
        joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in MuJoCo model: {joint_name}")
        return int(joint_id)

    def _name_to_actuator_id(self, actuator_name: str) -> int:
        actuator_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise ValueError(f"Actuator not found in MuJoCo model: {actuator_name}")
        return int(actuator_id)

    def _name_to_sensor_id(self, sensor_name: str) -> int:
        sensor_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id < 0:
            raise ValueError(f"Sensor not found in MuJoCo model: {sensor_name}")
        return int(sensor_id)

    def _name_to_site_id(self, site_name: str) -> int:
        site_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"Site not found in MuJoCo model: {site_name}")
        return int(site_id)

    def _name_to_body_id(self, body_name: str) -> int:
        body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body not found in MuJoCo model: {body_name}")
        return int(body_id)

    def _free_joint_qpos_addr(self) -> int | None:
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == self.mujoco.mjtJoint.mjJNT_FREE:
                return int(self.model.jnt_qposadr[joint_id])
        return None

    def _free_joint_qvel_addr(self) -> int | None:
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == self.mujoco.mjtJoint.mjJNT_FREE:
                return int(self.model.jnt_dofadr[joint_id])
        return None

    def _configure_actuators(self) -> None:
        for joint_name, joint_id, actuator_id, effort_limit in zip(
            self.joint_names,
            self.joint_ids,
            self.actuator_ids,
            self.effort_limits,
        ):
            is_ankle = "Ankle" in joint_name
            armature = self.armature_ankle if is_ankle else self.armature_default
            dof_id = self.model.jnt_dofadr[joint_id]
            self.model.dof_damping[dof_id] = 0.0
            self.model.dof_armature[dof_id] = armature

            self.model.actuator_gainprm[actuator_id] = 0.0
            self.model.actuator_gainprm[actuator_id, 0] = 1.0
            self.model.actuator_biasprm[actuator_id] = 0.0
            self.model.actuator_forcelimited[actuator_id] = 1
            self.model.actuator_forcerange[actuator_id] = [-float(effort_limit), float(effort_limit)]
            self.model.actuator_ctrllimited[actuator_id] = 1
            self.model.actuator_ctrlrange[actuator_id] = [-float(effort_limit), float(effort_limit)]

    def _configure_contact_properties(self) -> None:
        foot_friction = [self.foot_friction, self.torsional_friction, self.rolling_friction]
        floor_friction = [self.floor_friction, self.torsional_friction, self.rolling_friction]
        for geom_id in range(self.model.ngeom):
            geom_name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name == "floor":
                self.model.geom_friction[geom_id] = floor_friction
            elif geom_name is not None and "foot_collision" in geom_name:
                self.model.geom_friction[geom_id] = foot_friction

    def _configure_robot_visual_alpha(self) -> None:
        """Make robot render meshes transparent without affecting collision geoms or the floor."""
        visual_group = 2
        material_ids: set[int] = set()
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_group[geom_id]) != visual_group:
                continue
            material_id = int(self.model.geom_matid[geom_id])
            if material_id >= 0:
                material_ids.add(material_id)
            else:
                self.model.geom_rgba[geom_id, 3] = self.robot_model_alpha

        for material_id in material_ids:
            self.model.mat_rgba[material_id, 3] = self.robot_model_alpha

    def _prepare_com_offset(self) -> None:
        """Cache subtree inertias and root-z directions for absolute CoM offset updates."""
        self.com_offset_body_ids = np.empty(0, dtype=np.int32)
        self.com_offset_ipos = np.empty((0, 3), dtype=np.float64)
        self.com_offset_directions = np.empty((0, 3), dtype=np.float64)
        self.applied_com_z_offset = 0.0
        if self.com_body_id is None:
            return

        body_ids = []
        for body_id in range(self.model.nbody):
            ancestor_id = body_id
            while ancestor_id != 0 and ancestor_id != self.com_body_id:
                ancestor_id = int(self.model.body_parentid[ancestor_id])
            if ancestor_id == self.com_body_id and self.model.body_mass[body_id] > 0.0:
                body_ids.append(body_id)

        self.mujoco.mj_forward(self.model, self.data)
        root_z_world = self.data.xmat[self.com_body_id].reshape(3, 3)[:, 2].copy()
        self.com_offset_body_ids = np.asarray(body_ids, dtype=np.int32)
        self.com_offset_ipos = self.model.body_ipos[self.com_offset_body_ids].copy()
        self.com_offset_directions = np.asarray(
            [
                self.data.xmat[body_id].reshape(3, 3).T @ root_z_world
                for body_id in self.com_offset_body_ids
            ],
            dtype=np.float64,
        )

    def _set_com_z_offset(self, offset: float) -> None:
        if self.com_offset_body_ids.size == 0:
            return
        if offset == self.applied_com_z_offset:
            self.com_z_offset = offset
            return
        self.model.body_ipos[self.com_offset_body_ids] = (
            self.com_offset_ipos + self.com_offset_directions * offset
        )
        # Recompute model constants after changing inertial positions without disturbing live simulation data.
        self.mujoco.mj_setConst(self.model, self.mujoco.MjData(self.model))
        self.mujoco.mj_forward(self.model, self.data)
        self.applied_com_z_offset = offset
        self.com_z_offset = offset

    def _on_set_parameters(self, parameters: list[Parameter]) -> SetParametersResult:
        requested_offset = self.com_z_offset
        for parameter in parameters:
            if parameter.name != "com_z_offset":
                continue
            try:
                requested_offset = float(parameter.value)
            except (TypeError, ValueError):
                return SetParametersResult(successful=False, reason="com_z_offset must be a number")
            if not np.isfinite(requested_offset):
                return SetParametersResult(successful=False, reason="com_z_offset must be finite")

        if requested_offset != self.com_z_offset:
            self._set_com_z_offset(requested_offset)
            self.get_logger().info(f"Updated inertial CoM z offset to {requested_offset:.4f} m")
        return SetParametersResult(successful=True)

    def _reset_robot(self) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        if self.free_qpos_addr is not None:
            self.data.qpos[self.free_qpos_addr : self.free_qpos_addr + 3] = [0.0, 0.0, self.base_height]
            self.data.qpos[self.free_qpos_addr + 3 : self.free_qpos_addr + 7] = [1.0, 0.0, 0.0, 0.0]
        for joint_id, default_pos in zip(self.joint_ids, self.default_joint_pos):
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = float(default_pos)
            self.data.qvel[self.model.jnt_dofadr[joint_id]] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        if getattr(self, "policy_mode", None) == "disabled":
            self._capture_base_freeze()

    def _capture_base_freeze(self) -> None:
        if self.free_qpos_addr is None:
            self.frozen_base_qpos = None
            return
        self.frozen_base_qpos = self.data.qpos[self.free_qpos_addr : self.free_qpos_addr + 7].copy()
        self.frozen_base_qpos[2] = self.disabled_base_height

    def _apply_base_freeze(self) -> None:
        if self.policy_mode != "disabled" or self.frozen_base_qpos is None or self.free_qpos_addr is None:
            return
        self.data.qpos[self.free_qpos_addr : self.free_qpos_addr + 7] = self.frozen_base_qpos
        if self.free_qvel_addr is not None:
            self.data.qvel[self.free_qvel_addr : self.free_qvel_addr + 6] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def _joint_qpos(self) -> np.ndarray:
        return np.array(
            [self.data.qpos[self.model.jnt_qposadr[joint_id]] for joint_id in self.joint_ids],
            dtype=np.float32,
        )

    def _joint_qvel(self) -> np.ndarray:
        return np.array(
            [self.data.qvel[self.model.jnt_dofadr[joint_id]] for joint_id in self.joint_ids],
            dtype=np.float32,
        )

    def _sensor_data(self, sensor_id: int) -> np.ndarray:
        start = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[start : start + dim].copy()

    def _site_quat_xyzw(self, site_id: int) -> np.ndarray:
        quat_wxyz = np.zeros(4, dtype=np.float64)
        self.mujoco.mju_mat2Quat(quat_wxyz, self.data.site_xmat[site_id])
        return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)

    def _append_viewer_geom(
        self,
        geom_type: Any,
        size: np.ndarray,
        pos: np.ndarray,
        mat: np.ndarray,
        rgba: np.ndarray,
        label: str = "",
    ) -> Any | None:
        if self.viewer_handle is None or self.viewer_handle.user_scn is None:
            return None
        scene = self.viewer_handle.user_scn
        if scene.ngeom >= scene.maxgeom:
            return None
        geom = scene.geoms[scene.ngeom]
        self.mujoco.mjv_initGeom(geom, geom_type, size, pos, mat, rgba)
        geom.category = self.mujoco.mjtCatBit.mjCAT_DECOR
        geom.label = label
        scene.ngeom += 1
        return geom

    def _update_viewer_debug_visuals(self) -> None:
        if self.viewer_handle is None or self.viewer_handle.user_scn is None:
            return

        scene = self.viewer_handle.user_scn
        scene.ngeom = 0
        if not self.show_imu_visual and not self.show_com_visual:
            return

        identity = np.eye(3, dtype=np.float64).reshape(9)

        if self.com_body_id is not None:
            com_pos = self.data.subtree_com[self.com_body_id].astype(np.float64).copy()
            self._append_viewer_geom(
                self.mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, self.com_marker_radius, dtype=np.float64),
                com_pos,
                identity,
                np.array([1.0, 0.05, 0.65, 1.0], dtype=np.float32),
                "COM",
            )
            self._append_viewer_geom(
                self.mujoco.mjtGeom.mjGEOM_LABEL,
                np.zeros(3, dtype=np.float64),
                com_pos
                + np.array([0.0, 0.0, self.com_marker_radius * 1.8], dtype=np.float64),
                identity,
                np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                "COM",
            )

        if not self.show_imu_visual:
            return

        imu_pos = self.data.site_xpos[self.imu_site_id].astype(np.float64).copy()
        imu_rot = (
            self.data.site_xmat[self.imu_site_id]
            .reshape(3, 3)
            .astype(np.float64)
            .copy()
        )
        marker_size = np.full(3, self.imu_marker_radius, dtype=np.float64)
        label_pos = imu_pos + np.array(
            [0.0, 0.0, self.imu_marker_radius * 2.0],
            dtype=np.float64,
        )

        self._append_viewer_geom(
            self.mujoco.mjtGeom.mjGEOM_SPHERE,
            marker_size,
            imu_pos,
            identity,
            np.array([1.0, 0.9, 0.05, 1.0], dtype=np.float32),
            "IMU",
        )
        self._append_viewer_geom(
            self.mujoco.mjtGeom.mjGEOM_LABEL,
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            label_pos,
            identity,
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            "IMU",
        )

        axis_colors = (
            np.array([1.0, 0.05, 0.05, 1.0], dtype=np.float32),
            np.array([0.05, 0.85, 0.05, 1.0], dtype=np.float32),
            np.array([0.1, 0.35, 1.0, 1.0], dtype=np.float32),
        )
        for axis_index, color in enumerate(axis_colors):
            endpoint = imu_pos + imu_rot[:, axis_index] * self.imu_axis_length
            axis_geom = self._append_viewer_geom(
                self.mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                identity,
                color,
            )
            if axis_geom is not None:
                self.mujoco.mjv_connector(
                    axis_geom,
                    self.mujoco.mjtGeom.mjGEOM_ARROW,
                    self.imu_axis_radius,
                    imu_pos,
                    endpoint.astype(np.float64),
                )

    def _on_target(self, msg: JointState) -> None:
        if len(msg.name) != len(msg.position):
            self.get_logger().warning("Ignoring target JointState with mismatched name/position length", throttle_duration_sec=1.0)
            return
        index_by_name = {name: index for index, name in enumerate(msg.name)}
        missing = [name for name in self.joint_names if name not in index_by_name]
        if missing:
            self.get_logger().warning(f"Ignoring target JointState missing joints: {missing}", throttle_duration_sec=1.0)
            return
        target = np.array([msg.position[index_by_name[name]] for name in self.joint_names], dtype=np.float32)
        if not np.all(np.isfinite(target)):
            self.get_logger().warning("Ignoring target JointState with NaN or Inf", throttle_duration_sec=1.0)
            return
        self.target_joint_pos = target
        self.last_target_stamp = self.get_clock().now()

    def _on_torque(self, msg: Float64MultiArray) -> None:
        torque = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if torque.shape != (ACTION_DIM,) or not np.all(np.isfinite(torque)):
            self.get_logger().warning("Ignoring invalid torque command", throttle_duration_sec=1.0)
            return
        self.applied_torque = np.clip(torque, -self.effort_limits, self.effort_limits).astype(np.float32)
        self.last_torque_stamp = self.get_clock().now()

    def _on_policy_mode(self, msg: String) -> None:
        mode = msg.data.strip()
        if mode not in VALID_POLICY_MODES:
            self.get_logger().warning(
                f"Ignoring invalid policy mode '{msg.data}'. Expected one of {VALID_POLICY_MODES}",
                throttle_duration_sec=1.0,
            )
            return
        if mode == self.policy_mode:
            return
        self.get_logger().info(f"MuJoCo policy mode changed: {self.policy_mode} -> {mode}")
        self.policy_mode = mode
        if mode == "disabled":
            self._capture_base_freeze()
            self._apply_base_freeze()
        else:
            self.frozen_base_qpos = None

    def _write_actuator_torques(self, torques: np.ndarray) -> None:
        for actuator_id, torque in zip(self.actuator_ids, torques):
            self.data.ctrl[actuator_id] = float(torque)

    def _sim_tick(self) -> None:
        now = self.get_clock().now()
        self._apply_base_freeze()
        joint_pos = self._joint_qpos()
        joint_vel = self._joint_qvel()

        if self.control_mode == "position":
            target_age = (now.nanoseconds - self.last_target_stamp.nanoseconds) * 1e-9
            if target_age > self.target_timeout_s:
                self.target_joint_pos = joint_pos.copy()
            torque = self.kp * (self.target_joint_pos - joint_pos) - self.kd * joint_vel
            self.applied_torque = np.clip(torque, -self.effort_limits, self.effort_limits).astype(np.float32)
        else:
            torque_age = (now.nanoseconds - self.last_torque_stamp.nanoseconds) * 1e-9
            if torque_age > self.target_timeout_s:
                self.applied_torque = np.zeros(ACTION_DIM, dtype=np.float32)

        self._write_actuator_torques(self.applied_torque)

        self.mujoco.mj_step(self.model, self.data)
        self._apply_base_freeze()
        self.step_count += 1
        if self.step_count % self.publish_decimation == 0:
            self._publish_sensors()
        if self.publish_cmd_vel and self.step_count % self.command_decimation == 0:
            self._publish_command()

        if self.viewer_handle is not None:
            if not self.viewer_handle.is_running():
                self.get_logger().info("MuJoCo viewer closed; shutting down node")
                rclpy.shutdown()
                return
            with self.viewer_handle.lock():
                self._update_viewer_debug_visuals()
            self.viewer_handle.sync()

        if self.reset_on_fall and self.free_qpos_addr is not None and self.data.qpos[self.free_qpos_addr + 2] < self.fall_height:
            self.get_logger().warning("Resetting MuJoCo robot after fall", throttle_duration_sec=1.0)
            self._reset_robot()
            self.target_joint_pos = self.default_joint_pos.copy()
            self.applied_torque = np.zeros(ACTION_DIM, dtype=np.float32)

    def _publish_sensors(self) -> None:
        now = self.get_clock().now().to_msg()
        joint_pos = self._joint_qpos()
        joint_vel = self._joint_qvel()

        joint_msg = JointState()
        joint_msg.header.stamp = now
        joint_msg.name = list(self.joint_names)
        joint_msg.position = joint_pos.astype(float).tolist()
        joint_msg.velocity = joint_vel.astype(float).tolist()
        joint_msg.effort = self.applied_torque.astype(float).tolist()
        self.joint_pub.publish(joint_msg)

        imu_ang_vel = self._sensor_data(self.imu_sensor_id).astype(np.float32)
        imu_quat = self._site_quat_xyzw(self.imu_site_id)
        imu_msg = Imu()
        imu_msg.header.stamp = now
        imu_msg.header.frame_id = "imu"
        imu_msg.orientation.x = float(imu_quat[0])
        imu_msg.orientation.y = float(imu_quat[1])
        imu_msg.orientation.z = float(imu_quat[2])
        imu_msg.orientation.w = float(imu_quat[3])
        imu_msg.angular_velocity.x = float(imu_ang_vel[0])
        imu_msg.angular_velocity.y = float(imu_ang_vel[1])
        imu_msg.angular_velocity.z = float(imu_ang_vel[2])
        self.imu_pub.publish(imu_msg)

        debug = np.concatenate([joint_pos, joint_vel, self.target_joint_pos, self.applied_torque], dtype=np.float32)
        self.state_debug_pub.publish(Float32MultiArray(data=debug.tolist()))

        base_debug = np.array([float(self.data.time)], dtype=np.float32)
        if self.free_qpos_addr is not None:
            base_debug = np.concatenate(
                [
                    base_debug,
                    self.data.qpos[self.free_qpos_addr : self.free_qpos_addr + 7].astype(np.float32),
                ],
                dtype=np.float32,
            )
        if self.free_qvel_addr is not None:
            base_debug = np.concatenate(
                [
                    base_debug,
                    self.data.qvel[self.free_qvel_addr : self.free_qvel_addr + 6].astype(np.float32),
                ],
                dtype=np.float32,
            )
        self.base_state_debug_pub.publish(Float32MultiArray(data=base_debug.tolist()))

    def _publish_command(self) -> None:
        msg = Twist()
        msg.linear.x = float(self.command[0])
        msg.linear.y = float(self.command[1])
        msg.angular.z = float(self.command[2])
        self.cmd_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: MuJoCoBodyNode | None = None
    try:
        node = MuJoCoBodyNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            if node.viewer_handle is not None:
                node.viewer_handle.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
