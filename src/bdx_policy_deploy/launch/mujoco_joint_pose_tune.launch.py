from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("bdx_policy_deploy"))
    default_config = package_share / "config" / "bdx_mujoco_test.yaml"
    default_xml = package_share / "assets" / "mujoco" / "xmls" / "scene.xml"

    config = LaunchConfiguration("config")
    xml_path = LaunchConfiguration("xml_path")
    viewer = LaunchConfiguration("viewer")
    base_height = LaunchConfiguration("base_height")
    step_deg = LaunchConfiguration("step_deg")
    publish_rate_hz = LaunchConfiguration("publish_rate_hz")
    show_imu_visual = LaunchConfiguration("show_imu_visual")
    imu_axis_length = LaunchConfiguration("imu_axis_length")
    imu_axis_radius = LaunchConfiguration("imu_axis_radius")
    imu_marker_radius = LaunchConfiguration("imu_marker_radius")
    socket_bridge = LaunchConfiguration("socket_bridge")
    socket_host = LaunchConfiguration("socket_host")
    socket_port = LaunchConfiguration("socket_port")
    socket_protocol = LaunchConfiguration("socket_protocol")
    socket_rate_limit_hz = LaunchConfiguration("socket_rate_limit_hz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=str(default_config)),
            DeclareLaunchArgument("xml_path", default_value=str(default_xml)),
            DeclareLaunchArgument("viewer", default_value="true"),
            DeclareLaunchArgument("base_height", default_value="0.33"),
            DeclareLaunchArgument("step_deg", default_value="1.0"),
            DeclareLaunchArgument("publish_rate_hz", default_value="100.0"),
            DeclareLaunchArgument("show_imu_visual", default_value="true"),
            DeclareLaunchArgument("imu_axis_length", default_value="0.08"),
            DeclareLaunchArgument("imu_axis_radius", default_value="0.004"),
            DeclareLaunchArgument("imu_marker_radius", default_value="0.018"),
            DeclareLaunchArgument("socket_bridge", default_value="true"),
            DeclareLaunchArgument("socket_host", default_value="192.168.31.202"),
            DeclareLaunchArgument("socket_port", default_value="2333"),
            DeclareLaunchArgument("socket_protocol", default_value="udp"),
            DeclareLaunchArgument("socket_rate_limit_hz", default_value="0.0"),
            Node(
                package="bdx_policy_deploy",
                executable="mujoco_body_node",
                name="bdx_mujoco_body_node",
                output="screen",
                parameters=[
                    config,
                    {
                        "xml_path": xml_path,
                        "viewer": viewer,
                        "publish_cmd_vel": False,
                        "initial_policy_mode": "disabled",
                        "base_height": base_height,
                        "disabled_base_height": base_height,
                        "show_imu_visual": show_imu_visual,
                        "imu_axis_length": imu_axis_length,
                        "imu_axis_radius": imu_axis_radius,
                        "imu_marker_radius": imu_marker_radius,
                    },
                ],
            ),
            Node(
                package="bdx_policy_deploy",
                executable="joint_pose_command_node",
                name="bdx_joint_pose_command_node",
                output="screen",
                parameters=[
                    config,
                    {
                        "step_deg": step_deg,
                        "publish_rate_hz": publish_rate_hz,
                        "policy_mode": "disabled",
                        "publish_policy_mode": True,
                    }
                ],
            ),
            Node(
                package="bdx_policy_deploy",
                executable="joint_pose_socket_bridge_node",
                name="bdx_joint_pose_socket_bridge_node",
                output="screen",
                condition=IfCondition(socket_bridge),
                parameters=[
                    config,
                    {
                        "target_joint_state_topic": "/bdx_policy/target_joint_states",
                        "remote_host": socket_host,
                        "remote_port": socket_port,
                        "socket_protocol": socket_protocol,
                        "send_rate_limit_hz": socket_rate_limit_hz,
                    }
                ],
            ),
        ]
    )
