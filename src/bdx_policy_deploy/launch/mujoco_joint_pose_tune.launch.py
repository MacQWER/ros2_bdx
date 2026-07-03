from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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
    show_imu_visual = LaunchConfiguration("show_imu_visual")
    imu_axis_length = LaunchConfiguration("imu_axis_length")
    imu_axis_radius = LaunchConfiguration("imu_axis_radius")
    imu_marker_radius = LaunchConfiguration("imu_marker_radius")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=str(default_config)),
            DeclareLaunchArgument("xml_path", default_value=str(default_xml)),
            DeclareLaunchArgument("viewer", default_value="true"),
            DeclareLaunchArgument("base_height", default_value="0.33"),
            DeclareLaunchArgument("step_deg", default_value="1.0"),
            DeclareLaunchArgument("show_imu_visual", default_value="true"),
            DeclareLaunchArgument("imu_axis_length", default_value="0.08"),
            DeclareLaunchArgument("imu_axis_radius", default_value="0.004"),
            DeclareLaunchArgument("imu_marker_radius", default_value="0.018"),
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
                    {
                        "step_deg": step_deg,
                        "policy_mode": "disabled",
                        "publish_policy_mode": True,
                    }
                ],
            ),
        ]
    )
