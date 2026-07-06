from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("bdx_policy_deploy"))
    default_config = package_share / "config" / "bdx_mujoco_test.yaml"
    default_policy = package_share / "assets" / "policies" / "model_20260706.onnx"
    default_xml = package_share / "assets" / "mujoco" / "xmls" / "scene.xml"

    config = LaunchConfiguration("config")
    policy_path = LaunchConfiguration("policy_path")
    xml_path = LaunchConfiguration("xml_path")
    viewer = LaunchConfiguration("viewer")
    command_x = LaunchConfiguration("command_x")
    command_y = LaunchConfiguration("command_y")
    command_yaw = LaunchConfiguration("command_yaw")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=str(default_config)),
            DeclareLaunchArgument("policy_path", default_value=str(default_policy)),
            DeclareLaunchArgument("xml_path", default_value=str(default_xml)),
            DeclareLaunchArgument("viewer", default_value="false"),
            DeclareLaunchArgument("command_x", default_value="0.2"),
            DeclareLaunchArgument("command_y", default_value="0.0"),
            DeclareLaunchArgument("command_yaw", default_value="0.0"),
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
                        "command_x": command_x,
                        "command_y": command_y,
                        "command_yaw": command_yaw,
                    },
                ],
            ),
            Node(
                package="bdx_policy_deploy",
                executable="policy_node",
                name="bdx_policy_node",
                output="screen",
                parameters=[
                    config,
                    {"policy_path": policy_path},
                ],
            ),
        ]
    )
