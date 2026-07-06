from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("bdx_policy_deploy"))
    default_config = package_share / "config" / "bdx_policy.yaml"
    default_policy = package_share / "assets" / "policies" / "model_20260706.onnx"

    config = LaunchConfiguration("config")
    dry_run = LaunchConfiguration("dry_run")
    policy_path = LaunchConfiguration("policy_path")
    actuator_mode = LaunchConfiguration("actuator_mode")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=str(default_config)),
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument("policy_path", default_value=str(default_policy)),
            DeclareLaunchArgument("actuator_mode", default_value="position"),
            Node(
                package="bdx_policy_deploy",
                executable="policy_node",
                name="bdx_policy_node",
                output="screen",
                parameters=[
                    config,
                    {
                        "dry_run": dry_run,
                        "policy_path": policy_path,
                        "actuator_mode": actuator_mode,
                    },
                ],
            ),
        ]
    )
