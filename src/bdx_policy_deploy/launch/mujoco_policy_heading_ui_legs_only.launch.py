from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("bdx_policy_deploy"))
    default_config = package_share / "config" / "bdx_legs_only.yaml"
    default_policy = package_share / "assets" / "policies" / "bdx_legs_only_obs_norm.onnx"
    default_xml = package_share / "assets" / "mujoco" / "xmls" / "scene_legs_only.xml"

    config = LaunchConfiguration("config")
    policy_path = LaunchConfiguration("policy_path")
    xml_path = LaunchConfiguration("xml_path")
    viewer = LaunchConfiguration("viewer")
    show_com_visual = LaunchConfiguration("show_com_visual")
    robot_model_alpha = LaunchConfiguration("robot_model_alpha")
    initial_linear_x = LaunchConfiguration("initial_linear_x")
    initial_linear_y = LaunchConfiguration("initial_linear_y")
    initial_heading_deg = LaunchConfiguration("initial_heading_deg")
    initial_policy_mode = LaunchConfiguration("initial_policy_mode")
    real_obs_compare = LaunchConfiguration("real_obs_compare")
    real_obs_bind_port = LaunchConfiguration("real_obs_bind_port")
    real_obs_remote_host = LaunchConfiguration("real_obs_remote_host")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=str(default_config)),
            DeclareLaunchArgument("policy_path", default_value=str(default_policy)),
            DeclareLaunchArgument("xml_path", default_value=str(default_xml)),
            DeclareLaunchArgument("viewer", default_value="true"),
            DeclareLaunchArgument("show_com_visual", default_value="true"),
            DeclareLaunchArgument("robot_model_alpha", default_value="0.35"),
            DeclareLaunchArgument("initial_linear_x", default_value="0.0"),
            DeclareLaunchArgument("initial_linear_y", default_value="0.0"),
            DeclareLaunchArgument("initial_heading_deg", default_value="0.0"),
            DeclareLaunchArgument("initial_policy_mode", default_value="disabled"),
            DeclareLaunchArgument("real_obs_compare", default_value="true"),
            DeclareLaunchArgument("real_obs_bind_port", default_value="2333"),
            DeclareLaunchArgument("real_obs_remote_host", default_value="192.168.31.202"),
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
                        "show_com_visual": show_com_visual,
                        "robot_model_alpha": robot_model_alpha,
                        "publish_cmd_vel": False,
                        "initial_policy_mode": initial_policy_mode,
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
                    {
                        "policy_path": policy_path,
                        "initial_policy_mode": initial_policy_mode,
                    },
                ],
            ),
            Node(
                package="bdx_policy_deploy",
                executable="pygame_heading_command_node",
                name="bdx_pygame_heading_command_node",
                output="screen",
                parameters=[
                    {
                        "initial_linear_x": initial_linear_x,
                        "initial_linear_y": initial_linear_y,
                        "initial_heading_deg": initial_heading_deg,
                        "initial_policy_mode": initial_policy_mode,
                    }
                ],
            ),
            Node(
                package="bdx_policy_deploy",
                executable="real_observation_socket_compare_node",
                name="bdx_real_observation_socket_compare_node",
                output="screen",
                condition=IfCondition(real_obs_compare),
                parameters=[
                    config,
                    {
                        "bind_port": real_obs_bind_port,
                        "remote_host": real_obs_remote_host,
                        "initial_policy_mode": initial_policy_mode,
                        "compare_only_in_policy_mode": "zero_action",
                    },
                ],
            ),
        ]
    )
