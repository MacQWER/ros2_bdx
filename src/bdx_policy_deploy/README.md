# bdx_policy_deploy

ROS 2 package for running the BDX ONNX locomotion policy, testing it in MuJoCo, and manually commanding joints for hardware alignment.

## Environment

ROS 2 uses the system Python, not the conda Python. Build and run from the workspace with:

```bash
cd /home/ubuntu/dev/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select bdx_policy_deploy
source install/setup.bash
export ROS_LOG_DIR=/tmp/ros_logs
```

After rebuilding, source `install/setup.bash` again before running launch files.

## Main Nodes

- `policy_node`: subscribes to robot observations and `/cmd_vel`, runs the ONNX policy, and publishes joint targets.
- `mujoco_body_node`: uses MuJoCo as the robot body, publishes `/joint_states` and `/imu/data`, and applies PD torques to follow joint targets.
- `pygame_heading_command_node`: pygame UI for heading-based velocity commands.
- `joint_pose_command_node`: pygame slider UI for directly commanding joint angles.

## Policy Test With Heading UI

Launch MuJoCo, policy, and the heading command UI:

```bash
ros2 launch bdx_policy_deploy mujoco_policy_heading_ui.launch.py viewer:=true
```

The UI starts with velocity zero and mode `disabled`.

Controls:

```text
1: disabled    policy sends no target; MuJoCo base is fixed
2: zero_action policy sends default joint target
3: policy      normal ONNX policy running

W/S or Up/Down: vx +/- 0.2
A/D:            vy +/- 0.3
Q/E or Left/Right: target heading +/- 30 deg
R: align target heading to current heading
Space: stop and align heading
Esc: quit
```

The policy still receives `[vx, vy, yaw_rate]`; the UI converts target heading to yaw rate.

## Direct Joint Pose Tuning

Use this mode to hang the robot in simulation at `base_z=0.33` and directly command joint positions. This does not start the policy node.

```bash
ros2 launch bdx_policy_deploy mujoco_joint_pose_tune.launch.py viewer:=true
```

The joint tuner publishes:

```text
/bdx_policy/target_joint_states
```

MuJoCo tracks those targets using the PD logic inside `mujoco_body_node`.
The launch also starts `joint_pose_socket_bridge_node` by default, which forwards the same target joint positions to:

```text
udp://192.168.31.202:2333
```

The joint target publisher runs at 100 Hz by default, and socket forwarding follows that rate unless `socket_rate_limit_hz` is set.

Each packet is newline-terminated JSON with all joint positions in radians. Abbreviated example:

```json
{"type":"bdx_joint_pose","seq":0,"stamp":{"sec":0,"nanosec":0},"joint_names":["Left_Hip_Yaw"],"position_rad":[0.0]}
```

Socket options can be overridden at launch:

```bash
ros2 launch bdx_policy_deploy mujoco_joint_pose_tune.launch.py \
  publish_rate_hz:=100.0 socket_host:=192.168.31.202 socket_port:=2333 socket_protocol:=udp
```

Disable socket forwarding with:

```bash
ros2 launch bdx_policy_deploy mujoco_joint_pose_tune.launch.py socket_bridge:=false
```

The MuJoCo viewer highlights the IMU site by default with a yellow marker, an `IMU` label, and RGB axes:

```text
Red: IMU +X
Green: IMU +Y
Blue: IMU +Z
```

You can tune or hide the IMU visual:

```bash
ros2 launch bdx_policy_deploy mujoco_joint_pose_tune.launch.py \
  show_imu_visual:=true imu_axis_length:=0.08 imu_axis_radius:=0.004 imu_marker_radius:=0.018
```

Slider UI:

```text
Drag yellow sliders: set target joint angles
Blue marker: current measured joint angle
Left text: target / current / error
```

Keyboard fallback:

```text
1-0: select one of the 10 joints
W/S or Up/Down: switch selected joint
A/D or Left/Right: fine adjust selected joint
Shift + A/D: larger adjust
Home: reset selected joint
R or Backspace: reset all joints
Esc: quit
```

## Policy Modes

Policy mode is controlled by:

```text
/bdx_policy/mode  std_msgs/String
```

Supported values:

```text
disabled
zero_action
policy
```

For policy deployment, `action_clip: 0.0` disables raw action clipping. Joint targets are still clipped to configured joint limits.

Manual commands:

```bash
ros2 topic pub --once /bdx_policy/mode std_msgs/msg/String "{data: disabled}"
ros2 topic pub --once /bdx_policy/mode std_msgs/msg/String "{data: zero_action}"
ros2 topic pub --once /bdx_policy/mode std_msgs/msg/String "{data: policy}"
```

## Useful Topics

```text
/cmd_vel                         velocity command
/joint_states                    MuJoCo joint state
/imu/data                        MuJoCo IMU
/bdx_policy/target_joint_states  policy or joint tuner target
/bdx_policy/debug/observation    policy observation
/bdx_policy/debug/action         ONNX action after clipping
/bdx_policy/diagnostics          policy status
/bdx_mujoco/debug/base_state     [sim_time, base xyz, base quat wxyz, base velocity]
/bdx_mujoco/debug/state          [joint_pos, joint_vel, target_joint_pos, applied_torque]
```

`/bdx_mujoco/debug/base_state` index `3` is `base_z`.
