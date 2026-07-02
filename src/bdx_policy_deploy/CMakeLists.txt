cmake_minimum_required(VERSION 3.8)
project(bdx_policy_deploy)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/mujoco_body_node
  scripts/policy_node
  scripts/joint_pose_command_node
  scripts/pygame_heading_command_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  assets
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_policy_interface test/test_policy_interface.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()

ament_package()
