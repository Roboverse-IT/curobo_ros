"""
Test launch file for config/config_test/test_dual_arm.yaml.

gen_traj_test.launch.py with robot:=dual_ur10e, cuRobo's stock dual UR10e (two
arms, two tool frames). The other three pinned arguments, and why they are
pinned, are the same as there.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    curobo_ros_launch_dir = os.path.join(
        get_package_share_directory('curobo_ros'), 'launch')

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(curobo_ros_launch_dir, 'gen_traj.launch.py')
            ),
            launch_arguments={
                'gui': 'false',
                'voxel_size': '0.05',
                'mapper_extent_xyz': '[2.0, 2.0, 2.0]',
                'robot': 'dual_ur10e',
            }.items()
        ),
    ])
