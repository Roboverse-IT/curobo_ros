"""
Test launch file for config/config_test/test_dual_arm_attach.yaml.

gen_traj_test.launch.py with robot:=dual_ur10e_attach. Its cuRobo config keeps
cuRobo's relative urdf_path (resolved by cuRobo in its own assets), which
gen_traj.launch.py would hand to robot_state_publisher as is: the absolute path
of that URDF is passed here, read from cuRobo's install.
"""

import os

from ament_index_python.packages import get_package_share_directory
from curobo.content import get_assets_path
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
                'robot': 'dual_ur10e_attach',
                'urdf_path': os.path.join(
                    get_assets_path(), 'robot/ur_description/dual_ur10e.urdf'),
            }.items()
        ),
    ])
