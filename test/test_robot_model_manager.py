"""RobotModelManager: collision spheres when the robot also reports locked joints."""

from curobo._src.util_file import get_robot_configs_path, join_path
from curobo_ros.core.robot_model_manager import RobotModelManager
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

FRANKA = join_path(get_robot_configs_path(), 'franka.yml')  # fingers locked: 9 joints, 7 active
ARM = [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0]
NAMES = [f'panda_joint{i}' for i in range(1, 8)] + ['panda_finger_joint1', 'panda_finger_joint2']


class Robot:
    """Reports every joint of its description, as the robot strategies do."""

    def __init__(self, names, positions):
        self.names, self.positions = names, positions

    def get_joint_name(self):
        return self.names

    def get_joint_pose(self):
        return self.positions


def test_collision_spheres_from_all_robot_joints():
    """The spheres follow the arm joints, whatever the robot's joint order."""
    full = RobotModelManager(FRANKA, Robot(NAMES, ARM + [0.04, 0.04]), 'panda_link0')
    reversed_order = RobotModelManager(
        FRANKA, Robot(NAMES[::-1], (ARM + [0.04, 0.04])[::-1]), 'panda_link0')
    expected = full.get_kinematics_state(ARM).robot_spheres.reshape(-1, 4).tolist()

    assert full.get_collision_spheres() == expected
    assert reversed_order.get_collision_spheres() == expected
    spheres, mask = full.get_collision_spheres_with_attached(full.kin_model)
    assert spheres == expected and len(mask) == len(spheres)
