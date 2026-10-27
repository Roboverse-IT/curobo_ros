"""Locked joints set at run time give the model cuRobo builds with them locked there."""

import copy

from curobo.config_io import load_yaml
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState, Pose
from curobo_ros.robot.locked_joints import LockedJoints
from curobo_ros.robot.robot_description import bundled_curobo_config
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


def config(name, lock_joints=None):
    """Return a bundled cuRobo robot config, with these locks, posing every link."""
    cfg = load_yaml(bundled_curobo_config(name))
    kinematics = cfg['robot_cfg']['kinematics']
    if lock_joints is not None:
        kinematics['lock_joints'] = lock_joints
    links = Kinematics(KinematicsCfg.from_robot_yaml_file(copy.deepcopy(cfg)))
    kinematics['tool_frames'] = list(links.kinematics_config.link_name_to_idx_map)
    return cfg


def link_poses(kin):
    """Every link's pose (4x4) at a fixed, non-zero arm configuration."""
    q = torch.linspace(0.1, 0.6, len(kin.joint_names), device='cuda').unsqueeze(0)
    poses = kin.compute_kinematics(JointState(position=q, joint_names=kin.joint_names)).tool_poses
    return Pose(poses.position.reshape(-1, 3), poses.quaternion.reshape(-1, 4)).get_affine_matrix()


@pytest.mark.parametrize('robot, configured, measured', [
    # Two prismatic fingers.
    ('franka.yml', {'panda_finger_joint1': 0.04, 'panda_finger_joint2': 0.04},
     {'panda_finger_joint1': 0.01, 'panda_finger_joint2': 0.03}),
    # A revolute joint and its mimic (multiplier -1.5, offset 0.5).
    ('simple_mimic_robot.yml', {'chain_1_active_joint_1': 0.2}, {'chain_1_active_joint_1': -0.7}),
])
def test_update_gives_the_model_locked_at_the_values(robot, configured, measured):
    cfg = config(robot, configured)
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(copy.deepcopy(cfg)))
    locked = LockedJoints(kin, cfg)
    expected = Kinematics(KinematicsCfg.from_robot_yaml_file(config(robot, measured)))

    assert locked.names == list(configured)
    assert locked.values == pytest.approx(list(configured.values()))
    assert (link_poses(kin) - link_poses(expected)).abs().max() > 1e-3  # they differ

    locked.update([measured[n] for n in locked.names])

    assert locked.values == pytest.approx(list(measured.values()))
    torch.testing.assert_close(link_poses(kin), link_poses(expected), atol=1e-4, rtol=0.0)


def test_update_back_to_the_configured_values_restores_the_model():
    configured = {'chain_1_active_joint_1': 0.2}
    cfg = config('simple_mimic_robot.yml', configured)
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(copy.deepcopy(cfg)))
    before = link_poses(kin)
    locked = LockedJoints(kin, cfg)

    locked.update([1.1])
    locked.update([0.2])

    torch.testing.assert_close(link_poses(kin), before, atol=1e-5, rtol=0.0)


def test_a_model_without_locked_joints_has_none():
    cfg = config('ur10e.yml')
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(copy.deepcopy(cfg)))
    locked = LockedJoints(kin, cfg)

    assert locked.names == []
    locked.update([])


def test_update_needs_a_value_per_locked_joint():
    cfg = config('franka.yml')
    locked = LockedJoints(Kinematics(KinematicsCfg.from_robot_yaml_file(copy.deepcopy(cfg))), cfg)

    with pytest.raises(ValueError, match='expected 2 values'):
        locked.update([0.01])
