"""Locked joints set at run time give the model cuRobo builds with them locked there."""

import copy
import os
from pathlib import Path

from curobo.config_io import load_yaml
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.scene import Cuboid, Scene
from curobo.types import GoalToolPose, JointState, Pose
from curobo_ros.robot.locked_joints import LockedJoints
from curobo_ros.robot.robot_description import bundled_curobo_config
import pytest
import torch

CONFIG = Path(__file__).resolve().parents[1] / 'config'

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


def config(name, lock_joints=None):
    """Return the cuRobo config `name` (bundled or a path) with these locks, posing all links."""
    cfg = load_yaml(name if os.path.isabs(name) else bundled_curobo_config(name))
    kinematics = cfg['robot_cfg']['kinematics']
    if lock_joints is not None:
        kinematics['lock_joints'] = lock_joints
    links = Kinematics(KinematicsCfg.from_robot_yaml_file(copy.deepcopy(cfg)))
    kinematics['tool_frames'] = list(links.kinematics_config.link_name_to_idx_map)
    return cfg


def state(kin):
    """Return the kinematics at a fixed, non-zero arm configuration."""
    q = torch.linspace(0.1, 0.6, len(kin.joint_names), device='cuda').unsqueeze(0)
    return kin.compute_kinematics(JointState(position=q, joint_names=kin.joint_names))


def link_poses(kin):
    """Return every link's pose (4x4) at that configuration."""
    poses = state(kin).tool_poses
    return Pose(poses.position.reshape(-1, 3), poses.quaternion.reshape(-1, 4)).get_affine_matrix()


def spheres(kin):
    """Return the collision spheres at that configuration, as x, y, z, radius."""
    return state(kin).robot_spheres.reshape(-1, 4)


def link_spheres(kin, link):
    """Return the collision spheres of `link` at that configuration."""
    kc = kin.kinematics_config
    return spheres(kin)[kc.link_sphere_idx_map.reshape(-1) == kc.link_name_to_idx_map[link]]


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

    assert locked.update([measured[n] for n in locked.names])

    assert locked.values == pytest.approx(list(measured.values()))
    torch.testing.assert_close(link_poses(kin), link_poses(expected), atol=1e-4, rtol=0.0)
    # The collision spheres too (simple_mimic_robot has none).
    torch.testing.assert_close(spheres(kin), spheres(expected), atol=1e-4, rtol=0.0)
    assert not locked.update([measured[n] for n in locked.names])  # unchanged


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
    assert not locked.update([])


def test_update_needs_a_value_per_locked_joint():
    cfg = config('franka.yml')
    locked = LockedJoints(Kinematics(KinematicsCfg.from_robot_yaml_file(copy.deepcopy(cfg))), cfg)

    with pytest.raises(ValueError, match='expected 2 values'):
        locked.update([0.01])


def test_copy_to_writes_only_the_locked_joints_rows():
    cfg = config('franka.yml')
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(copy.deepcopy(cfg)))
    solver = Kinematics(KinematicsCfg.from_robot_yaml_file(copy.deepcopy(cfg)))
    locked = LockedJoints(kin, cfg)
    dst = solver.kinematics_config.fixed_transforms
    before, pointer = dst.clone(), dst.data_ptr()

    locked.update([0.01, 0.03])
    locked.copy_to(solver.kinematics_config)

    finger = [kin.kinematics_config.link_name_to_idx_map[n]
              for n in ('panda_leftfinger', 'panda_rightfinger')]
    changed = (dst != before).flatten(1).any(1).nonzero().reshape(-1).tolist()
    assert sorted(changed) == sorted(finger)
    assert dst.data_ptr() == pointer  # in place
    torch.testing.assert_close(link_poses(solver), link_poses(kin), atol=0.0, rtol=0.0)
    torch.testing.assert_close(spheres(solver), spheres(kin), atol=0.0, rtol=0.0)


def test_copy_to_refuses_a_model_of_another_robot():
    cfg = config('franka.yml')
    locked = LockedJoints(Kinematics(KinematicsCfg.from_robot_yaml_file(copy.deepcopy(cfg))), cfg)
    other = Kinematics(KinematicsCfg.from_robot_yaml_file(config('ur10e.yml')))

    with pytest.raises(ValueError, match='other links'):
        locked.copy_to(other.kinematics_config)


def test_a_locked_joint_of_one_arm_leaves_the_other_arm_and_its_attachment_alone():
    cfg = config(str(CONFIG / 'dual_ur10e_attach.yml'), {'wrist_3_joint_1': 0.0})
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(copy.deepcopy(cfg)))
    kc = kin.kinematics_config
    # An object held by each arm, off the wrist_3 axes.
    held = torch.tensor([[0.05, 0.0, 0.05, 0.03]] * 4, device='cuda')
    for link in ('attached_object_0', 'attached_object_1'):
        kc.update_link_spheres(link, held)
    locked = LockedJoints(kin, cfg)
    arm_0 = [n for n in kc.link_name_to_idx_map
             if not n.endswith('_1') and n not in ('tool1', 'world_base_link')]
    rows, poses = kc.fixed_transforms.clone(), link_poses(kin)
    sphere_table = kc.link_spheres.clone()
    held_0, held_1 = (link_spheres(kin, 'attached_object_0').clone(),
                      link_spheres(kin, 'attached_object_1').clone())

    assert locked.update([1.0])

    index = [kc.link_name_to_idx_map[n] for n in arm_0]
    changed = (kc.fixed_transforms != rows).flatten(1).any(1).nonzero().reshape(-1).tolist()
    assert changed == [kc.link_name_to_idx_map['wrist_3_link_1']]
    assert torch.equal(link_poses(kin)[index], poses[index])
    assert torch.equal(kc.link_spheres, sphere_table)
    assert torch.equal(link_spheres(kin, 'attached_object_0'), held_0)
    assert (link_spheres(kin, 'attached_object_1') - held_1).abs().max() > 1e-2
    assert (link_poses(kin)[kc.link_name_to_idx_map['tool1']]
            - poses[kc.link_name_to_idx_map['tool1']]).abs().max() > 1e-2


def test_a_warmed_cuda_graph_solver_sees_the_copied_values():
    """
    An IK replaying its captured CUDA graph takes the values copy_to writes.

    The goal is the hand pose of the default configuration; a small box sits
    where the open left finger is. With the fingers closed the hand reaches the
    goal; open, the finger is in the box.
    """
    cfg = load_yaml(bundled_curobo_config('franka.yml'))  # the hand is the only tool frame
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(copy.deepcopy(cfg)))
    locked = LockedJoints(kin, cfg)
    default = cfg['robot_cfg']['kinematics']['cspace']['default_joint_position'][:7]
    js = JointState(position=torch.tensor([default], device='cuda'), joint_names=kin.joint_names)
    hand = kin.compute_kinematics(js).tool_poses
    goal = Pose(hand.position.reshape(-1, 3)[:1], hand.quaternion.reshape(-1, 4)[:1])
    kc = kin.kinematics_config
    left = kc.link_sphere_idx_map.reshape(-1) == kc.link_name_to_idx_map['panda_leftfinger']
    finger = kin.compute_kinematics(js).robot_spheres.reshape(-1, 4)[left, :3].mean(0)
    box = Cuboid(name='box', pose=finger.tolist() + goal.quaternion[0].tolist(),
                 dims=[0.012, 0.012, 0.012])
    ik = InverseKinematics(InverseKinematicsCfg.create(
        robot=copy.deepcopy(cfg), scene_model=Scene(cuboid=[box]), num_seeds=20,
        use_cuda_graph=True, self_collision_check=True))
    target = GoalToolPose.from_poses({ik.kinematics.tool_frames[0]: goal})

    def reaches():
        result = ik.solve_pose(goal_tool_poses=target, current_state=None)
        return bool(result.success.reshape(-1)[0].item())

    assert locked.values == pytest.approx([0.04, 0.04])
    assert not reaches() and not reaches()  # warmed open: the finger is in the box
    for fingers, reached in (([0.0, 0.0], True), ([0.04, 0.04], False), ([0.0, 0.0], True)):
        locked.update(fingers)
        locked.copy_to(ik.kinematics.kinematics_config)
        assert reaches() == reached, fingers
