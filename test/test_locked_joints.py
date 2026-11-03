"""Locked joints that follow the measured joint state: synthetic models and a cuRobo robot."""

import copy
import math

from curobo_ros.core.locked_joints import (
    LockedJointFollower, LockedJointModelError, LockedJointTransforms, urdf_joint_limits)
import pytest
import torch


def _rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)


def _row(rotation, translation):
    return torch.cat([rotation, torch.tensor(translation, dtype=torch.float64).view(3, 1)], dim=1)


class Model:
    """Four links: a fixed one, a revolute knuckle and its mimic (x -1), a prismatic finger."""

    def __init__(self, bent=False):
        self.bent = bent

    def __call__(self, values):
        knuckle, finger = values.get('knuckle', 0.0), values.get('finger', 0.0)
        offset = [0.1 + (0.05 * knuckle ** 2 if self.bent else 0.0), 0.0, 0.2]
        return torch.stack([
            _row(_rot_z(0.3), [1.0, 2.0, 3.0]),
            _row(_rot_z(knuckle), offset),
            _row(_rot_z(-knuckle), [-0.1, 0.0, 0.2]),
            _row(torch.eye(3, dtype=torch.float64), [0.0, finger, 0.05]),
        ]).to(torch.float32)


DEFAULTS = {'knuckle': 0.0, 'finger': 0.04}
LIMITS = {'knuckle': (0.0, 0.8), 'finger': (0.0, 0.04)}


def transforms(model=None):
    model = model or Model()
    return LockedJointTransforms(DEFAULTS, model, model(DEFAULTS), limits=LIMITS)


def test_transforms_match_model_builds():
    """Any lock value gives the transforms a model build gives."""
    t = transforms()
    assert t.moving_rows == [1, 2, 3]
    for values in ({'knuckle': 0.55, 'finger': 0.01}, {'knuckle': 0.8, 'finger': 0.0}):
        expected = Model()(values).to(torch.float64)
        assert (t(values) - expected).abs().max() < 1e-6


def test_model_not_a_constant_axis_motion_fails():
    """A link that moves along a curve, not about a fixed axis, is refused at startup."""
    with pytest.raises(LockedJointModelError):
        transforms(Model(bent=True))


class Lock:
    def __init__(self, names, position):
        self.joint_names, self.position = names, position


class Config:
    """The two fields of cuRobo's KinematicsParams the follower writes."""

    def __init__(self, fixed, lock_names):
        self.fixed_transforms = fixed.clone()
        self.lock_jointstate = Lock(lock_names, torch.zeros(1, len(lock_names)))


def test_follower_writes_measured_values_in_place():
    """Measured values (clamped, rounded) land in every config without new tensors."""
    follower = LockedJointFollower(transforms(), limits=LIMITS, resolution=0.01)
    config = Config(Model()(DEFAULTS), ['other', 'knuckle', 'finger'])
    memory = config.fixed_transforms.data_ptr()

    values, sources = follower.values()
    assert values == DEFAULTS and set(sources.values()) == {'default'}

    follower.observe(['finger', 'knuckle', 'arm'], [0.0123, 0.9, 1.0])
    values, sources = follower.values()
    assert values == {'knuckle': 0.8, 'finger': 0.01}
    assert sources == {'knuckle': 'measured', 'finger': 'measured'}
    assert follower.apply(values, [config]) == []
    assert config.fixed_transforms.data_ptr() == memory
    assert (config.fixed_transforms - Model()(values)).abs().max() < 1e-6
    assert config.lock_jointstate.position.tolist() == [[0.0, pytest.approx(0.8),
                                                         pytest.approx(0.01)]]


def test_follower_refuses_another_layout():
    """A model whose other links differ is refused and left as it was."""
    follower = LockedJointFollower(transforms(), limits=LIMITS)
    other = Model()(DEFAULTS)
    other[0, 0, 3] += 0.5
    config = Config(other, ['knuckle', 'finger'])
    values, _ = follower.values()
    assert follower.apply(values, [config]) == [config]
    assert torch.equal(config.fixed_transforms, other)


def test_urdf_joint_limits():
    urdf = ('<robot name="r"><joint name="a" type="revolute"><limit lower="0" upper="0.8"/>'
            '</joint><joint name="b" type="fixed"/></robot>')
    assert urdf_joint_limits(urdf, ['a', 'b']) == {'a': (0.0, 0.8)}


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_franka_fingers_follow():
    """On cuRobo's Franka (fingers locked, prismatic) the swapped model's spheres are a
    fresh build's at the measured opening."""
    from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState

    config = load_yaml(join_path(get_robot_configs_path(), 'franka.yml'))
    lock = config['robot_cfg']['kinematics']['lock_joints']
    names = list(lock)

    def build(values):
        robot = copy.deepcopy(config)
        robot['robot_cfg']['kinematics']['lock_joints'].update(values)
        return Kinematics(KinematicsCfg.from_robot_yaml_file(robot))

    model = build({})
    limits = {n: (0.0, 0.04) for n in names}
    follower = LockedJointFollower(LockedJointTransforms(
        dict(lock), lambda v: build(v).kinematics_config.fixed_transforms,
        model.kinematics_config.fixed_transforms, limits=limits), limits=limits)
    follower.observe(names, [0.01, 0.01])
    values, _ = follower.values()
    assert follower.apply(values, [model.kinematics_config]) == []

    q = torch.zeros(1, model.get_dof(), device='cuda')
    js = JointState.from_position(q, joint_names=model.joint_names)
    spheres = model.compute_kinematics(js).robot_spheres
    expected = build(values).compute_kinematics(js).robot_spheres
    assert (spheres - expected).abs().max() < 1e-5
    assert (spheres - build({}).compute_kinematics(js).robot_spheres).abs().max() > 0.01
