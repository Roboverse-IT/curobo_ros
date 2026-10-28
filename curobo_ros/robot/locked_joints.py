"""The joints a cuRobo model locks, set to measured values at run time."""

import copy

from curobo.config_io import load_yaml
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState, Pose
import torch


class LockedJoints:
    """
    The locked joints of a cuRobo model (``lock_joints``, e.g. a gripper's), settable.

    cuRobo bakes a locked joint, and the joints that mimic it, into the fixed
    transforms of their links at the value in the robot config: the model keeps
    e.g. the gripper at that opening whatever the real one does. update()
    recomputes those transforms for new values, as cuRobo bakes them (forward
    kinematics on the same robot without the locks), and writes them into the
    model in place; copy_to() writes them into another model of the robot (a
    solver's), in place too, so a solver replaying a captured CUDA graph sees
    them.
    """

    def __init__(self, kin_model, robot_config):
        """
        Track the locked joints of `kin_model`, built from `robot_config`.

        robot_config is the cuRobo robot config (path or dict) the model was
        built from. A model without locked joints has empty `names`.
        """
        self._config = kin_model.kinematics_config
        cfg = copy.deepcopy(robot_config) if isinstance(robot_config, dict) \
            else load_yaml(robot_config)
        kinematics = cfg.get('robot_cfg', cfg)['kinematics']
        # cuRobo's lock_jointstate also holds cspace joints outside the chain,
        # which move no link: only the configured locks are baked in.
        lock = self._config.lock_jointstate
        locked = kinematics.get('lock_joints') or {}
        self.names = [n for n in lock.joint_names if n in locked] if lock is not None else []
        position = lock.position.reshape(-1).tolist() if lock is not None else []
        self.values = [position[lock.joint_names.index(n)] for n in self.names]
        if not self.names:
            return

        kinematics['lock_joints'] = {}
        # Every link's pose, in the model's link order.
        kinematics['tool_frames'] = list(self._config.link_name_to_idx_map)
        self._free = Kinematics(KinematicsCfg.from_robot_yaml_file(cfg))
        free_config = self._free.kinematics_config
        if free_config.link_name_to_idx_map != self._config.link_name_to_idx_map:
            raise ValueError('the robot config does not give the same links as the model')
        missing = [n for n in self.names if n not in self._free.joint_names]
        if missing:
            raise ValueError(f'the locked joints {missing} are not in the robot config cspace')

        self._dofs = [self._free.joint_names.index(n) for n in self.names]
        # The links each locked joint moves: its own and those of its mimics.
        child = [i for i, j in enumerate(free_config.joint_map.tolist()) if j in self._dofs]
        device = self._config.fixed_transforms.device
        self._child = torch.tensor(child, dtype=torch.long, device=device)
        self._parent = free_config.link_map.to(device=device, dtype=torch.long)[self._child]

    def update(self, values):
        """
        Set the locked joints to `values`, in `names` order.

        Return whether the values changed (no work when they did not).
        """
        values = [float(v) for v in values]
        if len(values) != len(self.names):
            raise ValueError(f'expected {len(self.names)} values for {self.names}, '
                             f'got {len(values)}')
        if values == self.values:
            return False
        fixed = self._config.fixed_transforms
        q = torch.zeros((1, len(self._free.joint_names)), dtype=fixed.dtype, device=fixed.device)
        q[0, self._dofs] = torch.tensor(values, dtype=fixed.dtype, device=fixed.device)
        poses = self._free.compute_kinematics(
            JointState(position=q, joint_names=self._free.joint_names)).tool_poses
        position, quaternion = poses.position.reshape(-1, 3), poses.quaternion.reshape(-1, 4)
        parent = Pose(position[self._parent], quaternion[self._parent])
        child = Pose(position[self._child], quaternion[self._child])
        fixed[self._child] = parent.inverse().multiply(child).get_affine_matrix().to(fixed.dtype)
        self.values = values
        return True

    def copy_to(self, kinematics_config):
        """
        Write the locked joints' transforms into `kinematics_config`, in place.

        kinematics_config is that of another model of the same robot, with the
        same locks (e.g. a solver's). Only the rows of the links the locked
        joints move are written: the other links, and the collision spheres,
        are left as they are.
        """
        if not self.names or kinematics_config is self._config:
            return
        src, dst = self._config.fixed_transforms, kinematics_config.fixed_transforms
        if (kinematics_config.link_name_to_idx_map != self._config.link_name_to_idx_map
                or dst.shape != src.shape):
            raise ValueError('the model has other links than the one the locked joints are of')
        dst[self._child] = src[self._child].to(dst.dtype)
