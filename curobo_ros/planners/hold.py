#!/usr/bin/env python3
"""
Tool frames that hold while the others move (``hold_tool_frames``), on a robot with
more than one tool frame, e.g. one arm still while the other grasps.

cuRobo plans every joint of the robot at once, and its pose criteria alone let a
holding arm drift by radians along the path. So a holding frame gets:

1. its current pose as goal, weighted along the whole path (``held()``);
2. its arm's start joints copied into every waypoint, with zero velocity,
   acceleration and jerk (``freeze``);
3. a check of the whole trajectory with the planner's own model, attached objects
   included (``recheck``): the moving arms planned around a holding arm that may
   have drifted, and must not touch it where it really stays.

A trajectory that fails the check counts as a failed attempt.
"""

import time
from types import SimpleNamespace

import torch

from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.types.device_cfg import DeviceCfg
from curobo.types import GoalToolPose, JointState, Pose

# A joint moved this much (rad or m) moves a tool frame that depends on it.
PROBE = 0.2
# Contact added over the first waypoint (m) that fails the check: numerical noise of
# the sphere models stays well below it.
TOLERANCE = 1e-3


def held() -> ToolPoseCriteria:
    """Pose criteria of a holding frame: its pose weighted along the whole path."""
    return ToolPoseCriteria(non_terminal_pose_axes_weight_factor=[1.0] * 6)


def frame_joints(mp) -> dict:
    """{tool frame: indices of the planned joints that move it}, cached on ``mp``."""
    cached = getattr(mp, '_curobo_ros_frame_joints', None)
    if cached is not None:
        return cached
    kin = mp.kinematics
    dof = kin.get_dof()
    frames = list(mp.tool_frames)
    q = kin.default_joint_state.position.reshape(-1)[:dof].clone().repeat(dof + 1, 1)
    q[1:] += PROBE * torch.eye(dof, device=q.device, dtype=q.dtype)
    poses = kin.compute_kinematics(JointState.from_position(q, joint_names=kin.joint_names)).tool_poses
    pos = poses.position.reshape(dof + 1, len(frames), 3)
    quat = poses.quaternion.reshape(dof + 1, len(frames), 4)
    moved = ((pos[1:] - pos[:1]).norm(dim=-1) > 1e-6) | ((quat[1:] - quat[:1]).norm(dim=-1) > 1e-6)
    out = {f: [j for j in range(dof) if bool(moved[j, k])] for k, f in enumerate(frames)}
    mp._curobo_ros_frame_joints = out
    return out


def split(mp, moving, holding, field="target_tool_frames"):
    """
    Checks a request's frames and returns the planned joints that hold.

    ``moving`` (the request's ``field``) and ``holding`` must name every tool frame of
    the robot once. A joint that moves both a holding and a moving frame cannot hold.
    """
    frames = list(mp.tool_frames)
    named = list(moving) + list(holding)
    if len(set(named)) != len(named) or set(named) != set(frames):
        raise ValueError(
            f"{field} {list(moving)} and hold_tool_frames {list(holding)} must name "
            f"every tool frame of the robot once: {frames}")
    joints = frame_joints(mp)
    hold = sorted({j for f in holding for j in joints[f]})
    shared = sorted(set(hold) & {j for f in moving for j in joints[f]})
    if shared:
        names = [mp.kinematics.joint_names[j] for j in shared]
        raise ValueError(f"joints {names} move a holding and a moving tool frame")
    return hold


def current_poses(mp, start_state, frames) -> dict:
    """{frame: Pose} of ``frames`` at the start configuration."""
    kin = mp.kinematics
    q = start_state.position.reshape(1, -1)[:, :kin.get_dof()]
    tool = mp.compute_kinematics(JointState.from_position(q.clone(), joint_names=kin.joint_names)).tool_poses
    all_frames = list(mp.tool_frames)
    pos = tool.position.reshape(len(all_frames), 3)
    quat = tool.quaternion.reshape(len(all_frames), 4)
    return {f: Pose(position=pos[all_frames.index(f)].view(1, 3).clone(),
                    quaternion=quat[all_frames.index(f)].view(1, 4).clone()) for f in frames}


def goal(mp, poses: dict) -> GoalToolPose:
    """GoalToolPose over every tool frame of the robot from {frame: Pose}."""
    return GoalToolPose.from_poses(poses, ordered_tool_frames=list(mp.tool_frames), num_goalset=1)


def freeze(trajectory: JointState, mp, start_state, hold) -> JointState:
    """``trajectory`` with the joints ``hold`` (planned-joint indices) at their start
    value in every waypoint, their velocity, acceleration and jerk zero."""
    kin = mp.kinematics
    start = start_state.position.reshape(-1)
    names = list(trajectory.joint_names) if trajectory.joint_names is not None else list(kin.joint_names)
    out = trajectory.clone()
    for j in hold:
        c = names.index(kin.joint_names[j])
        out.position[..., c] = start[j].to(out.position)
        for field in ('velocity', 'acceleration', 'jerk'):
            value = getattr(out, field, None)
            if value is not None:
                value[..., c] = 0.0
    return out


def _self_cost(mp) -> SelfCollisionCost:
    cost = getattr(mp, '_curobo_ros_self_cost', None)
    if cost is None:
        cost = SelfCollisionCost(SelfCollisionCostCfg(
            weight=1.0, self_collision_kin_config=mp.kinematics.get_self_collision_config()))
        mp._curobo_ros_self_cost = cost
    return cost


def recheck(mp, q, margin=0.0):
    """
    None when the joint trajectory ``q`` [T, dof] adds no contact over its first
    waypoint, else what it hits: self collision (the robot's collision pairs, the
    spheres grown by ``margin``) and the world, with what the robot holds attached.
    """
    kin = mp.kinematics
    q = q.reshape(-1, kin.get_dof()).contiguous()
    state = kin.compute_kinematics(JointState.from_position(q, joint_names=kin.joint_names))
    spheres = state.robot_spheres.reshape(1, q.shape[0], -1, 4).clone()
    if margin > 0.0:
        live = spheres[..., 3] > 0
        spheres[..., 3] = torch.where(live, spheres[..., 3] + margin / 2.0, spheres[..., 3])
    cost = _self_cost(mp)
    cost.setup_batch_tensors(1, q.shape[0])
    self_depth = cost.forward(spheres).reshape(-1)
    added = self_depth - self_depth[0].clamp(min=0.0)
    if float(added.max()) > TOLERANCE:
        t = int(added.argmax())
        return (f"self collision {float(self_depth[t]) * 1000:.1f} mm at waypoint {t} "
                f"of {q.shape[0]}")
    checker = getattr(mp, 'scene_collision_checker', None)
    if checker is None or not hasattr(checker, 'get_sphere_distance'):
        return None
    device_cfg = DeviceCfg(device=q.device, dtype=q.dtype)
    world = checker.get_sphere_distance(
        state, CollisionBuffer.from_shape(state.robot_spheres.shape, device_cfg),
        device_cfg.to_device([1.0]), activation_distance=device_cfg.to_device([0.0]))
    world = world.reshape(q.shape[0], -1)
    added = (world - world[:1]).clamp(min=0.0).max(dim=-1).values
    if float(added.max()) > TOLERANCE:
        t = int(added.argmax())
        return (f"world contact {float(world[t].max()) * 1000:.1f} mm at waypoint {t} "
                f"of {q.shape[0]}")
    return None


class Held:
    """A successful plan result with the holding joints frozen in its interpolated
    trajectory; every other attribute is the original result's."""

    def __init__(self, result, trajectory):
        self._result = result
        self._trajectory = trajectory
        self.success = torch.tensor([True])

    def get_interpolated_plan(self):
        return self._trajectory

    def __getattr__(self, name):
        return getattr(self._result, name)


def failed(status):
    return SimpleNamespace(success=torch.tensor([False]), status=status)


def finish(mp, result, start_state, hold, margin=0.0):
    """
    The plan ``result`` with the joints ``hold`` frozen and re-checked: a ``Held``
    result, the failed ``result`` as it is, or a failed result saying why.
    """
    if result is None or not bool(result.success.view(-1)[0]):
        return result
    if not hold:
        return result
    trajectory = freeze(result.get_interpolated_plan(), mp, start_state, hold)
    kin = mp.kinematics
    names = list(trajectory.joint_names) if trajectory.joint_names is not None else list(kin.joint_names)
    columns = [names.index(n) for n in kin.joint_names]
    q = trajectory.position.reshape(-1, trajectory.position.shape[-1])[:, columns]
    problem = recheck(mp, q, margin)
    if problem is not None:
        return failed(f"holding arm re-check: {problem}")
    return Held(result, trajectory)


def end_state(mp, result) -> JointState:
    """The last waypoint of a plan's interpolated trajectory, on the planned joints."""
    kin = mp.kinematics
    plan = result.get_interpolated_plan()
    names = list(plan.joint_names) if plan.joint_names is not None else list(kin.joint_names)
    last = plan.position.reshape(-1, plan.position.shape[-1])[-1:]
    columns = [names.index(n) for n in kin.joint_names]
    return JointState.from_position(last[:, columns].clone(), joint_names=list(kin.joint_names))


def joined(results):
    """One successful result whose interpolated trajectory is ``results``' trajectories one
    after the other (each starts where the previous one ends: its first waypoint is dropped)."""
    plans = [r.get_interpolated_plan() for r in results]
    out = plans[0].clone()
    for field in ('position', 'velocity', 'acceleration', 'jerk'):
        parts = [getattr(p, field, None) for p in plans]
        if any(x is None for x in parts):
            continue
        setattr(out, field, torch.cat([parts[0]] + [x[..., 1:, :] for x in parts[1:]], dim=-2))
    return Held(results[0], out)


def budget_left(node, started):
    """False once ``planning_time_budget`` (s, 0 = none) has passed since ``started``
    (time.monotonic())."""
    import time
    budget = float(node.get_parameter('planning_time_budget').value) if node.has_parameter(
        'planning_time_budget') else 0.0
    return budget <= 0.0 or time.monotonic() - started < budget
