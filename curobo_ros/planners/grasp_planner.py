#!/usr/bin/env python3
"""
cuRobo's grasp pipeline (``MotionPlanner.plan_grasp``) behind the ``plan_grasp`` action.

    goalset -> APPROACH (to the offset pose) -> GRASP (linear in) -> LIFT (linear out)

on the node's shared MotionPlanner. Restored from will-44's GraspPlanner
(removed in d253721) and adapted:

- legs carry the planned joints only, sampled at the interpolation dt and trimmed
  to their last real step (cuRobo pads them to its buffer length);
- every leg is checked before it is returned. cuRobo v0.8.0 can report success
  for a leg that does not reach its pose, or that leaves the straight line (the
  linear legs are a cost, not a constraint): a failed check re-plans, up to
  ``grasp_max_attempts`` times;
- ``time_dilation_factor`` slows the legs down;
- the attach goes through AttachmentServices (per link, every solver);
- with ``pose_goal_selection: minimal_motion`` the APPROACH leg is a joint-space
  plan to the approach IK solution nearest the start, over every candidate
  (cuRobo's goes wherever its random IK seeds land); GRASP and LIFT stay
  cuRobo's linear legs. When a leg fails or fails its check, the next nearest
  approach solution is tried;
- on a robot with more than one tool frame, ``grasp_tool_frames`` grasp together
  (candidate g of each) and ``hold_tool_frames`` hold (planners/hold.py): the same
  legs, the holding arms frozen in every leg and every leg re-checked.

Needs ``apply_curobo_patches`` (fix 4, ``core/curobo_patches.py``): without it
cuRobo v0.8.0's plan_grasp fails and leaves the IK unusable.
"""

import math
import time
import traceback
from types import SimpleNamespace

import torch
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.state.state_joint_trajectory_ops import get_joint_state_at_horizon_index
from curobo.types import GoalToolPose, JointState, Pose

from curobo_msgs.action import GraspPlan

from curobo_ros.core.time_dilation import check_factor, dilate

from . import hold, minimal_motion

LEGS = ("approach", "grasp", "lift")
# minimal_motion: approach solutions tried per attempt, nearest first.
GRASP_TRIES = 6


def _axis(name: str):
    axes = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
    if name not in axes:
        raise ValueError(f"axis must be x, y or z, got '{name}'")
    return axes[name]


def _angle(qa, qb) -> float:
    """Angle (rad) between two orientations given as w, x, y, z."""
    dot = abs(sum(a * b for a, b in zip(qa, qb)))
    return 2.0 * math.acos(min(1.0, dot))


class GraspPlanner:
    """Plans (and optionally executes) a grasp; checks every leg it returns."""

    def __init__(self, node, config_wrapper):
        self.node = node
        self.config_wrapper = config_wrapper

    @property
    def motion_planner(self):
        """The node's shared, warmed-up MotionPlanner."""
        return self.node.motion_planner

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(self, start_state: JointState, goal):
        """Plan with retries. Returns (legs, goal_index, message): ``legs`` maps
        each planned leg to (joint_names, positions [T, dof] tensor), or is None
        when no attempt passed the checks."""
        mp = self.motion_planner
        frames = list(mp.tool_frames)
        if len(frames) != 1:
            return self._plan_frames(start_state, goal)
        # Naming the only tool frame (a station built for several arms running one) is the
        # plain request; anything else needs a robot with more than one.
        if goal.hold_tool_frames or set(goal.grasp_tool_frames) - set(frames):
            raise ValueError(
                f"grasp_tool_frames and hold_tool_frames are for a robot with more than one "
                f"tool frame, this one has {frames}")
        n = len(goal.grasp_candidates)
        if n == 0:
            raise ValueError("grasp_candidates is empty (need >= 1 candidate grasp pose)")
        max_goalset = self.config_wrapper.max_goalset
        if n > 1 and n > max_goalset:
            raise ValueError(
                f"{n} grasp candidates but the planner was built with max_goalset "
                f"{max_goalset}: raise the max_goalset parameter")
        check_factor(goal.time_dilation_factor)

        attempts = max(1, int(self.node.get_parameter('grasp_max_attempts').value))
        nearest = self.node.get_parameter('pose_goal_selection').value == 'minimal_motion'
        message = ""
        for attempt in range(1, attempts + 1):
            if nearest:
                result = self._plan_minimal_motion(start_state, goal, frames[0])
            else:
                result = mp.plan_grasp(
                    grasp_poses=self._goalset(goal.grasp_candidates, frames[0]),
                    current_state=start_state,
                    grasp_approach_axis=goal.approach_axis or "z",
                    grasp_approach_offset=goal.approach_offset,
                    grasp_approach_in_tool_frame=goal.approach_in_tool_frame,
                    grasp_lift_axis=goal.lift_axis or "z",
                    grasp_lift_offset=goal.lift_offset,
                    grasp_lift_in_tool_frame=goal.lift_in_tool_frame,
                    plan_approach_to_grasp=goal.plan_approach_to_grasp,
                    plan_grasp_to_lift=goal.plan_grasp_to_lift,
                )
            if not self._flag(result.success):
                message = f"attempt {attempt}: {result.status}"
                self.node.get_logger().warn(f"Grasp {message}")
                continue
            goal_index = int(result.goalset_index.view(-1)[0].item())
            legs = self._legs(result, goal)
            problem = self._check(legs, start_state, goal, goal_index, frames[0])
            if problem is None:
                return legs, goal_index, f"Grasp planned (attempt {attempt})"
            message = f"attempt {attempt}: {problem}"
            self.node.get_logger().warn(f"Grasp leg check failed, {message}")
        return None, -1, f"Grasp planning failed after {attempts} attempt(s), last {message}"

    def _plan_minimal_motion(self, start_state, goal, frame):
        """
        APPROACH: joint-space plan to an approach IK solution, nearest the start
        first, over every candidate. GRASP and LIFT: cuRobo's linear legs from there
        (its plan_grasp steps 3 and 4). A solution whose legs fail to plan or fail the
        checks gives way to the next one. Returns the plan_grasp result fields
        ``_legs`` reads.
        """
        mp = self.motion_planner
        out = SimpleNamespace(success=torch.tensor([False]), status="", goalset_index=None)
        found = []
        for i in range(len(goal.grasp_candidates)):
            p, q = self._targets(goal, i, frame)["approach"]
            approach = GoalToolPose.from_poses({frame: Pose.from_list(p + q)})
            solutions, distance, margin = minimal_motion.ik_solutions(mp, approach, start_state)
            found.append((solutions, distance, margin, [i] * solutions.shape[0]))
        solutions, distance, margin = (torch.cat([f[k] for f in found]) for k in range(3))
        candidate = [i for f in found for i in f[3]]
        if solutions.shape[0] == 0:
            out.status = "minimal_motion: the IK found no approach solution"
            return out
        planner = getattr(self.node, 'joint_space_motion_planner', None) or mp
        max_attempts = max(1, int(self.node.get_parameter('max_attempts').value))
        failures = []
        for tried, k in enumerate(minimal_motion.nearest_first(solutions, distance, GRASP_TRIES), 1):
            goal_index = candidate[k]
            where = f"grasp candidate {goal_index}, approach IK solution {tried} of {solutions.shape[0]}"
            legs = self._plan_legs(planner, solutions[k], start_state, goal, goal_index, frame,
                                   max_attempts)
            problem = legs if isinstance(legs, str) else self._check(
                self._legs(legs, goal), start_state, goal, goal_index, frame)
            if problem is not None:
                failures.append(f"{tried}: {problem}")
                self.node.get_logger().warn(f"minimal_motion: {where}: {problem}, trying the next one")
                continue
            self.node.get_logger().info(
                f"minimal_motion: {where}, {float(distance[k]):.2f} rad from the start, "
                f"{float(margin[k]):.2f} rad from the nearest limit")
            legs.success = torch.tensor([True])
            legs.goalset_index = torch.tensor([goal_index])
            return legs
        out.status = ("minimal_motion: no approach solution planned ("
                      + ("; ".join(failures) or "none clear of the limits") + ")")
        return out

    def _plan_legs(self, planner, q, start_state, goal, goal_index, frame, max_attempts):
        """APPROACH to ``q``, then the linear legs. The plan_grasp result fields, or
        what failed."""
        approach = minimal_motion.plan_to(planner, q, start_state, max_attempts)
        if not minimal_motion.succeeded(approach):
            return "approach leg failed (no joint-space plan)"
        out = SimpleNamespace(success=torch.tensor([False]), status="", goalset_index=None)
        self._keep(out, "approach", approach)
        targets = self._targets(goal, goal_index, frame)
        legs = [("grasp", goal.approach_axis or "z", goal.approach_in_tool_frame)]
        if goal.plan_grasp_to_lift:
            legs.append(("lift", goal.lift_axis or "z", goal.lift_in_tool_frame))
        if not goal.plan_approach_to_grasp:
            legs = []
        last = approach
        for leg, axis, in_tool_frame in legs:
            p, q_target = targets[leg]
            result = self._linear(frame, Pose.from_list(p + q_target), axis, in_tool_frame,
                                  self._last(last))
            if not minimal_motion.succeeded(result):
                return f"{leg} leg failed ({self._why(result)})"
            self._keep(out, leg, result)
            last = result
        return out

    # ------------------------------------------------------------------
    # Several tool frames: grasp_tool_frames grasp, hold_tool_frames hold
    # ------------------------------------------------------------------

    def _plan_frames(self, start_state, goal):
        """plan() on a robot with more than one tool frame. Same return as plan()."""
        mp = self.motion_planner
        grasping, holding = list(goal.grasp_tool_frames), list(goal.hold_tool_frames)
        if not grasping:
            raise ValueError(
                f"a robot with tool frames {list(mp.tool_frames)} needs grasp_tool_frames "
                f"(and hold_tool_frames for the others)")
        if goal.attach_object_name:
            raise ValueError(
                "attach_object_name attaches to the default link: on a robot with several tool "
                "frames attach with attach_object and its attach_link")
        frozen = hold.split(mp, grasping, holding, 'grasp_tool_frames')
        n = len(goal.grasp_candidates)
        if n == 0 or n % len(grasping):
            raise ValueError(
                f"{n} grasp candidates for {len(grasping)} grasp frame(s): one block of G "
                f"candidates per frame")
        g_count = n // len(grasping)
        if g_count > 1 and g_count > self.config_wrapper.max_goalset:
            raise ValueError(
                f"{g_count} grasp candidates per frame but the planner was built with "
                f"max_goalset {self.config_wrapper.max_goalset}: raise the max_goalset parameter")
        check_factor(goal.time_dilation_factor)
        cand = {f: goal.grasp_candidates[k * g_count:(k + 1) * g_count] for k, f in enumerate(grasping)}
        attempts = max(1, int(self.node.get_parameter('grasp_max_attempts').value))
        nearest = self.node.get_parameter('pose_goal_selection').value == 'minimal_motion'
        started = time.monotonic()
        message = ""
        for attempt in range(1, attempts + 1):
            if attempt > 1 and not hold.budget_left(self.node, started):
                message += " (planning_time_budget spent)"
                break
            if nearest or holding:
                result = self._plan_frames_composed(start_state, goal, cand, g_count, grasping,
                                                    holding, frozen, started)
            else:
                result = mp.plan_grasp(
                    grasp_poses=self._goalset_frames(cand, grasping, g_count),
                    current_state=start_state,
                    grasp_approach_axis=goal.approach_axis or "z",
                    grasp_approach_offset=goal.approach_offset,
                    grasp_approach_in_tool_frame=goal.approach_in_tool_frame,
                    grasp_lift_axis=goal.lift_axis or "z",
                    grasp_lift_offset=goal.lift_offset,
                    grasp_lift_in_tool_frame=goal.lift_in_tool_frame,
                    plan_approach_to_grasp=goal.plan_approach_to_grasp,
                    plan_grasp_to_lift=goal.plan_grasp_to_lift,
                )
            if not self._flag(result.success):
                message = f"attempt {attempt}: {result.status}"
                self.node.get_logger().warn(f"Grasp {message}")
                continue
            goal_index = int(result.goalset_index.view(-1)[0].item())
            legs = self._legs(result, goal)
            problem = self._check_frames(legs, start_state, goal, goal_index, cand, grasping)
            if problem is None:
                return legs, goal_index, f"Grasp planned (attempt {attempt})"
            message = f"attempt {attempt}: {problem}"
            self.node.get_logger().warn(f"Grasp leg check failed, {message}")
        return None, -1, f"Grasp planning failed after {attempts} attempt(s), last {message}"

    def _check_frames(self, legs, start_state, goal, goal_index, cand, grasping):
        """_check per grasping frame, then the arm clearance (arm_clearance_m) over all legs."""
        for f in grasping:
            problem = self._check(legs, start_state, goal, goal_index, f, cand[f][goal_index])
            if problem is not None:
                return f"{f}: {problem}"
        clearance = hold.arm_clearance(self.node, self.motion_planner)
        if clearance > 0.0 and legs:
            q = torch.cat([q for leg in LEGS if leg in legs for _, q, _ in [legs[leg]]])
            return hold.clearance_problem(self.motion_planner, q, clearance)
        return None

    def _goalset_frames(self, cand, grasping, g_count) -> GoalToolPose:
        """Candidates of every grasping frame -> GoalToolPose [B=1, H=1, L, G, 3/4]."""
        device = self.config_wrapper._device
        dtype = self.config_wrapper._ops_dtype
        position = torch.tensor(
            [[[p.position.x, p.position.y, p.position.z] for p in cand[f]] for f in grasping],
            device=device, dtype=dtype)
        quaternion = torch.tensor(
            [[[p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z] for p in cand[f]]
             for f in grasping], device=device, dtype=dtype)
        return GoalToolPose(
            tool_frames=list(grasping),
            position=position.view(1, 1, len(grasping), g_count, 3),
            quaternion=quaternion.view(1, 1, len(grasping), g_count, 4),
        )

    def _plan_frames_composed(self, start_state, goal, cand, g_count, grasping, holding, frozen,
                              started):
        """
        APPROACH: joint-space plan to an IK solution of every grasping frame's approach
        pose of candidate g (the holding frames at their current pose, their joints at
        their start), nearest the start first, over every g. GRASP and LIFT: linear legs
        on the grasping frames, the holding ones weighted along the path. The holding
        joints are frozen and every leg re-checked (planners/hold.py).
        """
        mp = self.motion_planner
        out = SimpleNamespace(success=torch.tensor([False]), status="", goalset_index=None)
        held_poses = hold.current_poses(mp, start_state, holding)
        found = []
        for g in range(g_count):
            poses = {f: Pose.from_list(sum(self._targets(goal, g, f, cand[f][g])["approach"], []))
                     for f in grasping}
            poses.update(held_poses)
            solutions, distance, margin = minimal_motion.ik_solutions(
                mp, hold.goal(mp, poses), start_state, frozen)
            found.append((solutions, distance, margin, [g] * solutions.shape[0]))
        solutions, distance, margin = (torch.cat([f[k] for f in found]) for k in range(3))
        candidate = [g for f in found for g in f[3]]
        if solutions.shape[0] == 0:
            out.status = "minimal_motion: the IK found no approach solution"
            return out
        planner = getattr(self.node, 'joint_space_motion_planner', None) or mp
        max_attempts = max(1, int(self.node.get_parameter('max_attempts').value))
        failures = []
        for tried, k in enumerate(minimal_motion.nearest_first(solutions, distance, GRASP_TRIES), 1):
            if tried > 1 and not hold.budget_left(self.node, started):
                failures.append("planning_time_budget spent")
                break
            g = candidate[k]
            where = f"grasp candidate {g}, approach IK solution {tried} of {solutions.shape[0]}"
            legs = self._plan_legs_frames(planner, solutions[k], start_state, goal, g, cand, grasping,
                                          holding, held_poses, frozen, max_attempts)
            problem = legs if isinstance(legs, str) else self._check_frames(
                self._legs(legs, goal), start_state, goal, g, cand, grasping)
            if problem is not None:
                failures.append(f"{tried}: {problem}")
                self.node.get_logger().warn(f"minimal_motion: {where}: {problem}, trying the next one")
                continue
            self.node.get_logger().info(
                f"minimal_motion: {where} ({len(grasping)} grasping, {len(holding)} holding), "
                f"{float(distance[k]):.2f} rad from the start, "
                f"{float(margin[k]):.2f} rad from the nearest limit")
            legs.success = torch.tensor([True])
            legs.goalset_index = torch.tensor([g])
            return legs
        out.status = ("minimal_motion: no approach solution planned ("
                      + ("; ".join(failures) or "none clear of the limits") + ")")
        return out

    def _plan_legs_frames(self, planner, q, start_state, goal, g, cand, grasping, holding,
                          held_poses, frozen, max_attempts):
        """APPROACH to ``q``, then the linear legs, each frozen and re-checked. The
        plan_grasp result fields, or what failed."""
        mp = self.motion_planner
        approach = hold.finish(mp, minimal_motion.plan_to(planner, q, start_state, max_attempts),
                               start_state, frozen)
        if not minimal_motion.succeeded(approach):
            return f"approach leg failed ({getattr(approach, 'status', 'no joint-space plan')})"
        out = SimpleNamespace(success=torch.tensor([False]), status="", goalset_index=None)
        self._keep_plan(out, "approach", approach)
        legs = [("grasp", goal.approach_axis or "z", goal.approach_in_tool_frame)]
        if goal.plan_grasp_to_lift:
            legs.append(("lift", goal.lift_axis or "z", goal.lift_in_tool_frame))
        if not goal.plan_approach_to_grasp:
            legs = []
        last = approach
        for leg, axis, in_tool_frame in legs:
            poses = {f: Pose.from_list(sum(self._targets(goal, g, f, cand[f][g])[leg], []))
                     for f in grasping}
            start = self._end(last)
            result = self._linear_frames(poses, held_poses, axis, in_tool_frame, start, frozen)
            if not minimal_motion.succeeded(result):
                return f"{leg} leg failed ({getattr(result, 'status', None) or self._why(result)})"
            self._keep_plan(out, leg, result)
            last = result
        return out

    def _linear_frames(self, poses, held_poses, axis, in_tool_frame, start, frozen):
        """Linear legs of the grasping frames in ``poses``, the frames in ``held_poses``
        weighted along the path; the holding joints frozen and the leg re-checked, with
        the grasp contact links out of collision as for the planning."""
        mp = self.motion_planner
        links = mp.kinematics.config.kinematics_config.grasp_contact_link_names or []
        weight = float(self.node.get_parameter('grasp_line_weight').value)
        criteria = {f: ToolPoseCriteria.linear_motion(
            axis=axis, non_terminal_scale=weight, project_distance_to_goal=in_tool_frame) for f in poses}
        criteria.update({f: hold.held() for f in held_poses})
        mp.update_tool_pose_criteria(criteria)
        mp.disable_link_collision(links)
        try:
            planned = mp.plan_pose(hold.goal(mp, {**poses, **held_poses}), start)
            return hold.finish(mp, planned, start, frozen)
        finally:
            mp.enable_link_collision(links)
            mp.update_tool_pose_criteria({f: ToolPoseCriteria() for f in criteria})

    def _end(self, result) -> JointState:
        """The last waypoint of a (frozen) plan, on the planned joints."""
        kin = self.motion_planner.kinematics
        plan = result.get_interpolated_plan()
        names = list(plan.joint_names) if plan.joint_names is not None else list(kin.joint_names)
        last = plan.position.reshape(-1, plan.position.shape[-1])[-1:]
        columns = [names.index(n) for n in kin.joint_names]
        return JointState.from_position(last[:, columns].clone(), joint_names=list(kin.joint_names))

    @staticmethod
    def _keep_plan(out, leg, result) -> None:
        """A leg as its (frozen, trimmed) interpolated plan."""
        setattr(out, f"{leg}_interpolated_trajectory", result.get_interpolated_plan())
        setattr(out, f"{leg}_interpolated_last_tstep", None)

    @staticmethod
    def _why(result) -> str:
        """What a failed cuRobo result tells: its smallest pose errors."""
        if result is None:
            return "no result"
        parts = []
        for name, scale, unit in (("position_error", 1000.0, " mm"), ("rotation_error", 1.0, "")):
            value = getattr(result, name, None)
            if value is not None and value.numel():
                parts.append(f"{name.split('_')[0]} error {float(value.min()) * scale:.1f}{unit}")
        return ", ".join(parts) or "no success"

    def _linear(self, frame, pose, axis, in_tool_frame, start):
        """A linear leg as cuRobo's plan_grasp plans it: plan_pose with the
        linear-motion criteria, the grasp contact links out of collision."""
        mp = self.motion_planner
        links = mp.kinematics.config.kinematics_config.grasp_contact_link_names or []
        weight = float(self.node.get_parameter('grasp_line_weight').value)
        mp.update_tool_pose_criteria({frame: ToolPoseCriteria.linear_motion(
            axis=axis, non_terminal_scale=weight, project_distance_to_goal=in_tool_frame)})
        mp.disable_link_collision(links)
        try:
            return mp.plan_pose(GoalToolPose.from_poses(
                {frame: pose}, ordered_tool_frames=[frame], num_goalset=1), start)
        finally:
            mp.enable_link_collision(links)
            mp.update_tool_pose_criteria({frame: ToolPoseCriteria()})

    def _last(self, result) -> JointState:
        """The last planned state of a leg, on the active joints."""
        end = get_joint_state_at_horizon_index(result.js_solution, -1).squeeze(0)
        return self.motion_planner.kinematics.get_active_js(end)

    @staticmethod
    def _keep(out, leg, result) -> None:
        setattr(out, f"{leg}_interpolated_trajectory", result.interpolated_trajectory)
        setattr(out, f"{leg}_interpolated_last_tstep", result.interpolated_last_tstep)

    def _goalset(self, poses, frame) -> GoalToolPose:
        """geometry_msgs/Pose[] -> GoalToolPose [B=1, H=1, L=1, G, 3/4] on ``frame``."""
        device = self.config_wrapper._device
        dtype = self.config_wrapper._ops_dtype
        position = torch.tensor(
            [[p.position.x, p.position.y, p.position.z] for p in poses],
            device=device, dtype=dtype)
        quaternion = torch.tensor(
            [[p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z] for p in poses],
            device=device, dtype=dtype)
        n = len(poses)
        return GoalToolPose(
            tool_frames=[frame],
            position=position.view(1, 1, 1, n, 3),
            quaternion=quaternion.view(1, 1, 1, n, 4),
        )

    def _legs(self, result, goal):
        """The planned legs, trimmed, on the planned joints only: leg ->
        (joint_names, positions [T, dof], velocities [T, dof])."""
        wanted = {"approach": True, "grasp": goal.plan_approach_to_grasp,
                  "lift": goal.plan_approach_to_grasp and goal.plan_grasp_to_lift}
        kin = self.motion_planner.kinematics
        dof = kin.get_dof()
        legs = {}
        for leg in LEGS:
            js = getattr(result, f"{leg}_interpolated_trajectory", None)
            if not wanted[leg] or js is None:
                continue
            last = getattr(result, f"{leg}_interpolated_last_tstep", None)
            end = None if last is None else int(torch.as_tensor(last).view(-1)[0].item()) + 1

            def rows(t):
                return None if t is None else t.reshape(-1, t.shape[-1])[:end]

            names = list(js.joint_names) if js.joint_names is not None else list(kin.joint_names)
            active = kin.get_active_js(JointState(
                position=rows(js.position), velocity=rows(js.velocity), joint_names=names))
            vel = active.velocity
            legs[leg] = (list(active.joint_names), active.position.reshape(-1, dof),
                         None if vel is None else vel.reshape(-1, dof))
        return legs

    def _targets(self, goal, goal_index, frame, candidate=None):
        """Where each leg must end: (position, quaternion wxyz) per leg. ``candidate``:
        the grasp pose of ``frame`` when it is not grasp_candidates[goal_index]."""
        p = candidate if candidate is not None else goal.grasp_candidates[goal_index]
        grasp = Pose.from_list([p.position.x, p.position.y, p.position.z,
                                p.orientation.w, p.orientation.x, p.orientation.y,
                                p.orientation.z])

        def offset_pose(axis, offset, in_tool_frame):
            off = Pose.from_list([offset * v for v in _axis(axis)] + [1.0, 0.0, 0.0, 0.0])
            return grasp.multiply(off) if in_tool_frame else off.multiply(grasp)

        poses = {
            "approach": offset_pose(goal.approach_axis or "z", goal.approach_offset,
                                    goal.approach_in_tool_frame),
            "grasp": grasp,
            "lift": offset_pose(goal.lift_axis or "z", goal.lift_offset, goal.lift_in_tool_frame),
        }
        return {k: (v.position.view(-1).tolist(), v.quaternion.view(-1).tolist())
                for k, v in poses.items()}

    def _tcp(self, q, frame):
        """Tool-frame positions and quaternions (wxyz) along joint positions [T, dof]."""
        kin = self.motion_planner.kinematics
        poses = kin.get_link_poses(q.contiguous(), [frame])
        return (poses.position.reshape(-1, 3).tolist(),
                poses.quaternion.reshape(-1, 4).tolist())

    def _check(self, legs, start_state, goal, goal_index, frame, candidate=None):
        """None if every leg ends on its pose (and the linear ones stay on their
        line); otherwise what is wrong."""
        pos_tol = float(self.node.get_parameter('grasp_position_tolerance').value)
        rot_tol = float(self.node.get_parameter('grasp_orientation_tolerance').value)
        line_tol = float(self.node.get_parameter('grasp_line_tolerance').value)
        targets = self._targets(goal, goal_index, frame, candidate)
        previous_end = start_state.position.reshape(-1)[: self.motion_planner.kinematics.get_dof()]
        off_line = 0.0
        for leg in LEGS:
            if leg not in legs:
                continue
            _, q, _ = legs[leg]
            if float((q[0] - previous_end.to(q.device, q.dtype)).abs().max()) > 1e-3:
                return f"{leg} leg does not start where the previous one ends"
            previous_end = q[-1]
            positions, quaternions = self._tcp(q, frame)
            goal_p, goal_q = targets[leg]
            end_err = math.dist(positions[-1], goal_p)
            if end_err > pos_tol:
                return f"{leg} leg ends {end_err * 1000:.1f} mm from its pose"
            if _angle(quaternions[-1], goal_q) > rot_tol:
                return f"{leg} leg ends {math.degrees(_angle(quaternions[-1], goal_q)):.1f} deg off"
            if leg == "approach":
                continue
            # Linear legs: distance from the segment, orientation held.
            a = positions[0]
            axis = [g - s for g, s in zip(goal_p, a)]
            length = math.sqrt(sum(v * v for v in axis)) or 1.0
            u = [v / length for v in axis]
            lateral = beyond = 0.0
            for p, quat in zip(positions, quaternions):
                rel = [x - s for x, s in zip(p, a)]
                along = sum(r * k for r, k in zip(rel, u))
                lateral = max(lateral, math.sqrt(max(0.0, sum(r * r for r in rel) - along * along)))
                beyond = max(beyond, -along, along - length)
                if _angle(quat, goal_q) > rot_tol:
                    return (f"{leg} leg turns the tool "
                            f"{math.degrees(_angle(quat, goal_q)):.1f} deg on the way")
            off_line = max(off_line, lateral, beyond)
            if lateral > line_tol or beyond > line_tol:
                previous = targets[LEGS[LEGS.index(leg) - 1]][0]
                return (f"{leg} leg leaves its line by up to "
                        f"{max(lateral, beyond) * 1000:.1f} mm "
                        f"({'sideways' if lateral >= beyond else 'past its ends'}; it starts "
                        f"{math.dist(a, previous) * 1000:.1f} mm from its start pose)")
        self.node.get_logger().info(
            f"{frame}: linear legs at most {off_line * 1000:.1f} mm off their line")
        return None

    # ------------------------------------------------------------------
    # Result message
    # ------------------------------------------------------------------

    def fill_result(self, result_msg, legs, goal_index, goal):
        """Legs into the action result, sampled at the (dilated) interpolation dt."""
        from sensor_msgs.msg import JointState as JointStateMsg
        dt = self.interpolation_dt()
        factor = goal.time_dilation_factor
        out_dt = dt
        for leg in LEGS:
            msgs = []
            if leg in legs:
                names, q, v = legs[leg]
                pos = q.detach().cpu().tolist()
                vel = v.detach().cpu().tolist() if v is not None else None
                for i, row in enumerate(pos):
                    m = JointStateMsg()
                    m.name = names
                    m.position = row
                    if vel is not None:
                        m.velocity = vel[i]
                    msgs.append(m)
                out_dt = dilate(msgs, dt, factor)
            setattr(result_msg, f"{leg}_trajectory", msgs)
            setattr(result_msg, f"{leg}_success", leg in legs)
        result_msg.dt = out_dt
        result_msg.goal_index = goal_index

    def interpolation_dt(self) -> float:
        dt = getattr(getattr(self.motion_planner.trajopt_solver, 'config', None),
                     'interpolation_dt', None)
        if dt is None:
            dt = self.node.get_parameter('interpolation_dt').value
        return float(dt)

    # ------------------------------------------------------------------
    # Execution (stream approach -> grasp -> [attach] -> lift)
    # ------------------------------------------------------------------

    def execute(self, robot_context, legs, goal, goal_handle=None) -> bool:
        """Stream the legs to the robot; attach the object after the grasp leg."""
        try:
            n_legs = len(legs)
            for done, leg in enumerate(l for l in LEGS if l in legs):
                names, q, v = legs[leg]
                if not self._stream(robot_context, names, q, v, goal_handle, leg.upper(),
                                    done, n_legs):
                    return False
                if leg == "grasp" and goal.attach_object_name:
                    self._publish(goal_handle, "ATTACH", (done + 1) / max(n_legs, 1))
                    self.node.attachment_services.attach(
                        goal.attach_object_name,
                        JointState.from_position(q[-1:].clone(), joint_names=names))
            self._publish(goal_handle, "DONE", 1.0)
            return True
        except Exception as e:
            self.node.get_logger().error(f"Grasp execution error: {e}")
            self.node.get_logger().error(traceback.format_exc())
            robot_context.stop_robot()
            return False

    def _stream(self, robot_context, names, q, v, goal_handle, phase, leg_index, n_legs) -> bool:
        pos = q.detach().cpu().tolist()
        zeros = [[0.0] * len(pos[0]) for _ in pos]
        vel = v.detach().cpu().tolist() if v is not None else zeros
        robot_context.set_command(names, vel, zeros, pos)
        robot_context.send_trajectrory()
        while robot_context.get_progression() < 1.0:
            if goal_handle is not None and not goal_handle.is_active:
                robot_context.stop_robot()
                return False
            self._publish(goal_handle, phase,
                          (leg_index + robot_context.get_progression()) / max(n_legs, 1))
            time.sleep(0.01)
        time.sleep(0.05)  # let the robot settle on the last waypoint
        return True

    @staticmethod
    def _publish(goal_handle, phase, progression) -> None:
        if goal_handle is None:
            return
        fb = GraspPlan.Feedback()
        fb.phase = phase
        fb.progression = float(min(max(progression, 0.0), 1.0))
        goal_handle.publish_feedback(fb)

    @staticmethod
    def _flag(value) -> bool:
        if value is None:
            return False
        return bool(value.view(-1)[0].item()) if hasattr(value, 'view') else bool(value)
