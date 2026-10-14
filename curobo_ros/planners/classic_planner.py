#!/usr/bin/env python3
"""
Classic trajectory planner (v2: MotionPlanner.plan_pose).

v2 notes:
- MotionGen → MotionPlanner (wired via SinglePlanner._shared_motion_planner).
- MotionGenPlanConfig is gone: per-call params are kwargs on plan_pose().
- Pose → ToolPose, wrapped in GoalToolPose.
- PoseCostMetric was removed upstream in v2. Trajectory-axis constraints
  (`trajectory_constraints`) are re-wired via `ToolPoseCriteria` (held along
  the whole path) — see SinglePlanner._apply_pose_constraints.
"""

import itertools
import time

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo.types import JointState, Pose, GoalToolPose

from . import hold, minimal_motion

# Per-frame IK combinations planned at most (several moving tool frames).
MAX_COMBINATIONS = 4
from .single_planner import SinglePlanner


class ClassicPlanner(SinglePlanner):
    """
    Classic motion generation planner.

    Uses cuRobo's MotionPlanner to generate a complete collision-free trajectory
    from start to goal, which is then executed in open-loop.

    - Takes a single target pose.
    - Generates a full trajectory in one shot via plan_pose().
    - Executes the trajectory as-is (no post-processing).
    """

    def get_planner_name(self) -> str:
        return "Classic Motion Generation"

    def _plan_trajectory(
        self,
        start_state: JointState,
        goal_request,
        config: dict,
    ):
        """
        Generate trajectory using MotionPlanner.plan_pose().

        Args:
            start_state: Initial joint configuration.
            goal_request: TrajectoryGeneration request (uses target_pose).
            config: Dict with keys:
                - max_attempts (default 1)
                - timeout (default 5.0)
                - time_dilation_factor (default 0.5)

        Returns:
            MotionPlannerResult-like object.
        """
        tool_frames = list(getattr(goal_request, 'target_tool_frames', None) or [])
        holding = list(getattr(goal_request, 'hold_tool_frames', None) or [])
        selection = self.node.get_parameter('pose_goal_selection').value
        if holding and not tool_frames:
            raise ValueError("hold_tool_frames needs target_tool_frames: the goals of the other frames")
        if holding or (tool_frames and selection == 'minimal_motion'):
            return self._plan_frames(start_state, goal_request, tool_frames, holding,
                                     config.get('max_attempts', 1), selection)
        if tool_frames:
            goal = self._goal_per_tool_frame(goal_request, tool_frames)
        else:
            # Extract goal pose from request. v2 Pose.from_list expects
            # [x, y, z, qw, qx, qy, qz] (quaternion wxyz).
            pose = Pose.from_list([
                goal_request.target_pose.position.x,
                goal_request.target_pose.position.y,
                goal_request.target_pose.position.z,
                goal_request.target_pose.orientation.w,
                goal_request.target_pose.orientation.x,
                goal_request.target_pose.orientation.y,
                goal_request.target_pose.orientation.z,
            ])
            tool_frame = self.motion_planner.tool_frames[0]
            goal = GoalToolPose.from_poses({tool_frame: pose})

        max_attempts = config.get('max_attempts', 1)
        if selection not in ('planner', 'minimal_motion'):
            raise ValueError(
                f"pose_goal_selection must be 'planner' or 'minimal_motion', not '{selection}'")

        self.node.get_logger().info(f"Planning with max_attempts={max_attempts}")

        # v2: Cartesian axis constraints via ToolPoseCriteria (held along the
        # whole path). Reset afterwards since the MotionPlanner is shared.
        applied = self._apply_pose_constraints(goal_request)
        try:
            # A joint-space plan cannot hold path constraints, and a multi-frame
            # goal is one IK problem per arm: both stay with plan_pose.
            if selection == 'minimal_motion' and not applied and not tool_frames:
                return self._plan_minimal_motion(goal, start_state, max_attempts)
            return self.motion_planner.plan_pose(
                goal,
                start_state,
                max_attempts=max_attempts,
            )
        finally:
            if applied:
                self._reset_pose_criteria()

    def _plan_minimal_motion(self, goal, start_state, max_attempts):
        """
        Plan in joint space to the IK solution nearest the start, clear of the limits.

        The joint-space plan runs on the joint-space MotionPlanner when the node
        has one, so this planner's CUDA graphs never see a cspace problem.
        Returns plan_cspace's result, or None.
        """
        mp = self.motion_planner
        solutions, distance, margin = minimal_motion.ik_solutions(mp, goal, start_state)
        if solutions.shape[0] == 0:
            self.node.get_logger().warn('minimal_motion: the IK found no solution')
            return None
        planner = getattr(self.node, 'joint_space_motion_planner', None) or mp
        result, k, tried = minimal_motion.plan_nearest(
            planner, solutions, distance, start_state, max_attempts)
        if k is not None:
            self.node.get_logger().info(
                f'minimal_motion: IK solution {tried} of {solutions.shape[0]}, '
                f'{float(distance[k]):.2f} rad from the start, '
                f'{float(margin[k]):.2f} rad from the nearest limit')
            return result
        self.node.get_logger().warn(
            f'minimal_motion: {tried} IK solution(s) clear of the limits, none planned')
        return result

    def _plan_frames(self, start_state, goal_request, moving, holding, max_attempts, selection):
        """
        A goal per moving tool frame; the frames in ``holding`` hold (planners/hold.py).

        minimal_motion: the IK solution of the whole goal nearest the start (holding
        joints at their start values), planned in joint space, nearest first. Then, or
        with 'planner', plan_pose with the holding frames weighted along the path. Every
        plan has the holding joints frozen and is re-checked. trajectory_constraints
        (a straight leg) only with one moving frame, the others holding.
        """
        if selection not in ('planner', 'minimal_motion'):
            raise ValueError(
                f"pose_goal_selection must be 'planner' or 'minimal_motion', not '{selection}'")
        mp = self.motion_planner
        if len(moving) != len(goal_request.target_poses):
            raise ValueError(
                f"target_tool_frames has {len(moving)} frames but target_poses "
                f"{len(goal_request.target_poses)} poses: one pose per frame")
        frozen = hold.split(mp, moving, holding)
        poses = {f: Pose.from_list([p.position.x, p.position.y, p.position.z, p.orientation.w,
                                    p.orientation.x, p.orientation.y, p.orientation.z])
                 for f, p in zip(moving, goal_request.target_poses)}
        poses.update(hold.current_poses(mp, start_state, holding))
        goal = hold.goal(mp, poses)
        criteria = {f: hold.held() for f in holding}
        constraints = list(getattr(goal_request, 'trajectory_constraints', []) or [])
        if any(c == 1 for c in constraints):
            if len(moving) != 1 or len(constraints) != 6:
                raise ValueError(
                    "trajectory_constraints need one moving tool frame, the others holding, "
                    "and 6 entries [theta_x, theta_y, theta_z, x, y, z]")
            tx, ty, tz, x, y, z = (1.0 if c == 1 else 0.0 for c in constraints)
            criteria[moving[0]] = ToolPoseCriteria(non_terminal_pose_axes_weight_factor=[x, y, z, tx, ty, tz])
        margin = float(self.node.get_parameter('self_collision_margin_m').value)
        started = time.monotonic()
        log = self.node.get_logger()
        result = None
        planner = getattr(self.node, 'joint_space_motion_planner', None) or mp
        if selection == 'minimal_motion' and len(moving) > 1:
            # Several moving frames: the nearest IK solution of each frame on its own,
            # combined. A joint IK of all frames finds far fewer solutions, often far away.
            for tried, (dist, q) in enumerate(self._per_frame_solutions(mp, poses, start_state, moving), 1):
                if tried > MAX_COMBINATIONS or (tried > 1 and not hold.budget_left(self.node, started)):
                    break
                result = hold.finish(mp, minimal_motion.plan_to(planner, q, start_state, max_attempts),
                                     start_state, frozen, margin)
                if minimal_motion.succeeded(result):
                    log.info(f'minimal_motion ({len(moving)} moving, {len(holding)} holding): per-frame IK '
                             f'combination {tried}, {dist:.2f} rad from the start')
                    return result
                log.warn(f'minimal_motion: per-frame IK combination {tried}: '
                         f'{getattr(result, "status", "no plan")}')
        if selection == 'minimal_motion' and moving[0] not in criteria:
            solutions, distance, limit = minimal_motion.ik_solutions(mp, goal, start_state, frozen)
            for tried, k in enumerate(minimal_motion.nearest_first(solutions, distance), 1):
                if tried > 1 and not hold.budget_left(self.node, started):
                    break
                result = hold.finish(mp, minimal_motion.plan_to(planner, solutions[k], start_state, max_attempts),
                                     start_state, frozen, margin)
                if minimal_motion.succeeded(result):
                    log.info(f'minimal_motion ({len(moving)} moving, {len(holding)} holding): IK solution '
                             f'{tried} of {solutions.shape[0]}, {float(distance[k]):.2f} rad from the start, '
                             f'{float(limit[k]):.2f} rad from the nearest limit')
                    return result
                log.warn(f'minimal_motion: IK solution {tried}: {getattr(result, "status", "no plan")}')
            if solutions.shape[0] == 0:
                log.warn('minimal_motion: the IK found no solution')
        for attempt in range(2):
            if (attempt or result is not None) and not hold.budget_left(self.node, started):
                break
            if criteria:
                mp.update_tool_pose_criteria(criteria)
            try:
                planned = mp.plan_pose(goal, start_state, max_attempts=max_attempts)
            finally:
                if criteria:
                    mp.update_tool_pose_criteria({f: ToolPoseCriteria() for f in criteria})
            result = hold.finish(mp, planned, start_state, frozen, margin)
            if minimal_motion.succeeded(result):
                return result
            log.warn(f'plan_pose ({len(moving)} moving, {len(holding)} holding), try {attempt + 1}: '
                     f'{getattr(result, "status", "no plan")}')
        return result

    @staticmethod
    def _per_frame_solutions(mp, poses, start_state, moving):
        """[(distance, q)] nearest first: the IK solutions of each moving frame alone (the
        other frames at their current pose, their joints at their start), up to
        minimal_motion.MAX_CANDIDATES per frame, combined."""
        joints = hold.frame_joints(mp)
        frames = list(mp.tool_frames)
        current = hold.current_poses(mp, start_state, frames)
        dof = mp.kinematics.get_dof()
        start = start_state.position.reshape(-1)[:dof]
        per_frame = []
        for f in moving:
            others = [g for g in frames if g != f]
            fixed = sorted({j for g in others for j in joints[g]})
            goal = hold.goal(mp, {**{g: current[g] for g in others}, f: poses[f]})
            solutions, distance, _ = minimal_motion.ik_solutions(mp, goal, start_state, fixed)
            kept = [(solutions[k], float(distance[k]))
                    for k in minimal_motion.nearest_first(solutions, distance)]
            if not kept:
                return []
            per_frame.append(kept)
        out = []
        for choice in itertools.product(*per_frame):
            q = start.clone()
            for f, (solution, _) in zip(moving, choice):
                q[joints[f]] = solution[joints[f]].to(q)
            out.append((sum(d * d for _, d in choice) ** 0.5, q))
        return sorted(out, key=lambda c: c[0])

    def _goal_per_tool_frame(self, goal_request, tool_frames):
        """One goal per tool frame (TrajectoryGeneration.target_tool_frames):
        target_poses[k] is the goal of tool_frames[k], all reached by one
        trajectory. cuRobo plans every tool frame of the robot at once, so all of
        them must be named; a frame that must stay put gets its current pose."""
        all_frames = list(self.motion_planner.tool_frames)
        if len(tool_frames) != len(goal_request.target_poses):
            raise ValueError(
                f"target_tool_frames has {len(tool_frames)} frames but target_poses "
                f"{len(goal_request.target_poses)} poses: one pose per frame")
        if len(set(tool_frames)) != len(tool_frames) or set(tool_frames) != set(all_frames):
            raise ValueError(
                f"target_tool_frames {tool_frames} must name every tool frame of the "
                f"robot once: {all_frames}")
        poses = {}
        for frame, p in zip(tool_frames, goal_request.target_poses):
            poses[frame] = Pose.from_list([
                p.position.x, p.position.y, p.position.z,
                p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z,
            ])
        return GoalToolPose.from_poses(poses, ordered_tool_frames=all_frames, num_goalset=1)

    # _process_trajectory(): default (no-op) from SinglePlanner is fine.
    # execute() / cancel(): inherited from SinglePlanner.
