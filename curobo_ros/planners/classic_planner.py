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

from curobo.types import JointState, Pose, GoalToolPose

from . import minimal_motion
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
        selection = self.node.get_parameter('pose_goal_selection').value
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
