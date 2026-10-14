#!/usr/bin/env python3
"""
Joint space trajectory planner (v2).

v2 notes:
- MotionGen.plan_single_js() → MotionPlanner.plan_cspace() (joint goal).
- MotionGenPlanConfig is gone; per-call params are kwargs on plan_cspace().
- v2 plan_cspace() no longer accepts timeout/time_dilation_factor/
  enable_graph/enable_opt — those tunables live on the trajopt YAML.
"""

import time

import torch
from curobo.types import JointState

from . import hold
from .single_planner import SinglePlanner

# A holding arm's target joint may differ from its start by this much (rad or m).
SAME_JOINT = 1e-4


class JointSpacePlanner(SinglePlanner):
    """
    Joint-space planner using MotionPlanner.plan_joint_state() (v2).

    Plans directly in joint space — no IK, no singularities to worry about.
    Ideal when the goal is already expressed as a joint configuration.
    """

    def get_planner_name(self) -> str:
        return "Joint Space Motion Generation"

    @property
    def motion_planner(self):
        """The node's joint-space MotionPlanner when it has one
        (joint_space_own_planner), else the shared one."""
        own = getattr(self.node, 'joint_space_motion_planner', None)
        return own if own is not None else self._shared_motion_planner

    def _plan_trajectory(
        self,
        start_state: JointState,
        goal_request,
        config: dict,
    ):
        if not hasattr(goal_request, 'target_joint_positions'):
            raise ValueError(
                "JointSpacePlanner requires 'target_joint_positions' in the request."
            )

        goal_joint_positions = goal_request.target_joint_positions
        if not goal_joint_positions:
            raise ValueError("target_joint_positions is empty.")

        robot_dof = self.motion_planner.kinematics.get_dof()
        if len(goal_joint_positions) != robot_dof:
            raise ValueError(
                f"Joint count mismatch: received {len(goal_joint_positions)} joints, "
                f"but robot has {robot_dof} DOF"
            )
        if any(not (-1e6 < x < 1e6) or x != x for x in goal_joint_positions):
            raise ValueError(
                f"Invalid joint positions (NaN/Inf): {goal_joint_positions}"
            )

        goal_state = JointState.from_position(
            torch.tensor(
                [goal_joint_positions],
                dtype=start_state.position.dtype,
                device=start_state.position.device,
            )
        )

        max_attempts = config.get('max_attempts', 1)
        enable_graph_attempt = config.get('enable_graph_attempt', 1)

        start_pos = start_state.position[0].cpu().tolist()
        goal_pos = list(goal_joint_positions)
        self.node.get_logger().info("Planning joint space trajectory:")
        self.node.get_logger().info(f"  Start: {[f'{x:.3f}' for x in start_pos]}")
        self.node.get_logger().info(f"  Goal:  {[f'{x:.3f}' for x in goal_pos]}")
        self.node.get_logger().info(
            f"  Config: max_attempts={max_attempts}, "
            f"enable_graph_attempt={enable_graph_attempt}"
        )

        holding = list(getattr(goal_request, 'hold_tool_frames', None) or [])
        if not holding:
            started = time.monotonic()
            result = self.motion_planner.plan_cspace(
                goal_state,
                start_state,
                max_attempts=max_attempts,
                enable_graph_attempt=enable_graph_attempt,
            )
            if len(self.motion_planner.tool_frames) > 1 and not (
                    result is not None and bool(result.success.view(-1)[0])) and hold.budget_left(
                    self.node, started):
                return self._via_default(goal_state, start_state, max_attempts, enable_graph_attempt, result)
            return result
        # Arms that hold (planners/hold.py): target joints equal to the start ones,
        # frozen along the plan, then the plan re-checked.
        mp = self.motion_planner
        moving = [f for f in mp.tool_frames if f not in holding]
        frozen = hold.split(mp, moving, holding)
        start = start_state.position.reshape(-1)
        moved = [mp.kinematics.joint_names[j] for j in frozen
                 if abs(goal_joint_positions[j] - float(start[j])) > SAME_JOINT]
        if moved:
            raise ValueError(f"hold_tool_frames {holding}: the target moves their joints {moved}")
        margin = float(self.node.get_parameter('self_collision_margin_m').value)
        started = time.monotonic()
        result = None
        for attempt in range(2):
            if attempt and not hold.budget_left(self.node, started):
                break
            result = hold.finish(mp, mp.plan_cspace(goal_state, start_state, max_attempts=max_attempts,
                                                    enable_graph_attempt=enable_graph_attempt),
                                 start_state, frozen, margin)
            if result is not None and bool(result.success.view(-1)[0]):
                return result
            self.node.get_logger().warn(
                f"joint-space plan, {len(holding)} frame(s) holding, try {attempt + 1}: "
                f"{getattr(result, 'status', 'no plan')}")
        return result

    def _via_default(self, goal_state, start_state, max_attempts, enable_graph_attempt, failed):
        """
        Several tool frames: a long joint-space move fails from some configurations
        (cuRobo v0.8.0, measured on a dual-arm AMR: ~10-20% of the returns from both
        chests). Through the robot's default configuration it plans in most of them:
        two plans, one trajectory. Returns ``failed`` when either leg fails.
        """
        mp = self.motion_planner
        kin = mp.kinematics
        via = JointState.from_position(
            kin.default_joint_state.position.reshape(1, -1)[:, :kin.get_dof()].clone().to(
                start_state.position),
            joint_names=list(kin.joint_names))
        first = mp.plan_cspace(via, start_state, max_attempts=max_attempts,
                               enable_graph_attempt=enable_graph_attempt)
        if first is None or not bool(first.success.view(-1)[0]):
            self.node.get_logger().warn("joint-space plan failed, and so did the leg to the default configuration")
            return failed
        second = mp.plan_cspace(goal_state, hold.end_state(mp, first), max_attempts=max_attempts,
                                enable_graph_attempt=enable_graph_attempt)
        if second is None or not bool(second.success.view(-1)[0]):
            self.node.get_logger().warn("joint-space plan failed, and so did the leg from the default configuration")
            return failed
        self.node.get_logger().info("joint-space plan through the default configuration (direct plan failed)")
        return hold.joined([first, second])
