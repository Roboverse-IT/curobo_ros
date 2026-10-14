#!/usr/bin/env python3
"""
Minimal joint motion: plan in joint space to the IK solution nearest the start.

cuRobo's IK seeds are random, so the solution behind a pose plan can sit several
radians away on another arm branch. Used by the classic planner (pose goals) and
the grasp planner (approach pose) when ``pose_goal_selection`` is ``minimal_motion``.
"""

import math

import torch

from curobo.types import JointState

# A goal this close to a joint limit makes the next joint-space plan fail
# (cuRobo v0.8.0), and the IK hands such goals out freely.
LIMIT_MARGIN = 0.05
# IK seeds scattered this much around the start find the solutions near it;
# cuRobo's own seeds are random and land several radians away.
SEED_NOISE = 0.6
# IK solutions tried, nearest first; two closer than this count as one.
MAX_CANDIDATES = 3
SAME_SOLUTION = 0.05


def ik_solutions(mp, goal, start_state, hold=()):
    """
    IK solutions of ``goal``: (solutions [N, dof], distance [N], margin [N]).

    The IK runs twice with its configured number of seeds (its own random ones,
    then seeds around the start), so its CUDA graph keeps its shape. ``distance``
    is from the start, inf for a solution within LIMIT_MARGIN of a limit. The
    joints ``hold`` (planned-joint indices of the arms that hold, whose goal is
    their current pose) keep their start values.
    """
    dof = mp.kinematics.get_dof()
    start = start_state.position.view(1, -1)[:, :dof]
    low, high = (limit.view(-1)[:dof].to(start)
                 for limit in mp.kinematics.get_joint_limits().position)
    n = mp.ik_solver.config.num_seeds
    noise = torch.randn(
        (n, dof), generator=torch.Generator(device=start.device).manual_seed(0),
        device=start.device, dtype=start.dtype) * SEED_NOISE
    noise[0] = 0.0
    near = torch.maximum(torch.minimum(start + noise, high), low).view(1, n, dof)
    found = []
    for seeds in (None, near):
        ik = mp.ik_solver.solve_pose(
            goal, current_state=start_state, seed_config=seeds, return_seeds=n)
        solved = ik.solution.reshape(-1, ik.solution.shape[-1])[ik.success.view(-1)]
        found.append(solved[:, :dof])
    solutions = torch.cat(found)
    if len(hold):
        solutions[:, list(hold)] = start[:, list(hold)]
    # A joint with more than a full turn of range: the same angle on the turn
    # nearest the start, when that turn is inside the limits.
    turns = torch.round((start - solutions) / (2 * math.pi))
    turns[:, (high - low) < 2 * math.pi + 0.1] = 0.0
    turned = solutions + 2 * math.pi * turns
    inside = (turned >= low + LIMIT_MARGIN) & (turned <= high - LIMIT_MARGIN)
    solutions = torch.where(inside, turned, solutions)
    margin = torch.minimum(solutions - low, high - solutions).min(dim=-1).values
    distance = (solutions - start).norm(dim=-1)
    distance[margin < LIMIT_MARGIN] = float('inf')
    return solutions, distance, margin


def nearest_first(solutions, distance, limit=MAX_CANDIDATES):
    """Indices of the solutions, nearest first: finite distance only, two closer than
    SAME_SOLUTION count as one, at most ``limit``."""
    kept = []
    for k in torch.argsort(distance).tolist():
        if len(kept) == limit or not torch.isfinite(distance[k]):
            return
        if any(float((solutions[k] - solutions[t]).abs().max()) < SAME_SOLUTION for t in kept):
            continue
        kept.append(k)
        yield k


def plan_to(planner, q, start_state, max_attempts):
    """``plan_cspace`` from the start to the joint configuration ``q``."""
    return planner.plan_cspace(
        JointState.from_position(q.view(1, -1).clone()), start_state, max_attempts=max_attempts)


def succeeded(result):
    return result is not None and bool(result.success.any())


def plan_nearest(planner, solutions, distance, start_state, max_attempts):
    """
    ``plan_cspace`` to the solutions nearest first, up to MAX_CANDIDATES distinct ones.

    Returns (result, k, tried): ``k`` is the index of the planned solution, None
    when none planned; ``result`` is the last plan_cspace result, or None.
    """
    result, tried = None, 0
    for tried, k in enumerate(nearest_first(solutions, distance), 1):
        result = plan_to(planner, solutions[k], start_state, max_attempts)
        if succeeded(result):
            return result, k, tried
    return result, None, tried
