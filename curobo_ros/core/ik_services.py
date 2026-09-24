#!/usr/bin/env python3
"""
IK services for the unified planner node (v2).

Provides a lazy-initialized IK solver that shares the obstacle manager
and robot config of the trajectory planner.

Services exposed (prefixed with the node name):
  /<node>/warmup_ik  (WarmupIK) - init IK solver with given batch size (default 1)
  /<node>/ik         (Ik)       - single pose → joint state
  /<node>/ik_batch   (IkBatch)  - N poses → joint states

v2 notes:
- IKSolver / IKSolverConfig → InverseKinematics / InverseKinematicsCfg.
- CollisionCheckerType is gone; collision is part of InverseKinematicsCfg.create.
- load_from_robot_config(...) replaced by Cfg.create(robot=<yaml_path>, scene_model=<Scene>, ...).
"""

import torch
import std_msgs.msg
from sensor_msgs.msg import JointState

from curobo.types import Pose as CuroboPose, GoalToolPose, JointState as CuRoboJS
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg

from curobo_msgs.srv import Ik, IkBatch, WarmupIK


class IKServices:
    """
    Manages the IK solver and its ROS services on an existing node.

    Depends on:
    - config_wrapper.robot_config_file   (YAML path — v2 factories accept it directly)
    - config_wrapper.obstacle_manager    (Scene, shared with MotionPlanner)
    - config_wrapper.collision_cache     (single v2 cache integer)

    The solver is created only when warmup_ik is called.
    Obstacle updates are propagated via update_world().
    """

    def __init__(self, node, config_wrapper):
        self._node = node
        self._config = config_wrapper

        self._ik_solver: InverseKinematics | None = None
        self._ik_batch_size: int = 0  # 0 = not yet warmed up

        # Device / dtype resolved from config_wrapper (set by RobotModelManager).
        self._device = getattr(config_wrapper, '_device', torch.device('cuda'))
        self._dtype = getattr(config_wrapper, '_ops_dtype', torch.float32)

        name = node.get_name()
        node.create_service(WarmupIK, f'{name}/warmup_ik', self._warmup_ik_callback)
        node.create_service(Ik,       f'{name}/ik',        self._ik_callback)
        node.create_service(IkBatch,  f'{name}/ik_batch',  self._ik_batch_callback)

        node.get_logger().info("IKServices registered (not yet initialized - call warmup_ik)")

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup_ik_callback(self, request: WarmupIK.Request, response: WarmupIK.Response):
        batch_size = max(1, request.batch_size)
        try:
            self._init(batch_size)
            response.success = True
            response.message = f"IK solver ready (batch_size={batch_size})"
        except Exception as e:
            self._node.get_logger().error(f"IK warmup failed: {e}")
            response.success = False
            response.message = str(e)
        return response

    # ------------------------------------------------------------------
    # Services
    # ------------------------------------------------------------------

    def _ik_callback(self, request: Ik.Request, response: Ik.Response):
        if self._ik_solver is None:
            response.success = False
            response.error_msg.data = "IK not initialized. Call warmup_ik first."
            return response

        ok, result = self._solve([request.pose])
        if not ok:
            response.success = False
            response.error_msg.data = "IK solve failed"
            return response

        js = JointState()
        js.position = result.solution.cpu().numpy()[0][0].tolist()
        valid = std_msgs.msg.Bool()
        valid.data = bool(result.success.cpu().numpy()[0][0])
        response.joint_states = js
        response.joint_states_valid = valid
        response.success = True
        return response

    def _ik_batch_callback(self, request: IkBatch.Request, response: IkBatch.Response):
        if self._ik_solver is None:
            response.success = False
            response.error_msg.data = "IK not initialized. Call warmup_ik first."
            return response

        # tool_frames / return_seeds / start_pose all default to "unset": the
        # path below them is the one that existed before these fields did.
        if request.tool_frames or request.return_seeds > 0 or request.start_pose.position:
            return self._ik_batch_frames(request, response)

        ok, result = self._solve(request.poses)
        if not ok:
            response.success = False
            response.error_msg.data = "IK batch solve failed"
            return response

        for i, j in enumerate(result.solution.cpu().numpy()):
            js = JointState()
            js.position = j[0].tolist()
            valid = std_msgs.msg.Bool()
            valid.data = bool(result.success.cpu().numpy()[i][0])
            response.joint_states.append(js)
            response.joint_states_valid.append(valid)
        response.success = True
        return response

    def _ik_batch_frames(self, request: IkBatch.Request, response: IkBatch.Response):
        """IkBatch with tool_frames, return_seeds or start_pose set (see IkBatch.srv):
        one problem per group of len(tool_frames) poses, return_seeds solutions
        per problem (problem-major), seeded from start_pose when given."""
        def fail(msg):
            self._node.get_logger().error(f"IK batch: {msg}")
            response.success = False
            response.error_msg.data = msg
            return response

        kin = self._ik_solver.kinematics
        all_frames = list(kin.tool_frames)
        frames = list(request.tool_frames) or all_frames[:1]
        if len(set(frames)) != len(frames) or any(f not in all_frames for f in frames):
            return fail(f"tool_frames {frames} must name distinct tool frames of {all_frames}")
        if request.tool_frames and set(frames) != set(all_frames):
            # cuRobo solves every tool frame of the robot at once.
            return fail(f"tool_frames must list every tool frame of the robot: {all_frames}")
        n = len(frames)
        if not request.poses or len(request.poses) % n:
            return fail(f"{len(request.poses)} poses is not a positive multiple of {n} tool frames")
        problems = len(request.poses) // n
        if problems != self._ik_batch_size:
            try:
                self._init(problems)
            except Exception as e:
                self._ik_batch_size = 0
                return fail(f"reinit for batch_size={problems} failed: {e}")

        # v2 Pose quaternion is wxyz; ROS geometry_msgs is xyzw.
        poses = {}
        for k, frame in enumerate(frames):
            group = request.poses[k::n]
            poses[frame] = CuroboPose(
                position=torch.tensor(
                    [[p.position.x, p.position.y, p.position.z] for p in group],
                    dtype=self._dtype, device=self._device),
                quaternion=torch.tensor(
                    [[p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]
                     for p in group],
                    dtype=self._dtype, device=self._device),
            )
        if request.tool_frames:
            goal = GoalToolPose.from_poses(poses, ordered_tool_frames=all_frames, num_goalset=1)
        else:
            goal = GoalToolPose.from_poses(poses)

        current_state = None
        if request.start_pose.position:
            q = list(request.start_pose.position)
            names = list(kin.joint_names)
            if len(q) != len(names):
                return fail(f"start_pose has {len(q)} positions, the robot {len(names)} joints {names}")
            current_state = CuRoboJS.from_position(
                torch.tensor([q] * problems, dtype=self._dtype, device=self._device),
                joint_names=names)

        seeds = max(1, request.return_seeds)
        try:
            result = self._ik_solver.solve_pose(
                goal_tool_poses=goal, current_state=current_state, return_seeds=seeds)
        except Exception as e:
            return fail(f"solve failed: {e}")
        torch.cuda.synchronize()

        solutions = result.solution.reshape(problems, seeds, -1).cpu().numpy()
        valid = result.success.reshape(problems, seeds).cpu().numpy()
        for j in range(problems):
            for s in range(seeds):
                js = JointState()
                js.position = solutions[j, s].tolist()
                ok = std_msgs.msg.Bool()
                ok.data = bool(valid[j, s])
                response.joint_states.append(js)
                response.joint_states_valid.append(ok)
        response.success = True
        return response

    # ------------------------------------------------------------------
    # World update (called by the node when obstacles change)
    # ------------------------------------------------------------------

    def update_world(self):
        """Propagate obstacle changes to the IK solver. No-op if not initialized."""
        if self._ik_solver is None:
            return
        scene = self._config.obstacle_manager.get_scene()
        self._ik_solver.update_world(scene)
        self._node.get_logger().info("IKServices: world updated")

    def rebuild(self):
        """Recreate the IK solver after a collision-cache change.

        The collision cache is fixed at solver creation, so a cache change
        requires a rebuild (reusing the last batch size). No-op if the solver
        was never initialized.
        """
        if self._ik_solver is None:
            return
        self._init(max(1, self._ik_batch_size))
        self._node.get_logger().info("IKServices: solver rebuilt after cache change")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init(self, batch_size: int):
        """Create (or recreate) the IK solver for the given batch size."""
        # Primitives only at construction; update_world() pushes the perception
        # layer by copy afterwards. See primitives_only_scene().
        scene = self._config.obstacle_manager.primitives_only_scene()
        robot_yml = self._config.robot_config_file

        self._node.get_logger().info(f"Initializing IK solver (batch_size={batch_size})...")

        cfg = InverseKinematicsCfg.create(
            robot=robot_yml,
            scene_model=scene,
            num_seeds=20,
            position_tolerance=0.005,
            orientation_tolerance=0.05,
            self_collision_check=True,
            collision_cache=self._config.collision_cache,
            use_cuda_graph=False,
        )
        self._ik_solver = InverseKinematics(cfg)
        self._ik_batch_size = batch_size

        # Warmup: solve a batch of random configs to prime CUDA kernels.
        q_sample = self._ik_solver.sample_configs(batch_size)
        js = CuRoboJS.from_position(
            q_sample, joint_names=self._ik_solver.kinematics.joint_names
        )
        kin_state = self._ik_solver.compute_kinematics(js)
        goal = kin_state.tool_poses.as_goal()
        self._ik_solver.solve_pose(goal_tool_poses=goal)
        torch.cuda.synchronize()

        # A new solver starts from the robot YAML with every obstacle on: give it
        # the robot's spheres (links off, objects attached) and the obstacles off.
        if hasattr(self._node, 'apply_solver_states'):
            self._node.apply_solver_states()

        self._node.get_logger().info("IK solver ready")

    def _solve(self, poses):
        """
        Solve IK for a list of geometry_msgs/Pose.
        Reinitializes the solver if the batch size has changed.
        Returns (success: bool, result).
        """
        if not poses:
            self._node.get_logger().error("IK: empty pose list")
            return False, None

        n = len(poses)
        if n != self._ik_batch_size:
            try:
                self._init(n)
            except Exception as e:
                self._node.get_logger().error(f"IK reinit for batch_size={n} failed: {e}")
                self._ik_batch_size = 0
                return False, None

        # v2 Pose quaternion is wxyz; ROS geometry_msgs is xyzw.
        positions = [[p.position.x, p.position.y, p.position.z] for p in poses]
        orientations = [[p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]
                        for p in poses]

        pose2d = CuroboPose(
            position=torch.tensor(positions, dtype=self._dtype, device=self._device),
            quaternion=torch.tensor(orientations, dtype=self._dtype, device=self._device),
        )
        tool_frame = self._ik_solver.kinematics.tool_frames[0]
        goal = GoalToolPose.from_poses({tool_frame: pose2d})

        try:
            result = self._ik_solver.solve_pose(goal_tool_poses=goal)
        except Exception:
            try:
                self._init(n)
                result = self._ik_solver.solve_pose(goal_tool_poses=goal)
            except Exception as e:
                self._node.get_logger().error(f"IK solve failed: {e}")
                self._ik_batch_size = 0
                return False, None

        torch.cuda.synchronize()
        return True, result
