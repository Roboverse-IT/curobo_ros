#!/usr/bin/env python3
"""
Attach/detach services for the unified planner node.

Attaches a scene obstacle to the arm's ``attached_object`` link as collision
spheres (fitted by ``sphere_fit`` via cuRobo's attachment_manager) so it moves
with the arm for subsequent open-loop transport planning, until ``detach``
releases it. The robot YAML must declare the ``attached_object`` link (see
docs/tutorials/03-collision-objects.md).

Standalone feature: independent of any particular planning pipeline. Useful
for pre-positioned objects, simulation, and tests — the caller supplies the
scene obstacle name and the joint state to fit spheres at (usually the
robot's current pose).

Registers its own services (`<node>/attach_object`, `<node>/detach_object`,
`<node>/detach_link`), following the same self-registering pattern as
``IKServices`` / ``FKServices``.

A robot with one attach link per arm (dual-arm) names the link in the request
(``AttachObject.attach_link``) and releases it with ``detach_link``; left empty,
everything works on ``attached_object`` as before.

The fit is made on the MotionPlanner, then carried to every other solver (IK
services, MPC, retargeter), each of which has its own copy of the robot's
spheres: see the node's apply_robot_sphere_states / apply_obstacle_states.
"""

import os

import torch
from std_srvs.srv import Trigger
from curobo.scene import Capsule, Cuboid, Cylinder, Mesh, Sphere
from curobo.types import JointState
from curobo.sphere_fit import SphereFitType, estimate_sphere_count

from curobo_msgs.srv import AttachObject, RemoveObject

ATTACH_LINK = "attached_object"

# AttachObject.sphere_fit_type -> cuRobo's fit; FIT_DEFAULT (0) passes none, so
# the attachment manager keeps its own default as before.
_SPHERE_FIT = {
    AttachObject.Request.FIT_SURFACE: SphereFitType.SURFACE,
    AttachObject.Request.FIT_VOXEL: SphereFitType.VOXEL,
    AttachObject.Request.FIT_MORPHIT: SphereFitType.MORPHIT,
}


class AttachmentServices:
    """Attach/detach a scene obstacle to the robot's ``attached_object`` link."""

    def __init__(self, node, config_wrapper):
        self.node = node
        self.config_wrapper = config_wrapper
        self._attached = {}  # attach link -> name of the obstacle attached to it

        name = node.get_name()
        self.attach_object_srv = node.create_service(
            AttachObject, f'{name}/attach_object', self._attach_object_callback)
        self.detach_object_srv = node.create_service(
            Trigger, f'{name}/detach_object', self._detach_object_callback)
        # Per link, for robots with more than one attach link. RemoveObject is
        # reused as is: its `name` is the link to release.
        self.detach_link_srv = node.create_service(
            RemoveObject, f'{name}/detach_link', self._detach_link_callback)

    @property
    def attached_name(self):
        """Name of the obstacle attached to ``attached_object``, or None."""
        return self._attached.get(ATTACH_LINK)

    @property
    def attached_names(self):
        """Names of every attached obstacle, whatever the link."""
        return set(self._attached.values())

    @property
    def motion_planner(self):
        """The node's shared MotionPlanner (may be None before warmup)."""
        return self.node.motion_planner

    def kinematics(self):
        """The MotionPlanner's kinematics (carries attached-object spheres), or
        None if the planner isn't ready. Used by ros_service_manager to render
        the fitted attached-object spheres in the collision-sphere viz.
        """
        try:
            return self._attachment_manager()._kinematics
        except Exception:
            return None

    def _attachment_manager(self):
        """cuRobo AttachmentManager for the shared MotionPlanner.

        In this build ``MotionPlanner.attachment_manager`` forwards to
        ``trajopt_solver.attachment_manager``, but TrajOptSolver composes
        SolverCore and doesn't re-expose it — reach into ``.core`` instead.
        """
        mp = self.motion_planner
        am = getattr(mp, 'attachment_manager', None)  # property may raise -> None
        if am is None:
            am = mp.trajopt_solver.core.attachment_manager
        return am

    def _has_attach_link(self, link: str = ATTACH_LINK) -> bool:
        """Whether the loaded robot YAML declares the attach link."""
        mp = self.motion_planner
        if mp is None:
            return False
        kc = mp.kinematics.kinematics_config
        return link in kc.link_name_to_idx_map

    # ------------------------------------------------------------------
    # Attach / detach
    # ------------------------------------------------------------------

    def attach(self, object_name: str, grasp_end_state: JointState,
               link: str = ATTACH_LINK, fit_type=None, num_spheres: int = 0,
               obstacle=None) -> None:
        """Fit + attach the named scene obstacle to the arm at the given config.

        The obstacle is fetched from the ObstacleManager's authoritative Scene
        (so it works even if the solver's loaded scene_model is out of sync) and
        passed directly to ``attach`` (not ``attach_from_scene``, which looks up
        by name in the solver's scene_model). ``disable_obstacle_names`` also
        disables the world obstacle so it isn't double-counted with the attached
        spheres. Affects the shared MotionPlanner only (open-loop transport).

        Sphere-fitting is a GPU operation — held under node.gpu_lock so it can't
        overlap a concurrent CUDA-graph capture or depth-camera integrate.

        ``link`` is the attach link (one per arm on a dual-arm robot);
        ``fit_type`` a cuRobo SphereFitType, None for the manager's default;
        ``num_spheres`` > 0 asks for that many spheres (capped at the link's
        budget), 0 keeps the estimate below. An obstacle read from USD is fitted
        from its geometry as read, not from the copy padded for the SDF kernel.
        ``obstacle`` given: that geometry (a payload, not a scene obstacle) is
        attached under ``object_name``, and there is no world copy to switch off.

        The spheres are fitted in the world and moved into the attach link's
        frame here, at ``grasp_end_state``. cuRobo v0.8.0's attach writes
        world-frame spheres as link-local ones when no pose offset is given (the
        payload lands about a metre from the hand), and its offset path measures
        from the first tool frame, whatever the attach link.
        """
        if not self._has_attach_link(link):
            raise ValueError(
                f"Robot YAML does not declare the '{link}' link — "
                "attach is unavailable for this robot (see "
                "docs/tutorials/03-collision-objects.md)."
            )
        obstacle_manager = self.config_wrapper.obstacle_manager
        in_scene = obstacle is None
        if in_scene:
            scene = obstacle_manager.get_scene()
            obstacle = (getattr(obstacle_manager, 'unpadded_meshes', {}).get(object_name)
                        or self._find_obstacle(scene, object_name))
            if obstacle is None:
                raise ValueError(f"Obstacle '{object_name}' not found in the scene")
        with self.node.gpu_lock:
            am = self._attachment_manager()
            # cuRobo's automatic fit (num_spheres=None) sizes itself to the
            # obstacle's geometry, not to what the robot YAML allocated for
            # ATTACH_LINK, and aborts the attach when it overruns. Capping the
            # fit at the slot count makes attach always succeed -- but a cap
            # that truncates silently hands back a collision model far coarser
            # than the payload's real shape, and the caller is about to plan
            # motions against exactly that model. So cap, and say so.
            n_slots = am.kinematics_params.get_sphere_index_from_link_name(
                link).shape[0]
            n_needed = num_spheres if num_spheres > 0 else self._estimate_sphere_need(obstacle)
            if n_needed > n_slots:
                self.node.get_logger().warn(
                    f"Attached object '{object_name}' needs about {n_needed} "
                    f"collision spheres but link '{link}' allocates only "
                    f"{n_slots}: its collision model is a coarse approximation "
                    f"that may not cover corners or protrusions. Raise "
                    f"'extra_collision_spheres.{link}' in the robot YAML "
                    f"before planning with this payload on real hardware."
                )
            # Only cap when the estimate exceeds the budget; below it, let the
            # estimate stand so a small object isn't padded to the full slot
            # count. n_needed == 0 means the estimate failed -> fall back.
            n_fit = min(n_needed, n_slots) if n_needed > 0 else n_slots
            fit = {} if fit_type is None else {'sphere_fit_type': fit_type}
            spheres = am.fit_spheres([obstacle], num_spheres=n_fit, **fit)
            am.update(self._to_link_frame(am._kinematics, grasp_end_state, link, spheres),
                      grasp_end_state, link_name=link)
            if in_scene and object_name not in obstacle_manager.obstacle_names:
                # e.g. from the world file: apply_obstacle_states does not know it
                self._set_world_copy(object_name, False)
            # The fit lands on the planner's spheres: copy it to the node's
            # robot model, from which every solver takes its spheres.
            self.config_wrapper.kin_model.kinematics_config.update_link_spheres(
                link, am.kinematics_params.get_link_spheres(link))
            self._attached[link] = object_name
            self._apply_to_solvers(object_name)
        self.node.get_logger().info(
            f"Attached '{object_name}' to link '{link}'")

    @staticmethod
    def _link_pose(kin, q, link):
        """World pose (3x3 rotation, translation) of ``link`` at configuration
        ``q`` [1, dof]: the nearest tool frame up its chain, then the fixed
        transforms down to it. Raises if a joint moves in between."""
        kc = kin.kinematics_config
        names = {i: n for n, i in kc.link_name_to_idx_map.items()}
        tools = set(kin.tool_frames)
        idx, name, chain = kc.link_name_to_idx_map[link], link, []
        while name not in tools:
            if int(kc.joint_map[idx]) != -1:
                raise ValueError(
                    f"'{link}' is not fixed to a tool frame {sorted(tools)}: "
                    "its pose is not known here")
            chain.append(kc.fixed_transforms[idx])
            idx = int(kc.link_map[idx])
            if idx < 0 or idx not in names:
                raise ValueError(f"'{link}' has no tool frame above it")
            name = names[idx]
        pose = kin.get_link_poses(q, [name])
        w, x, y, z = pose.quaternion.reshape(-1)[:4].tolist()
        rot = torch.tensor([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], device=q.device, dtype=q.dtype)
        trans = pose.position.reshape(-1)[:3].to(q.dtype)
        for fixed in reversed(chain):   # from the tool frame down to the link
            fixed = fixed.to(q.dtype)
            trans = trans + rot @ fixed[:3, 3]
            rot = rot @ fixed[:3, :3]
        return rot, trans

    def _to_link_frame(self, kin, joint_state, link, spheres):
        """World-frame spheres [n, 4] into ``link``'s frame at ``joint_state``."""
        q = joint_state.position.reshape(1, -1)[:, : kin.get_dof()]
        rot, trans = self._link_pose(kin, q, link)
        local = spheres.clone()
        local[:, :3] = (spheres[:, :3].to(rot.dtype) - trans) @ rot
        return local

    def _set_world_copy(self, name, enabled) -> None:
        """Switch an obstacle the node does not track by name on or off in every solver."""
        for label, checker in self.node._scene_collision_checkers():
            try:
                checker.enable_obstacle(name, enabled)
            except Exception as e:
                self.node.get_logger().error(f"obstacle '{name}' in {label}: {e}")

    def payload(self, request, joint_state):
        """The geometry an AttachObject request describes (object_name empty),
        posed in the world: ``pose`` is in ``header.frame_id`` ('' -> the robot
        base), a robot link fixed to a tool frame."""
        p = request.pose
        local = [p.position.x, p.position.y, p.position.z,
                 p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]
        frame = request.header.frame_id
        kin = self._attachment_manager()._kinematics
        if frame:
            q = joint_state.position.reshape(1, -1)[:, : kin.get_dof()]
            rot, trans = self._link_pose(kin, q, frame)
            pos = trans + rot @ torch.tensor(local[:3], device=rot.device, dtype=rot.dtype)
            frame_q = self._rot_to_quat(rot)
            pose = pos.tolist() + self._quat_mul(frame_q, local[3:])
        else:
            pose = local
        name = f"payload@{request.attach_link or ATTACH_LINK}"
        d = request.dimensions
        if request.type == request.CUBOID:
            return name, Cuboid(name=name, pose=pose, dims=[d.x, d.y, d.z])
        if request.type == request.SPHERE:
            return name, Sphere(name=name, pose=pose, radius=d.x)
        if request.type == request.CAPSULE:
            return name, Capsule(name=name, pose=pose, base=[0, 0, 0], tip=[0, 0, d.y], radius=d.x)
        if request.type == request.CYLINDER:
            return name, Cylinder(name=name, pose=pose, radius=d.x, height=d.y)
        if request.type == request.MESH:
            if not os.path.isabs(request.mesh_file_path) or not os.path.exists(request.mesh_file_path):
                raise ValueError(f"mesh_file_path must be an existing absolute path: "
                                 f"'{request.mesh_file_path}'")
            return name, Mesh(name=name, pose=pose, file_path=request.mesh_file_path,
                              scale=[d.x, d.y, d.z])
        raise ValueError(f"unknown payload type {request.type}")

    @staticmethod
    def _quat_mul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]

    @staticmethod
    def _rot_to_quat(r):
        """3x3 rotation -> quaternion [w, x, y, z]."""
        m = r.tolist()
        tr = m[0][0] + m[1][1] + m[2][2]
        if tr > 0:
            s = (tr + 1.0) ** 0.5 * 2
            return [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
        if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
            s = (1.0 + m[0][0] - m[1][1] - m[2][2]) ** 0.5 * 2
            return [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]
        if m[1][1] > m[2][2]:
            s = (1.0 + m[1][1] - m[0][0] - m[2][2]) ** 0.5 * 2
            return [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s]
        s = (1.0 + m[2][2] - m[0][0] - m[1][1]) ** 0.5 * 2
        return [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s]

    def _apply_to_solvers(self, object_name) -> None:
        """Carry an attach or a detach to every solver: the robot's spheres, and
        the obstacles' on/off state.

        The obstacles are set from the node's own record (attached ones off,
        the others as set_obstacle_collision left them), not from cuRobo's
        attachment manager: that remembers only the obstacles of the LAST
        attach, so with two attach links a detach re-enables the object the
        other arm still holds and leaves the released one off.
        """
        node = self.node
        if hasattr(node, 'apply_robot_sphere_states'):
            node.apply_robot_sphere_states()
        if hasattr(node, 'apply_obstacle_states'):
            if object_name:
                node.apply_obstacle_states([object_name])
            node.apply_obstacle_states()

    @staticmethod
    def _estimate_sphere_need(obstacle) -> int:
        """cuRobo's own estimate of how many spheres this geometry needs.

        Same heuristic the AttachmentManager applies when ``num_spheres=None``
        (bounding-box volume, 1 sphere per 15 cm3, capped at 100). Returns 0 if
        it can't be computed, so a failure here never blocks an attach.
        """
        try:
            return estimate_sphere_count(
                obstacle.get_trimesh_mesh(transform_with_pose=True))
        except Exception:
            return 0

    @staticmethod
    def _find_obstacle(scene, name: str):
        """Find an obstacle by name across the Scene's typed buckets.

        The ObstacleManager stores obstacles in typed lists (cuboid/sphere/…),
        not in Scene.objects, so Scene.get_obstacle (which reads .objects) misses
        runtime-added ones — search the buckets directly.
        """
        for bucket in ('cuboid', 'sphere', 'capsule', 'cylinder', 'mesh'):
            for obj in (getattr(scene, bucket, None) or []):
                if obj.name == name:
                    return obj
        if getattr(scene, 'objects', None):
            return scene.get_obstacle(name)
        return None

    def detach(self, link: str = ATTACH_LINK) -> str:
        """Release the attached object (reset link spheres, re-enable the obstacle)."""
        with self.node.gpu_lock:
            self._attachment_manager().detach(link)
            self.config_wrapper.kin_model.kinematics_config.reset_link_spheres(link)
            released = self._attached.pop(link, None)
            obstacle_names = self.config_wrapper.obstacle_manager.obstacle_names
            if (released and not released.startswith("payload@")
                    and released not in obstacle_names):
                self._set_world_copy(released, True)
            self._apply_to_solvers(released)
        self.node.get_logger().info(f"Detached '{released}' from '{link}'")
        return released or ""

    # ------------------------------------------------------------------
    # Service callbacks
    # ------------------------------------------------------------------

    def _attach_object_callback(self, request, response):
        """Attach a scene obstacle to the arm at its current joint configuration."""
        try:
            if self.motion_planner is None:
                self.node._warmup_classic()
            # start_pose given: attach at that configuration; empty: at the
            # robot's current one, as before.
            _, state = self.node._resolve_start_state(request)
            name, obstacle = request.object_name, None
            if not name:
                name, obstacle = self.payload(request, state)
            self.attach(name, state,
                        link=request.attach_link or ATTACH_LINK,
                        fit_type=_SPHERE_FIT.get(request.sphere_fit_type),
                        num_spheres=request.num_spheres, obstacle=obstacle)
            response.success = True
            response.message = f"Attached '{name}'"
        except Exception as e:
            response.success = False
            response.message = f"Attach error: {e}"
            self.node.get_logger().error(response.message)
        return response

    def _detach_link_callback(self, request, response):
        """Release what is attached to one link (``request.name``)."""
        link = request.name
        try:
            if not self._has_attach_link(link):
                raise ValueError(f"the robot has no attach link '{link}'")
            released = self.detach(link)
            response.success = True
            response.message = (
                f"Detached '{released}' from '{link}'" if released
                else f"Nothing attached to '{link}'")
        except Exception as e:
            response.success = False
            response.message = f"Detach error: {e}"
            self.node.get_logger().error(response.message)
        return response

    def _detach_object_callback(self, request, response):
        """Release a previously attached object from the arm."""
        try:
            released = self.detach()
            response.success = True
            response.message = (
                f"Detached '{released}'" if released else "Nothing attached")
        except Exception as e:
            response.success = False
            response.message = f"Detach error: {e}"
            self.node.get_logger().error(response.message)
        return response
