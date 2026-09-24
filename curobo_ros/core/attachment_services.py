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

from std_srvs.srv import Trigger
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
               link: str = ATTACH_LINK, fit_type=None, num_spheres: int = 0) -> None:
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
        """
        if not self._has_attach_link(link):
            raise ValueError(
                f"Robot YAML does not declare the '{link}' link — "
                "attach is unavailable for this robot (see "
                "docs/tutorials/03-collision-objects.md)."
            )
        obstacle_manager = self.config_wrapper.obstacle_manager
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
            am.attach(
                joint_states=grasp_end_state,
                obstacles=[obstacle],
                link_name=link,
                num_spheres=n_fit,
                disable_obstacle_names=[object_name],
                **fit,
            )
            # The fit lands on the planner's spheres: copy it to the node's
            # robot model, from which every solver takes its spheres.
            self.config_wrapper.kin_model.kinematics_config.update_link_spheres(
                link, am.kinematics_params.get_link_spheres(link))
            self._attached[link] = object_name
            self._apply_to_solvers(object_name)
        self.node.get_logger().info(
            f"Attached '{object_name}' to link '{link}'")

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
            _, current_state = self.node._resolve_start_state(None)
            self.attach(request.object_name, current_state,
                        link=request.attach_link or ATTACH_LINK,
                        fit_type=_SPHERE_FIT.get(request.sphere_fit_type),
                        num_spheres=request.num_spheres)
            response.success = True
            response.message = f"Attached '{request.object_name}'"
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
