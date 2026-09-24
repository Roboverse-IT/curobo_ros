"""Local fixes to cuRobo v0.8.0, applied only when a robot descriptor asks for
them (``apply_curobo_patches: true``, see robot/robot_description.py).

Reused as-is from the dual-arm bench where they were found and verified; only
the text is translated. Three fixes:

1) a joint reindex that does not touch ``knot``
2) the ``warp.torch`` import
3) an ``enable_obstacle`` that can also disable meshes

--- 3) ``enable_obstacle`` and meshes ---

WHY. ``SceneData.enable_obstacle`` tries the obstacle types in turn -- cuboids,
meshes, voxels -- and catches ``RuntimeError`` to move on to the next one. But
the name lookups raise ``ValueError``: ``CuboidData.get_idx`` does a bare
``list.index``, and ``MeshData.get_idx`` catches ``ValueError`` only to re-raise
it through ``log_and_raise``, which raises ``ValueError`` again. So the FIRST
attempt, on the cuboids, blows up and the meshes are never reached.

In practice: **only a cuboid obstacle can be disabled** whenever a cuboid store
exists, and a collision cache with a cuboid entry always allocates one. And
since ``AttachmentManager.attach(..., disable_obstacle_names=...)`` goes through
here, attaching an object read as a mesh and removing it from the world
**always fails**, with ``ValueError: '<name>' is not in list`` talking about the
cuboids while the object is a mesh.

WHAT IT DOES. Replaces ``enable_obstacle`` with a version that tries the same
types in the same order catching ``ValueError`` too, and raises only if the name
exists in none of them.

To remove when NVIDIA fixes it upstream: changing ``except RuntimeError`` into
``except ValueError`` there is enough.

--- 1) reindex and ``knot`` ---

WHY. ``reindex_joint_state_inplace`` builds the index from the joint NAMES, so
in the space of ALL joints, and hands it to ``jit_inplace_reindex``, which also
applies it to ``knot`` -- the B-spline control points, which live in the space
of the ACTIVE joints. As long as the active joints are the first N of all the
joints the index stays in range and nobody notices: that is the case of EVERY
robot shipped with cuRobo (franka 0..6 of 7, ur10e 0..5 of 6, dual_ur10e 0..11
of 12). On a robot where a locked joint sits before the active ones in the
chain (e.g. a lifting column between the base and the arms: index 0, pushing 14
active joints to indices 1..14 of 17), index 14 on a 14-column ``knot`` is out
of range by one, and ``plan_grasp`` dies with
``indexSelectSmallIndex: Assertion 'srcIndex < srcSelectDimSize' failed``.

WHAT IT DOES. Replaces ``jit_inplace_reindex`` with a version that reorders
``knot`` only when it really lives in the same space as the other tensors;
otherwise it leaves it as it is, which is already the right active order.

To remove when NVIDIA fixes it upstream.
"""
from typing import Optional
import torch


def apply() -> bool:
    """Apply the fixes. Returns True if they were applied by this call."""
    # 2) Version drift between curobo and warp. curobo declares
    #    `warp-lang>=0.10.0` with no ceiling, so the latest gets installed; but
    #    since warp 1.14 the torch interop moved from `warp.torch` to
    #    `warp._src.torch`, while curobo still calls `wp.torch.device_from_torch`
    #    in data_mesh.py. A scene of primitives only shows nothing: it blows up
    #    as soon as the scene holds a MESH, i.e. as soon as it is a real scene.
    import warp as wp
    if not hasattr(wp, "torch"):
        from warp._src import torch as _wp_torch
        wp.torch = _wp_torch

    import curobo._src.state.state_joint_ops as ops

    if getattr(ops, "_knot_patch_applied", False):
        return False

    original = ops.jit_inplace_reindex

    def jit_inplace_reindex_safe(position, velocity, acceleration, jerk, knot, new_index):
        # `knot` lives in the active-joint space: if it has a number of columns
        # different from `position`, the name-built index does not concern it.
        if knot is not None and knot.shape[-1] != position.shape[-1]:
            p, v, a, j, _ = original(position, velocity, acceleration, jerk, None, new_index)
            return p, v, a, j, knot
        return original(position, velocity, acceleration, jerk, knot, new_index)

    ops.jit_inplace_reindex = jit_inplace_reindex_safe
    ops._knot_patch_applied = True

    # 3) enable_obstacle never reaches the meshes
    from curobo._src.geom.data.data_scene import SceneData

    def enable_obstacle_any_type(self, name: str, enabled: bool = True, env_idx: int = 0) -> None:
        for store in (self.cuboids, getattr(self, "meshes", None), getattr(self, "voxels", None)):
            if store is None:
                continue
            try:
                store.set_enabled(name, enabled, env_idx)
                return
            except (RuntimeError, ValueError):
                continue
        raise ValueError(f"obstacle '{name}' not found in any type (env {env_idx})")

    SceneData.enable_obstacle = enable_obstacle_any_type
    return True
