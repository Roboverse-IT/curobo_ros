"""Clean-up of an obstacle scene before handing it to cuRobo v0.8.0.

Reused as-is from the dual-arm bench where it was found and verified; only the
text is translated. Used by ObstacleManager for obstacles read from a USD file.

--- The problem ---

cuRobo's mesh SDF kernel saturates on the size of the mesh itself.
In `curobo/_src/geom/data/data_mesh.py`, `compute_local_sdf()`:

    bounding_box_size = wp.vec3(obs_set.dims[flat_idx,0], ...)
    max_distance = wp.length(bounding_box_size) * 0.5
    result = wp.mesh_query_point(mesh_id, local_pt, max_distance)
    if not result.result:
        return max_distance          # <-- SATURATED value, not the real distance

When the query point is farther than `max_distance` from the mesh, the query
finds nothing and the function returns `max_distance`. That value is then
compared with the RADIUS of the robot's collision sphere. So:

    r_sphere > 0.5 * |diagonal of the mesh bbox|
        =>  the sphere is in collision AT EVERY POINT IN SPACE,
            with a constant apparent penetration r_sphere - 0.5*|diagonal|.

That is: **any mesh smaller than the robot's largest sphere makes the scene
unplannable**, wherever it is. There is no error message: IK simply stops
finding solutions and `plan_pose` returns `None`.

On a robot whose largest sphere is 130 mm (a wrist), a 55 mm part to be picked
is well under that threshold -- the objects to manipulate are exactly the small
ones.

--- The fix ---

`dims` is computed from the bounding box of the warp mesh's POINTS
(`data_mesh.py`, `get_bounds()` reads `self.mesh.points`), while the surface is
defined by the FACES. Adding vertices that no face references therefore widens
the bounding box -- and so `max_distance` -- without moving the collision
geometry by a micron.

The only effect is the intended one: the query searches farther (slightly
slower) and, when it finds nothing, returns a value large enough to read as
"no collision".

To remove when NVIDIA fixes it upstream. That the value returned on a miss must
be >= any sphere radius is a condition cuRobo neither documents nor checks.
"""

from __future__ import annotations

import copy
import math
from typing import Any, List, Optional, Tuple

import numpy as np


def half_diagonal(mesh: Any) -> float:
    """Half the diagonal of the mesh's bounding box -- i.e. `max_distance`."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    return float(np.linalg.norm(v.max(0) - v.min(0)) * 0.5)


def pad_mesh(mesh: Any, min_half_diagonal: float) -> Any:
    """Copy of the mesh with a wider bounding box, same surface.

    Adds only two vertices -- the opposite corners of the wanted box -- that no
    face references.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    lo, hi = v.min(0), v.max(0)
    centre, half = (hi + lo) / 2.0, (hi - lo) / 2.0
    # component by component: just enough for |diagonal|/2 >= threshold
    need = min_half_diagonal / math.sqrt(3.0)
    new_half = np.maximum(half, need)
    out = copy.copy(mesh)
    out.vertices = np.vstack([v, centre - new_half, centre + new_half]).tolist()
    return out


def sanitize_scene(
    scene: Any,
    max_robot_sphere_radius: float,
    margin: float = 0.05,
) -> Tuple[Any, List[str]]:
    """Widen the bbox of the meshes too small for cuRobo's SDF kernel.

    Args:
        scene: a `SceneCfg` read from a USD (or from any other source).
        max_robot_sphere_radius: the radius of the robot's largest collision
            sphere, `planner.kinematics.robot_spheres[..., 3].max()`.
        margin: clearance over the threshold. Needed because the threshold is a
            comparison between the sphere radius and `max_distance`: sitting
            exactly on it would leave the apparent penetration at zero but the
            gradient dirty.

    Returns:
        (cleaned scene, readable list of what was touched).
    """
    if not getattr(scene, "mesh", None):
        return scene, []

    threshold = max_robot_sphere_radius + margin
    fixed, report = [], []
    for m in scene.mesh:
        h = half_diagonal(m)
        if h < threshold:
            fixed.append(pad_mesh(m, threshold))
            report.append(
                f"{m.name.split('/')[-1]}: half-diagonal {h*1000:.0f} mm "
                f"< {threshold*1000:.0f} mm -> bbox widened"
            )
        else:
            fixed.append(m)

    out = copy.copy(scene)
    out.mesh = fixed
    return out, report


def audit_scene(scene: Any, max_robot_sphere_radius: float) -> List[str]:
    """List the meshes that would give false collision, without changing anything."""
    out = []
    for m in getattr(scene, "mesh", None) or []:
        h = half_diagonal(m)
        if h < max_robot_sphere_radius:
            out.append(
                f"{m.name.split('/')[-1]}: half-diagonal {h*1000:.0f} mm "
                f"< sphere radius {max_robot_sphere_radius*1000:.0f} mm"
            )
    return out
