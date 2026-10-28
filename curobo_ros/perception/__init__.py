"""Perception algorithms that do not depend on ROS messages or executors."""

from curobo_ros.perception.projected_mesh_mask import (
    ProjectedMeshMask, resolve_urdf_mesh_paths)

__all__ = ['ProjectedMeshMask', 'resolve_urdf_mesh_paths']
