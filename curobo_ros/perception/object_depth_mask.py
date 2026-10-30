"""Generic, depth-consistent masking of known non-robot objects.

The robot self-mask remains owned by ``robot_segmentation``.  This component
removes a separately registered object's pixels only when their measured
camera-z agrees with the nearest rasterized object surface.  A 2-D silhouette
is intentionally not sufficient: it would erase a background seen through a
concavity or beside a partially occluded object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from curobo_ros.perception.projected_mesh_mask import (
    _quaternion_to_rotation_matrix, rasterize_nearest_depth)


def _box_geometry(dimensions: Sequence[float]):
    dims = np.asarray(dimensions, dtype=np.float64).reshape(3)
    if not np.isfinite(dims).all() or np.any(dims <= 0.0):
        raise ValueError('cuboid dimensions must be positive finite values')
    x, y, z = dims / 2.0
    vertices = np.array([
        [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
        [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ], dtype=np.int64)
    return vertices, faces


def _load_obj(path: str | Path):
    vertices, faces = [], []
    with open(path) as stream:
        for line in stream:
            fields = line.strip().split()
            if not fields or fields[0].startswith('#'):
                continue
            if fields[0] == 'v' and len(fields) >= 4:
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif fields[0] == 'f' and len(fields) >= 4:
                indices = [int(value.split('/')[0]) for value in fields[1:]]
                indices = [value - 1 if value > 0 else len(vertices) + value
                           for value in indices]
                for index in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[index], indices[index + 1]])
    return vertices, faces


@dataclass(frozen=True)
class ObjectGeometry:
    """Triangle surface in the object's local coordinate frame."""

    vertices: np.ndarray
    faces: np.ndarray

    @classmethod
    def cuboid(cls, dimensions: Sequence[float]):
        return cls(*_box_geometry(dimensions))

    @classmethod
    def mesh(cls, vertices: Iterable[Iterable[float]], faces: Iterable[Iterable[int]]):
        vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if not len(vertices) or not len(faces):
            raise ValueError('mesh geometry must contain vertices and triangular faces')
        if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError('mesh geometry contains invalid vertices or face indices')
        return cls(vertices, faces)

    @classmethod
    def mesh_file(cls, path: str | Path):
        try:
            import trimesh
            mesh = trimesh.load(str(path), force='mesh', process=False)
            if hasattr(mesh, 'geometry'):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            return cls.mesh(mesh.vertices, mesh.faces)
        except Exception:
            return cls.mesh(*_load_obj(path))

    def world_aabb(self, pose, margin_m=0.0):
        """Conservative base-frame AABB, for local TSDF invalidation."""
        margin = float(margin_m)
        if margin < 0.0:
            raise ValueError('margin_m must be >= 0')
        value = np.asarray(pose, dtype=np.float64).reshape(7)
        q = value[3:] / np.linalg.norm(value[3:])
        w, x, y, z = q
        rotation = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        points = self.vertices.astype(np.float64) @ rotation.T + value[:3]
        return points.min(axis=0) - margin, points.max(axis=0) + margin


def _pose_tensor(pose, device):
    """Validate [x,y,z,qw,qx,qy,qz] and produce a base<-object transform."""
    if hasattr(pose, 'position') and hasattr(pose, 'quaternion'):
        value = torch.cat((
            pose.position.reshape(-1, 3)[0], pose.quaternion.reshape(-1, 4)[0]))
        value = value.to(device=device, dtype=torch.float32)
    else:
        value = torch.as_tensor(pose, dtype=torch.float32, device=device).reshape(-1)
    if value.numel() != 7 or not bool(torch.isfinite(value).all()):
        raise ValueError('pose must be seven finite [x,y,z,qw,qx,qy,qz] values')
    norm = torch.linalg.vector_norm(value[3:])
    if not bool(torch.isfinite(norm)) or float(norm) < 1.0e-8:
        raise ValueError('pose quaternion is invalid')
    value = value.clone()
    value[3:] /= norm
    transform = torch.eye(4, dtype=torch.float32, device=device)
    transform[:3, :3] = _quaternion_to_rotation_matrix(value[3:].reshape(1, 4))[0]
    transform[:3, 3] = value[:3]
    return transform


@dataclass
class _ObjectEntry:
    geometry: ObjectGeometry
    pose: Sequence[float] | None
    pose_at_stamp: Callable[[int], Sequence[float] | None] | None


class ObjectDepthMasker:
    """Mask multiple known objects from an already robot-masked depth image.

    Poses are base-frame ``[x,y,z,qw,qx,qy,qz]`` values.  A dynamic object's
    callback receives the acquired image stamp in nanoseconds and must return
    the pose at that exact stamp, or ``None``.  In the latter case the object
    is deliberately skipped rather than using a stale pose.
    """

    def __init__(self, depth_tolerance_m=0.015, dilation_px=1):
        if depth_tolerance_m < 0.0:
            raise ValueError('depth_tolerance_m must be >= 0')
        if not isinstance(dilation_px, int) or dilation_px < 0:
            raise ValueError('dilation_px must be an integer >= 0')
        self.depth_tolerance_m = float(depth_tolerance_m)
        self.dilation_px = dilation_px
        self._objects = {}
        self._cuda_geometry = {}
        self.last_mask = None

    def set_object(self, name, geometry, pose=None, pose_at_stamp=None):
        if not name:
            raise ValueError('object name must be non-empty')
        if not isinstance(geometry, ObjectGeometry):
            raise TypeError('geometry must be ObjectGeometry')
        if (pose is None) == (pose_at_stamp is None):
            raise ValueError('provide exactly one of pose or pose_at_stamp')
        if pose_at_stamp is not None and not callable(pose_at_stamp):
            raise TypeError('pose_at_stamp must be callable')
        # Validate fixed input now, rather than failing during a camera callback.
        if pose is not None:
            _pose_tensor(pose, torch.device('cpu'))
        self._objects[name] = _ObjectEntry(geometry, pose, pose_at_stamp)

    def remove_object(self, name):
        self._objects.pop(name, None)

    def clear(self):
        self._objects.clear()

    def _geometry(self, geometry, device):
        key = (id(geometry), str(device))
        cached = self._cuda_geometry.get(key)
        if cached is None:
            cached = (
                torch.as_tensor(geometry.vertices, dtype=torch.float32, device=device).contiguous(),
                torch.as_tensor(geometry.faces, dtype=torch.int64, device=device).contiguous())
            self._cuda_geometry[key] = cached
        return cached

    def mask_depth(self, depth_image, intrinsics, camera_pose_base, stamp_ns=None):
        """Return the Boolean mask for a CUDA ``HxW`` depth image in metres."""
        if depth_image.ndim != 2 or depth_image.device.type != 'cuda':
            raise ValueError('depth_image must be an HxW CUDA tensor')
        device = depth_image.device
        k = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
        if k.numel() != 9:
            raise ValueError('intrinsics must be a 3x3 matrix')
        k = k.reshape(3, 3)
        camera_from_base = torch.linalg.inv(_pose_tensor(camera_pose_base, device))
        nearest = torch.full_like(depth_image, float('inf'), dtype=torch.float32)
        for entry in self._objects.values():
            pose = entry.pose if entry.pose_at_stamp is None else entry.pose_at_stamp(stamp_ns)
            if pose is None:
                continue
            base_from_object = _pose_tensor(pose, device)
            vertices, faces = self._geometry(entry.geometry, device)
            camera_vertices = vertices @ (camera_from_base @ base_from_object)[:3, :3].T
            camera_vertices += (camera_from_base @ base_from_object)[:3, 3]
            front = rasterize_nearest_depth(
                camera_vertices, faces,
                (k[0, 0], k[1, 1], k[0, 2], k[1, 2]), depth_image.shape)
            torch.minimum(nearest, front, out=nearest)
        mask = torch.isfinite(nearest) & torch.isfinite(depth_image) & (depth_image > 0.0)
        mask &= (depth_image - nearest).abs() <= self.depth_tolerance_m
        if self.dilation_px:
            radius = self.dilation_px
            mask = torch.nn.functional.max_pool2d(
                mask[None, None].to(dtype=torch.float32), 2 * radius + 1,
                stride=1, padding=radius)[0, 0].to(dtype=torch.bool)
        self.last_mask = mask
        return mask

    def apply(self, depth_image, intrinsics, camera_pose_base, stamp_ns=None):
        """Zero matching object pixels, returning ``(masked_depth, mask)``."""
        mask = self.mask_depth(depth_image, intrinsics, camera_pose_base, stamp_ns)
        return depth_image.masked_fill(mask, 0.0), mask
