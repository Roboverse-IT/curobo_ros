"""GPU projection and silhouette rasterization for robot link meshes."""

import glob
import hashlib
import os
import tempfile
import xml.etree.ElementTree as ET

from ament_index_python.packages import (
    PackageNotFoundError, get_package_share_directory)
from curobo._src.util.warp import init_warp
from curobo.content import get_assets_path
import numpy as np
import torch

try:
    import warp as wp
except ImportError:  # pragma: no cover - Warp is part of the cuRobo runtime
    wp = None


_RESOLVED_DIR = '/tmp/curobo_ros_resolved'


def _package_asset(package, tail, search_roots):
    """Resolve one package-relative asset from ROS, cuRobo, or source roots."""
    try:
        candidate = os.path.join(get_package_share_directory(package), tail)
        if os.path.isfile(candidate):
            return candidate
    except (PackageNotFoundError, ValueError):
        pass

    candidate = os.path.join(get_assets_path(), 'robot', package, tail)
    if os.path.isfile(candidate):
        return candidate

    for root in search_roots:
        candidates = [os.path.join(root, package, tail)]
        if os.path.basename(root) == package:
            candidates.insert(0, os.path.join(root, tail))
        candidates.extend(glob.glob(
            os.path.join(root, '**', package, tail), recursive=True))
        for candidate in candidates:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
    return None


def resolve_urdf_mesh_paths(urdf_path, search_roots=()):
    """Write a URDF copy whose loadable ``package://`` meshes are absolute.

    ROS package shares are preferred, followed by cuRobo's bundled robot
    assets and explicitly supplied source roots.  For generated URDFs living
    below a project directory, a sibling ``src`` directory is also searched.
    URIs with no matching file remain unchanged so callers can either skip the
    affected link or reject it according to their configured strictness.

    Returns ``(resolved_urdf_path, unresolved_uris)``.
    """
    urdf_path = os.path.abspath(urdf_path)
    if not os.path.isfile(urdf_path):
        raise ValueError(f'URDF is not an existing file: {urdf_path!r}')

    roots = []
    for root in search_roots:
        root = os.path.abspath(os.path.expanduser(root))
        if root not in roots:
            roots.append(root)
    ancestor = os.path.dirname(urdf_path)
    while True:
        source_root = os.path.join(ancestor, 'src')
        if os.path.isdir(source_root):
            if source_root not in roots:
                roots.append(source_root)
            break
        parent = os.path.dirname(ancestor)
        if parent == ancestor:
            break
        ancestor = parent

    tree = ET.parse(urdf_path)
    unresolved = set()
    changed = False
    cache = {}
    for mesh in tree.getroot().iter('mesh'):
        filename = mesh.get('filename', '')
        if not filename.startswith('package://'):
            continue
        rest = filename[len('package://'):]
        package, separator, tail = rest.partition('/')
        if not package or not separator or not tail:
            unresolved.add(filename)
            continue
        key = (package, tail)
        if key not in cache:
            cache[key] = _package_asset(package, tail, roots)
        resolved = cache[key]
        if resolved is None:
            unresolved.add(filename)
            continue
        mesh.set('filename', resolved)
        changed = True

    if not changed:
        return urdf_path, sorted(unresolved)

    digest_input = '|'.join([urdf_path, *roots])
    digest = hashlib.sha1(digest_input.encode()).hexdigest()[:10]
    name = os.path.splitext(os.path.basename(urdf_path))[0]
    os.makedirs(_RESOLVED_DIR, exist_ok=True)
    resolved_path = os.path.join(
        _RESOLVED_DIR, f'{name}-mesh-{digest}.urdf')
    handle, temporary_path = tempfile.mkstemp(
        prefix=f'.{name}-mesh-', suffix='.urdf', dir=_RESOLVED_DIR)
    os.close(handle)
    try:
        tree.write(temporary_path, encoding='unicode', xml_declaration=True)
        os.replace(temporary_path, resolved_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return resolved_path, sorted(unresolved)


if wp is not None:
    @wp.kernel
    def _transform_project_vertices(
        local_vertices: wp.array(dtype=wp.vec3),
        vertex_link_ids: wp.array(dtype=wp.int32),
        link_transforms: wp.array(dtype=wp.float32, ndim=3),
        camera_vertices: wp.array(dtype=wp.vec3),
        projected: wp.array(dtype=wp.vec2),
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        raster_stride: float,
    ):
        """Transform and project one cached link-local hull vertex."""
        tid = wp.tid()
        link = vertex_link_ids[tid]
        p = local_vertices[tid]
        x = (
            link_transforms[link, 0, 0] * p[0]
            + link_transforms[link, 0, 1] * p[1]
            + link_transforms[link, 0, 2] * p[2]
            + link_transforms[link, 0, 3]
        )
        y = (
            link_transforms[link, 1, 0] * p[0]
            + link_transforms[link, 1, 1] * p[1]
            + link_transforms[link, 1, 2] * p[2]
            + link_transforms[link, 1, 3]
        )
        z = (
            link_transforms[link, 2, 0] * p[0]
            + link_transforms[link, 2, 1] * p[1]
            + link_transforms[link, 2, 2] * p[2]
            + link_transforms[link, 2, 3]
        )
        w = (
            link_transforms[link, 3, 0] * p[0]
            + link_transforms[link, 3, 1] * p[1]
            + link_transforms[link, 3, 2] * p[2]
            + link_transforms[link, 3, 3]
        )
        if wp.abs(w) > 1.0e-8:
            x = x / w
            y = y / w
            z = z / w
        camera_vertices[tid] = wp.vec3(x, y, z)
        if z > 1.0e-5:
            projected[tid] = wp.vec2(
                (fx * x / z + cx) / raster_stride,
                (fy * y / z + cy) / raster_stride,
            )
        else:
            projected[tid] = wp.vec2(0.0, 0.0)

    @wp.kernel
    def _rasterize_projected_triangles(
        vertices: wp.array(dtype=wp.vec2),
        faces: wp.array(dtype=wp.int32, ndim=2),
        face_link_ids: wp.array(dtype=wp.int32),
        output: wp.array(dtype=wp.int32),
        width: int,
        height: int,
    ):
        """Rasterize one projected triangle into a low-resolution owner map."""
        tid = wp.tid()
        a = vertices[faces[tid, 0]]
        b = vertices[faces[tid, 1]]
        c = vertices[faces[tid, 2]]
        xmin = int(wp.floor(wp.min(a[0], wp.min(b[0], c[0]))))
        xmax = int(wp.ceil(wp.max(a[0], wp.max(b[0], c[0]))))
        ymin = int(wp.floor(wp.min(a[1], wp.min(b[1], c[1]))))
        ymax = int(wp.ceil(wp.max(a[1], wp.max(b[1], c[1]))))
        xmin = wp.max(xmin, 0)
        ymin = wp.max(ymin, 0)
        xmax = wp.min(xmax, width - 1)
        ymax = wp.min(ymax, height - 1)
        denominator = (
            (b[1] - c[1]) * (a[0] - c[0])
            + (c[0] - b[0]) * (a[1] - c[1])
        )
        if wp.abs(denominator) > 1.0e-8:
            for y in range(ymin, ymax + 1):
                for x in range(xmin, xmax + 1):
                    wa = (
                        (b[1] - c[1]) * (float(x) - c[0])
                        + (c[0] - b[0]) * (float(y) - c[1])
                    ) / denominator
                    wb = (
                        (c[1] - a[1]) * (float(x) - c[0])
                        + (a[0] - c[0]) * (float(y) - c[1])
                    ) / denominator
                    wc = 1.0 - wa - wb
                    if wa >= 0.0 and wb >= 0.0 and wc >= 0.0:
                        wp.atomic_max(
                            output, y * width + x, face_link_ids[tid] + 1)

    @wp.kernel
    def _rasterize_projected_triangle_depth(
        vertices: wp.array(dtype=wp.vec2),
        camera_vertices: wp.array(dtype=wp.vec3),
        faces: wp.array(dtype=wp.int32, ndim=2),
        output: wp.array(dtype=float),
        width: int,
        height: int,
    ):
        """Rasterize triangles into a nearest camera-z buffer.

        ``z`` is interpolated perspectively.  Linearly interpolating vertex
        z in screen space is only correct for a fronto-parallel surface; the
        reciprocal form below is the camera-z of the ray/triangle
        intersection at this pixel.
        """
        tid = wp.tid()
        ia, ib, ic = faces[tid, 0], faces[tid, 1], faces[tid, 2]
        a, b, c = vertices[ia], vertices[ib], vertices[ic]
        za, zb, zc = (camera_vertices[ia][2], camera_vertices[ib][2],
                      camera_vertices[ic][2])
        xmin = wp.max(int(wp.floor(wp.min(a[0], wp.min(b[0], c[0])))), 0)
        xmax = wp.min(int(wp.ceil(wp.max(a[0], wp.max(b[0], c[0])))), width - 1)
        ymin = wp.max(int(wp.floor(wp.min(a[1], wp.min(b[1], c[1])))), 0)
        ymax = wp.min(int(wp.ceil(wp.max(a[1], wp.max(b[1], c[1])))), height - 1)
        denominator = ((b[1] - c[1]) * (a[0] - c[0])
                       + (c[0] - b[0]) * (a[1] - c[1]))
        if wp.abs(denominator) > 1.0e-8:
            for y in range(ymin, ymax + 1):
                for x in range(xmin, xmax + 1):
                    wa = ((b[1] - c[1]) * (float(x) - c[0])
                          + (c[0] - b[0]) * (float(y) - c[1])) / denominator
                    wb = ((c[1] - a[1]) * (float(x) - c[0])
                          + (a[0] - c[0]) * (float(y) - c[1])) / denominator
                    wc = 1.0 - wa - wb
                    if wa >= -1.0e-6 and wb >= -1.0e-6 and wc >= -1.0e-6:
                        inverse_depth = wa / za + wb / zb + wc / zc
                        if inverse_depth > 0.0:
                            wp.atomic_min(output, y * width + x,
                                          1.0 / inverse_depth)


def _quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Return row-major rotation matrices for cuRobo wxyz quaternions."""
    qw, qx, qy, qz = quaternion.unbind(dim=-1)
    return torch.stack((
        1.0 - 2.0 * (qy * qy + qz * qz),
        2.0 * (qx * qy - qw * qz),
        2.0 * (qx * qz + qw * qy),
        2.0 * (qx * qy + qw * qz),
        1.0 - 2.0 * (qx * qx + qz * qz),
        2.0 * (qy * qz - qw * qx),
        2.0 * (qx * qz - qw * qy),
        2.0 * (qy * qz + qw * qx),
        1.0 - 2.0 * (qx * qx + qy * qy),
    ), dim=-1).reshape(-1, 3, 3)


def rasterize_nearest_depth(camera_vertices, faces, intrinsics, image_shape):
    """Return a CUDA nearest-surface camera-z image for arbitrary triangles.

    ``camera_vertices`` is ``(N, 3)`` in the optical frame, and ``faces`` is
    ``(M, 3)``.  The function deliberately has no robot/link concepts so it
    can serve object masking and other depth-aware projection users.  Invalid
    and camera-plane-crossing faces are omitted rather than extrapolated.
    """
    if wp is None:
        raise RuntimeError('nearest-depth rasterization requires Warp')
    if camera_vertices.ndim != 2 or camera_vertices.shape[1] != 3:
        raise ValueError('camera_vertices must have shape (N, 3)')
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('faces must have shape (M, 3)')
    if camera_vertices.device.type != 'cuda' or faces.device != camera_vertices.device:
        raise ValueError('vertices and faces must be on the same CUDA device')
    height, width = map(int, image_shape)
    if height <= 0 or width <= 0:
        raise ValueError('image_shape must contain positive height and width')
    init_warp()
    vertices = camera_vertices.to(dtype=torch.float32).contiguous()
    faces64 = faces.to(device=vertices.device, dtype=torch.int64).contiguous()
    if faces64.numel() and (int(faces64.min()) < 0 or int(faces64.max()) >= len(vertices)):
        raise ValueError('face index is outside camera_vertices')
    valid_vertices = torch.isfinite(vertices).all(dim=1) & (vertices[:, 2] > 1.0e-5)
    valid_faces = faces64[valid_vertices[faces64].all(dim=1)]
    front = torch.full((height, width), float('inf'), dtype=torch.float32,
                       device=vertices.device)
    if not len(valid_faces):
        return front
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    projected = torch.empty((len(vertices), 2), dtype=torch.float32,
                            device=vertices.device)
    projected[:, 0] = fx * vertices[:, 0] / vertices[:, 2] + cx
    projected[:, 1] = fy * vertices[:, 1] / vertices[:, 2] + cy
    stream = wp.stream_from_torch(torch.cuda.current_stream(device=vertices.device))
    wp.launch(
        _rasterize_projected_triangle_depth,
        dim=int(valid_faces.shape[0]),
        inputs=[
            wp.from_torch(projected, dtype=wp.vec2),
            wp.from_torch(vertices, dtype=wp.vec3),
            wp.from_torch(valid_faces.to(dtype=torch.int32).contiguous()),
            wp.from_torch(front.reshape(-1)), width, height,
        ],
        device=wp.device_from_torch(vertices.device), stream=stream)
    return front


class ProjectedMeshMask:
    """Cache convex link hulls and rasterize camera silhouettes on CUDA."""

    def __init__(self, link_names, link_meshes, device, margin=0.0,
                 raster_stride=8, strict=True):
        """Create a backend from link-local cuRobo mesh objects."""
        if wp is None:
            raise RuntimeError(
                'Projected mesh segmentation requires Warp from the cuRobo '
                'runtime')
        init_warp()
        if not link_names or len(link_names) != len(link_meshes):
            raise ValueError(
                'link_names and link_meshes must have equal non-zero length')
        if len(set(link_names)) != len(link_names):
            raise ValueError(
                f'mesh link names contain duplicates: {link_names}')
        if not isinstance(margin, (int, float)) or isinstance(margin, bool) \
                or margin < 0.0:
            raise ValueError(
                f'mesh margin must be a number >= 0, got {margin!r}')
        if (not isinstance(raster_stride, int)
                or isinstance(raster_stride, bool) or raster_stride < 1):
            raise ValueError(
                'mesh raster stride must be an integer >= 1, got '
                f'{raster_stride!r}')

        requested_link_names = list(link_names)
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.margin = float(margin)
        self.raster_stride = raster_stride
        self.link_names, self._geometry, skipped = self._load_geometry(
            requested_link_names, link_meshes, strict)
        self.skipped_link_names = skipped
        if not self.link_names:
            raise ValueError(
                'none of the configured mesh links have loadable visual '
                'geometry')
        self._buffers = None

    @classmethod
    def from_kinematics(cls, kinematics, link_names, device, margin=0.0,
                        raster_stride=8, strict=False):
        """Load selected meshes from a configured cuRobo Kinematics model."""
        selected_names = []
        selected_meshes = []
        missing = []
        for name in link_names:
            try:
                mesh = kinematics.get_link_mesh(name)
            except KeyError:
                mesh = None
            if mesh is None:
                missing.append(name)
            else:
                selected_names.append(name)
                selected_meshes.append(mesh)
        if strict and missing:
            raise ValueError(
                f'configured mesh links have no visual geometry: {missing}')
        if not selected_names:
            raise ValueError(
                'none of the configured mesh links have visual geometry: '
                f'{link_names}')
        backend = cls(
            selected_names,
            selected_meshes,
            device,
            margin=margin,
            raster_stride=raster_stride,
            strict=strict,
        )
        backend.skipped_link_names = missing + backend.skipped_link_names
        return backend

    def _load_geometry(self, link_names, link_meshes, strict):
        """Load, convexify, optionally inflate, and upload link geometry."""
        loaded_names = []
        geometry = []
        skipped = []
        for link_name, mesh in zip(link_names, link_meshes):
            if mesh is None:
                skipped.append(link_name)
                continue
            try:
                source = mesh.get_trimesh_mesh(
                    process=False, transform_with_pose=True)
            except (FileNotFoundError, OSError, ValueError):
                if strict:
                    raise ValueError(
                        f"mesh link '{link_name}' geometry cannot be loaded "
                        f"from {getattr(mesh, 'file_path', None)!r}")
                skipped.append(link_name)
                continue
            hull = source.convex_hull
            if hull.vertices.size == 0 or hull.faces.size == 0:
                if strict:
                    raise ValueError(
                        f"mesh link '{link_name}' has empty geometry")
                skipped.append(link_name)
                continue
            vertices = np.asarray(hull.vertices, dtype=np.float64)
            faces = np.asarray(hull.faces, dtype=np.int64)
            if self.margin > 0.0:
                corners = vertices[faces]
                area_normals = np.cross(
                    corners[:, 1] - corners[:, 0],
                    corners[:, 2] - corners[:, 0],
                )
                normals = np.zeros_like(vertices)
                for corner in range(3):
                    np.add.at(normals, faces[:, corner], area_normals)
                lengths = np.linalg.norm(normals, axis=1, keepdims=True)
                normals /= np.maximum(lengths, 1.0e-12)
                vertices = vertices + normals * self.margin
            geometry.append((
                torch.as_tensor(
                    vertices, dtype=torch.float32,
                    device=self.device).contiguous(),
                torch.as_tensor(
                    faces, dtype=torch.int64,
                    device=self.device).contiguous(),
            ))
            loaded_names.append(link_name)
        return loaded_names, geometry, skipped

    def _prepare_buffers(self, height, width):
        """Allocate shape-dependent buffers and warm both Warp kernels once."""
        if self._buffers is not None \
                and self._buffers['height'] == height \
                and self._buffers['width'] == width:
            return self._buffers

        stride = self.raster_stride
        raster_height = (height + stride - 1) // stride
        raster_width = (width + stride - 1) // stride
        faces = []
        face_link_ids = []
        vertex_link_ids = []
        offset = 0
        for link_id, (vertices, local_faces) in enumerate(self._geometry):
            faces.append(local_faces + offset)
            face_link_ids.append(torch.full(
                (local_faces.shape[0],), link_id,
                dtype=torch.int32, device=self.device))
            vertex_link_ids.append(torch.full(
                (vertices.shape[0],), link_id,
                dtype=torch.int32, device=self.device))
            offset += int(vertices.shape[0])
        faces64 = torch.cat(faces).contiguous()
        total_vertices = sum(int(vertices.shape[0])
                             for vertices, _ in self._geometry)
        buffers = {
            'height': height,
            'width': width,
            'raster_height': raster_height,
            'raster_width': raster_width,
            'faces64': faces64,
            'faces32': faces64.to(dtype=torch.int32).contiguous(),
            'face_link_ids': torch.cat(face_link_ids).contiguous(),
            'local_vertices': torch.cat(
                [vertices for vertices, _ in self._geometry]).contiguous(),
            'vertex_link_ids': torch.cat(vertex_link_ids).contiguous(),
            'link_transforms': torch.empty(
                (len(self._geometry), 4, 4),
                dtype=torch.float32, device=self.device),
            'camera_vertices': torch.empty(
                (total_vertices, 3), dtype=torch.float32, device=self.device),
            'projected': torch.empty(
                (total_vertices, 2), dtype=torch.float32, device=self.device),
            'raster_owner': torch.zeros(
                raster_height * raster_width,
                dtype=torch.int32, device=self.device),
            'mask': torch.empty(
                (height, width), dtype=torch.bool, device=self.device),
        }
        self._buffers = buffers
        self._warm_kernels()
        return buffers

    def _warm_kernels(self):
        """Compile kernels outside the first measured segmentation frame."""
        buffers = self._buffers
        buffers['projected'].zero_()
        buffers['camera_vertices'].zero_()
        buffers['link_transforms'].zero_()
        buffers['raster_owner'].zero_()
        stream = wp.stream_from_torch(
            torch.cuda.current_stream(device=self.device))
        wp.launch(
            _transform_project_vertices,
            dim=int(buffers['local_vertices'].shape[0]),
            inputs=[
                wp.from_torch(buffers['local_vertices'], dtype=wp.vec3),
                wp.from_torch(buffers['vertex_link_ids']),
                wp.from_torch(buffers['link_transforms']),
                wp.from_torch(buffers['camera_vertices'], dtype=wp.vec3),
                wp.from_torch(buffers['projected'], dtype=wp.vec2),
                1.0, 1.0, 0.0, 0.0, float(self.raster_stride),
            ],
            device=wp.device_from_torch(self.device),
            stream=stream,
        )
        wp.launch(
            _rasterize_projected_triangles,
            dim=int(buffers['faces32'].shape[0]),
            inputs=[
                wp.from_torch(buffers['projected'], dtype=wp.vec2),
                wp.from_torch(buffers['faces32']),
                wp.from_torch(buffers['face_link_ids']),
                wp.from_torch(buffers['raster_owner']),
                int(buffers['raster_width']),
                int(buffers['raster_height']),
            ],
            device=wp.device_from_torch(self.device),
            stream=stream,
        )
        torch.cuda.synchronize(self.device)

    def _camera_from_link(self, base_from_camera, positions, quaternions):
        """Fill camera<-link transforms from stamped base-frame link poses."""
        buffers = self._buffers
        base_from_camera = base_from_camera.to(
            device=self.device, dtype=torch.float32)
        positions = positions.to(device=self.device, dtype=torch.float32)
        quaternions = quaternions.to(device=self.device, dtype=torch.float32)
        if positions.shape != (len(self.link_names), 3):
            raise ValueError(
                f'link positions must have shape ({len(self.link_names)}, 3), '
                f'got {tuple(positions.shape)}')
        if quaternions.shape != (len(self.link_names), 4):
            raise ValueError(
                'link quaternions must have shape '
                f'({len(self.link_names)}, 4), '
                f'got {tuple(quaternions.shape)}')
        camera_rotation_inverse = base_from_camera[:3, :3].transpose(0, 1)
        camera_position = base_from_camera[:3, 3]
        link_rotation = _quaternion_to_rotation_matrix(quaternions)
        rotation = torch.matmul(
            camera_rotation_inverse.unsqueeze(0), link_rotation)
        translation = torch.matmul(
            camera_rotation_inverse,
            (positions - camera_position).transpose(0, 1),
        ).transpose(0, 1)
        transforms = buffers['link_transforms']
        transforms.zero_()
        transforms[:, :3, :3] = rotation
        transforms[:, :3, 3] = translation
        transforms[:, 3, 3] = 1.0
        return transforms

    def mask(self, depth_image, intrinsics, base_from_camera,
             link_positions, link_quaternions):
        """Return the valid-depth pixels covered by any projected link hull."""
        if depth_image.ndim != 2:
            raise ValueError(
                f'depth_image must have shape (H, W), got {depth_image.shape}')
        if depth_image.device != self.device:
            raise ValueError(
                f'depth image is on {depth_image.device}, backend is on '
                f'{self.device}')
        height, width = map(int, depth_image.shape)
        buffers = self._prepare_buffers(height, width)
        transforms = self._camera_from_link(
            base_from_camera, link_positions, link_quaternions)
        fx, fy, cx, cy = (float(value) for value in intrinsics)
        stream = wp.stream_from_torch(
            torch.cuda.current_stream(device=self.device))
        wp.launch(
            _transform_project_vertices,
            dim=int(buffers['local_vertices'].shape[0]),
            inputs=[
                wp.from_torch(buffers['local_vertices'], dtype=wp.vec3),
                wp.from_torch(buffers['vertex_link_ids']),
                wp.from_torch(transforms),
                wp.from_torch(buffers['camera_vertices'], dtype=wp.vec3),
                wp.from_torch(buffers['projected'], dtype=wp.vec2),
                fx, fy, cx, cy, float(self.raster_stride),
            ],
            device=wp.device_from_torch(self.device),
            stream=stream,
        )

        vertices = buffers['camera_vertices']
        valid_vertex = (
            (vertices[:, 2] > 1.0e-5)
            & torch.isfinite(vertices).all(dim=1)
        )
        valid_face = valid_vertex[buffers['faces64']].all(dim=1)
        if not bool(valid_face.any()):
            return buffers['mask'].zero_()
        faces = buffers['faces64'][valid_face]
        projected_faces = buffers['projected'][faces]
        bbox = projected_faces.max(dim=1).values \
            - projected_faces.min(dim=1).values
        face_area = bbox[:, 0] * bbox[:, 1]
        keep = (
            torch.isfinite(projected_faces).all(dim=(1, 2))
            & (face_area <= 500.0)
        )
        if not bool(keep.any()):
            return buffers['mask'].zero_()
        faces = faces[keep].to(dtype=torch.int32).contiguous()
        face_link_ids = buffers['face_link_ids'][valid_face][keep].contiguous()
        buffers['raster_owner'].zero_()
        wp.launch(
            _rasterize_projected_triangles,
            dim=int(faces.shape[0]),
            inputs=[
                wp.from_torch(buffers['projected'], dtype=wp.vec2),
                wp.from_torch(faces),
                wp.from_torch(face_link_ids),
                wp.from_torch(buffers['raster_owner']),
                int(buffers['raster_width']),
                int(buffers['raster_height']),
            ],
            device=wp.device_from_torch(self.device),
            stream=stream,
        )
        owner = buffers['raster_owner'].view(
            buffers['raster_height'], buffers['raster_width'])
        owner = torch.nn.functional.max_pool2d(
            torch.nn.functional.pad(
                owner.to(dtype=torch.float32)[None, None], (0, 1, 0, 1)),
            kernel_size=2,
            stride=1,
        )[0, 0]
        full = owner.repeat_interleave(
            self.raster_stride, dim=0).repeat_interleave(
                self.raster_stride, dim=1)[:height, :width]
        buffers['mask'].copy_(full > 0)
        buffers['mask'].logical_and_(depth_image > 0.0)
        return buffers['mask']
