"""GPU object depth mask parity with dynamic_scene.object_segmenter."""

import sys

import numpy as np
import pytest
import torch

from curobo_ros.perception.object_depth_mask import ObjectDepthMasker, ObjectGeometry


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

sys.path.insert(0, '/ws/dynamic_scene/src')
from object_segmenter import ObjectDepthSegmenter, ObjectGeometry as ReferenceGeometry  # noqa: E402


K = np.array([[80.0, 0.0, 8.0], [0.0, 80.0, 8.0], [0.0, 0.0, 1.0]])
CAMERA = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
POSE = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def gpu_mask(geometry, depth, pose=POSE, **kwargs):
    masker = ObjectDepthMasker(**kwargs)
    masker.set_object('object', geometry, pose=pose)
    mask = masker.mask_depth(
        torch.as_tensor(depth, device='cuda'),
        torch.as_tensor(K, device='cuda'), CAMERA, stamp_ns=123)
    return mask.cpu().numpy()


def reference_mask(geometry, depth, pose=POSE, **kwargs):
    reference = ObjectDepthSegmenter(
        ReferenceGeometry.mesh(geometry.vertices, geometry.faces),
        object_pose_base=pose, **kwargs)
    return reference.mask_depth(depth, K, CAMERA)


@pytest.mark.parametrize('geometry', [
    ObjectGeometry.cuboid([0.1, 0.1, 0.1]),
    ObjectGeometry.mesh(
        [[-0.05, -0.05, 0.0], [0.05, -0.05, 0.0],
         [0.05, 0.05, 0.0], [-0.05, 0.05, 0.0]],
        [[0, 1, 2], [0, 2, 3]]),
])
def test_cuboid_and_mesh_match_standalone_for_fronto_parallel_surface(geometry):
    """The direct comparison is exact where both interpolants are equivalent."""
    front = ObjectDepthSegmenter(
        ReferenceGeometry.mesh(geometry.vertices, geometry.faces),
        object_pose_base=POSE, depth_tolerance_m=0.01, dilation_px=0)._front_depth(K, CAMERA, (17, 17))
    depth = np.full((17, 17), 2.0, dtype=np.float32)
    depth[np.isfinite(front)] = front[np.isfinite(front)]
    expected = reference_mask(geometry, depth, depth_tolerance_m=0.01, dilation_px=0)
    actual = gpu_mask(geometry, depth, depth_tolerance_m=0.01, dilation_px=0)
    np.testing.assert_array_equal(actual, expected)


def test_depth_consistency_preserves_background_and_foreground_occluder():
    geometry = ObjectGeometry.cuboid([0.1, 0.1, 0.1])
    front = ObjectDepthSegmenter(
        ReferenceGeometry.mesh(geometry.vertices, geometry.faces),
        object_pose_base=POSE, depth_tolerance_m=0.01, dilation_px=0)._front_depth(K, CAMERA, (17, 17))
    depth = np.full((17, 17), 2.0, dtype=np.float32)  # background survives
    pixels = np.argwhere(np.isfinite(front))
    assert len(pixels) > 3
    y, x = pixels[len(pixels) // 2]
    depth[y, x] = front[y, x]  # actual object is removed
    fy, fx = pixels[0]
    depth[fy, fx] = front[fy, fx] - 0.20  # foreground occluder survives
    mask = gpu_mask(geometry, depth, depth_tolerance_m=0.01, dilation_px=0)
    assert mask[y, x]
    assert not mask[fy, fx]
    assert not mask[0, 0]


def test_overlapping_triangles_choose_the_nearest_surface():
    geometry = ObjectGeometry.mesh(
        [[-0.06, -0.06, 0.0], [0.06, -0.06, 0.0], [0.0, 0.06, 0.0],
         [-0.06, -0.06, 0.20], [0.06, -0.06, 0.20], [0.0, 0.06, 0.20]],
        [[0, 1, 2], [3, 4, 5]])
    depth = np.full((17, 17), 2.0, dtype=np.float32)
    depth[8, 8] = 1.0
    mask = gpu_mask(geometry, depth, depth_tolerance_m=0.01, dilation_px=0)
    assert mask[8, 8]
    depth[8, 8] = 1.2
    assert not gpu_mask(geometry, depth, depth_tolerance_m=0.01, dilation_px=0)[8, 8]


def test_multiple_objects_dilation_and_unavailable_timestamp_pose():
    masker = ObjectDepthMasker(depth_tolerance_m=0.01, dilation_px=1)
    box = ObjectGeometry.cuboid([0.04, 0.04, 0.04])
    masker.set_object('left', box, pose=[-0.05, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    masker.set_object('right', box, pose=[0.05, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    masker.set_object('unavailable', box, pose_at_stamp=lambda stamp: None)
    depth = torch.full((17, 17), 2.0, dtype=torch.float32, device='cuda')
    # Both visible boxes have front camera-z 0.98 m.
    depth[8, 4] = depth[8, 12] = 0.98
    mask = masker.mask_depth(depth, torch.as_tensor(K, device='cuda'), CAMERA, 10).cpu()
    assert mask[8, 4] and mask[8, 12]
    assert mask[8, 3] and mask[8, 13]  # dilation
    assert not mask[0, 0]


def test_out_of_frustum_object_masks_nothing():
    depth = np.full((17, 17), 1.0, dtype=np.float32)
    mask = gpu_mask(ObjectGeometry.cuboid([0.1, 0.1, 0.1]), depth,
                    pose=[10.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    depth_tolerance_m=0.01, dilation_px=0)
    assert not mask.any()
