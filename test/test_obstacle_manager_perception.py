"""ObstacleManager perception: the Mapper is built from the node parameters."""

from curobo.types import CameraObservation, DeviceCfg, Pose
from curobo_ros.core.obstacle_manager import ObstacleManager
from curobo_ros.perception import ObjectDepthMasker, ObjectGeometry
import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


class Logger:
    """Records the warnings logged through node.get_logger()."""

    def __init__(self):
        self.warnings = []

    def warn(self, message, **kwargs):
        self.warnings.append(message)

    def info(self, message, **kwargs):
        pass

    def debug(self, message, **kwargs):
        pass


@pytest.fixture
def make_node():
    rclpy.init()
    nodes = []

    def make(overrides=()):
        node = rclpy.create_node('obstacle_manager_perception_test',
                                 parameter_overrides=list(overrides))
        logger = Logger()
        node.get_logger = lambda: logger
        nodes.append(node)
        return node, logger

    yield make
    for node in nodes:
        node.destroy_node()
    rclpy.shutdown()


def observation(height, width):
    pose = Pose.from_list([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                          device_cfg=DeviceCfg(device='cuda'))  # 1 m up, looking down
    intrinsics = torch.tensor([[600.0, 0.0, width / 2], [0.0, 600.0, height / 2],
                               [0.0, 0.0, 1.0]], device='cuda').unsqueeze(0)
    return CameraObservation(
        depth_image=torch.full((1, height, width), 0.8, device='cuda'),
        rgb_image=torch.zeros((1, height, width, 3), dtype=torch.uint8, device='cuda'),
        intrinsics=intrinsics, pose=pose)


def test_one_mapper_takes_depth_frames_of_any_size(make_node):
    """No image size is configured: cameras of different resolutions share the Mapper."""
    node, _ = make_node()
    manager = ObstacleManager(node, None)
    manager.setup_perception(num_cameras=2, total_frame_rate_hz=60.0)

    for height, width in [(480, 640), (720, 1280), (480, 640)]:
        node.mapper.integrate(observation(height, width))
    torch.cuda.synchronize()

    stats = node.mapper.get_stats()
    assert stats['frame_count'] == 3
    assert stats['active_blocks'] > 0
    assert stats['allocation_failures'] == 0


def test_voxel_grid_contains_the_camera_voxels(make_node):
    """A depth frame of a plane 0.8 m below a camera at 1 m fills the grid layer at z=0.2 m."""
    node, logger = make_node()
    manager = ObstacleManager(node, None)
    manager.setup_perception(num_cameras=1, total_frame_rate_hz=30.0)
    node.mapper.integrate(observation(480, 640))
    torch.cuda.synchronize()

    voxel_size = 0.05
    voxel_grid, grid_min, size = manager._compute_dense_voxel_grid(node, voxel_size)

    assert logger.warnings == []
    occupied = np.argwhere(voxel_grid)
    assert len(occupied) > 0
    z = grid_min[2] + (occupied[:, 2] + 0.5) * voxel_size
    assert abs(float(np.median(z)) - 0.2) <= voxel_size


def test_removed_parameters_are_reported(make_node):
    """Overrides of undeclared parameters are otherwise kept by rclpy without a word."""
    node, logger = make_node([Parameter('mapper_image_width', value=1280),
                              Parameter('mapper_image_height', value=720),
                              Parameter('decay_factor', value=0.9),
                              Parameter('mapper_voxel_size', value=0.02)])
    ObstacleManager(node, None)

    assert len(logger.warnings) == 3
    assert "'decay_factor'" in logger.warnings[0]
    assert "'mapper_image_height'" in logger.warnings[1]
    assert "'mapper_image_width'" in logger.warnings[2]
    assert not node.has_parameter('mapper_image_width')


def test_frustum_decay_is_forwarded_to_the_native_mapper(make_node):
    node, _ = make_node([Parameter('frustum_decay_factor', value=0.95)])
    manager = ObstacleManager(node, None)
    manager.setup_perception(num_cameras=1, total_frame_rate_hz=30.0)

    assert manager.mapper.config.frustum_decay_factor == pytest.approx(0.95)


def test_tsdf_invalidation_rejects_bad_bounds_and_handles_empty_map(make_node):
    node, _ = make_node()
    manager = ObstacleManager(node, None)
    manager.setup_perception(num_cameras=1, total_frame_rate_hz=30.0)

    with pytest.raises(ValueError, match='invalidation bounds'):
        manager.invalidate_tsdf_region([1, 0, 0], [0, 1, 1])
    assert manager.invalidate_tsdf_region([0, 0, 0], [1, 1, 1]) == 0


def test_tsdf_invalidation_clears_only_selected_sparse_voxels(make_node):
    node, _ = make_node()
    manager = ObstacleManager(node, None)
    manager.setup_perception(num_cameras=1, total_frame_rate_hz=30.0)
    node.mapper.integrate(observation(64, 64))
    torch.cuda.synchronize()
    data = node.mapper.tsdf.data
    before = data.block_data.clone()
    n = int(data.num_allocated.item())
    active = torch.nonzero(data.block_to_hash_slot[:n] >= 0).reshape(-1)
    assert len(active) > 0
    pool_idx = int(active[0].item())
    coords = data.block_coords[:n * 3].view(n, 3)
    # Clear a single voxel at the centre of an allocated sparse block.
    block = coords[pool_idx].float()
    size = float(data.voxel_size)
    origin = data.origin.float()
    if data.grid_shape is None:
        point = origin + (block * 8.0 + 0.5) * size
    else:
        shape = torch.tensor(
            (data.grid_shape[2], data.grid_shape[1], data.grid_shape[0]),
            device=block.device, dtype=torch.float32)
        point = origin + (block * 8.0 + 0.5 - shape / 2.0) * size
    cleared = manager.invalidate_tsdf_region((point - size * 0.1).cpu(),
                                              (point + size * 0.1).cpu())
    assert cleared == 1
    assert torch.count_nonzero(data.block_data[pool_idx, 1:] != before[pool_idx, 1:]) == 0
    unchanged = torch.arange(n, device='cuda') != pool_idx
    assert torch.count_nonzero(data.block_data[:n][unchanged] != before[:n][unchanged]) == 0
    assert float(data.block_sums[pool_idx].item()) == pytest.approx(
        float(data.block_data[pool_idx, :, 1].sum(dtype=torch.float32).item()))
    assert node.mapper._last_voxel_grid is None


def test_masked_object_is_not_reintegrated_after_local_invalidation(make_node):
    """The generic mask + existing invalidator remove a payload without erasing
    the mapper globally; an all-masked follow-up cannot recreate it."""
    node, _ = make_node()
    manager = ObstacleManager(node, None)
    manager.setup_perception(num_cameras=1, total_frame_rate_hz=30.0)
    raw = observation(64, 64)
    # The camera is 1 m above base looking down.  This broad cuboid's top face
    # is at camera-z 0.775 m and fills this small field of view.
    raw.depth_image.fill_(0.775)
    masker = ObjectDepthMasker(depth_tolerance_m=0.01, dilation_px=0)
    masker.set_object('payload', ObjectGeometry.cuboid([0.2, 0.2, 0.05]),
                      pose=[0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])
    masked, mask = masker.apply(raw.depth_image[0], raw.intrinsics[0], raw.pose)
    assert bool(mask.all())

    node.mapper.integrate(raw)
    torch.cuda.synchronize()
    assert node.mapper.get_stats()['active_blocks'] > 0
    cleared = manager.invalidate_tsdf_region([-0.11, -0.11, 0.14],
                                              [0.11, 0.11, 0.26])
    assert cleared > 0
    data = node.mapper.tsdf.data
    n = int(data.num_allocated.item())
    active = data.block_to_hash_slot[:n] >= 0
    pool = torch.arange(n, device='cuda')[active]
    blocks = data.block_coords[:n * 3].view(n, 3)[active].to(torch.int64)
    local = torch.arange(512, device='cuda', dtype=torch.int64)
    xyz = torch.stack((local % 8, (local // 8) % 8, local // 64), dim=1)
    voxels = blocks[:, None, :] * 8 + xyz[None, :, :]
    shape = torch.as_tensor(
        (data.grid_shape[2], data.grid_shape[1], data.grid_shape[0]), device='cuda')
    centers = data.origin.float()[None, None, :] + (
        voxels.float() + 0.5 - shape / 2.0) * float(data.voxel_size)
    region = ((centers >= torch.tensor([-0.11, -0.11, 0.14], device='cuda'))
              & (centers <= torch.tensor([0.11, 0.11, 0.26], device='cuda'))).all(dim=-1)
    assert int(region.sum()) == cleared
    assert torch.count_nonzero(data.block_data[pool][region]) == 0

    raw.depth_image = masked.unsqueeze(0)
    node.mapper.integrate(raw)
    torch.cuda.synchronize()
    # Mapper bookkeeping/decay can change unrelated blocks, but a zero-depth
    # image must not put any TSDF value back into the invalidated payload area.
    assert torch.count_nonzero(data.block_data[pool][region]) == 0
