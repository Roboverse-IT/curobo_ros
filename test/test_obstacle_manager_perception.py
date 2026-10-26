"""ObstacleManager perception: the Mapper is built from the node parameters."""

from curobo.types import CameraObservation, DeviceCfg, Pose
from curobo_ros.core.obstacle_manager import ObstacleManager
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
