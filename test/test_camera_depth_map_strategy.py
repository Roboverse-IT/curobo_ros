"""DepthMapCameraStrategy: each depth image is posed with the TF at its own stamp."""

from builtin_interfaces.msg import Time
from curobo_ros.cameras.camera_depth_map_strategy import DepthMapCameraStrategy
from geometry_msgs.msg import TransformStamped
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import Image
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

BASE, CAMERA = 'base', 'camera_optical_frame'
K = [500.0, 0.0, 2.0, 0.0, 500.0, 2.0, 0.0, 0.0, 1.0]


class Mapper:
    """Records the observations the strategy integrates."""

    def __init__(self):
        self.observations = []

    def integrate(self, observation):
        self.observations.append(observation)


@pytest.fixture
def node():
    rclpy.init()
    node = rclpy.create_node('camera_strategy_test')
    node.declare_parameter('base_link', BASE)
    node.mapper = Mapper()
    yield node
    node.destroy_node()
    rclpy.shutdown()


def camera(node, frame_id=CAMERA, extrinsics=None):
    return DepthMapCameraStrategy(
        node, 'cam', topic='/cam/depth', frame_id=frame_id, intrinsics=K, extrinsics=extrinsics)


def publish_tf(strategy, sec, x):
    t = TransformStamped()
    t.header.stamp = Time(sec=sec)
    t.header.frame_id, t.child_frame_id = BASE, CAMERA
    t.transform.translation.x = x
    t.transform.rotation.w = 1.0
    strategy.tf_buffer.set_transform(t, 'test')


def depth(sec, nanosec=0, frame_id=CAMERA):
    msg = Image(height=4, width=4, encoding='16UC1', step=8)
    msg.data = np.full((4, 4), 1000, dtype=np.uint16).tobytes()
    msg.header.stamp = Time(sec=sec, nanosec=nanosec)
    msg.header.frame_id = frame_id
    return msg


def integrated_x(node):
    return [o.pose.position.reshape(-1)[0].item() for o in node.mapper.observations]


def test_pose_is_the_transform_at_the_image_stamp(node):
    """Not the latest transform: the camera moved after the image was taken."""
    strategy = camera(node)
    publish_tf(strategy, 10, 0.1)
    publish_tf(strategy, 11, 0.3)

    strategy.callback_depth_map(depth(10))
    strategy.callback_depth_map(depth(10, nanosec=500_000_000))

    assert integrated_x(node) == pytest.approx([0.1, 0.2])


def test_frames_without_a_pose_at_their_stamp_are_dropped(node):
    """Zero stamp (tf2 would return the latest) and a stamp past the TF buffer."""
    strategy = camera(node)
    publish_tf(strategy, 10, 0.1)
    publish_tf(strategy, 11, 0.3)

    strategy.callback_depth_map(depth(0))
    strategy.callback_depth_map(depth(12))

    assert node.mapper.observations == []


def test_frames_from_another_frame_are_dropped(node):
    """With TF and with static extrinsics, which both describe the configured frame."""
    strategy = camera(node)
    publish_tf(strategy, 10, 0.1)
    static = camera(node, extrinsics=[0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    strategy.callback_depth_map(depth(10, frame_id='other_frame'))
    static.callback_depth_map(depth(10, frame_id='other_frame'))
    assert node.mapper.observations == []

    static.callback_depth_map(depth(10))
    assert integrated_x(node) == pytest.approx([0.5])


def test_frame_id_is_required(node):
    with pytest.raises(ValueError, match='frame_id'):
        camera(node, frame_id='')
