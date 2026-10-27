"""DepthMapCameraStrategy: each depth image is posed with the TF at its own stamp."""

from builtin_interfaces.msg import Time
from curobo_ros.cameras.camera_depth_map_strategy import DepthMapCameraStrategy
from geometry_msgs.msg import TransformStamped
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import Image
import time
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


def camera(node, frame_id=CAMERA, extrinsics=None, tf_wait_max_s=0.05):
    return DepthMapCameraStrategy(
        node, 'cam', topic='/cam/depth', frame_id=frame_id, intrinsics=K, extrinsics=extrinsics,
        tf_wait_max_s=tf_wait_max_s)


def spin_for(node, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.01)


def tf_watches(strategy):
    """TF watches registered with the buffer: never more than one."""
    return len(strategy.tf_buffer._new_data_callbacks)


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
    for nanosec in range(0, 10**9, 10**8):  # TF is down: 10 frames time out
        strategy.callback_depth_map(depth(12, nanosec))
        spin_for(node, 0.06)

    assert node.mapper.observations == []
    assert tf_watches(strategy) == 1


def test_frame_waits_for_the_tf_at_its_stamp(node):
    """The image arrives before the TF sample after its stamp, as on a real station."""
    strategy = camera(node)
    publish_tf(strategy, 10, 0.1)

    strategy.callback_depth_map(depth(10, nanosec=500_000_000))
    assert node.mapper.observations == []
    publish_tf(strategy, 11, 0.3)
    spin_for(node, 0.02)

    assert integrated_x(node) == pytest.approx([0.2])
    assert tf_watches(strategy) == 0


def test_waiting_frame_goes_first_or_is_dropped_when_the_next_arrives(node):
    """Frames are integrated in acquisition order; a still-unposable one is dropped."""
    strategy = camera(node, tf_wait_max_s=10.0)
    publish_tf(strategy, 10, 0.1)

    strategy.callback_depth_map(depth(10, nanosec=500_000_000))
    publish_tf(strategy, 11, 0.3)  # no spin: the waiting frame is not integrated yet
    strategy.callback_depth_map(depth(10, nanosec=800_000_000))
    assert integrated_x(node) == pytest.approx([0.2, 0.26])

    strategy.callback_depth_map(depth(11, nanosec=500_000_000))
    strategy.callback_depth_map(depth(11, nanosec=600_000_000))
    publish_tf(strategy, 12, 0.5)
    spin_for(node, 0.02)
    assert integrated_x(node) == pytest.approx([0.2, 0.26, 0.42])
    assert tf_watches(strategy) == 0


def test_a_frame_older_than_any_tf_does_not_hold_up_the_next(node):
    """The camera can start before the robot's TF: no TF ever covers its first frame."""
    strategy = camera(node, tf_wait_max_s=10.0)
    publish_tf(strategy, 10, 0.1)

    strategy.callback_depth_map(depth(5))
    strategy.callback_depth_map(depth(10, nanosec=500_000_000))
    publish_tf(strategy, 11, 0.3)
    spin_for(node, 0.02)

    assert integrated_x(node) == pytest.approx([0.2])
    assert tf_watches(strategy) == 0


def test_tf_wait_zero_drops_at_once(node):
    strategy = camera(node, tf_wait_max_s=0)
    publish_tf(strategy, 10, 0.1)

    strategy.callback_depth_map(depth(10, nanosec=500_000_000))
    publish_tf(strategy, 11, 0.3)
    spin_for(node, 0.02)

    assert node.mapper.observations == []
    assert tf_watches(strategy) == 0


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


def test_frame_id_and_tf_wait_are_validated(node):
    with pytest.raises(ValueError, match='frame_id'):
        camera(node, frame_id='')
    with pytest.raises(ValueError, match='tf_wait_max_s'):
        camera(node, tf_wait_max_s=-0.1)
    with pytest.raises(ValueError, match='tf_wait_max_s'):
        camera(node, tf_wait_max_s='0.05')
