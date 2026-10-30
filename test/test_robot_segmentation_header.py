"""Robot segmentation: everything published for a depth frame carries that frame's header."""

from builtin_interfaces.msg import Time
from curobo_msgs.srv import SetMask
from curobo_ros.core.robot_segmentation import DepthMapRobotSegmentation
from geometry_msgs.msg import TransformStamped
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Header
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

CAMERA = 'camera_optical_frame'
STAMP = Time(sec=1234, nanosec=567_000_000)


@pytest.fixture
def node():
    rclpy.init()
    node = DepthMapRobotSegmentation()
    yield node
    node.destroy_node()
    rclpy.shutdown()


def capture(publisher, monkeypatch, subscribed=False):
    sent = []
    monkeypatch.setattr(publisher, 'publish', sent.append)
    if subscribed:
        monkeypatch.setattr(publisher, 'get_subscription_count', lambda: 1)
    return sent


def test_outputs_carry_the_depth_header(node, monkeypatch):
    base = node.get_parameter('robot_base_frame').value
    camera_tf = TransformStamped()  # camera 1 m above the base, looking down
    camera_tf.header.frame_id, camera_tf.child_frame_id = base, CAMERA
    camera_tf.transform.translation.z = 1.0
    camera_tf.transform.rotation.x, camera_tf.transform.rotation.w = 1.0, 0.0  # 180 deg about x
    node.tf_buffer.set_transform_static(camera_tf, 'test')

    # A mask over the whole view, so the debug cloud has points to publish.
    mask = SetMask.Request(type=SetMask.Request.CUBOID, name='all')
    mask.pose.orientation.w = 1.0
    mask.dimensions.x = mask.dimensions.y = mask.dimensions.z = 4.0
    assert node.set_mask_callback(mask, SetMask.Response()).success

    images = capture(node.publisher_, monkeypatch)
    markers = capture(node.sphere_marker_pub, monkeypatch, subscribed=True)
    clouds = capture(node.robot_pointcloud_pub, monkeypatch, subscribed=True)

    # The joints around the depth stamp, as the robot driver publishes them.
    for t in (STAMP.nanosec - 2_000_000, STAMP.nanosec + 2_000_000):
        node.listener_callback_joint_states(JointState(
            header=Header(stamp=Time(sec=STAMP.sec, nanosec=t)),
            name=list(node._kin_model.joint_names),
            position=[0.0] * len(node._kin_model.joint_names)))
    node.listener_callback_camera_info(CameraInfo(
        width=64, height=48, k=[60.0, 0.0, 32.0, 0.0, 60.0, 24.0, 0.0, 0.0, 1.0]))
    depth = Image(header=Header(stamp=STAMP, frame_id=CAMERA), height=48, width=64,
                  encoding='32FC1', step=64 * 4,
                  data=np.full((48, 64), 0.8, dtype=np.float32).tobytes())
    node.listener_callback_depth(depth)

    assert len(images) == 1
    assert images[0].header == depth.header
    assert len(markers) == 1 and markers[0].markers
    assert all(m.header.stamp == STAMP and m.header.frame_id == base
               for m in markers[0].markers)
    assert len(clouds) == 1
    assert clouds[0].header.stamp == STAMP and clouds[0].header.frame_id == base
