"""Robot segmentation: a depth frame is masked with the TF and joints at its stamp, or dropped."""

import time

from builtin_interfaces.msg import Time
from curobo.types import JointState as CuroboJointState
from curobo_msgs.srv import SetMask
from curobo_ros.core.robot_segmentation import DepthMapRobotSegmentation
from geometry_msgs.msg import TransformStamped
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

CAMERA = 'camera_optical_frame'
STAMP_NS = 1234_567_000_000
MS = 1_000_000


def stamp(t_ns):
    return Time(sec=t_ns // 10**9, nanosec=t_ns % 10**9)


@pytest.fixture
def node(request):
    params = getattr(request, 'param', [])
    rclpy.init(args=['--ros-args'] + [a for p in params for a in ('-p', p)])
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


def base(node):
    return node.get_parameter('robot_base_frame').value


def camera_tf(node, t_ns, z, static=False):
    """Camera z m above the base, looking down."""
    tf = TransformStamped()
    tf.header.frame_id, tf.child_frame_id = base(node), CAMERA
    tf.header.stamp = stamp(t_ns)
    tf.transform.translation.z = z
    tf.transform.rotation.x, tf.transform.rotation.w = 1.0, 0.0  # 180 deg about x
    if static:
        node.tf_buffer.set_transform_static(tf, 'test')
    else:
        node.tf_buffer.set_transform(tf, 'test')


def joints(node, t_ns, q, names=None):
    names = list(node._kin_model.joint_names) if names is None else names
    node.listener_callback_joint_states(JointState(
        header=Header(stamp=stamp(t_ns)), name=names, position=[float(v) for v in q]))


def depth(t_ns=STAMP_NS):
    return Image(header=Header(stamp=stamp(t_ns), frame_id=CAMERA), height=48, width=64,
                 encoding='32FC1', step=64 * 4,
                 data=np.full((48, 64), 0.8, dtype=np.float32).tobytes())


def camera_info(node):
    node.listener_callback_camera_info(CameraInfo(
        width=64, height=48, k=[60.0, 0.0, 32.0, 0.0, 60.0, 24.0, 0.0, 0.0, 1.0]))


def mask_all(node, frame_id=''):
    """Mask the whole view, so every depth point goes to the debug cloud."""
    mask = SetMask.Request(type=SetMask.Request.CUBOID, name='all', frame_id=frame_id)
    mask.pose.orientation.w = 1.0
    mask.dimensions.x = mask.dimensions.y = mask.dimensions.z = 10.0
    assert node.set_mask_callback(mask, SetMask.Response()).success


def cloud_z(cloud):
    return np.frombuffer(bytes(cloud.data), dtype=np.float32).reshape(-1, 3)[:, 2]


def spin_for(node, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.005)


def warnings(node, monkeypatch):
    logged = []
    logger = node.get_logger()
    monkeypatch.setattr(logger, 'warn', lambda msg, **kw: logged.append(msg))
    return logged


def ready(node, static_camera=True):
    """Feed camera info and the joints around the stamp, and the camera TF unless told not to."""
    camera_info(node)
    if static_camera:
        camera_tf(node, 0, 1.0, static=True)
    n = len(node._kin_model.joint_names)
    joints(node, STAMP_NS - 2 * MS, [0.0] * n)
    joints(node, STAMP_NS + 2 * MS, [0.0] * n)


def test_spheres_are_at_the_joints_at_the_depth_stamp(node, monkeypatch):
    camera_info(node)
    camera_tf(node, 0, 1.0, static=True)
    names = list(node._kin_model.joint_names)
    q = np.linspace(0.1, 0.6, len(names))
    # Reversed, with an extra joint: a positional read would mix the joints up.
    joints(node, STAMP_NS - 2 * MS, list(np.zeros(len(names))) + [0.7],
           names=names[::-1] + ['finger_joint'])
    joints(node, STAMP_NS + 2 * MS, list(2 * q[::-1]) + [0.7],
           names=names[::-1] + ['finger_joint'])
    markers = capture(node.sphere_marker_pub, monkeypatch)

    node.listener_callback_depth(depth())

    assert len(markers) == 1
    spheres = node._kin_model.compute_kinematics(CuroboJointState(
        position=torch.tensor(q, dtype=torch.float32, device='cuda').unsqueeze(0),
        joint_names=names)).robot_spheres.reshape(-1, 4).cpu().numpy()
    published = np.array([[m.pose.position.x, m.pose.position.y, m.pose.position.z]
                          for m in markers[0].markers])
    np.testing.assert_allclose(published, spheres[:, :3], atol=1e-4)


def test_the_camera_pose_is_the_tf_at_the_depth_stamp(node, monkeypatch):
    ready(node, static_camera=False)
    camera_tf(node, STAMP_NS - 100 * MS, 1.0)
    camera_tf(node, STAMP_NS + 100 * MS, 1.2)
    mask_all(node)
    clouds = capture(node.robot_pointcloud_pub, monkeypatch, subscribed=True)

    node.listener_callback_depth(depth())

    assert len(clouds) == 1
    # Camera at 1.1 m at the stamp (1.2 m latest), depth 0.8 m straight down.
    np.testing.assert_allclose(np.median(cloud_z(clouds[0])), 0.3, atol=1e-3)


def test_a_frame_without_the_camera_tf_at_its_stamp_is_dropped(node, monkeypatch):
    ready(node, static_camera=False)
    camera_tf(node, STAMP_NS - 100 * MS, 1.0)  # only before the stamp
    images = capture(node.publisher_, monkeypatch)
    logged = warnings(node, monkeypatch)

    node.listener_callback_depth(depth())
    spin_for(node, 0.15)

    assert images == []
    assert any(f'No TF {CAMERA}' in w and 'Dropping this depth frame' in w for w in logged)


def test_a_frame_without_a_mask_tf_at_its_stamp_is_dropped(node, monkeypatch):
    ready(node)
    mask_all(node, frame_id='gripper')  # no TF for it
    images = capture(node.publisher_, monkeypatch)
    logged = warnings(node, monkeypatch)

    node.listener_callback_depth(depth())
    spin_for(node, 0.15)

    assert images == []
    assert any('No TF gripper' in w and 'Dropping this depth frame' in w for w in logged)


def test_a_frame_waits_for_the_joints_at_its_stamp(node, monkeypatch):
    camera_info(node)
    camera_tf(node, 0, 1.0, static=True)
    n = len(node._kin_model.joint_names)
    joints(node, STAMP_NS - 2 * MS, [0.0] * n)
    images = capture(node.publisher_, monkeypatch)

    node.listener_callback_depth(depth())
    spin_for(node, 0.01)
    assert images == []  # waiting

    joints(node, STAMP_NS + 2 * MS, [0.0] * n)
    spin_for(node, 0.01)
    assert len(images) == 1
    assert images[0].header.stamp == stamp(STAMP_NS)


def test_a_frame_without_joints_at_its_stamp_is_dropped(node, monkeypatch):
    camera_info(node)
    camera_tf(node, 0, 1.0, static=True)
    joints(node, STAMP_NS - 2 * MS, [0.0] * len(node._kin_model.joint_names))
    images = capture(node.publisher_, monkeypatch)
    logged = warnings(node, monkeypatch)

    node.listener_callback_depth(depth())
    spin_for(node, 0.15)

    assert images == []
    assert any('No joint positions at the depth stamp' in w for w in logged)


def test_a_frame_in_a_joint_state_gap_is_dropped(node, monkeypatch):
    camera_info(node)
    camera_tf(node, 0, 1.0, static=True)
    n = len(node._kin_model.joint_names)
    joints(node, STAMP_NS - 40 * MS, [0.0] * n)
    joints(node, STAMP_NS + 40 * MS, [0.0] * n)  # 80 ms apart, max 50 ms
    images = capture(node.publisher_, monkeypatch)
    logged = warnings(node, monkeypatch)

    node.listener_callback_depth(depth())

    assert images == []
    assert any('0.080 s apart' in w for w in logged)


@pytest.mark.parametrize('node', [['stamp_wait_max_s:=0.0']], indirect=True)
def test_stamp_wait_zero_drops_at_once(node, monkeypatch):
    camera_info(node)
    camera_tf(node, 0, 1.0, static=True)
    joints(node, STAMP_NS - 2 * MS, [0.0] * len(node._kin_model.joint_names))
    images = capture(node.publisher_, monkeypatch)
    logged = warnings(node, monkeypatch)

    node.listener_callback_depth(depth())

    assert images == []
    assert any('No joint positions at the depth stamp' in w for w in logged)


@pytest.mark.parametrize('param', ['stamp_wait_max_s:=-0.1', 'joint_state_max_gap_s:=0.0'])
def test_wait_and_gap_are_validated(param):
    rclpy.init(args=['--ros-args', '-p', param])
    try:
        with pytest.raises(ValueError, match=param.split(':=')[0]):
            DepthMapRobotSegmentation()
    finally:
        rclpy.shutdown()


def test_tf_and_joint_states_are_stored_while_the_node_is_busy(node):
    # Nothing spins the node, as while it segments a frame: TF and joint
    # states still arrive, on their own threads.
    sender = rclpy.create_node('sender')
    try:
        tf_pub = sender.create_publisher(TFMessage, '/tf', 100)
        js_pub = sender.create_publisher(
            JointState, node.subscription_joint_states.topic_name, 100)
        tf = TransformStamped()
        tf.header.frame_id, tf.child_frame_id = base(node), CAMERA
        tf.header.stamp = stamp(STAMP_NS)
        tf.transform.rotation.w = 1.0
        names = list(node._kin_model.joint_names)
        at_stamp = rclpy.time.Time(nanoseconds=STAMP_NS)
        end = time.monotonic() + 5.0
        while time.monotonic() < end and not (
                node.tf_buffer.can_transform(base(node), CAMERA, at_stamp)
                and node._joint_history.covers(at_stamp)):
            tf_pub.publish(TFMessage(transforms=[tf]))
            js_pub.publish(JointState(header=Header(stamp=stamp(STAMP_NS)), name=names,
                                      position=[0.0] * len(names)))
            time.sleep(0.05)
    finally:
        sender.destroy_node()

    assert node.tf_buffer.can_transform(base(node), CAMERA, at_stamp)
    assert node._joint_history.covers(at_stamp)


def test_destroy_node_stops_the_listener_threads():
    rclpy.init()
    try:
        node = DepthMapRobotSegmentation()
        threads = [node._tf_thread._thread, node._joint_state_thread._thread]
        assert all(t.is_alive() for t in threads)
        node.destroy_node()
        assert not any(t.is_alive() for t in threads)
    finally:
        rclpy.shutdown()
