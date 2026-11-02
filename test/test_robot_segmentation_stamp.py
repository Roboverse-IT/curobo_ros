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
    markers = capture(node.sphere_marker_pub, monkeypatch, subscribed=True)

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


@pytest.mark.parametrize('param', [
    'segmentation_mode:=unknown',
    'mesh_margin:=-0.1',
    'mesh_raster_stride:=0',
])
def test_mesh_configuration_is_validated(param):
    rclpy.init(args=['--ros-args', '-p', param])
    try:
        with pytest.raises(ValueError, match=param.split(':=')[0]):
            DepthMapRobotSegmentation()
    finally:
        rclpy.shutdown()


@pytest.mark.parametrize('node', [[
    'robot:=ur10e_xrdf',
    'robot_base_frame:=base_link',
    'segmentation_mode:=mesh',
    'mesh_link_names:=[wrist_3_link]',
]], indirect=True)
def test_mesh_backend_loads_selected_robot_geometry(node):
    assert node._segmentation_mode == 'mesh'
    assert node._mesh_link_names == ['wrist_3_link']
    assert node._mesh_segmenter.link_names == ['wrist_3_link']
    assert node._mesh_segmenter._geometry[0][0].shape[0] > 3


def test_mesh_and_combined_modes_select_their_backends(node, monkeypatch):
    class FakeMesh:
        pass

    def center_pixel(depth_image, camera_intrinsics, kinematics_state):
        mask = torch.zeros_like(depth_image, dtype=torch.bool)
        mask[depth_image.shape[0] // 2, depth_image.shape[1] // 2] = True
        return mask

    camera_info(node)
    msg = depth()
    node.depth_header = msg.header
    node._frame_transforms = {
        CAMERA: torch.eye(4, dtype=torch.float32, device='cuda')}
    node._mesh_segmenter = FakeMesh()
    monkeypatch.setattr(node, '_mesh_mask_from_state', center_pixel)
    node.distance_threshold = 100.0
    image = torch.full((48, 64), 0.8, dtype=torch.float32, device='cuda')
    q = torch.zeros(
        len(node._kin_model.joint_names), dtype=torch.float32, device='cuda')

    node._segmentation_mode = 'mesh'
    mesh_only = node._mask_depth_image(image, q)
    node._segmentation_mode = 'combined'
    combined = node._mask_depth_image(image, q)

    assert torch.count_nonzero(mesh_only == 0.0) == 1
    assert torch.count_nonzero(combined) == 0


def test_mask_preserves_every_unmasked_input_pixel(node, monkeypatch):
    camera_info(node)
    node.depth_header = depth().header
    node._frame_transforms = {
        CAMERA: torch.eye(4, dtype=torch.float32, device='cuda')}
    node._mesh_segmenter = None
    node.distance_threshold = -100.0  # all sphere distances are retained
    image = (0.5 + torch.arange(48 * 64, device='cuda', dtype=torch.float32)
             .reshape(48, 64) * 0.0001)
    original = image.clone()
    q = torch.zeros(
        len(node._kin_model.joint_names), dtype=torch.float32, device='cuda')

    def center_pixel(depth_image, camera_intrinsics, kinematics_state):
        mask = torch.zeros_like(depth_image, dtype=torch.bool)
        mask[24, 32] = True
        return mask

    monkeypatch.setattr(node, '_mesh_mask_from_state', center_pixel)
    node._segmentation_mode = 'sphere'
    assert torch.equal(node._mask_depth_image(image, q), original)
    node._mesh_segmenter = object()
    node._segmentation_mode = 'combined'
    expected = original.clone()
    expected[24, 32] = 0.0
    assert torch.equal(node._mask_depth_image(image, q), expected)
    assert torch.equal(image, original)

    node._segmentation_mode = 'mesh'
    monkeypatch.setattr(node, 'depth_to_pointcloud',
                        lambda *args: pytest.fail('mesh-only mode built a point cloud'))
    assert torch.equal(node._mask_depth_image(image, q), expected)


def test_sphere_mask_matches_broadcast_distance_reference(node):
    node.depth_header = Header(frame_id=CAMERA)
    node._frame_transforms = {
        CAMERA: torch.eye(4, dtype=torch.float32, device='cuda')}
    q = torch.zeros(
        len(node._kin_model.joint_names), dtype=torch.float32, device='cuda')
    state = node._kin_model.compute_kinematics(CuroboJointState(
        position=q.unsqueeze(0), joint_names=node._kin_model.joint_names))
    spheres = state.robot_spheres.reshape(-1, 4)
    # Exercise both sides of the configured mask threshold around multiple
    # current collision spheres, with a base==camera transform for this unit.
    offsets = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], device='cuda')
    points = (spheres[:8, :3].unsqueeze(1) + offsets).reshape(-1, 3)

    actual = node._mask_pointcloud(points, state)
    reference = torch.min(
        torch.norm(points.unsqueeze(1) - spheres[:, :3].unsqueeze(0), dim=2)
        - spheres[:, 3].unsqueeze(0), dim=1).values > node.distance_threshold

    assert torch.equal(actual, reference)


def test_depth_message_preserves_recorded_integer_millimetres(node):
    # The moving station bag contains 502, 506, and 511 mm readings. Their
    # float32 metre representations fall just below the integer on * 1000.
    node.depth_header = depth().header
    millimetres = np.array([[0, 502, 506, 511]], dtype=np.uint16)
    metres = torch.as_tensor(
        millimetres.astype(np.float32) / 1000.0, device='cuda')

    message = node.depth_tensor_to_image_msg(metres)

    assert message.encoding == '16UC1'
    np.testing.assert_array_equal(
        np.frombuffer(message.data, dtype=np.uint16).reshape(1, 4),
        millimetres)


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


FRANKA_DESCRIPTOR = """name: franka_test
curobo_config: franka.yml
strategy: emulator
strategy_params:
  joint_states_topic: /franka_test/joint_states
"""
FINGERS = ['panda_finger_joint1', 'panda_finger_joint2']  # locked at 0.04 in franka.yml


@pytest.fixture
def franka(tmp_path, request):
    """Run the node on cuRobo's Franka, whose model locks the two finger joints."""
    descriptor = tmp_path / 'franka_test.yaml'
    descriptor.write_text(FRANKA_DESCRIPTOR + getattr(request, 'param', ''))
    rclpy.init(args=['--ros-args', '-p', f'robot:={descriptor}'])
    node = DepthMapRobotSegmentation()
    yield node
    node.destroy_node()
    rclpy.shutdown()


def franka_spheres(fingers):
    """Return the Franka's spheres at zero arm joints, its fingers locked at `fingers`."""
    from curobo.config_io import load_yaml
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo_ros.robot.robot_description import bundled_curobo_config
    cfg = load_yaml(bundled_curobo_config('franka.yml'))
    cfg['robot_cfg']['kinematics']['lock_joints'] = dict(zip(FINGERS, fingers))
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(cfg))
    return kin.compute_kinematics(CuroboJointState(
        position=torch.zeros((1, len(kin.joint_names)), device='cuda'),
        joint_names=kin.joint_names)).robot_spheres.reshape(-1, 4).cpu().numpy()


def published_centres(markers):
    return np.array([[m.pose.position.x, m.pose.position.y, m.pose.position.z]
                     for m in markers.markers])


def test_locked_joints_are_masked_at_their_values_at_the_depth_stamp(franka, monkeypatch):
    camera_info(franka)
    camera_tf(franka, 0, 1.0, static=True)
    names = list(franka._kin_model.joint_names) + FINGERS
    for t_ns, fingers in ((STAMP_NS - 2 * MS, 0.01), (STAMP_NS + 2 * MS, 0.03)):
        franka.listener_callback_joint_states(JointState(
            header=Header(stamp=stamp(t_ns)), name=names, position=[0.0] * 7 + [fingers] * 2))
    markers = capture(franka.sphere_marker_pub, monkeypatch, subscribed=True)

    franka.listener_callback_depth(depth())

    assert len(markers) == 1
    # The fingers at 0.02 (interpolated), not at the configured 0.04.
    configured, measured = franka_spheres([0.04] * 2)[:, :3], franka_spheres([0.02] * 2)[:, :3]
    assert np.abs(configured - measured).max() > 0.01
    np.testing.assert_allclose(published_centres(markers[0]), measured, atol=1e-4)


def test_locked_joints_keep_their_configured_values_until_measured(franka, monkeypatch):
    ready(franka)  # arm joints only
    markers = capture(franka.sphere_marker_pub, monkeypatch, subscribed=True)

    franka.listener_callback_depth(depth())

    assert len(markers) == 1
    np.testing.assert_allclose(published_centres(markers[0]), franka_spheres([0.04] * 2)[:, :3],
                               atol=1e-4)


def test_a_frame_without_the_measured_locked_joints_at_its_stamp_is_dropped(franka, monkeypatch):
    ready(franka)  # the arm joints around the stamp
    franka.listener_callback_joint_states(JointState(  # the fingers only well before it
        header=Header(stamp=stamp(STAMP_NS - 100 * MS)), name=FINGERS, position=[0.02] * 2))
    images = capture(franka.publisher_, monkeypatch)
    logged = warnings(franka, monkeypatch)

    franka.listener_callback_depth(depth())
    spin_for(franka, 0.15)

    assert images == []
    assert any(f'No {FINGERS} positions at the depth stamp' in w for w in logged)


# A driver publishing the fingers under its own names, in mm of opening.
DRIVER_FINGERS = """joint_states:
  names: {panda_finger_joint1: left_mm, panda_finger_joint2: right_mm}
  conversions:
    panda_finger_joint1: {reported: [0.0, 80.0], model: [0.0, 0.04]}
    panda_finger_joint2: {reported: [0.0, 80.0], model: [0.0, 0.04]}
"""


@pytest.mark.parametrize('franka', [DRIVER_FINGERS], indirect=True)
def test_locked_joints_are_read_through_the_descriptor_joint_states_map(franka, monkeypatch):
    camera_info(franka)
    camera_tf(franka, 0, 1.0, static=True)
    names = list(franka._kin_model.joint_names) + ['left_mm', 'right_mm']
    for t_ns in (STAMP_NS - 2 * MS, STAMP_NS + 2 * MS):
        franka.listener_callback_joint_states(JointState(
            header=Header(stamp=stamp(t_ns)), name=names, position=[0.0] * 7 + [40.0] * 2))
    markers = capture(franka.sphere_marker_pub, monkeypatch, subscribed=True)

    franka.listener_callback_depth(depth())

    assert len(markers) == 1
    np.testing.assert_allclose(published_centres(markers[0]), franka_spheres([0.02] * 2)[:, :3],
                               atol=1e-4)


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
