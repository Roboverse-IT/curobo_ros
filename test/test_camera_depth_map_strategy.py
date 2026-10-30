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


def camera(node, frame_id=CAMERA, extrinsics=None, tf_wait_max_s=0.05,
           depth_filter=None):
    return DepthMapCameraStrategy(
        node, 'cam', topic='/cam/depth', frame_id=frame_id, intrinsics=K, extrinsics=extrinsics,
        tf_wait_max_s=tf_wait_max_s, depth_filter=depth_filter)


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


def depth_float(values, sec=10):
    """Build a 32-bit floating-point depth image."""
    values = np.asarray(values, dtype=np.float32)
    msg = Image(
        height=values.shape[0], width=values.shape[1], encoding='32FC1',
        step=values.shape[1] * 4, data=values.tobytes())
    msg.header.stamp = Time(sec=sec)
    msg.header.frame_id = CAMERA
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


def test_depth_filter_matches_curobo_filter_depth(node):
    """The configured camera path must equal direct cuRobo filtering."""
    from curobo.perception import FilterDepth

    config = {
        'enabled': True,
        'depth_min': 0.15,
        'depth_max': 2.0,
        'flying_pixel_threshold': 0.5,
        'bilateral_kernel_size': 5,
        'bilateral_sigma_spatial': 10.0,
        'bilateral_sigma_depth': 0.1,
    }
    strategy = camera(
        node, extrinsics=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        depth_filter=config)
    raw = np.full((8, 8), 0.8, dtype=np.float32)
    raw[0, 0], raw[0, 1], raw[3, 3], raw[7, 7] = 0.05, 3.0, np.nan, 1.4

    strategy.callback_depth_map(depth_float(raw))

    direct = FilterDepth(
        image_shape=raw.shape,
        depth_minimum_distance=config['depth_min'],
        depth_maximum_distance=config['depth_max'],
        flying_pixel_threshold=config['flying_pixel_threshold'],
        bilateral_kernel_size=config['bilateral_kernel_size'],
        bilateral_sigma_spatial=config['bilateral_sigma_spatial'],
        bilateral_sigma_depth=config['bilateral_sigma_depth'],
        device='cuda')
    expected, _ = direct(torch.nan_to_num(
        torch.from_numpy(raw).to(device='cuda'), nan=0.0).unsqueeze(0))
    integrated = node.mapper.observations[-1].depth_image
    torch.testing.assert_close(integrated, expected)
    assert integrated[0, 0, 0] == 0.0
    assert integrated[0, 0, 1] == 0.0


@pytest.mark.parametrize('filter_options', [
    {'flying_pixel_threshold': None, 'bilateral_kernel_size': None},
    {'flying_pixel_threshold': 0.5, 'bilateral_kernel_size': None},
    {'flying_pixel_threshold': 0.5, 'bilateral_kernel_size': 5},
])
def test_depth_filter_keeps_segmentation_holes_invalid(node, filter_options):
    strategy = camera(
        node, extrinsics=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        depth_filter={'enabled': True, 'depth_min': 0.15, 'depth_max': 2.0,
                      **filter_options})
    masked = np.full((16, 16), 0.8, dtype=np.float32)
    masked[4:12, 4:12] = 0.0

    strategy.callback_depth_map(depth_float(masked))

    integrated = node.mapper.observations[-1].depth_image[0]
    assert torch.count_nonzero(integrated[4:12, 4:12]) == 0


def test_depth_filter_is_disabled_by_default(node):
    """Omitting the filter must preserve every source depth value."""
    strategy = camera(
        node, extrinsics=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    raw = np.array([[0.05, 3.0], [np.nan, 0.8]], dtype=np.float32)

    strategy.callback_depth_map(depth_float(raw))

    integrated = node.mapper.observations[-1].depth_image.cpu().numpy()[0]
    np.testing.assert_allclose(integrated, raw, equal_nan=True)


def test_object_masker_runs_before_filter_and_uses_image_stamp(node):
    """The mapper boundary accepts a generic post-robot object-mask stage."""
    class Masker:
        def __init__(self):
            self.stamps = []

        def apply(self, image, intrinsics, pose, stamp_ns):
            self.stamps.append(stamp_ns)
            out = image.clone()
            out[0, 0] = 0.0
            return out, torch.zeros_like(image, dtype=torch.bool)

    node.object_depth_masker = Masker()
    strategy = camera(
        node, extrinsics=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        depth_filter={'enabled': True, 'depth_min': 0.15, 'depth_max': 2.0,
                      'flying_pixel_threshold': None, 'bilateral_kernel_size': None})
    strategy.callback_depth_map(depth_float(np.full((8, 8), 0.8, dtype=np.float32), sec=12))
    assert node.object_depth_masker.stamps == [12_000_000_000]
    assert node.mapper.observations[-1].depth_image[0, 0, 0] == 0.0


def test_generic_object_masker_removes_only_matching_depth(node):
    from curobo_ros.perception import ObjectDepthMasker, ObjectGeometry

    node.object_depth_masker = ObjectDepthMasker(
        depth_tolerance_m=0.01, dilation_px=0)
    node.object_depth_masker.set_object(
        'payload', ObjectGeometry.cuboid([0.001, 0.001, 0.02]),
        pose=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    strategy = camera(
        node, extrinsics=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    strategy.callback_depth_map(depth_float(np.full((8, 8), 0.99, dtype=np.float32), sec=13))
    integrated = node.mapper.observations[-1].depth_image[0]
    assert 0 < torch.count_nonzero(integrated) < integrated.numel()


def test_temporal_decay_uses_shared_accepted_image_time(node):
    """Interleaved cameras share one elapsed-time decay clock on their mapper."""
    node.declare_parameter('decay_half_life_s', 2.0)

    class Config:
        time_decay = None

    class TimedMapper:
        def __init__(self):
            self.integrator = type('Integrator', (), {'config': Config()})()

    node.mapper = TimedMapper()
    first, second = camera(node), camera(node)
    first._set_temporal_decay(node.mapper, depth(10))
    assert node.mapper.integrator.config.time_decay == 1.0
    second._set_temporal_decay(node.mapper, depth(11))
    assert node.mapper.integrator.config.time_decay == pytest.approx(2 ** -0.5)
    first._set_temporal_decay(node.mapper, depth(11))
    assert node.mapper.integrator.config.time_decay == 1.0
    second._set_temporal_decay(node.mapper, depth(15))
    assert node.mapper.integrator.config.time_decay == pytest.approx(2 ** -2.0)


@pytest.mark.parametrize('config, match', [
    ({'enabled': 'yes'}, 'enabled'),
    ({'depth_min': 2.0, 'depth_max': 1.0}, 'depth_max'),
    ({'flying_pixel_threshold': 1.1}, 'flying_pixel_threshold'),
    ({'bilateral_kernel_size': 4}, 'bilateral_kernel_size'),
    ({'unknown': 1}, 'unknown depth_filter keys'),
])
def test_depth_filter_configuration_is_validated(node, config, match):
    """Reject unsafe or unsupported FilterDepth settings at startup."""
    with pytest.raises(ValueError, match=match):
        camera(node, depth_filter=config)


def test_camera_yaml_routes_depth_filter_to_the_strategy(node, tmp_path):
    """Load the complete per-camera filter mapping from cameras.yaml."""
    from curobo_ros.core.camera_system_manager import CameraSystemManager

    config = tmp_path / 'cameras.yaml'
    config.write_text(f"""cameras:
  - name: filtered
    type: depth_camera
    topic: /filtered/depth
    frame_id: {CAMERA}
    intrinsics: {K}
    extrinsics: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    frame_rate_hz: 30.0
    depth_filter:
      enabled: true
      depth_min: 0.2
      depth_max: 1.5
      flying_pixel_threshold: null
      bilateral_kernel_size: null
""")

    manager = CameraSystemManager(node, str(config))

    strategy = manager.camera_context.cameras['filtered']
    assert strategy._depth_filter_config == {
        'enabled': True,
        'depth_min': 0.2,
        'depth_max': 1.5,
        'flying_pixel_threshold': None,
        'bilateral_kernel_size': None,
        'bilateral_sigma_spatial': 10.0,
        'bilateral_sigma_depth': 0.1,
    }
