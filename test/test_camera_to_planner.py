"""A depth camera's obstacle reaches the planner: depth -> Mapper -> ESDF -> collision decision."""

import threading
import time

from curobo.types import JointState as CuroboJointState
from curobo_msgs.srv import GetCollisionDistance, SetPlanner, TrajectoryGeneration
from curobo_ros.core.unified_planner_node import UnifiedPlannerNode
from geometry_msgs.msg import TransformStamped
import numpy as np
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from sensor_msgs.msg import Image, JointState
from tf2_msgs.msg import TFMessage
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

CAMERA = 'test_camera_optical_frame'
WIDTH, HEIGHT, FOCAL = 160, 120, 120.0
DEPTH_M, HALF_SIDE_M = 0.8, 0.1  # a 0.2 m square, 0.8 m below the camera

# Joint-space goals of the UR10e: Q_IN puts a wrist sphere where the camera
# sees the square, the other two keep every sphere more than 0.7 m from it.
Q_IN = [0.0, -1.2, 1.8, -2.2, -1.57, 0.0]
Q_START = [1.57, -1.2, 1.8, -2.2, -1.57, 0.0]
Q_OUT = [3.0, -1.2, 1.8, -2.2, -1.57, 0.0]

CAMERAS_YAML = f"""cameras:
  - name: test_camera
    type: depth_camera
    topic: /test_camera/depth
    frame_id: {CAMERA}
    intrinsics: [{FOCAL}, 0.0, {WIDTH / 2}, 0.0, {FOCAL}, {HEIGHT / 2}, 0.0, 0.0, 1.0]
    extrinsics: []
    frame_rate_hz: 30.0
"""


class Planner:
    """The planner node and a client of its services, each spun on its own thread."""

    def __init__(self, cameras_yaml, robot='ur10e_xrdf', extra_params=()):
        rclpy.init(args=['--ros-args', '-p', f'robot:={robot}',
                         '-p', f'cameras_config_file:={cameras_yaml}',
                         *extra_params])
        self.node = UnifiedPlannerNode()
        self.client = rclpy.create_node('camera_to_planner_client')
        self._node_executor = MultiThreadedExecutor()
        self._node_executor.add_node(self.node)
        self._node_thread = threading.Thread(
            target=self._node_executor.spin, daemon=True)
        self._node_thread.start()
        self._client_executor = SingleThreadedExecutor()  # spun by call()
        self._client_executor.add_node(self.client)
        self._distance = self.client.create_client(
            GetCollisionDistance, '/unified_planner/get_collision_distance')
        self._plan = self.client.create_client(
            TrajectoryGeneration, '/unified_planner/generate_trajectory')
        self._set_planner = self.client.create_client(
            SetPlanner, '/unified_planner/set_planner')

        deadline = time.monotonic() + 180.0
        while not self.node.get_parameter('node_is_available').value:
            assert time.monotonic() < deadline, 'the planner did not start'
            time.sleep(0.1)
        self.joint_names = list(self.node.config_wrapper_motion.kin_model.joint_names)

    def destroy(self):
        self._node_executor.shutdown(timeout_sec=5.0)
        self._node_thread.join(timeout=5.0)
        if self._node_thread.is_alive():
            raise RuntimeError('planner node executor did not stop')
        self._client_executor.shutdown()
        self.client.destroy_node()
        self.node.destroy_node()
        rclpy.shutdown()

    def call(self, client, request):
        assert client.wait_for_service(timeout_sec=10.0)
        future = client.call_async(request)
        self._client_executor.spin_until_future_complete(future, timeout_sec=60.0)
        assert future.done(), f'no answer from {client.srv_name}'
        return future.result()

    def joint_state(self, q):
        return JointState(name=self.joint_names, position=[float(x) for x in q])

    def distance(self, *configurations):
        """Return the largest sphere collision cost of each configuration (0 when clear)."""
        response = self.call(self._distance, GetCollisionDistance.Request(
            trajectory=[self.joint_state(q) for q in configurations]))
        data = np.array(response.data).reshape(response.nb_config, response.nb_sphere)
        return data.max(axis=1).tolist()

    def plan(self, goal):
        """Plan in joint space from Q_START to `goal`."""
        return self.call(self._plan, TrajectoryGeneration.Request(
            start_pose=self.joint_state(Q_START),
            target_joint_positions=[float(x) for x in goal]))

    def use_joint_space_planner(self):
        response = self.call(self._set_planner, SetPlanner.Request(
            planner_type=SetPlanner.Request.JOINT_SPACE))
        assert response.success, response.message

    def outermost_sphere(self, q):
        """Return the active collision sphere (x, y, z, r) farthest from the base axis at `q`."""
        kin = self.node.config_wrapper_motion.kin_model
        state = kin.compute_kinematics(CuroboJointState(
            position=torch.tensor([q], dtype=torch.float32, device='cuda'),
            joint_names=self.joint_names))
        spheres = state.robot_spheres.reshape(-1, 4).cpu().numpy()
        spheres = spheres[spheres[:, 3] > 0]
        return spheres[np.argmax(np.linalg.norm(spheres[:, :2], axis=1))]


class Camera:
    """Streams the TF of a camera looking down at `target` and its depth image of a square."""

    def __init__(self, base_frame, target):
        self.node = rclpy.create_node('camera_to_planner_camera')
        self._tf = self.node.create_publisher(TFMessage, '/tf', 10)
        self._depth = self.node.create_publisher(Image, '/test_camera/depth', 10)
        self._camera = TransformStamped()
        self._camera.header.frame_id, self._camera.child_frame_id = base_frame, CAMERA
        translation = self._camera.transform.translation
        translation.x, translation.y, translation.z = (
            float(target[0]), float(target[1]), float(target[2]) + DEPTH_M)
        rotation = self._camera.transform.rotation
        rotation.x, rotation.w = 1.0, 0.0  # optical z (forward) along the base -z

        image = np.zeros((HEIGHT, WIDTH), np.uint16)
        r = int(HALF_SIDE_M / DEPTH_M * FOCAL)
        image[HEIGHT // 2 - r:HEIGHT // 2 + r, WIDTH // 2 - r:WIDTH // 2 + r] = DEPTH_M * 1000
        self._image = Image(height=HEIGHT, width=WIDTH, encoding='16UC1', step=2 * WIDTH,
                            data=image.tobytes())
        self._image.header.frame_id = CAMERA

        self.node.create_timer(1.0 / 30.0, self._publish)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        threading.Thread(target=self._executor.spin, daemon=True).start()

    def _publish(self):
        stamp = self.node.get_clock().now().to_msg()
        self._camera.header.stamp = self._image.header.stamp = stamp
        self._tf.publish(TFMessage(transforms=[self._camera]))
        self._depth.publish(self._image)

    def destroy(self):
        self._executor.shutdown()
        self.node.destroy_node()


@pytest.fixture(scope='module')
def planner(tmp_path_factory):
    cameras_yaml = tmp_path_factory.mktemp('camera_to_planner') / 'cameras.yaml'
    cameras_yaml.write_text(CAMERAS_YAML)
    planner = Planner(cameras_yaml)
    planner.use_joint_space_planner()
    yield planner
    planner.destroy()


@pytest.fixture(scope='module')
def camera(planner):
    """Stream the camera, from once the planner has integrated a few of its frames."""
    camera = Camera(planner.node.get_parameter('base_link').value,
                    planner.outermost_sphere(Q_IN))
    deadline = time.monotonic() + 10.0
    while planner.node.mapper.get_stats()['frame_count'] < 15:
        assert time.monotonic() < deadline, 'the depth frames were not integrated'
        time.sleep(0.05)
    while planner.node._esdf_prewarm.snapshot()[0] != 'ready':
        assert time.monotonic() < deadline, 'ESDF pre-warm did not finish'
        time.sleep(0.05)
    yield camera
    camera.destroy()


def test_camera_less_collision_world_is_empty(planner):
    """No mapper integration means no ESDF pre-warm or change to query behavior."""
    assert planner.node._esdf_prewarm.snapshot()[0] == 'waiting_for_integration'
    assert planner.distance(Q_IN, Q_OUT) == [0.0, 0.0]


def test_collision_distance_sees_the_camera_obstacle(planner, camera):
    """The pre-warmed ESDF reaches the collision service before any plan."""
    inside, outside = planner.distance(Q_IN, Q_OUT)

    assert inside > 0.0
    assert outside == 0.0


def test_a_goal_in_the_camera_obstacle_is_rejected(planner, camera):
    response = planner.plan(Q_IN)

    assert not response.success
    assert 'world contact' in response.message


def test_a_goal_clear_of_the_camera_obstacle_is_planned_clear_of_it(planner, camera):
    response = planner.plan(Q_OUT)

    assert response.success, response.message
    trajectory = [list(point.position) for point in response.trajectory]
    assert max(planner.distance(*trajectory)) == 0.0
