"""The joints a robot model locks (a gripper's) follow its joint states in every solver.

A franka driven through joint_pose: its joint states carry the fingers, which
cuRobo's franka.yml locks open. A small box sits where the open left finger is,
with the hand at the goal pose: open, the fingers are in the box; closed, they
are clear of it.
"""

import threading
import time

from curobo.types import JointState as CuroboJointState
from curobo_msgs.action import SendTrajectory
from curobo_msgs.srv import AddObject, GetCollisionDistance, TrajectoryGeneration, WarmupFK, \
    WarmupIK
from curobo_ros.core.unified_planner_node import UnifiedPlannerNode
from geometry_msgs.msg import Pose
import numpy as np
import pytest
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from sensor_msgs.msg import JointState
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

TOPIC = '/locked_joints_test'
DESCRIPTOR = f"""curobo_config: franka.yml
strategy: joint_pose
strategy_params:
  command_topic: {TOPIC}/command
  joint_states_topic: {TOPIC}/joint_states
"""
FINGERS = ['panda_finger_joint1', 'panda_finger_joint2']
OPEN, CLOSED = [0.04, 0.04], [0.0, 0.0]
GOAL = [0.0, -1.3, 0.0, -2.5, 0.0, 1.5, 0.8]   # franka.yml's default configuration
START = [0.8, -1.3, 0.0, -2.5, 0.0, 1.5, 0.8]  # the hand away from the box


class Planner:
    """The planner node, a client of its services and the robot's joint states."""

    def __init__(self, descriptor):
        rclpy.init(args=['--ros-args', '-p', f'robot:={descriptor}'])
        self.node = UnifiedPlannerNode()
        self.client = rclpy.create_node('locked_joints_test_client')
        self._node_executor = MultiThreadedExecutor()
        self._node_executor.add_node(self.node)
        threading.Thread(target=self._node_executor.spin, daemon=True).start()
        self._client_executor = SingleThreadedExecutor()  # spun by call()
        self._client_executor.add_node(self.client)
        self._joint_states = self.client.create_publisher(JointState, f'{TOPIC}/joint_states', 10)
        name = self.node.get_name()
        self._distance = self.client.create_client(
            GetCollisionDistance, f'/{name}/get_collision_distance')
        self._plan = self.client.create_client(
            TrajectoryGeneration, f'/{name}/generate_trajectory')
        self._add = self.client.create_client(AddObject, f'/{name}/add_object')
        self._warmup_ik = self.client.create_client(WarmupIK, f'/{name}/warmup_ik')
        self._warmup_fk = self.client.create_client(WarmupFK, f'/{name}/warmup_fk')
        self._execute = ActionClient(self.client, SendTrajectory, f'/{name}/execute_trajectory')

        deadline = time.monotonic() + 180.0
        while not self.node.get_parameter('node_is_available').value:
            assert time.monotonic() < deadline, 'the planner did not start'
            time.sleep(0.1)
        self.kin = self.node.config_wrapper_motion.kin_model
        self.joint_names = list(self.kin.joint_names)
        self.d = self.node.get_parameter('collision_activation_distance').value

    def destroy(self):
        self._node_executor.shutdown()
        self._client_executor.shutdown()
        self.client.destroy_node()
        self.node.destroy_node()
        rclpy.shutdown()

    def call(self, client, request):
        assert client.wait_for_service(timeout_sec=10.0)
        future = client.call_async(request)
        self._client_executor.spin_until_future_complete(future, timeout_sec=120.0)
        assert future.done(), f'no answer from {client.srv_name}'
        return future.result()

    def robot_at(self, fingers, arm=GOAL):
        """Publish the robot's joint states, until the node has them."""
        msg = JointState(name=self.joint_names + FINGERS, position=list(arm) + list(fingers))
        deadline = time.monotonic() + 5.0
        while self.node.robot_context.get_joint_positions(FINGERS) != fingers:
            assert time.monotonic() < deadline, 'the joint states did not arrive'
            self._joint_states.publish(msg)
            time.sleep(0.05)

    def distance(self, *configurations):
        """Return each sphere's collision cost (config x sphere), by default the robot's."""
        response = self.call(self._distance, GetCollisionDistance.Request(trajectory=[
            JointState(name=self.joint_names, position=list(q)) for q in configurations]))
        return np.array(response.data).reshape(response.nb_config, response.nb_sphere)

    def hand_pose(self, q):
        state = self.kin.compute_kinematics(CuroboJointState(
            position=torch.tensor([q], dtype=torch.float32, device='cuda'),
            joint_names=self.joint_names))
        return state.tool_poses.position.reshape(-1, 3)[0], \
            state.tool_poses.quaternion.reshape(-1, 4)[0], \
            state.robot_spheres.reshape(-1, 4)

    def plan_request(self):
        position, quaternion, _ = self.hand_pose(GOAL)
        target = Pose()
        target.position.x, target.position.y, target.position.z = position.tolist()
        (target.orientation.w, target.orientation.x, target.orientation.y,
         target.orientation.z) = quaternion.tolist()
        return JointState(name=self.joint_names, position=START), target

    def plan(self):
        start, target = self.plan_request()
        return self.call(self._plan, TrajectoryGeneration.Request(
            start_pose=start, target_pose=target))

    def execute(self):
        start, target = self.plan_request()
        assert self._execute.wait_for_server(timeout_sec=10.0)
        sent = self._execute.send_goal_async(SendTrajectory.Goal(
            start_pose=start, target_pose=target, allow_cached=True))
        self._client_executor.spin_until_future_complete(sent, timeout_sec=10.0)
        goal = sent.result()
        result = goal.get_result_async()
        self._client_executor.spin_until_future_complete(result, timeout_sec=120.0)
        if not result.done():
            # Still executing (nothing moves the robot): stop it, or teardown waits on it.
            self._client_executor.spin_until_future_complete(
                goal.cancel_goal_async(), timeout_sec=10.0)
            self._client_executor.spin_until_future_complete(result, timeout_sec=10.0)
            pytest.fail('execute_trajectory did not finish: it executed a plan')
        return result.result().result


@pytest.fixture(scope='module')
def planner(tmp_path_factory):
    descriptor = tmp_path_factory.mktemp('locked_joints') / 'franka_test.yaml'
    descriptor.write_text(DESCRIPTOR)
    planner = Planner(str(descriptor))
    # The box, at the open left finger (the model's configured opening).
    assert planner.node.locked_joints.values == pytest.approx(OPEN)
    kc = planner.kin.kinematics_config
    _, quaternion, spheres = planner.hand_pose(GOAL)
    left = kc.link_sphere_idx_map.reshape(-1) == kc.link_name_to_idx_map['panda_leftfinger']
    box = Pose()
    box.position.x, box.position.y, box.position.z = spheres[left, :3].mean(0).tolist()
    box.orientation.w, box.orientation.x, box.orientation.y, box.orientation.z = \
        quaternion.tolist()
    added_box = AddObject.Request(type=AddObject.Request.CUBOID, name='box', pose=box)
    added_box.dimensions.x = added_box.dimensions.y = added_box.dimensions.z = 0.012
    added = planner.call(planner._add, added_box)
    assert added.success, added.message
    yield planner
    planner.destroy()


def test_the_collision_query_sees_the_measured_fingers(planner):
    planner.robot_at(OPEN)
    current, at_goal = planner.distance()[0], planner.distance(GOAL)[0]
    np.testing.assert_array_equal(current, at_goal)
    assert current.max() > planner.d / 2  # a finger penetrates the box

    planner.robot_at(CLOSED)
    current, at_goal = planner.distance()[0], planner.distance(GOAL)[0]
    np.testing.assert_array_equal(current, at_goal)
    assert current.max() <= planner.d / 2  # nothing penetrates


def test_planning_sees_the_measured_fingers(planner):
    planner.robot_at(OPEN)
    assert not planner.plan().success

    planner.robot_at(CLOSED)
    response = planner.plan()
    assert response.success, response.message
    trajectory = [list(point.position) for point in response.trajectory]
    assert planner.distance(*trajectory).max() <= planner.d / 2


def test_a_cached_plan_is_not_reused_once_the_fingers_moved(planner):
    planner.robot_at(CLOSED)
    assert planner.plan().success  # cached for execute_trajectory

    planner.robot_at(OPEN)
    result = planner.execute()

    assert not result.success
    assert result.message.startswith('Planning failed'), result.message


def test_every_solver_takes_the_measured_fingers_built_before_or_after(planner):
    node = planner.node
    child = node.locked_joints._child
    canonical = planner.kin.kinematics_config.fixed_transforms

    def models():
        found = {label: s.kinematics.kinematics_config for label, s in node._solvers()}
        found['fk_services'] = node.fk_services._fk_model.kinematics_config
        return found

    planner.robot_at(OPEN)
    planner.distance()  # refreshes the solvers
    configured = canonical[child].clone()
    planner.robot_at(CLOSED)
    planner.distance()
    assert not torch.equal(canonical[child], configured)

    # Built from the robot YAML (the fingers open) after the fingers closed.
    assert planner.call(planner._warmup_ik, WarmupIK.Request(batch_size=1)).success
    assert planner.call(planner._warmup_fk, WarmupFK.Request(batch_size=1)).success
    mpc = node.planner_manager.get_planner('mpc')
    with node.gpu_lock:
        mpc.ensure_solver()
        mpc._ensure_ik_solver()
    assert {'motion_planner', 'ik_services', 'mpc', 'mpc_goal_ik', 'fk_services'} \
        <= set(models())
    for label, kc in models().items():
        assert torch.equal(kc.fixed_transforms[child], canonical[child]), label

    planner.robot_at(OPEN)
    planner.distance()
    assert torch.equal(canonical[child], configured)
    for label, kc in models().items():
        assert torch.equal(kc.fixed_transforms[child], canonical[child]), label
