
"""
Auto-generated test file
Generated from YAML specification
DO NOT EDIT - Changes will be overwritten
"""

import unittest
import pytest
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rosidl_runtime_py import set_message_fields
import launch
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node as LaunchNode
import launch_testing
import launch_testing.actions
import time
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from curobo_msgs.srv import WarmupFK
from curobo_msgs.srv import Fk
from curobo_msgs.srv import WarmupIK
from curobo_msgs.srv import IkBatch
from curobo_msgs.srv import TrajectoryGeneration

@launch_testing.ready_to_test_action_timeout(90.0)
@pytest.mark.launch_test
def generate_test_description():
    """Generate launch description for test."""
    
    launch_file_0_path = PathJoinSubstitution([
        FindPackageShare('curobo_ros'),
        'launch',
        'gen_traj_test_dual_arm.launch.py'
    ])

    launch_file_0 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file_0_path)
    )
    
    return LaunchDescription([
        launch_file_0,
        launch_testing.util.KeepAliveProc(),
        launch_testing.actions.ReadyToTest(),
    ]), locals()



class GeneratedTestSuite(unittest.TestCase):
    """Generated test class"""

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls._wait_until_ready()

    @classmethod
    def _wait_until_ready(cls):
        """Block until '/unified_planner/generate_trajectory' is advertised, or fail the suite.

        Graph introspection, never a call. A node that builds heavy
        state in its constructor advertises nothing until that work
        is done, so the service appearing in the graph *is* the
        readiness signal -- and because nothing is invoked, a service
        whose callback is broken cannot make this hang. Pick a
        service the node creates late in its startup.

        This replaces guessing with `startup_delay`, which raced
        machine load and failed on whichever suite happened to be
        slowest that run.
        """
        node = rclpy.create_node('readiness_probe')
        try:
            deadline = time.monotonic() + 180.0
            while time.monotonic() < deadline:
                advertised = [
                    name for name, _
                    in node.get_service_names_and_types()
                ]
                if '/unified_planner/generate_trajectory' in advertised:
                    return
                time.sleep(0.5)
            raise AssertionError(
                "System under test never became ready: "
                "'/unified_planner/generate_trajectory' was not advertised within 180.0s. "
                "Raise 'ready_wait' if startup is legitimately "
                "slower, otherwise the node failed to start -- "
                "check the launch output above.")
        finally:
            node.destroy_node()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_node')

    def tearDown(self):
        self.node.destroy_node()


    def test_01_warmup_fk(self):
        """01 warmup FK"""

        # Create service client
        client = self.node.create_client(WarmupFK, '/unified_planner/warmup_fk')

        # Wait for service to be available
        timeout = 10.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/warmup_fk' not available after {timeout}s")

        # Create request
        request = WarmupFK.Request()
        set_message_fields(request, {'batch_size': 1})

        # Call service
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)

        # Check if call completed
        if not future.done():
            self.fail("Service call to '/unified_planner/warmup_fk' timed out")

        # Get response
        response = future.result()
        if response is None:
            self.fail("Service call to '/unified_planner/warmup_fk' failed")


        self.assertEqual(
            response.success,
            True,
            f"Field 'success' doesn't match expected value"
        )

    def test_02_fk_of_both_tool_frames(self):
        """02 FK of both tool frames"""

        # Create service client
        client = self.node.create_client(Fk, '/unified_planner/fk')

        # Wait for service to be available
        timeout = 10.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/fk' not available after {timeout}s")

        # Create request
        request = Fk.Request()
        set_message_fields(request, {'joint_states': [{'position': [-1.57, -2.2, 1.9, -1.383, -1.57, 0.0, 1.57, -0.63, -1.9, -1.7, 1.57, 0.0]}], 'tool_frames': ['tool1', 'tool0']})

        # Call service
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)

        # Check if call completed
        if not future.done():
            self.fail("Service call to '/unified_planner/fk' timed out")

        # Get response
        response = future.result()
        if response is None:
            self.fail("Service call to '/unified_planner/fk' failed")


        self.assertEqual(
            len(list(response.poses)),
            2,
            f"Field 'poses' length doesn't match expected 2"
        )

    def test_03_fk_without_tool_frames_is_the_first_frame_only(self):
        """03 FK without tool_frames is the first frame only"""

        # Create service client
        client = self.node.create_client(Fk, '/unified_planner/fk')

        # Wait for service to be available
        timeout = 10.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/fk' not available after {timeout}s")

        # Create request
        request = Fk.Request()
        set_message_fields(request, {'joint_states': [{'position': [-1.57, -2.2, 1.9, -1.383, -1.57, 0.0, 1.57, -0.63, -1.9, -1.7, 1.57, 0.0]}]})

        # Call service
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)

        # Check if call completed
        if not future.done():
            self.fail("Service call to '/unified_planner/fk' timed out")

        # Get response
        response = future.result()
        if response is None:
            self.fail("Service call to '/unified_planner/fk' failed")


        self.assertEqual(
            len(list(response.poses)),
            1,
            f"Field 'poses' length doesn't match expected 1"
        )

    def test_04_warmup_ik(self):
        """04 warmup IK"""

        # Create service client
        client = self.node.create_client(WarmupIK, '/unified_planner/warmup_ik')

        # Wait for service to be available
        timeout = 10.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/warmup_ik' not available after {timeout}s")

        # Create request
        request = WarmupIK.Request()
        set_message_fields(request, {'batch_size': 1})

        # Call service
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)

        # Check if call completed
        if not future.done():
            self.fail("Service call to '/unified_planner/warmup_ik' timed out")

        # Get response
        response = future.result()
        if response is None:
            self.fail("Service call to '/unified_planner/warmup_ik' failed")


        self.assertEqual(
            response.success,
            True,
            f"Field 'success' doesn't match expected value"
        )

    def test_05_ik_of_both_tool_frames_eight_seeds_from_start(self):
        """05 IK of both tool frames, eight seeds, from START"""

        # Create service client
        client = self.node.create_client(IkBatch, '/unified_planner/ik_batch')

        # Wait for service to be available
        timeout = 90.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/ik_batch' not available after {timeout}s")

        # Create request
        request = IkBatch.Request()
        set_message_fields(request, {'poses': [{'position': {'x': 0.2728, 'y': -0.3256, 'z': 0.9043}, 'orientation': {'x': -0.706, 'y': 0.706, 'z': -0.0394, 'w': 0.0399}}, {'position': {'x': -0.0786, 'y': 0.3258, 'z': 0.8166}, 'orientation': {'x': -0.686, 'y': 0.6873, 'z': -0.1693, 'w': 0.1685}}], 'tool_frames': ['tool1', 'tool0'], 'return_seeds': 8, 'start_pose': {'position': [-1.57, -2.2, 1.9, -1.383, -1.57, 0.0, 1.57, -0.63, -1.9, -1.7, 1.57, 0.0]}})

        # Call service
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)

        # Check if call completed
        if not future.done():
            self.fail("Service call to '/unified_planner/ik_batch' timed out")

        # Get response
        response = future.result()
        if response is None:
            self.fail("Service call to '/unified_planner/ik_batch' failed")


        self.assertEqual(
            response.success,
            True,
            f"Field 'success' doesn't match expected value"
        )

        self.assertEqual(
            len(list(response.joint_states)),
            8,
            f"Field 'joint_states' length doesn't match expected 8"
        )

    def test_06_one_trajectory_for_both_tool_frames(self):
        """06 one trajectory for both tool frames"""

        # Create service client
        client = self.node.create_client(TrajectoryGeneration, '/unified_planner/generate_trajectory')

        # Wait for service to be available
        timeout = 90.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/generate_trajectory' not available after {timeout}s")

        # Create request
        request = TrajectoryGeneration.Request()
        set_message_fields(request, {'start_pose': {'position': [-1.57, -2.2, 1.9, -1.383, -1.57, 0.0, 1.57, -0.63, -1.9, -1.7, 1.57, 0.0]}, 'target_poses': [{'position': {'x': 0.2728, 'y': -0.3256, 'z': 0.9043}, 'orientation': {'x': -0.706, 'y': 0.706, 'z': -0.0394, 'w': 0.0399}}, {'position': {'x': -0.0786, 'y': 0.3258, 'z': 0.8166}, 'orientation': {'x': -0.686, 'y': 0.6873, 'z': -0.1693, 'w': 0.1685}}], 'target_tool_frames': ['tool1', 'tool0']})

        # Call service
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)

        # Check if call completed
        if not future.done():
            self.fail("Service call to '/unified_planner/generate_trajectory' timed out")

        # Get response
        response = future.result()
        if response is None:
            self.fail("Service call to '/unified_planner/generate_trajectory' failed")


        self.assertEqual(
            response.success,
            True,
            f"Field 'success' doesn't match expected value"
        )

    def test_07_a_goal_for_one_frame_of_two_is_refused(self):
        """07 a goal for one frame of two is refused"""

        # Create service client
        client = self.node.create_client(TrajectoryGeneration, '/unified_planner/generate_trajectory')

        # Wait for service to be available
        timeout = 10.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/generate_trajectory' not available after {timeout}s")

        # Create request
        request = TrajectoryGeneration.Request()
        set_message_fields(request, {'start_pose': {'position': [-1.57, -2.2, 1.9, -1.383, -1.57, 0.0, 1.57, -0.63, -1.9, -1.7, 1.57, 0.0]}, 'target_poses': [{'position': {'x': 0.2728, 'y': -0.3256, 'z': 0.9043}, 'orientation': {'x': -0.706, 'y': 0.706, 'z': -0.0394, 'w': 0.0399}}], 'target_tool_frames': ['tool1']})

        # Call service
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)

        # Check if call completed
        if not future.done():
            self.fail("Service call to '/unified_planner/generate_trajectory' timed out")

        # Get response
        response = future.result()
        if response is None:
            self.fail("Service call to '/unified_planner/generate_trajectory' failed")


        self.assertEqual(
            response.success,
            False,
            f"Field 'success' doesn't match expected value"
        )

        self.assertTrue(
            response.message.startswith("Planning error: target_tool_frames"),
            f"Field 'message' value {{  response.message  }} doesn't match pattern 'Planning error: target_tool_frames%'"
        )

@launch_testing.post_shutdown_test()
class PostShutdownTests(unittest.TestCase):
    """Post-shutdown tests to validate clean exit"""

    def test_exit_codes(self, proc_info):
        """Test that all processes exited without critical errors"""
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -9]
        )

