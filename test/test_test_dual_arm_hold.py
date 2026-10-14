
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
from curobo_msgs.srv import TrajectoryGeneration
from curobo_msgs.srv import SetPlanner
from curobo_msgs.action import GraspPlan

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


    def test_01_tool1_moves_tool0_holds(self):
        """01 tool1 moves, tool0 holds"""

        # Create service client
        client = self.node.create_client(TrajectoryGeneration, '/unified_planner/generate_trajectory')

        # Wait for service to be available
        timeout = 90.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/generate_trajectory' not available after {timeout}s")

        # Create request
        request = TrajectoryGeneration.Request()
        set_message_fields(request, {'start_pose': {'position': [-1.57, -2.2, 1.9, -1.383, -1.57, 0.0, 1.57, -0.63, -1.9, -1.7, 1.57, 0.0]}, 'target_poses': [{'position': {'x': 0.2728, 'y': -0.3256, 'z': 0.9043}, 'orientation': {'x': -0.706, 'y': 0.706, 'z': -0.0394, 'w': 0.0399}}], 'target_tool_frames': ['tool1'], 'hold_tool_frames': ['tool0']})

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

    def test_02_a_frame_both_moving_and_holding_is_refused(self):
        """02 a frame both moving and holding is refused"""

        # Create service client
        client = self.node.create_client(TrajectoryGeneration, '/unified_planner/generate_trajectory')

        # Wait for service to be available
        timeout = 10.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/generate_trajectory' not available after {timeout}s")

        # Create request
        request = TrajectoryGeneration.Request()
        set_message_fields(request, {'start_pose': {'position': [-1.57, -2.2, 1.9, -1.383, -1.57, 0.0, 1.57, -0.63, -1.9, -1.7, 1.57, 0.0]}, 'target_poses': [{'position': {'x': 0.2728, 'y': -0.3256, 'z': 0.9043}, 'orientation': {'x': -0.706, 'y': 0.706, 'z': -0.0394, 'w': 0.0399}}], 'target_tool_frames': ['tool1'], 'hold_tool_frames': ['tool1']})

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

    def test_03_hold_tool_frames_without_target_tool_frames_is_refused(self):
        """03 hold_tool_frames without target_tool_frames is refused"""

        # Create service client
        client = self.node.create_client(TrajectoryGeneration, '/unified_planner/generate_trajectory')

        # Wait for service to be available
        timeout = 10.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/generate_trajectory' not available after {timeout}s")

        # Create request
        request = TrajectoryGeneration.Request()
        set_message_fields(request, {'start_pose': {'position': [-1.57, -2.2, 1.9, -1.383, -1.57, 0.0, 1.57, -0.63, -1.9, -1.7, 1.57, 0.0]}, 'target_pose': {'position': {'x': 0.2728, 'y': -0.3256, 'z': 0.9043}, 'orientation': {'x': -0.706, 'y': 0.706, 'z': -0.0394, 'w': 0.0399}}, 'hold_tool_frames': ['tool0']})

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
            response.message.startswith("Planning error: hold_tool_frames needs target_tool_frames"),
            f"Field 'message' value {{  response.message  }} doesn't match pattern 'Planning error: hold_tool_frames needs target_tool_frames%'"
        )

    def test_04_joint_space_planner(self):
        """04 joint-space planner"""

        # Create service client
        client = self.node.create_client(SetPlanner, '/unified_planner/set_planner')

        # Wait for service to be available
        timeout = 10.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/set_planner' not available after {timeout}s")

        # Create request
        request = SetPlanner.Request()
        set_message_fields(request, {'planner_type': 5})

        # Call service
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)

        # Check if call completed
        if not future.done():
            self.fail("Service call to '/unified_planner/set_planner' timed out")

        # Get response
        response = future.result()
        if response is None:
            self.fail("Service call to '/unified_planner/set_planner' failed")


        self.assertEqual(
            response.success,
            True,
            f"Field 'success' doesn't match expected value"
        )

    def test_05_joint_space_tool0_holds_with_its_joints_unchanged(self):
        """05 joint space, tool0 holds with its joints unchanged"""

        # Create service client
        client = self.node.create_client(TrajectoryGeneration, '/unified_planner/generate_trajectory')

        # Wait for service to be available
        timeout = 90.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/generate_trajectory' not available after {timeout}s")

        # Create request
        request = TrajectoryGeneration.Request()
        set_message_fields(request, {'start_pose': {'position': [-1.57, -2.2, 1.9, -1.383, -1.57, 0.0, 1.57, -0.63, -1.9, -1.7, 1.57, 0.0]}, 'target_joint_positions': [-1.37, -2.0, 1.9, -1.383, -1.57, 0.0, 1.57, -0.63, -1.9, -1.7, 1.57, 0.0], 'hold_tool_frames': ['tool0']})

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

    def test_06_joint_space_a_holding_arm_whose_target_moves_is_refused(self):
        """06 joint space, a holding arm whose target moves is refused"""

        # Create service client
        client = self.node.create_client(TrajectoryGeneration, '/unified_planner/generate_trajectory')

        # Wait for service to be available
        timeout = 10.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/generate_trajectory' not available after {timeout}s")

        # Create request
        request = TrajectoryGeneration.Request()
        set_message_fields(request, {'start_pose': {'position': [-1.57, -2.2, 1.9, -1.383, -1.57, 0.0, 1.57, -0.63, -1.9, -1.7, 1.57, 0.0]}, 'target_joint_positions': [-1.37, -2.0, 1.9, -1.383, -1.57, 0.0, 1.77, -0.63, -1.9, -1.7, 1.57, 0.0], 'hold_tool_frames': ['tool0']})

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
            response.message.startswith("Planning error: hold_tool_frames"),
            f"Field 'message' value {{  response.message  }} doesn't match pattern 'Planning error: hold_tool_frames%'"
        )

    def test_07_back_to_the_classic_planner(self):
        """07 back to the classic planner"""

        # Create service client
        client = self.node.create_client(SetPlanner, '/unified_planner/set_planner')

        # Wait for service to be available
        timeout = 10.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/set_planner' not available after {timeout}s")

        # Create request
        request = SetPlanner.Request()
        set_message_fields(request, {'planner_type': 0})

        # Call service
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)

        # Check if call completed
        if not future.done():
            self.fail("Service call to '/unified_planner/set_planner' timed out")

        # Get response
        response = future.result()
        if response is None:
            self.fail("Service call to '/unified_planner/set_planner' failed")


        self.assertEqual(
            response.success,
            True,
            f"Field 'success' doesn't match expected value"
        )

    def test_08_tool1_grasps_tool0_holds(self):
        """08 tool1 grasps, tool0 holds"""

        # Storage for feedback messages
        self.received_feedback = []

        def feedback_callback(feedback_msg):
            self.received_feedback.append(feedback_msg.feedback)

        # Create action client
        client = ActionClient(self.node, GraspPlan, '/unified_planner/plan_grasp')

        # Wait for the action server to be available
        timeout = 120.0
        if not client.wait_for_server(timeout_sec=timeout):
            self.fail(f"Action server '/unified_planner/plan_grasp' not available after {timeout}s")

        # Create and send the goal
        goal = GraspPlan.Goal()
        set_message_fields(goal, {'start_pose': {'position': [-1.57, -2.2, 1.9, -1.383, -1.57, 0.0, 1.57, -0.63, -1.9, -1.7, 1.57, 0.0]}, 'grasp_candidates': [{'position': {'x': 0.2728, 'y': -0.3256, 'z': 0.9043}, 'orientation': {'x': -0.706, 'y': 0.706, 'z': -0.0394, 'w': 0.0399}}], 'grasp_tool_frames': ['tool1'], 'hold_tool_frames': ['tool0'], 'approach_offset': -0.05, 'lift_offset': 0.05, 'lift_in_tool_frame': False, 'execute': False})

        send_future = client.send_goal_async(goal, feedback_callback=feedback_callback)
        rclpy.spin_until_future_complete(self.node, send_future, timeout_sec=timeout)

        goal_handle = send_future.result()
        if goal_handle is None:
            self.fail("Goal request to '/unified_planner/plan_grasp' timed out")
        self.assertTrue(goal_handle.accepted, "Goal was rejected by '/unified_planner/plan_grasp'")

        # Wait for the result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=timeout)

        if result_future.result() is None:
            self.fail("Action '/unified_planner/plan_grasp' did not return a result in time")

        result = result_future.result().result


        self.assertEqual(
            result.success,
            True,
            f"Field 'success' doesn't match expected value"
        )

        self.assertEqual(
            result.goal_index,
            0,
            f"Field 'goal_index' doesn't match expected value"
        )

    def test_09_a_grasp_on_a_two_frame_robot_needs_grasp_tool_frames(self):
        """09 a grasp on a two-frame robot needs grasp_tool_frames"""

        # Storage for feedback messages
        self.received_feedback = []

        def feedback_callback(feedback_msg):
            self.received_feedback.append(feedback_msg.feedback)

        # Create action client
        client = ActionClient(self.node, GraspPlan, '/unified_planner/plan_grasp')

        # Wait for the action server to be available
        timeout = 60.0
        if not client.wait_for_server(timeout_sec=timeout):
            self.fail(f"Action server '/unified_planner/plan_grasp' not available after {timeout}s")

        # Create and send the goal
        goal = GraspPlan.Goal()
        set_message_fields(goal, {'start_pose': {'position': [-1.57, -2.2, 1.9, -1.383, -1.57, 0.0, 1.57, -0.63, -1.9, -1.7, 1.57, 0.0]}, 'grasp_candidates': [{'position': {'x': 0.2728, 'y': -0.3256, 'z': 0.9043}, 'orientation': {'x': -0.706, 'y': 0.706, 'z': -0.0394, 'w': 0.0399}}], 'execute': False})

        send_future = client.send_goal_async(goal, feedback_callback=feedback_callback)
        rclpy.spin_until_future_complete(self.node, send_future, timeout_sec=timeout)

        goal_handle = send_future.result()
        if goal_handle is None:
            self.fail("Goal request to '/unified_planner/plan_grasp' timed out")
        self.assertTrue(goal_handle.accepted, "Goal was rejected by '/unified_planner/plan_grasp'")

        # Wait for the result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=timeout)

        if result_future.result() is None:
            self.fail("Action '/unified_planner/plan_grasp' did not return a result in time")

        result = result_future.result().result


        self.assertEqual(
            result.success,
            False,
            f"Field 'success' doesn't match expected value"
        )

        self.assertTrue(
            result.message.startswith("Grasp error: a robot with tool frames"),
            f"Field 'message' value {{  result.message  }} doesn't match pattern 'Grasp error: a robot with tool frames%'"
        )

    def test_10_every_tool_frame_grasps_or_holds(self):
        """10 every tool frame grasps or holds"""

        # Storage for feedback messages
        self.received_feedback = []

        def feedback_callback(feedback_msg):
            self.received_feedback.append(feedback_msg.feedback)

        # Create action client
        client = ActionClient(self.node, GraspPlan, '/unified_planner/plan_grasp')

        # Wait for the action server to be available
        timeout = 60.0
        if not client.wait_for_server(timeout_sec=timeout):
            self.fail(f"Action server '/unified_planner/plan_grasp' not available after {timeout}s")

        # Create and send the goal
        goal = GraspPlan.Goal()
        set_message_fields(goal, {'start_pose': {'position': [-1.57, -2.2, 1.9, -1.383, -1.57, 0.0, 1.57, -0.63, -1.9, -1.7, 1.57, 0.0]}, 'grasp_candidates': [{'position': {'x': 0.2728, 'y': -0.3256, 'z': 0.9043}, 'orientation': {'x': -0.706, 'y': 0.706, 'z': -0.0394, 'w': 0.0399}}], 'grasp_tool_frames': ['tool1'], 'execute': False})

        send_future = client.send_goal_async(goal, feedback_callback=feedback_callback)
        rclpy.spin_until_future_complete(self.node, send_future, timeout_sec=timeout)

        goal_handle = send_future.result()
        if goal_handle is None:
            self.fail("Goal request to '/unified_planner/plan_grasp' timed out")
        self.assertTrue(goal_handle.accepted, "Goal was rejected by '/unified_planner/plan_grasp'")

        # Wait for the result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=timeout)

        if result_future.result() is None:
            self.fail("Action '/unified_planner/plan_grasp' did not return a result in time")

        result = result_future.result().result


        self.assertEqual(
            result.success,
            False,
            f"Field 'success' doesn't match expected value"
        )

        self.assertTrue(
            result.message.startswith("Grasp error: grasp_tool_frames"),
            f"Field 'message' value {{  result.message  }} doesn't match pattern 'Grasp error: grasp_tool_frames%'"
        )

    def test_11_attach_object_name_is_refused_on_a_two_frame_grasp(self):
        """11 attach_object_name is refused on a two-frame grasp"""

        # Storage for feedback messages
        self.received_feedback = []

        def feedback_callback(feedback_msg):
            self.received_feedback.append(feedback_msg.feedback)

        # Create action client
        client = ActionClient(self.node, GraspPlan, '/unified_planner/plan_grasp')

        # Wait for the action server to be available
        timeout = 60.0
        if not client.wait_for_server(timeout_sec=timeout):
            self.fail(f"Action server '/unified_planner/plan_grasp' not available after {timeout}s")

        # Create and send the goal
        goal = GraspPlan.Goal()
        set_message_fields(goal, {'start_pose': {'position': [-1.57, -2.2, 1.9, -1.383, -1.57, 0.0, 1.57, -0.63, -1.9, -1.7, 1.57, 0.0]}, 'grasp_candidates': [{'position': {'x': 0.2728, 'y': -0.3256, 'z': 0.9043}, 'orientation': {'x': -0.706, 'y': 0.706, 'z': -0.0394, 'w': 0.0399}}], 'grasp_tool_frames': ['tool1'], 'hold_tool_frames': ['tool0'], 'attach_object_name': 'part', 'execute': False})

        send_future = client.send_goal_async(goal, feedback_callback=feedback_callback)
        rclpy.spin_until_future_complete(self.node, send_future, timeout_sec=timeout)

        goal_handle = send_future.result()
        if goal_handle is None:
            self.fail("Goal request to '/unified_planner/plan_grasp' timed out")
        self.assertTrue(goal_handle.accepted, "Goal was rejected by '/unified_planner/plan_grasp'")

        # Wait for the result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=timeout)

        if result_future.result() is None:
            self.fail("Action '/unified_planner/plan_grasp' did not return a result in time")

        result = result_future.result().result


        self.assertEqual(
            result.success,
            False,
            f"Field 'success' doesn't match expected value"
        )

        self.assertTrue(
            result.message.startswith("Grasp error: attach_object_name"),
            f"Field 'message' value {{  result.message  }} doesn't match pattern 'Grasp error: attach_object_name%'"
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

