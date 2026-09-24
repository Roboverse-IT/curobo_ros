
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
from curobo_msgs.srv import AddObject
from curobo_msgs.srv import AttachObject
from curobo_msgs.srv import RemoveObject

@launch_testing.ready_to_test_action_timeout(90.0)
@pytest.mark.launch_test
def generate_test_description():
    """Generate launch description for test."""
    
    launch_file_0_path = PathJoinSubstitution([
        FindPackageShare('curobo_ros'),
        'launch',
        'gen_traj_test_dual_arm_attach.launch.py'
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


    def test_01_add_a_cuboid(self):
        """01 add a cuboid"""

        # Create service client
        client = self.node.create_client(AddObject, '/unified_planner/add_object')

        # Wait for service to be available
        timeout = 90.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/add_object' not available after {timeout}s")

        # Create request
        request = AddObject.Request()
        set_message_fields(request, {'type': 0, 'name': 'table', 'pose': {'position': {'x': 1.5, 'y': 0.0, 'z': 0.2}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}, 'dimensions': {'x': 0.4, 'y': 0.4, 'z': 0.4}, 'color': {'r': 0.5, 'g': 0.5, 'b': 0.5, 'a': 1.0}})

        # Call service
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)

        # Check if call completed
        if not future.done():
            self.fail("Service call to '/unified_planner/add_object' timed out")

        # Get response
        response = future.result()
        if response is None:
            self.fail("Service call to '/unified_planner/add_object' failed")


        self.assertEqual(
            response.success,
            True,
            f"Field 'success' doesn't match expected value"
        )

    def test_02_add_the_part_for_arm_1(self):
        """02 add the part for arm 1"""

        # Create service client
        client = self.node.create_client(AddObject, '/unified_planner/add_object')

        # Wait for service to be available
        timeout = 90.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/add_object' not available after {timeout}s")

        # Create request
        request = AddObject.Request()
        set_message_fields(request, {'type': 5, 'name': 'part_1', 'mesh_file_path': '/home/ros2_ws/src/curobo_ros/config/config_test/cube_55mm.usda', 'pose': {'position': {'x': 1.4, 'y': 0.1, 'z': 0.43}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}, 'dimensions': {'x': 1.0, 'y': 1.0, 'z': 1.0}, 'color': {'r': 1.0, 'g': 0.8, 'b': 0.0, 'a': 1.0}})

        # Call service
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)

        # Check if call completed
        if not future.done():
            self.fail("Service call to '/unified_planner/add_object' timed out")

        # Get response
        response = future.result()
        if response is None:
            self.fail("Service call to '/unified_planner/add_object' failed")


        self.assertEqual(
            response.success,
            True,
            f"Field 'success' doesn't match expected value"
        )

    def test_03_add_the_part_for_arm_0(self):
        """03 add the part for arm 0"""

        # Create service client
        client = self.node.create_client(AddObject, '/unified_planner/add_object')

        # Wait for service to be available
        timeout = 90.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/add_object' not available after {timeout}s")

        # Create request
        request = AddObject.Request()
        set_message_fields(request, {'type': 5, 'name': 'part_0', 'mesh_file_path': '/home/ros2_ws/src/curobo_ros/config/config_test/cube_55mm.usda', 'pose': {'position': {'x': 1.4, 'y': -0.1, 'z': 0.43}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}}, 'dimensions': {'x': 1.0, 'y': 1.0, 'z': 1.0}, 'color': {'r': 1.0, 'g': 0.8, 'b': 0.0, 'a': 1.0}})

        # Call service
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)

        # Check if call completed
        if not future.done():
            self.fail("Service call to '/unified_planner/add_object' timed out")

        # Get response
        response = future.result()
        if response is None:
            self.fail("Service call to '/unified_planner/add_object' failed")


        self.assertEqual(
            response.success,
            True,
            f"Field 'success' doesn't match expected value"
        )

    def test_04_arm_1_holds_part_1_surface_fit_4_spheres(self):
        """04 arm 1 holds part_1 (surface fit, 4 spheres)"""

        # Create service client
        client = self.node.create_client(AttachObject, '/unified_planner/attach_object')

        # Wait for service to be available
        timeout = 90.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/attach_object' not available after {timeout}s")

        # Create request
        request = AttachObject.Request()
        set_message_fields(request, {'object_name': 'part_1', 'attach_link': 'attached_object_1', 'sphere_fit_type': 1, 'num_spheres': 4})

        # Call service
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)

        # Check if call completed
        if not future.done():
            self.fail("Service call to '/unified_planner/attach_object' timed out")

        # Get response
        response = future.result()
        if response is None:
            self.fail("Service call to '/unified_planner/attach_object' failed")


        self.assertEqual(
            response.success,
            True,
            f"Field 'success' doesn't match expected value"
        )

        self.assertEqual(
            response.message,
            "Attached 'part_1'",
            f"Field 'message' doesn't match expected value"
        )

    def test_05_arm_0_holds_part_0_at_the_same_time_voxel_fit_estimated_count(self):
        """05 arm 0 holds part_0 at the same time (voxel fit, estimated count)"""

        # Create service client
        client = self.node.create_client(AttachObject, '/unified_planner/attach_object')

        # Wait for service to be available
        timeout = 90.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/attach_object' not available after {timeout}s")

        # Create request
        request = AttachObject.Request()
        set_message_fields(request, {'object_name': 'part_0', 'attach_link': 'attached_object_0', 'sphere_fit_type': 2})

        # Call service
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)

        # Check if call completed
        if not future.done():
            self.fail("Service call to '/unified_planner/attach_object' timed out")

        # Get response
        response = future.result()
        if response is None:
            self.fail("Service call to '/unified_planner/attach_object' failed")


        self.assertEqual(
            response.success,
            True,
            f"Field 'success' doesn't match expected value"
        )

        self.assertEqual(
            response.message,
            "Attached 'part_0'",
            f"Field 'message' doesn't match expected value"
        )

    def test_06_release_arm_1(self):
        """06 release arm 1"""

        # Create service client
        client = self.node.create_client(RemoveObject, '/unified_planner/detach_link')

        # Wait for service to be available
        timeout = 10.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/detach_link' not available after {timeout}s")

        # Create request
        request = RemoveObject.Request()
        set_message_fields(request, {'name': 'attached_object_1'})

        # Call service
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)

        # Check if call completed
        if not future.done():
            self.fail("Service call to '/unified_planner/detach_link' timed out")

        # Get response
        response = future.result()
        if response is None:
            self.fail("Service call to '/unified_planner/detach_link' failed")


        self.assertEqual(
            response.success,
            True,
            f"Field 'success' doesn't match expected value"
        )

        self.assertEqual(
            response.message,
            "Detached 'part_1' from 'attached_object_1'",
            f"Field 'message' doesn't match expected value"
        )

    def test_07_release_arm_0(self):
        """07 release arm 0"""

        # Create service client
        client = self.node.create_client(RemoveObject, '/unified_planner/detach_link')

        # Wait for service to be available
        timeout = 10.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/detach_link' not available after {timeout}s")

        # Create request
        request = RemoveObject.Request()
        set_message_fields(request, {'name': 'attached_object_0'})

        # Call service
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)

        # Check if call completed
        if not future.done():
            self.fail("Service call to '/unified_planner/detach_link' timed out")

        # Get response
        response = future.result()
        if response is None:
            self.fail("Service call to '/unified_planner/detach_link' failed")


        self.assertEqual(
            response.success,
            True,
            f"Field 'success' doesn't match expected value"
        )

        self.assertEqual(
            response.message,
            "Detached 'part_0' from 'attached_object_0'",
            f"Field 'message' doesn't match expected value"
        )

    def test_08_a_link_the_robot_does_not_have_is_refused(self):
        """08 a link the robot does not have is refused"""

        # Create service client
        client = self.node.create_client(RemoveObject, '/unified_planner/detach_link')

        # Wait for service to be available
        timeout = 10.0
        if not client.wait_for_service(timeout_sec=timeout):
            self.fail(f"Service '/unified_planner/detach_link' not available after {timeout}s")

        # Create request
        request = RemoveObject.Request()
        set_message_fields(request, {'name': 'no_such_link'})

        # Call service
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)

        # Check if call completed
        if not future.done():
            self.fail("Service call to '/unified_planner/detach_link' timed out")

        # Get response
        response = future.result()
        if response is None:
            self.fail("Service call to '/unified_planner/detach_link' failed")


        self.assertEqual(
            response.success,
            False,
            f"Field 'success' doesn't match expected value"
        )

        self.assertTrue(
            response.message.startswith("Detach error:"),
            f"Field 'message' value {{  response.message  }} doesn't match pattern 'Detach error: %'"
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

