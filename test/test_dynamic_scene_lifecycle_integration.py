"""Offline UR10e/Robotiq attachment -> mapper lifecycle integration test.

This intentionally exercises the public ROS services and the real CUDA mapper;
the only synthetic parts are a depth camera and its timestamped joint feedback.
"""

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import rclpy
import torch
from curobo.types import JointState as CuroboJointState
from curobo_msgs.srv import AddObject, AttachObject
from geometry_msgs.msg import TransformStamped
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage

from curobo_ros.core.attachment_services import AttachmentServices
from curobo_ros.perception import ObjectGeometry
from curobo_ros.perception.object_depth_mask import _pose_tensor
from curobo_ros.perception.projected_mesh_mask import rasterize_nearest_depth
from test_camera_to_planner import CAMERA, CAMERAS_YAML, FOCAL, HEIGHT, Planner, WIDTH


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

Q0 = [0.0, -1.2, 1.8, -2.2, -1.57, 0.0]
Q1 = [0.18, -1.15, 1.72, -2.15, -1.57, 0.08]
# This remains a small payload for the 24 allocated attachment slots, while
# deliberately overlapping the modeled tool region at its attachment link.
CUBE_DIMS = np.array([0.20, 0.20, 0.20])
BACKGROUND_DEPTH_M = 1.20


def _link_pose(planner, q):
    """Actual cuRobo FK for the test model's virtual attachment link."""
    kin = planner.node.config_wrapper_motion.kin_model
    state = CuroboJointState(
        position=torch.tensor([q], dtype=torch.float32, device='cuda'),
        joint_names=planner.joint_names)
    rotation, translation = AttachmentServices._link_pose(
        kin, state.position, 'attached_object')
    return translation.detach().cpu().tolist() + AttachmentServices._rot_to_quat(rotation)


def _weight_in_aabb(manager, lower, upper):
    """Sum positive TSDF weights in an AABB (the mapper's sparse coordinates)."""
    data = manager.mapper.tsdf.data
    n = int(data.num_allocated.item())
    if n == 0:
        return 0.0
    active = data.block_to_hash_slot[:n] >= 0
    if not bool(active.any()):
        return 0.0
    pool = torch.arange(n, device=data.block_coords.device)[active]
    coords = data.block_coords[:n * 3].view(n, 3)[active].to(torch.int64)
    local = torch.arange(512, device=coords.device, dtype=torch.int64)
    xyz = torch.stack((local % 8, (local // 8) % 8, local // 64), dim=1)
    voxels = coords[:, None, :] * 8 + xyz[None, :, :]
    if data.grid_shape is None:
        centers = data.origin.float()[None, None, :] + voxels.float() * float(data.voxel_size)
    else:
        shape = torch.as_tensor((data.grid_shape[2], data.grid_shape[1], data.grid_shape[0]),
                                device=coords.device, dtype=torch.float32)
        centers = (data.origin.float()[None, None, :]
                   + (voxels.float() + 0.5 - shape / 2.0) * float(data.voxel_size))
    region = ((centers >= torch.as_tensor(lower, device=centers.device))
              & (centers <= torch.as_tensor(upper, device=centers.device))).all(dim=-1)
    return float(data.block_data[pool, :, 1][region].sum(dtype=torch.float32).item())


class JointFeedback:
    """Timestamped feedback on the descriptor's real emulator topic."""

    def __init__(self, names, q):
        self.node = rclpy.create_node('ur10e_attachment_joint_feedback')
        self.publisher = self.node.create_publisher(
            JointState, '/ur10e_robotiq_attach_test/joint_states', 100)
        self.names, self.q = list(names), list(q)
        self.node.create_timer(0.01, self.publish)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def publish(self):
        msg = JointState(name=self.names, position=[float(x) for x in self.q])
        msg.header.stamp = self.node.get_clock().now().to_msg()
        self.publisher.publish(msg)

    def set(self, q):
        self.q = list(q)

    def destroy(self):
        self.executor.shutdown()
        self.node.destroy_node()


class SyntheticDepthCamera:
    """A camera with a rasterized cube over a persistent planar background."""

    def __init__(self, base_frame, geometry, pose, gpu_lock):
        self.node = rclpy.create_node('ur10e_attachment_lifecycle_camera')
        self.tf = self.node.create_publisher(TFMessage, '/tf', 10)
        self.depth = self.node.create_publisher(Image, '/test_camera/depth', 10)
        self.geometry = geometry
        self.pose = pose
        # This test-only renderer shares the process/CUDA context with the
        # planner. Production cameras live in separate processes; here its GPU
        # work must honor the same lock contract as mapper integration.
        self.gpu_lock = gpu_lock
        self.last_maskable_depth = None
        self.last_stamp_ns = None
        self._tf = TransformStamped()
        self._tf.header.frame_id, self._tf.child_frame_id = base_frame, CAMERA
        self.node.create_timer(1.0 / 30.0, self.publish)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def set_pose(self, pose):
        self.pose = pose

    def _render(self):
        # The optical frame points along base -Z, as in the existing helper.
        camera_pose = [self.pose[0], self.pose[1], self.pose[2] + 0.80, 0., 1., 0., 0.]
        base_from_camera = _pose_tensor(camera_pose, torch.device('cuda'))
        base_from_object = _pose_tensor(self.pose, torch.device('cuda'))
        camera_from_base = torch.linalg.inv(base_from_camera)
        vertices = torch.as_tensor(self.geometry.vertices, device='cuda')
        faces = torch.as_tensor(self.geometry.faces, device='cuda')
        transform = camera_from_base @ base_from_object
        camera_vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
        front = rasterize_nearest_depth(
            camera_vertices, faces, (FOCAL, FOCAL, WIDTH / 2, HEIGHT / 2), (HEIGHT, WIDTH))
        depth = torch.full((HEIGHT, WIDTH), BACKGROUND_DEPTH_M, device='cuda')
        depth = torch.where(torch.isfinite(front), front, depth)
        return camera_pose, depth

    def publish(self):
        with self.gpu_lock:
            camera_pose, depth = self._render()
        stamp = self.node.get_clock().now().to_msg()
        self.last_stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        t = self._tf.transform.translation
        t.x, t.y, t.z = camera_pose[:3]
        q = self._tf.transform.rotation
        q.w, q.x, q.y, q.z = camera_pose[3:]
        self._tf.header.stamp = stamp
        image = Image(height=HEIGHT, width=WIDTH, encoding='16UC1', step=2 * WIDTH,
                      data=(depth.clamp_min(0).mul(1000).to(torch.uint16).cpu().numpy()).tobytes())
        image.header.frame_id, image.header.stamp = CAMERA, stamp
        self.last_maskable_depth = depth.cpu().numpy()
        self.tf.publish(TFMessage(transforms=[self._tf]))
        self.depth.publish(image)

    def destroy(self):
        self.executor.shutdown()
        self.node.destroy_node()


def _wait_for_frames(planner, previous, count=8):
    deadline = time.monotonic() + 20.0
    while planner.node.mapper.get_stats()['frame_count'] < previous + count:
        assert time.monotonic() < deadline, 'synthetic depth frames were not integrated'
        time.sleep(0.05)


def _aabb(geometry, pose):
    return geometry.world_aabb(pose, margin_m=0.0)


def test_ur10e_robotiq_attachment_mapper_lifecycle(tmp_path):
    cameras = tmp_path / 'camera.yaml'
    cameras.write_text(CAMERAS_YAML)
    descriptor = '/ws/curobo_ros/test/fixtures/ur10e_robotiq_attach_test.yaml'
    planner = Planner(str(cameras), robot=descriptor, extra_params=(
        '-p', 'attached_object_masking_enabled:=true',
        '-p', 'attached_object_mask_dilation_px:=0',
    ))
    feedback = camera = None
    try:
        geometry = ObjectGeometry.cuboid(CUBE_DIMS)
        pose0, pose1 = _link_pose(planner, Q0), _link_pose(planner, Q1)
        feedback = JointFeedback(planner.joint_names, Q0)
        time.sleep(0.15)  # establish two stamped history samples before camera frames
        camera = SyntheticDepthCamera(
            planner.node.get_parameter('base_link').value, geometry, pose0,
            planner.node.gpu_lock)
        _wait_for_frames(planner, 0, 12)

        manager = planner.node.config_wrapper_motion.obstacle_manager
        lower0, upper0 = _aabb(geometry, pose0)
        object_weight_before = _weight_in_aabb(manager, lower0, upper0)
        # A point well behind the cube is visible in every synthetic image.
        background_weight_before = _weight_in_aabb(
            manager, np.array(pose0[:3]) + [-0.30, -0.30, -0.45],
            np.array(pose0[:3]) + [0.30, 0.30, -0.30])
        assert object_weight_before > 0.0
        assert background_weight_before > 0.0
        # First ESDF construction captures a CUDA graph.  Stop the producer
        # before it, then use the node's lock-owning refresh API; calling the
        # mapper adapter directly races depth integration and poisons capture.
        camera.destroy(); camera = None
        planner.node.refresh_perception_world()
        assert manager.get_scene().voxel
        initial_distance = planner.distance(Q0)[0]
        assert initial_distance > 0.0

        add = planner.client.create_client(AddObject, '/unified_planner/add_object')
        box = AddObject.Request(type=AddObject.Request.CUBOID, name='lifecycle_cube')
        box.pose.position.x, box.pose.position.y, box.pose.position.z = pose0[:3]
        box.pose.orientation.w, box.pose.orientation.x, box.pose.orientation.y, box.pose.orientation.z = pose0[3:]
        box.dimensions.x, box.dimensions.y, box.dimensions.z = CUBE_DIMS.tolist()
        assert planner.call(add, box).success
        # The collision values below must be TSDF-derived, not the analytic cube.
        manager.set_obstacles_enabled(['lifecycle_cube'], False)
        planner.node.apply_obstacle_states(['lifecycle_cube'])
        distance_before = planner.distance(Q0)[0]
        assert distance_before > 0.0

        attach = planner.client.create_client(AttachObject, '/unified_planner/attach_object')
        request = AttachObject.Request(object_name='lifecycle_cube', attach_link='attached_object')
        request.start_pose = planner.joint_state(Q0)
        attached = planner.call(attach, request)
        assert attached.success, attached.message
        service = planner.node.attachment_services
        assert service.attached_name == 'lifecycle_cube'
        record = service._mapper_attachments['attached_object']
        np.testing.assert_allclose(record['object_in_link'], np.eye(4), atol=2e-5)
        assert 'lifecycle_cube' in planner.node.object_depth_masker._objects
        object_weight_invalidated = _weight_in_aabb(manager, lower0, upper0)
        background_weight_invalidated = _weight_in_aabb(
            manager, np.array(pose0[:3]) + [-0.30, -0.30, -0.45],
            np.array(pose0[:3]) + [0.30, 0.30, -0.30])
        assert object_weight_invalidated == 0.0
        assert background_weight_invalidated > 0.0
        distance_invalidated = planner.distance(Q0)[0]
        assert distance_invalidated < distance_before

        previous = planner.node.mapper.get_stats()['frame_count']
        camera = SyntheticDepthCamera(
            planner.node.get_parameter('base_link').value, geometry, pose0,
            planner.node.gpu_lock)
        _wait_for_frames(planner, previous)
        mask0 = planner.node.object_depth_masker.last_mask
        assert mask0 is not None and int(mask0.sum().item()) > 0
        assert np.all(camera.last_maskable_depth[mask0.cpu().numpy()] < BACKGROUND_DEPTH_M)
        assert _weight_in_aabb(manager, lower0, upper0) == 0.0

        # A new stamped state drives FK for the new image time; there is no
        # latest-state fallback in ObjectDepthMasker.
        feedback.set(Q1)
        camera.set_pose(pose1)
        previous = planner.node.mapper.get_stats()['frame_count']
        _wait_for_frames(planner, previous)
        tracked = planner.node.get_attachment_link_pose_at_stamp('attached_object', camera.last_stamp_ns)
        assert tracked is not None
        np.testing.assert_allclose(tracked[:3], pose1[:3], atol=3e-3)
        mask1 = planner.node.object_depth_masker.last_mask
        assert mask1 is not None and int(mask1.sum().item()) > 0
        lower1, upper1 = _aabb(geometry, pose1)
        # At the moved pose the retained planar background may pass through the
        # cuboid's *world AABB*, so its aggregate TSDF weight is not an object
        # occupancy metric.  With dilation off, exact pixel equality proves the
        # complete rendered object surface was removed before integration.
        assert int(mask1.sum().item()) == int(np.count_nonzero(
            camera.last_maskable_depth < BACKGROUND_DEPTH_M))
        distance_masked = planner.distance(Q1)[0]
        # The background plane is intentionally retained and may remain within
        # the configured activation band at Q1.  It must nevertheless be less
        # costly than the original reconstructed payload contact.
        assert distance_masked < distance_before

        camera.destroy(); camera = None
        detach = planner.client.create_client(Trigger, '/unified_planner/detach_object')
        detached = planner.call(detach, Trigger.Request())
        assert detached.success, detached.message
        assert service.attached_name is None
        assert 'lifecycle_cube' not in planner.node.object_depth_masker._objects
        assert 'lifecycle_cube' in manager.disabled_obstacles  # never restore stale pre-grasp pose

        previous = planner.node.mapper.get_stats()['frame_count']
        camera = SyntheticDepthCamera(
            planner.node.get_parameter('base_link').value, geometry, pose1,
            planner.node.gpu_lock)
        _wait_for_frames(planner, previous)
        object_weight_reconstructed = _weight_in_aabb(manager, lower1, upper1)
        assert object_weight_reconstructed > 0.0
        distance_reconstructed = planner.distance(Q1)[0]
        assert distance_reconstructed > distance_masked
        Path('/tmp/ur10e_lifecycle_measurements.json').write_text(json.dumps({
            'tsdf_object_weight_before': object_weight_before,
            'tsdf_background_weight_before': background_weight_before,
            'tsdf_object_weight_after_invalidation': object_weight_invalidated,
            'tsdf_background_weight_after_invalidation': background_weight_invalidated,
            'tsdf_object_weight_after_detach_reconstruction': object_weight_reconstructed,
            'distance_initial': initial_distance,
            'distance_before_attach': distance_before,
            'distance_after_invalidation': distance_invalidated,
            'distance_masked_at_q1': distance_masked,
            'distance_after_detach_reconstruction': distance_reconstructed,
            'mask_pixels_q0': int(mask0.sum().item()),
            'mask_pixels_q1': int(mask1.sum().item()),
        }, sort_keys=True))
    finally:
        if camera is not None:
            camera.destroy()
        if feedback is not None:
            feedback.destroy()
        planner.destroy()
