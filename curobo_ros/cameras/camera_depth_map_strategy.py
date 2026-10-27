from curobo_ros.cameras.camera_strategy import CameraStrategy
from curobo.types import CameraObservation, Pose
from curobo_ros.core.stamp_wait import StampWait

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from rclpy.wait_for_message import wait_for_message
import rclpy
from scipy.spatial.transform import Rotation
from tf2_ros import TransformException

import torch
import numpy as np

# How long a depth frame may wait for the TF at its stamp. Covers TF publishers
# down to 20 Hz: the TF sample after the stamp then arrives at most 50 ms later.
DEFAULT_TF_WAIT_MAX_S = 0.05


class DepthMapCameraStrategy(CameraStrategy):
    '''
    Camera strategy for Intel RealSense depth cameras.

    This strategy subscribes to depth image and camera info topics to provide
    camera observations for cuRobo collision checking.
    '''
    def __init__(self, node, camera_name, topic='/camera/depth/image_rect_raw',
                 camera_info_topic='/camera/depth/camera_info',
                 frame_id='',
                 intrinsics=None,
                 extrinsics=None,
                 tf_wait_max_s=DEFAULT_TF_WAIT_MAX_S
                 ):
        """
        Initialize a depthmap camera strategy.

        Args:
            node: ROS2 node for creating subscriptions and logging
            camera_name: Name of the camera
            topic: Topic name for depth image
            camera_info_topic: Topic name for camera info
            frame_id: Frame of the depth image (its header.frame_id). Required:
                frames from any other frame are dropped, and it is the TF
                child frame (or the frame the static extrinsics describe).
            intrinsics: Optional camera intrinsics from config (list or dict)
            extrinsics: Optional camera extrinsics from config (list)
            tf_wait_max_s: How long (s, steady clock) a depth frame waits for
                the TF at its stamp when that TF has not arrived yet; the frame
                is dropped after it. 0 drops it at once. Unused with static
                extrinsics.
        """
        super().__init__(node, camera_name, topic, camera_info_topic, frame_id, intrinsics, extrinsics)

        # Without a frame neither the TF lookup nor the static extrinsics refer to
        # anything, and the incoming header.frame_id cannot be checked.
        if not frame_id:
            raise ValueError(
                f"Camera '{camera_name}': 'frame_id' is required (the frame_id of "
                f'the depth images on {topic})')
        if not isinstance(tf_wait_max_s, (int, float)) or tf_wait_max_s < 0:
            raise ValueError(
                f"Camera '{camera_name}': 'tf_wait_max_s' must be a number >= 0, "
                f'got {tf_wait_max_s!r}')
        self._tf_wait_max_s = float(tf_wait_max_s)

        # Robot base frame for the camera-pose TF lookup. Read from the node's
        # 'base_link' parameter (the planner's configurable base frame) instead
        # of hardcoding 'base_link' — the Doosan base is 'base_0', and 'base_link'
        # collides with the Ridgeback frame when the mobile base is present.
        self.base_frame = (
            node.get_parameter('base_link').get_parameter_value().string_value
            if node.has_parameter('base_link') else 'base_0')

        self.depth_map = None
        self.intrinsics = None

        # Try to use intrinsics from config first
        self.intrinsics = self._parse_intrinsics(intrinsics)

        # Fallback to camera_info topic if not provided in config
        if self.intrinsics is None:
            node.get_logger().info(f"No intrinsics in config, waiting for camera info on {camera_info_topic}...")
            res, camera_info_msg = wait_for_message(CameraInfo, node, camera_info_topic, time_to_wait=5.0)

            if res:
                # Extract intrinsics from camera_info and create 3x3 matrix
                # K is a flattened 3x3 matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
                K = camera_info_msg.k

                # Create intrinsics as a 3x3 matrix (shape: (3, 3))
                # CameraObservation will add batch dimension automatically if needed
                self.intrinsics = torch.tensor([
                    [K[0], K[1], K[2]],  # [fx,  0, cx]
                    [K[3], K[4], K[5]],  # [ 0, fy, cy]
                    [K[6], K[7], K[8]]   # [ 0,  0,  1]
                ], dtype=self.tensor_args.dtype, device=self.tensor_args.device)

                node.get_logger().info(f"Camera intrinsics from topic: fx={K[0]:.2f}, fy={K[4]:.2f}, cx={K[2]:.2f}, cy={K[5]:.2f}")
            else:
                node.get_logger().error(f"Failed to receive camera info from {camera_info_topic}")
                raise RuntimeError("Could not get camera intrinsics")
        else:
            node.get_logger().info("Using intrinsics from config file")

        # Parse extrinsics from config
        self.camera_pose_static = self._parse_extrinsics(extrinsics)
        if self.camera_pose_static is not None:
            node.get_logger().info("Using static extrinsics from config file")



        # Create subscription to depth topic
        self.sub_depth = self.node.create_subscription(
            Image, topic, self.callback_depth_map, 1)

        # A depth frame can arrive before the TF at its stamp: it waits (at most
        # one, the newest, for at most tf_wait_max_s) instead of being dropped.
        self._stamp_wait = StampWait(
            self.node, self.tf_buffer, self._tf_wait_max_s,
            process=self._integrate_stamped, missing_tf=self._missing_tf,
            callback_group=self.sub_depth.callback_group)

        # Image processing
        self.bridge = CvBridge()

        node.get_logger().info(f"DepthMap camera initialized with depth topic: {topic}")


    def callback_depth_map(self, msg):
        """
        Callback for receiving depth image data.

        Args:
            msg: Image message
        """
        # Intrinsics and extrinsics belong to the configured frame: a depth
        # image from any other frame would be projected with the wrong pose.
        if msg.header.frame_id != self._frame_id:
            self.node.get_logger().error(
                f"Depth frame_id '{msg.header.frame_id}' on {self._topic} does not "
                f"match the configured frame_id '{self._frame_id}'. Dropping this "
                'depth frame (no integration).',
                throttle_duration_sec=2.0)
            return
        if self.camera_pose_static is not None:
            self._integrate(msg, self.camera_pose_static)
            return

        # The camera pose is the one at the image's acquisition time, not the
        # latest one: a moving camera keeps moving between acquisition and this
        # callback. A zero stamp would make tf2 return the latest transform, so
        # it is rejected instead.
        stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        if stamp.nanoseconds == 0:
            self.node.get_logger().error(
                f'Depth image on {self._topic} has no header stamp. '
                'Dropping this depth frame (no integration).',
                throttle_duration_sec=2.0)
            return

        self._stamp_wait.submit(msg, stamp)

    def _missing_tf(self, msg, stamp):
        """Return (base, camera) if the camera TF at `stamp` has not arrived yet, else None."""
        if self.tf_buffer.can_transform(self.base_frame, self._frame_id, stamp):
            return None
        return (self.base_frame, self._frame_id)

    def _integrate_stamped(self, msg, stamp):
        """Integrate `msg` with the TF at `stamp`; drop it if that TF is unavailable."""
        try:
            t = self.tf_buffer.lookup_transform(self.base_frame, self._frame_id, stamp)
        except TransformException as ex:
            # Do NOT fall back to an identity or the latest pose: that would
            # integrate this frame's depth where the camera was not, placing
            # phantom obstacles in the collision world (and leaving the real
            # observed volume uncovered) — worse than skipping the frame.
            self.node.get_logger().warn(
                f'Could not transform {self.base_frame} to {self._frame_id} '
                f'at {stamp.nanoseconds * 1e-9:.6f} s: '
                f'{ex}. Dropping this depth frame (no integration).',
                throttle_duration_sec=2.0)
            return
        translation = t.transform.translation
        position = [translation.x, translation.y, translation.z]
        rotation_msg = t.transform.rotation
        quat_ros = [rotation_msg.x, rotation_msg.y, rotation_msg.z, rotation_msg.w]
        quat_scipy = Rotation.from_quat(quat_ros).as_quat()  # [x, y, z, w]
        # cuRobo expects [x, y, z, qw, qx, qy, qz]
        self._integrate(
            msg, position + [quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])

    def _integrate(self, msg, pose):
        """
        Integrate one depth image into the Mapper.

        Args:
            msg: Image message
            pose: Camera pose in the base frame, a cuRobo Pose or a plain
                [x, y, z, qw, qx, qy, qz] list (converted on the GPU below).
        """
        try:
            # ---- CPU-only work first (safe to run during a CUDA graph capture) ----
            # Convert ROS image to numpy (millimetres 16UC1 or metres 32FC1).
            if msg.encoding == "16UC1":
                depth_img = self.bridge.imgmsg_to_cv2(msg, "16UC1")
                depth_img_float = depth_img.astype(np.float32) / 1000.0
            elif msg.encoding == "32FC1":
                depth_img_float = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            else:
                self.node.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}, trying 32FC1")
                depth_img_float = self.bridge.imgmsg_to_cv2(msg, "32FC1")

            # ---- GPU section: EVERY CUDA op is under the lock ----
            # depth->cuda, pose->cuda, rgb alloc AND mapper.integrate are all GPU
            # ops. If curobo is capturing a CUDA graph (plan / MPC cold-start) it
            # holds gpu_lock, so skip the WHOLE frame — issuing ANY of these ops
            # mid-capture raises cudaErrorStreamCaptureUnsupported and poisons the
            # process's CUDA context. (Previously only integrate() was guarded, so
            # the depth->cuda copy above still raced the capture.)
            mapper = getattr(self.node, 'mapper', None)
            if mapper is None:
                self.node.get_logger().warn(
                    "No mapper on node - depth frame ignored. "
                    "Unified planner should expose `node.mapper` (curobo.perception.Mapper).",
                    throttle_duration_sec=5.0)
                return
            gpu_lock = getattr(self.node, 'gpu_lock', None)
            if gpu_lock is not None and not gpu_lock.acquire(blocking=False):
                self.node.get_logger().debug(
                    "depth frame skipped (GPU capture in progress)",
                    throttle_duration_sec=2.0)
                return
            try:
                depth_tensor = torch.from_numpy(depth_img_float).to(
                    device=self.tensor_args.device, dtype=self.tensor_args.dtype)

                if isinstance(pose, Pose):
                    self.camera_pose = pose
                else:
                    self.camera_pose = Pose.from_list(pose, device_cfg=self.tensor_args)

                # v2 Mapper requires a leading camera dimension on every field and a
                # paired rgb image (zero buffer — no colour for collision mapping).
                depth_b = depth_tensor.unsqueeze(0) if depth_tensor.ndim == 2 else depth_tensor
                intrinsics_b = (
                    self.intrinsics.unsqueeze(0) if self.intrinsics.ndim == 2 else self.intrinsics)
                rgb_b = torch.zeros(
                    (depth_b.shape[0], depth_b.shape[1], depth_b.shape[2], 3),
                    dtype=torch.uint8, device=self.tensor_args.device)
                data_camera = CameraObservation(
                    depth_image=depth_b, rgb_image=rgb_b,
                    intrinsics=intrinsics_b, pose=self.camera_pose)

                mapper.integrate(data_camera)
                torch.cuda.synchronize()
                self.depth_map = depth_tensor
            finally:
                if gpu_lock is not None:
                    gpu_lock.release()

        except CvBridgeError as e:
            self.node.get_logger().error(f"CvBridge error: {e}")
        except Exception as e:
            self.node.get_logger().error(f"Error processing depth image: {e}")
