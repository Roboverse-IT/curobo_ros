import os
import struct

import numpy as np
import torch

# ROS2
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs.msg import JointState as JointStateMsg
from std_msgs.msg import Header
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray

# cuRobo imports (v2)
from curobo.config_io import load_yaml
from curobo.content import get_assets_path
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState

from curobo_msgs.srv import RemoveObject, SetMask

from scipy.spatial.transform import Rotation

from curobo_ros.core.listener_thread import ListenerThread
from curobo_ros.core.stamp_wait import StampWait
from curobo_ros.perception import (
    ProjectedMeshMask, resolve_urdf_mesh_paths)
from curobo_ros.robot.joint_state_history import (
    DEFAULT_JOINT_STATE_MAX_GAP_S, JointStateHistory)
from curobo_ros.robot.locked_joints import LockedJoints
from curobo_ros.robot.robot_context import RobotContext
from curobo_ros.robot.robot_description import resolve_curobo_config, resolve_request_file


class DepthMapRobotSegmentation(Node):
    def __init__(self, distance_threshold=0.05, ops_dtype=torch.float32):
        """
        Initializes the depth map segmentation node.
        This node removes the robot from depth images using joint states.

        Args:
            distance_threshold: Minimum distance (in meters) to consider a point as not part of the robot
            ops_dtype: Data type for tensor operations
        """
        super().__init__('curobo_depth_map_robot_segmentation')

        # ---- Node params first (no robot/kinematics dependency) ----
        self.declare_parameter('depth_image_topic', '/depth_to_rgb/image_raw')
        self.declare_parameter('camera_info_topic', '/depth_to_rgb/camera_info')
        self.declare_parameter('robot_base_frame', 'base_0')
        # Inflation added to every mask shape's half-extents / radius, separate
        # from distance_threshold below.
        self.declare_parameter('mask_margin', 0.0)
        # Minimum distance (m) to a robot collision sphere for a depth point to
        # be kept (i.e. considered NOT part of the robot). Was a constructor-only
        # kwarg, unreachable from a launch file — now a runtime-settable param.
        self.declare_parameter('distance_threshold', distance_threshold)
        # Interchangeable robot geometry backends. Sphere preserves the
        # historical behavior; mesh projects configured visual geometry;
        # combined removes the union.
        self.declare_parameter('segmentation_mode', 'sphere')
        self.declare_parameter(
            'mesh_link_names', rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter(
            'mesh_asset_roots', rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter('mesh_margin', 0.0)
        self.declare_parameter('mesh_raster_stride', 8)
        # A depth frame is segmented with the TF and joint states at its stamp.
        # When they have not all arrived yet it waits, at most this long (s,
        # steady clock); then it is dropped. 0 drops it at once. Covers TF
        # publishers down to 20 Hz, as the planner's camera tf_wait_max_s.
        self.declare_parameter('stamp_wait_max_s', 0.05)
        # Largest gap (s) between the two joint states around a depth stamp
        # that is still interpolated; a frame in a larger gap is dropped.
        self.declare_parameter('joint_state_max_gap_s', DEFAULT_JOINT_STATE_MAX_GAP_S)
        self.mask_margin = self.get_parameter('mask_margin').get_parameter_value().double_value
        self.distance_threshold = self.get_parameter(
            'distance_threshold').get_parameter_value().double_value
        self._segmentation_mode = self.get_parameter(
            'segmentation_mode').get_parameter_value().string_value
        mesh_links_fallback = rclpy.Parameter(
            'mesh_link_names', rclpy.Parameter.Type.STRING_ARRAY, [])
        requested_mesh_links = list(self.get_parameter_or(
            'mesh_link_names', mesh_links_fallback).value)
        mesh_roots_fallback = rclpy.Parameter(
            'mesh_asset_roots', rclpy.Parameter.Type.STRING_ARRAY, [])
        mesh_asset_roots = list(self.get_parameter_or(
            'mesh_asset_roots', mesh_roots_fallback).value)
        mesh_margin = self.get_parameter(
            'mesh_margin').get_parameter_value().double_value
        mesh_raster_stride = self.get_parameter(
            'mesh_raster_stride').get_parameter_value().integer_value

        if self._segmentation_mode not in ('sphere', 'mesh', 'combined'):
            raise ValueError(
                "'segmentation_mode' must be sphere, mesh, or combined, got "
                f'{self._segmentation_mode!r}')
        if mesh_margin < 0.0:
            raise ValueError(f"'mesh_margin' must be >= 0, got {mesh_margin}")
        if mesh_raster_stride < 1:
            raise ValueError(
                "'mesh_raster_stride' must be >= 1, got "
                f'{mesh_raster_stride}')
        if len(set(requested_mesh_links)) != len(requested_mesh_links):
            raise ValueError(
                "'mesh_link_names' contains duplicates: "
                f'{requested_mesh_links}')
        for root in mesh_asset_roots:
            if not os.path.isabs(root) or not os.path.isdir(root):
                raise ValueError(
                    "'mesh_asset_roots' entries must be existing absolute "
                    f'directories, got {root!r}')

        stamp_wait_max_s = self.get_parameter('stamp_wait_max_s').get_parameter_value().double_value
        joint_state_max_gap_s = self.get_parameter(
            'joint_state_max_gap_s').get_parameter_value().double_value
        if stamp_wait_max_s < 0.0:
            raise ValueError(f"'stamp_wait_max_s' must be >= 0, got {stamp_wait_max_s}")
        if joint_state_max_gap_s <= 0.0:
            raise ValueError(
                f"'joint_state_max_gap_s' must be > 0, got {joint_state_max_gap_s}")

        depth_image_topic = self.get_parameter('depth_image_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value

        self._ops_dtype = ops_dtype
        self._device = torch.device('cuda')

        # ---- Robot: single descriptor-driven source (was: a hardcoded m1013
        # path here, independent of RobotContext's own robot resolution — two
        # sources of truth that could silently disagree). RobotContext declares
        # the 'robot' param and loads the descriptor; robot_config_file below
        # defaults to that SAME descriptor instead of a hardcoded path.
        self.robot_context = RobotContext(self)

        default_robot_config = self.robot_context.description.curobo_config_path
        if not self.has_parameter('robot_config_file'):
            self.declare_parameter('robot_config_file', default_robot_config)
        robot_config_file = self.get_parameter('robot_config_file').get_parameter_value().string_value
        if not robot_config_file:
            robot_config_file = default_robot_config

        # An explicit robot_config_file override bypasses load_robot_description's
        # own path resolution — resolve urdf_path/asset_root_path here too, or a
        # package://-relative/custom-relative path silently falls back to cuRobo's
        # bundled-assets resolution instead. See resolve_curobo_config.
        robot_config_file = resolve_curobo_config(robot_config_file)

        # v2: Kinematics replaces CudaRobotModel; from_robot_yaml_file loads the
        # robot YAML directly — no manual load_yaml/RobotConfig.from_dict dance.
        mesh_links = []
        if self._segmentation_mode == 'sphere':
            kinematics_cfg = KinematicsCfg.from_robot_yaml_file(
                robot_config_file)
        else:
            # cuRobo loads meshes and returns their poses in the same FK state
            # only when both fields are set before Kinematics construction.
            config = load_yaml(robot_config_file)
            config_robot = config.get('robot_cfg', config)
            config_kinematics = config_robot.get('kinematics', {})
            urdf_path = config_kinematics.get('urdf_path')
            if not urdf_path:
                raise ValueError(
                    'Mesh segmentation requires kinematics.urdf_path')
            asset_root = config_kinematics.get('asset_root_path')
            if asset_root and not os.path.isabs(asset_root):
                asset_candidate = os.path.join(get_assets_path(), asset_root)
                if os.path.isdir(asset_candidate):
                    asset_root = asset_candidate
                    config_kinematics['asset_root_path'] = asset_root
            if not os.path.isabs(urdf_path):
                urdf_candidates = [
                    os.path.join(os.path.dirname(robot_config_file), urdf_path),
                    os.path.join(get_assets_path(), urdf_path),
                ]
                if asset_root and os.path.isabs(asset_root):
                    urdf_candidates.insert(0, os.path.join(asset_root, urdf_path))
                urdf_path = next(
                    (path for path in urdf_candidates if os.path.isfile(path)),
                    urdf_path,
                )
            search_roots = list(mesh_asset_roots)
            if asset_root and os.path.isdir(asset_root):
                search_roots.append(asset_root)
            resolved_urdf, unresolved_meshes = resolve_urdf_mesh_paths(
                urdf_path, search_roots)
            config_kinematics['urdf_path'] = resolved_urdf
            if unresolved_meshes:
                self.get_logger().warning(
                    'Mesh segmentation could not resolve these URDF assets; '
                    'auto-selected links using them will be skipped: '
                    f'{unresolved_meshes}')
            configured_mesh_links = list(
                config_kinematics.get('mesh_link_names') or ())
            mesh_links = requested_mesh_links or configured_mesh_links or list(
                config_kinematics.get('collision_link_names') or ())
            if not mesh_links:
                raise ValueError(
                    'Mesh segmentation has no links: set mesh_link_names or '
                    'configure kinematics.mesh_link_names/collision_link_names')
            config_kinematics['mesh_link_names'] = mesh_links
            config_kinematics['load_tool_frames_with_mesh'] = True
            kinematics_cfg = KinematicsCfg.from_robot_yaml_file(config)

        kin_model = Kinematics(kinematics_cfg)
        self._kin_model = kin_model
        # Adopt the canonical joint order/DOF into the shared descriptor, same as
        # unified_planner_node does — makes RobotContext.get_joint_pose()/
        # get_joint_name() DOF-consistent with this node's own kinematics.
        self.robot_context.bind_kinematics(kin_model)

        self._mesh_link_names = mesh_links
        self._mesh_segmenter = None
        if self._segmentation_mode in ('mesh', 'combined'):
            self._mesh_segmenter = ProjectedMeshMask.from_kinematics(
                kin_model,
                mesh_links,
                self._device,
                margin=mesh_margin,
                raster_stride=mesh_raster_stride,
                strict=bool(requested_mesh_links),
            )
            self._mesh_link_names = self._mesh_segmenter.link_names
            if self._mesh_segmenter.skipped_link_names:
                self.get_logger().info(
                    'Mesh segmentation skipped links without visual geometry: '
                    f'{self._mesh_segmenter.skipped_link_names}')
            self.get_logger().info(
                f'Robot segmentation mode={self._segmentation_mode}, mesh '
                f'links={self._mesh_link_names}, margin={mesh_margin:.4f} m, '
                f'raster_stride={mesh_raster_stride}')

        # The joint states come straight from the robot's topic, by name and
        # with their stamps, so each depth frame is segmented with the joints at
        # its own stamp (the control strategy only keeps the latest state).
        joint_states_topic = self.robot_context.description.strategy_params.get(
            'joint_states_topic')
        if not joint_states_topic:
            raise ValueError(
                f"Robot '{self.robot_context.description.name}': the descriptor has no "
                'strategy_params.joint_states_topic, which segmentation needs')
        # Driver names and values mapped to the model's (descriptor joint_states).
        self._joint_state_map = self.robot_context.description.joint_state_map
        self._joint_history = JointStateHistory(kin_model.joint_names, joint_state_max_gap_s)
        # The joints the model locks (e.g. a gripper's fingers) keep their
        # configured values until the topic carries them; from then on each
        # frame is masked with their values at its stamp too.
        self._locked_joints = LockedJoints(kin_model, robot_config_file)
        self._locked_history = (
            JointStateHistory(self._locked_joints.names, joint_state_max_gap_s)
            if self._locked_joints.names else None)
        self._locked_measured = False
        self._locked_configured_logged = False

        # TF and joint states are handled on threads of their own (see
        # ListenerThread): on this node's executor they queued behind
        # segmentation, and a depth frame was dropped before its TF arrived.
        self.tf_buffer = tf2_ros.Buffer()
        self._tf_thread = ListenerThread(f'{self.get_name()}_tf_listener')
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self._tf_thread.node)

        # DOF-agnostic (was a hardcoded 6-zero vector + empty joint_names).
        self.q_js = JointState(
            position=torch.zeros(len(kin_model.joint_names), dtype=self._ops_dtype, device=self._device),
            joint_names=list(kin_model.joint_names))

        self.depth_image = None
        self.camera_info = None
        # Header of the depth frame being segmented. Everything published for
        # that frame carries its stamp, so consumers can match it to the TF and
        # joint states of the moment the depth was captured.
        self.depth_header = None
        self._frame_transforms = {}
        self.bridge = CvBridge()

        # Extra user-defined masks (e.g. a grasped object) removed from the depth
        # in addition to the robot. Keyed by name; each entry rides a TF frame so
        # it follows the arm. See SetMask.srv / _shape_inside_mask.
        self._masks = {}

        # Publisher for the masked depth image
        self.publisher_ = self.create_publisher(Image, "masked_depth_image", 10)

        # Publisher for collision spheres visualization
        self.sphere_marker_pub = self.create_publisher(MarkerArray, 'collision_spheres', 10)

        # Publisher for robot point cloud (debug)
        self.robot_pointcloud_pub = self.create_publisher(PointCloud2, 'robot_pointcloud_debug', 10)

        # Subscription to depth image
        self.subscription_depth = self.create_subscription(
            Image,
            depth_image_topic,
            self.listener_callback_depth,
            1)

        # A depth frame waits (at most one, the newest) for the TF and joint
        # states at its stamp; see stamp_wait_max_s.
        self._stamp_wait = StampWait(
            self, self.tf_buffer, stamp_wait_max_s,
            process=self._segment_at_stamp, missing_tf=self._missing_tf,
            ready=lambda msg, stamp: self._joints_cover(stamp),
            callback_group=self.subscription_depth.callback_group)

        # On their own thread (see the TF listener above); the queue holds 0.2 s
        # of them at 500 Hz.
        self._joint_state_thread = ListenerThread(f'{self.get_name()}_joint_states')
        self.subscription_joint_states = self._joint_state_thread.node.create_subscription(
            JointStateMsg,
            joint_states_topic,
            self.listener_callback_joint_states,
            100)

        # Subscription to camera info
        self.subscription_camera_info = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.listener_callback_camera_info,
            1)

        # Services to add/update and remove extra depth masks (e.g. grasped
        # object). Sync callbacks: pure dict mutation, no GPU -> safe under spin.
        self.set_mask_srv = self.create_service(
            SetMask, 'set_mask', self.set_mask_callback)
        self.remove_mask_srv = self.create_service(
            RemoveObject, 'remove_mask', self.remove_mask_callback)

        # Segmentation is driven by the depth-frame subscriber callback (runs at
        # the camera rate), not a fixed timer — see listener_callback_depth.

        self.get_logger().info("Depth map segmentation node initialized")

    # ── Mask services ────────────────────────────────────────────────────────

    def set_mask_callback(self, request, response):
        """Add or update a named mask shape (SetMask.srv).

        Dimension conventions mirror obstacle_manager.add_object:
        CUBOID [dx,dy,dz], SPHERE [r,_,_], CAPSULE [r,h,_] (segment [0,0,0]->[0,0,h]),
        CYLINDER [r,h,_] (centered, axis z), MESH [sx,sy,sz].
        """
        d = request.dimensions
        mesh_file_path = request.mesh_file_path
        if request.type == 4:
            try:
                mesh_file_path = resolve_request_file(mesh_file_path)
            except ValueError as ex:
                response.success = False
                response.message = str(ex)
                return response
        elif d.x <= 0 or d.y <= 0 or d.z <= 0:
            response.success = False
            response.message = 'Mask dimensions must be positive'
            return response

        p = request.pose.position
        o = request.pose.orientation
        self._masks[request.name] = {
            'type': request.type,
            'frame_id': request.frame_id,
            'mesh_file_path': mesh_file_path,
            'pos': torch.tensor([p.x, p.y, p.z],
                                dtype=self._ops_dtype, device=self._device),
            'quat': [o.x, o.y, o.z, o.w],  # scipy order
            'dims': [d.x, d.y, d.z],
        }
        frame = request.frame_id if request.frame_id else \
            self.get_parameter('robot_base_frame').get_parameter_value().string_value
        response.success = True
        response.message = f"Mask '{request.name}' set (type={request.type}, frame='{frame}')"
        self.get_logger().info(response.message)
        return response

    def remove_mask_callback(self, request, response):
        """Remove a named mask (RemoveObject.srv). Idempotent."""
        if request.name in self._masks:
            del self._masks[request.name]
            response.success = True
            response.message = f"Mask '{request.name}' removed"
        else:
            response.success = True
            response.message = f"Mask '{request.name}' not present (nothing to remove)"
        self.get_logger().info(response.message)
        return response

    def listener_callback_camera_info(self, msg):
        """
        Callback for receiving camera info data.
        Stores the camera intrinsics matrix.
        """
        self.camera_info = msg

    def destroy_node(self):
        """Stop the TF and joint-state threads, then destroy the node."""
        self._joint_state_thread.destroy()
        self._tf_thread.destroy()
        super().destroy_node()

    def listener_callback_joint_states(self, msg):
        """Store a joint state; re-check a depth frame waiting for it (joint-state thread)."""
        msg = self._joint_state_map.to_model(msg)
        # A state without the locked joints is not for them: no warning.
        if self._locked_history is not None and self._locked_history.add(msg) is None:
            self._locked_measured = True
        reason = self._joint_history.add(msg)
        if reason is not None:
            self.get_logger().warn(
                f'Joint state on {self.subscription_joint_states.topic_name} ignored: {reason}',
                throttle_duration_sec=5.0)
            return
        self._stamp_wait.notify()

    def _joints_cover(self, stamp):
        """Return True once the joint states (locked joints once measured) reach `stamp`."""
        return self._joint_history.covers(stamp) and (
            not self._locked_measured or self._locked_history.covers(stamp))

    def listener_callback_depth(self, msg):
        """
        Callback for receiving depth image data.
        The frame is segmented once the TF and joint states at its stamp are
        there (see stamp_wait_max_s), or dropped.
        """
        # Segmentation is driven by frame ARRIVAL (i.e. camera rate, ~5Hz)
        # rather than a 100Hz timer that republished the same frame -- nvblox
        # then re-integrated the same depth data ~6x (wasted GPU work, competing
        # with the MPPI solve). See debug 2026-07-15.
        if self.camera_info is None:
            self.get_logger().warn(
                'No camera info received yet. Dropping this depth frame (nothing published).',
                throttle_duration_sec=5.0)
            return
        # A zero stamp would make tf2 return the latest transform.
        stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        if stamp.nanoseconds == 0:
            self._drop('the depth image has no header stamp', stamp)
            return
        self._stamp_wait.submit(msg, stamp)

    def _missing_tf(self, msg, stamp):
        """Return (base, frame) of a transform `msg` lacks at `stamp`, or None."""
        base = self.get_parameter('robot_base_frame').get_parameter_value().string_value
        for frame in self._frames_needed(msg):
            if not self.tf_buffer.can_transform(base, frame, stamp):
                return (base, frame)
        return None

    def _frames_needed(self, msg):
        """Return the depth frame and every mask frame: all are posed in the base frame."""
        frames = [msg.header.frame_id]
        for m in self._masks.values():
            if m['frame_id'] and m['frame_id'] not in frames:
                frames.append(m['frame_id'])
        return frames

    def _drop(self, reason, stamp):
        self.get_logger().warn(
            f'{reason}. Dropping this depth frame (stamp {stamp.nanoseconds * 1e-9:.6f} s, '
            'nothing published).',
            throttle_duration_sec=2.0)

    def _segment_at_stamp(self, msg, stamp):
        """
        Segment `msg` with the TF and joint states at `stamp`; drop it if either is missing.

        Masking with any other pose would cut the robot where it was not at
        capture time, and leave it in the depth where it was.
        """
        q, reason = self._joint_history.at(stamp)
        if q is None:
            self._drop(f'No joint positions at the depth stamp: {reason}', stamp)
            return
        if self._locked_measured:
            locked, reason = self._locked_history.at(stamp)
            if locked is None:
                self._drop(f'No {self._locked_joints.names} positions at the depth stamp: '
                           f'{reason}', stamp)
                return
            self._locked_joints.update(locked)
        elif self._locked_joints.names and not self._locked_configured_logged:
            self._locked_configured_logged = True
            self.get_logger().info(
                f'{self._locked_joints.names} not on '
                f'{self.subscription_joint_states.topic_name} yet: masked at their configured '
                f'values {self._locked_joints.values}')
        base = self.get_parameter('robot_base_frame').get_parameter_value().string_value
        transforms = {}
        for frame in self._frames_needed(msg):
            try:
                transforms[frame] = self._tf_matrix(base, frame, stamp)
            except tf2_ros.TransformException as ex:
                self._drop(f'No TF {frame} -> {base} at the depth stamp: {ex}', stamp)
                return

        try:
            # Convert ROS Image message to numpy array
            # Assuming depth is in millimeters (16UC1) or meters (32FC1)
            if msg.encoding == "16UC1":
                depth_img = self.bridge.imgmsg_to_cv2(msg, "16UC1")
                # Convert from millimeters to meters
                depth_img = depth_img.astype(np.float32) / 1000.0
            elif msg.encoding == "32FC1":
                depth_img = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            else:
                self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
                return

            # Convert to torch tensor
            self.depth_image = torch.from_numpy(depth_img).to(
                dtype=self._ops_dtype, device=self._device)
            self.depth_header = msg.header
            # base <- frame at the depth stamp, for the depth frame and every
            # mask frame of this depth frame.
            self._frame_transforms = transforms

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        self.q_js.position = torch.tensor(q, dtype=self._ops_dtype, device=self._device)
        self.segment_and_publish()

    def segment_and_publish(self):
        """
        Performs the segmentation by masking the robot in the depth image.
        Publishes the filtered depth image as a ROS2 message.
        """
        masked_depth = self._mask_depth_image(self.depth_image, self.q_js.position)
        masked_depth_msg = self.depth_tensor_to_image_msg(masked_depth)
        self.publisher_.publish(masked_depth_msg)

    def depth_to_pointcloud(self, depth_image: torch.Tensor, camera_intrinsics: dict) -> torch.Tensor:
        """
        Converts a depth image to a 3D point cloud.

        Args:
            depth_image: torch.Tensor of shape (H, W) with depth values in meters
            camera_intrinsics: dict with keys 'fx', 'fy', 'cx', 'cy'

        Returns:
            torch.Tensor of shape (N, 3) representing 3D points (x, y, z)
        """
        H, W = depth_image.shape

        # Get camera intrinsics
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']

        # Create pixel coordinate grid
        v_coords, u_coords = torch.meshgrid(
            torch.arange(H, dtype=self._ops_dtype, device=self._device),
            torch.arange(W, dtype=self._ops_dtype, device=self._device),
            indexing='ij')

        # Get valid depth values (non-zero and non-nan)
        valid_mask = (depth_image > 0) & (~torch.isnan(depth_image)) & (~torch.isinf(depth_image))

        # Extract valid pixels
        u_valid = u_coords[valid_mask]
        v_valid = v_coords[valid_mask]
        z_valid = depth_image[valid_mask]

        # Convert to 3D coordinates using pinhole camera model
        # x = (u - cx) * z / fx
        # y = (v - cy) * z / fy
        x = (u_valid - cx) * z_valid / fx
        y = (v_valid - cy) * z_valid / fy
        z = z_valid

        # Stack into (N, 3) point cloud
        points = torch.stack([x, y, z], dim=1)

        return points, valid_mask

    def pointcloud_to_depth(self, points: torch.Tensor, camera_intrinsics: dict,
                           image_shape: tuple) -> torch.Tensor:
        """
        Converts a 3D point cloud back to a depth image.

        Args:
            points: torch.Tensor of shape (N, 3) representing 3D points
            camera_intrinsics: dict with keys 'fx', 'fy', 'cx', 'cy'
            image_shape: tuple (H, W) for the output depth image

        Returns:
            torch.Tensor of shape (H, W) representing the depth image
        """
        H, W = image_shape

        # Get camera intrinsics
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']

        # Initialize depth image with zeros
        depth_image = torch.zeros((H, W), dtype=self._ops_dtype, device=self._device)

        # Project 3D points to 2D pixel coordinates
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Avoid division by zero
        valid_z = z > 0
        x_valid = x[valid_z]
        y_valid = y[valid_z]
        z_valid = z[valid_z]

        # Project to pixel coordinates
        u = (x_valid * fx / z_valid + cx).long()
        v = (y_valid * fy / z_valid + cy).long()

        # Filter points within image bounds
        valid_pixels = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u_valid = u[valid_pixels]
        v_valid = v[valid_pixels]
        z_final = z_valid[valid_pixels]

        # Fill depth image (handle occlusions by keeping closest depth)
        # Vectorized approach: sort points by depth (far to near) and scatter
        # Points written last (nearest) will overwrite farther points at same pixel

        if len(z_final) > 0:
            # Sort by depth (descending order: farthest to nearest)
            # This way, when we scatter, nearest points overwrite farther ones
            sorted_indices = torch.argsort(z_final, descending=True)
            u_sorted = u_valid[sorted_indices]
            v_sorted = v_valid[sorted_indices]
            z_sorted = z_final[sorted_indices]

            # Convert 2D indices (v, u) to 1D linear indices
            linear_indices = v_sorted * W + u_sorted

            # Scatter depth values into flattened depth image
            # Later values (closer points) will overwrite earlier ones (farther points)
            depth_image.view(-1).scatter_(0, linear_indices, z_sorted)

        return depth_image

    def _mask_depth_image(self, depth_image: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Masks the depth image by removing robot points.

        Strategy:
        1. Convert depth image to point cloud
        2. Apply robot masking on point cloud (reusing existing logic)
        3. Convert masked point cloud back to depth image

        Args:
            depth_image: torch.Tensor of shape (H, W)
            q: torch.Tensor of joint positions

        Returns:
            torch.Tensor of shape (H, W) with robot pixels masked (set to 0)
        """
        # Get camera intrinsics from camera_info
        camera_intrinsics = {
            'fx': self.camera_info.k[0],
            'fy': self.camera_info.k[4],
            'cx': self.camera_info.k[2],
            'cy': self.camera_info.k[5]
        }

        # Compute FK once. Sphere and mesh backends consume this same state, so
        # both geometries correspond to the depth stamp's interpolated joints.
        q_batched = q.unsqueeze(0) if q.ndim == 1 else q
        js = JointState(position=q_batched, joint_names=self._kin_model.joint_names)
        kinematics_state = self._kin_model.compute_kinematics(js)
        robot_spheres = kinematics_state.robot_spheres.reshape(-1, 4)
        self.publish_collision_spheres(robot_spheres)

        # Step 1: Convert depth image to point cloud.
        points, valid_mask = self.depth_to_pointcloud(depth_image, camera_intrinsics)

        # Step 2: Apply the sphere backend and the independent analytic object
        # masks. Mesh-only mode preserves the input pixels when no object masks
        # are configured instead of doing a needless point-cloud round trip.
        use_spheres = self._segmentation_mode in ('sphere', 'combined')
        if use_spheres or self._masks:
            filtered_points, point_keep, points_base = self._mask_pointcloud(
                points, kinematics_state, include_spheres=use_spheres)
            masked_depth = self.pointcloud_to_depth(
                filtered_points,
                camera_intrinsics,
                depth_image.shape)
        else:
            points_base = self._transform_points_to_base(points)
            point_keep = torch.ones(
                points.shape[0], dtype=torch.bool, device=self._device)
            masked_depth = depth_image.clone()

        full_mask = torch.zeros_like(depth_image, dtype=torch.bool)
        full_mask[valid_mask] = ~point_keep

        # Step 3: Project the configured link hulls and union their silhouette
        # with the sphere/object result.
        if self._mesh_segmenter is not None:
            mesh_mask = self._mesh_mask_from_state(
                depth_image, camera_intrinsics, kinematics_state)
            masked_depth.masked_fill_(mesh_mask, 0.0)
            full_mask.logical_or_(mesh_mask)

        if self.robot_pointcloud_pub.get_subscription_count() > 0:
            masked_points = points_base[full_mask[valid_mask]]
            if masked_points.shape[0] > 0:
                robot_frame = self.get_parameter(
                    'robot_base_frame').get_parameter_value().string_value
                self.robot_pointcloud_pub.publish(self._create_pointcloud2_msg(
                    masked_points, robot_frame, self.depth_header.stamp))

        return masked_depth

    def _mesh_mask_from_state(self, depth_image, camera_intrinsics,
                              kinematics_state):
        """Render configured link poses from one synchronized FK state."""
        tool_poses = kinematics_state.tool_poses
        if tool_poses is None:
            raise RuntimeError('cuRobo FK did not return mesh link poses')
        missing = [name for name in self._mesh_link_names
                   if name not in tool_poses.tool_frames]
        if missing:
            raise RuntimeError(
                f'cuRobo FK did not return mesh link poses for {missing}')
        tool_poses = tool_poses.reorder_links(self._mesh_link_names)
        positions = tool_poses.position[:, 0, :, :].reshape(-1, 3)
        quaternions = tool_poses.quaternion[:, 0, :, :].reshape(-1, 4)
        base_from_camera = self._frame_transforms[self.depth_header.frame_id]
        intrinsics = (
            camera_intrinsics['fx'], camera_intrinsics['fy'],
            camera_intrinsics['cx'], camera_intrinsics['cy'])
        return self._mesh_segmenter.mask(
            depth_image,
            intrinsics,
            base_from_camera,
            positions,
            quaternions,
        )

    def _mask_pointcloud(self, point_cloud: torch.Tensor, kinematics_state,
                         include_spheres=True):
        """
        Applies masking operation to remove points belonging to the robot.
        This is the same logic as in the original robot_segmentation.py

        Args:
            point_cloud: torch.Tensor of shape (N, 3)
            kinematics_state: cuRobo FK result for the depth stamp
            include_spheres: whether collision spheres contribute to the mask

        Returns:
            Tuple of filtered camera points, keep mask, and base-frame points
        """
        # v2: `robot_spheres` replaces `link_spheres_tensor`, shape [B, H, N, 4]
        robot_spheres = kinematics_state.robot_spheres.reshape(-1, 4)

        # The point cloud is in the camera optical frame, but robot_spheres are
        # in the robot base frame. Transform the points into the base frame for
        # the distance test; keep the original camera-frame cloud so the depth
        # is reconstructed in the camera frame downstream.
        points_base = self._transform_points_to_base(point_cloud)

        if include_spheres:
            points = points_base.unsqueeze(1)  # (N, 1, 3)
            spheres_centers = robot_spheres[:, :3].unsqueeze(0)  # (1, S, 3)
            spheres_radii = robot_spheres[:, 3].unsqueeze(0)  # (1, S)
            distances = torch.norm(
                points - spheres_centers, dim=2) - spheres_radii
            min_distances, _ = torch.min(distances, dim=1)
            mask = min_distances > self.distance_threshold
        else:
            mask = torch.ones(
                point_cloud.shape[0], dtype=torch.bool, device=self._device)

        # Also drop points inside any user-defined mask shape (e.g. a grasped
        # object) so they never reach the mapper / ESDF. Uses the same base-frame
        # cloud; each mask follows its TF frame.
        inside_shapes = self._shape_inside_mask(points_base)
        if inside_shapes is not None:
            mask = mask & ~inside_shapes

        return point_cloud[mask], mask, points_base

    def _transform_points_to_base(self, points: torch.Tensor):
        """Transform camera-frame points (N, 3) into the robot base frame via TF.

        Robot collision spheres are expressed in the kinematic base frame while
        the depth point cloud is in the camera optical frame; without this
        transform the masking compares mismatched coordinate systems. The TF
        comes from the hand-eye calibration (base -> camera), at the depth stamp.
        """
        T = self._frame_transforms[self.depth_header.frame_id]

        homog = torch.cat(
            [points, torch.ones((points.shape[0], 1),
                                dtype=self._ops_dtype, device=self._device)],
            dim=1)  # (N, 4)
        return (homog @ T.T)[:, :3]

    def _tf_matrix(self, target_frame: str, source_frame: str, stamp):
        """4x4 homogeneous transform ``target <- source`` from TF at ``stamp``.

        Shared by the robot base transform and the per-mask frame tracking.
        Raises tf2_ros.TransformException if the transform is unavailable at ``stamp``.
        """
        tf = self.tf_buffer.lookup_transform(target_frame, source_frame, stamp)

        tr = tf.transform.translation
        rot = tf.transform.rotation
        R = Rotation.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
        T = torch.eye(4, dtype=self._ops_dtype, device=self._device)
        T[:3, :3] = torch.tensor(R, dtype=self._ops_dtype, device=self._device)
        T[:3, 3] = torch.tensor([tr.x, tr.y, tr.z],
                                dtype=self._ops_dtype, device=self._device)
        return T

    def _pose_matrix(self, pos: torch.Tensor, quat) -> torch.Tensor:
        """4x4 homogeneous transform from a position tensor (3,) and a scipy
        quaternion [x, y, z, w]."""
        R = Rotation.from_quat(quat).as_matrix()
        T = torch.eye(4, dtype=self._ops_dtype, device=self._device)
        T[:3, :3] = torch.tensor(R, dtype=self._ops_dtype, device=self._device)
        T[:3, 3] = pos
        return T

    def _shape_inside_mask(self, points_base: torch.Tensor):
        """Boolean (N,) mask, True where a base-frame point lies inside ANY
        user-defined mask shape. Returns None when no masks are defined.

        Each mask pose is expressed in its ``frame_id`` (re-resolved at every
        depth stamp so it follows the arm); an empty frame_id means the robot
        base frame.
        """
        if not self._masks:
            return None

        N = points_base.shape[0]
        homog = torch.cat(
            [points_base, torch.ones((N, 1),
                                     dtype=self._ops_dtype, device=self._device)],
            dim=1)  # (N, 4)
        inside_any = torch.zeros(N, dtype=torch.bool, device=self._device)

        for m in self._masks.values():
            T_frame_mask = self._pose_matrix(m['pos'], m['quat'])
            if m['frame_id']:
                T_base_mask = self._frame_transforms[m['frame_id']] @ T_frame_mask
            else:
                T_base_mask = T_frame_mask

            # Points into the mask-local frame, then analytic inside test.
            local = (homog @ torch.inverse(T_base_mask).T)[:, :3]
            inside = self._inside_shape(local, m)
            if inside is not None:
                inside_any |= inside

        return inside_any

    def _inside_shape(self, local: torch.Tensor, m: dict):
        """Analytic point-inside test in the shape's local frame (+mask_margin).
        Returns bool (N,), or None for unsupported types."""
        t = self.mask_margin
        dims = m['dims']
        typ = m['type']

        if typ == 0:
            hx, hy, hz = dims[0] / 2 + t, dims[1] / 2 + t, dims[2] / 2 + t
            return ((local[:, 0].abs() <= hx)
                    & (local[:, 1].abs() <= hy)
                    & (local[:, 2].abs() <= hz))

        if typ == 1:
            return torch.norm(local, dim=1) <= dims[0] + t

        if typ == 2:  # centered, axis z, z in [-h/2, h/2]
            r, hz = dims[0] + t, dims[1] / 2 + t
            radial = torch.norm(local[:, :2], dim=1)
            return (local[:, 2].abs() <= hz) & (radial <= r)

        if typ == 3:  # segment [0,0,0]->[0,0,h] along +z
            r, h = dims[0] + t, dims[1]
            z_clamped = local[:, 2].clamp(0.0, h)
            dz = local[:, 2] - z_clamped
            dist = torch.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2 + dz ** 2)
            return dist <= r

        if typ == 4:
            self.get_logger().warn(
                "MESH mask not supported analytically yet - skipped. "
                "(future: curobo WorldMeshCollision zero-radius point SDF)",
                throttle_duration_sec=5.0)
            return None

        return None

    def depth_tensor_to_image_msg(self, depth_tensor: torch.Tensor) -> Image:
        """
        Converts a depth tensor to a ROS2 Image message.

        Args:
            depth_tensor: torch.Tensor of shape (H, W)

        Returns:
            Image message
        """
        # Convert to numpy and scale back to millimeters for 16UC1 encoding
        depth_np = depth_tensor.cpu().numpy()
        depth_mm = (depth_np * 1000.0).astype(np.uint16)

        # Create Image message
        msg = self.bridge.cv2_to_imgmsg(depth_mm, encoding="16UC1")
        msg.header = self.depth_header

        return msg

    def publish_collision_spheres(self, robot_spheres):
        """
        Publishes the robot's collision spheres as markers for visualization in RViz.

        Args:
            robot_spheres: torch.Tensor of shape (N, 4) where each row is [x, y, z, radius]
        """
        robot_spheres = robot_spheres.cpu().numpy().tolist()
        marker_array = MarkerArray()

        for i, sphere in enumerate(robot_spheres):
            marker = Marker()
            marker.header.frame_id = self.get_parameter('robot_base_frame').get_parameter_value().string_value
            marker.header.stamp = self.depth_header.stamp
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = i
            marker.pose.position.x = sphere[0]
            marker.pose.position.y = sphere[1]
            marker.pose.position.z = sphere[2]
            marker.scale.x = sphere[3] * 2  # Diameter
            marker.scale.y = sphere[3] * 2
            marker.scale.z = sphere[3] * 2
            marker.color.a = 0.5  # Transparency
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)

        self.sphere_marker_pub.publish(marker_array)


    def _create_pointcloud2_msg(self, points: torch.Tensor, frame_id: str, timestamp) -> PointCloud2:
        """
        Create a PointCloud2 message from a torch tensor of 3D points.

        Args:
            points: torch.Tensor of shape (N, 3) containing XYZ coordinates
            frame_id: Frame ID for the point cloud
            timestamp: ROS timestamp

        Returns:
            PointCloud2 message
        """
        # Convert tensor to numpy on CPU
        points_np = points.cpu().numpy()

        # Create PointCloud2 message
        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = timestamp
        msg.header.frame_id = frame_id

        # Define point cloud fields (X, Y, Z)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        msg.is_bigendian = False
        msg.point_step = 12  # 3 * 4 bytes (float32)
        msg.row_step = msg.point_step * points_np.shape[0]
        msg.is_dense = True
        msg.height = 1
        msg.width = points_np.shape[0]

        # Pack point data
        buffer = []
        for point in points_np:
            buffer.append(struct.pack('fff', point[0], point[1], point[2]))

        msg.data = b''.join(buffer)

        return msg


def main(args=None):
    """
    Main entry point for the depth map segmentation node.
    """
    rclpy.init(args=args)
    node = DepthMapRobotSegmentation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    # See unified_planner_node.main: rclpy's SIGINT handler has already shut the
    # context down, so an unguarded shutdown() exits 1 on a clean Ctrl-C.
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
