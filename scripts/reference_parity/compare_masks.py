"""Compare existing standalone and ROS mask paths on identical bag frames."""
import os
import sys
sys.path.insert(0, '/ws/dynamic_scene/src')
sys.path.insert(0, '/ws/curobo_ros')

import numpy as np
import rclpy
import torch
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, JointState as RosJointState
from std_msgs.msg import Header
from curobo.types import CameraObservation, Pose
import mapper_config
import robot_segmenter
from curobo_ros.core.robot_segmentation import DepthMapRobotSegmentation

GRIPPER_LINKS = [
    'robotiq_140_base_link', 'left_outer_knuckle', 'left_outer_finger',
    'left_inner_knuckle', 'left_inner_finger', 'left_inner_finger_pad',
    'right_outer_knuckle', 'right_outer_finger', 'right_inner_knuckle',
    'right_inner_finger', 'right_inner_finger_pad',
]
data = np.load(sys.argv[1], allow_pickle=False)
os.chdir('/ws/dynamic_scene')
cfg, _ = mapper_config.load_config()
cfg.robot_segmenter.gripper_sphere_buffer = float(
    os.environ.get('PARITY_REF_GRIPPER_BUFFER', '0.0'))
cfg.robot_segmenter.profile = False
robot_segmenter._resolve_robot_asset_root = lambda fallback_dir=None: (
    '/ws/dynamic_scene/src/ros2_robotiq_gripper')
reference = robot_segmenter.build_segmenter(cfg, torch.device('cuda'))
args = ['--ros-args', '-p', 'robot:=/opt/e2e/ur_station_seg_map.yaml',
        '-p', 'robot_base_frame:=base_link', '-p', 'segmentation_mode:=combined',
        '-p', 'mesh_margin:=0.004', '-p', 'mesh_raster_stride:=8']
if os.environ.get('PARITY_AUTO_MESH') == '1':
    asset_root = os.environ.get('PARITY_MESH_ASSET_ROOT')
    if asset_root:
        args += ['-p', f'mesh_asset_roots:=[{asset_root}]']
else:
    args += ['-p', 'mesh_link_names:=[' + ','.join(GRIPPER_LINKS) + ']']
rclpy.init(args=args)
node = DepthMapRobotSegmentation()
mesh = node._mesh_segmenter
print('ref_active', reference.kinematics.joint_names,
      'ros_active', node._kin_model.joint_names,
      'ros_locked', node._locked_joints.names, flush=True)
outputs = {}
modes = (('combined', 'combined'),) if os.environ.get('PARITY_COMBINED_ONLY') == '1' else (
    ('mesh_only', 'mesh'), ('native_only', 'sphere'), ('combined', 'combined'))
try:
    for i in range(int(data['n'])):
        depth = torch.as_tensor(data[f'depth_{i}'], device='cuda').clone()
        K = torch.as_tensor(data['K'], device='cuda', dtype=torch.float32)
        xyz = data[f'camera_xyz_{i}']
        xyzw = data[f'camera_xyzw_{i}']
        quat = np.roll(xyzw, 1)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = Rotation.from_quat(xyzw).as_matrix()
        T[:3, 3] = xyz
        frame_id = str(data[f'frame_id_{i}'])
        node.camera_info = CameraInfo(k=data['K'].reshape(-1).tolist())
        node.depth_header = Header(frame_id=frame_id)
        node._frame_transforms = {frame_id: torch.as_tensor(T, device='cuda')}
        q_raw = data[f'joint_q_{i}'].tolist()
        names = data['joint_names'].tolist()
        msg = RosJointState(name=names, position=q_raw)
        ref_js = robot_segmenter.joint_state_from_msg(
            msg, torch.device('cuda'), required_joint_names=reference.kinematics.joint_names,
            position_maps=reference._joint_position_maps)
        ros_msg = node._joint_state_map.to_model(msg)
        ros_q = dict(zip(ros_msg.name, ros_msg.position))
        model_finger = float(ros_q['finger_joint'])
        node._locked_joints.update([model_finger])
        q = torch.as_tensor([ros_q[name] for name in node._kin_model.joint_names],
                            dtype=torch.float32, device='cuda')
        ref_state = reference.kinematics.compute_kinematics(
            reference.kinematics.get_active_js(ref_js))
        from curobo.types import JointState as CuroboJointState
        ros_state = node._kin_model.compute_kinematics(CuroboJointState(
            position=q.unsqueeze(0), joint_names=node._kin_model.joint_names))
        ref_spheres = ref_state.robot_spheres.reshape(-1, 4)
        ros_spheres = ros_state.robot_spheres.reshape(-1, 4)
        print('fk', i, 'sphere_shape', tuple(ref_spheres.shape),
              tuple(ros_spheres.shape), 'max_xyz_delta_m',
              float(torch.max(torch.abs(ref_spheres[:, :3] - ros_spheres[:, :3]))),
              'max_radius_delta_m',
              float(torch.max(torch.abs(ref_spheres[:, 3] - ros_spheres[:, 3]))),
              flush=True)
        pose = Pose(position=torch.as_tensor(xyz[None], device='cuda'),
                    quaternion=torch.as_tensor(quat[None], device='cuda'))
        print('frame', i, 'stamp_ns', int(data[f'stamp_ns_{i}']),
              'finger_ref', float(ref_js.position[0, names.index('finger_joint')]),
              'finger_ros', model_finger, flush=True)
        for mode_ref, mode_ros in modes:
            obs = CameraObservation(name='wrist', depth_image=depth.clone(),
                                    intrinsics=K.clone(), pose=pose)
            reference._mask_mode = mode_ref
            node._segmentation_mode = mode_ros
            node._mesh_segmenter = None if mode_ros == 'sphere' else mesh
            ref_mask = robot_segmenter.mask_robot(reference, obs, ref_js)
            ros_depth = node._mask_depth_image(depth, q)
            torch.cuda.synchronize()
            ros_mask = (depth > 0) & (ros_depth == 0)
            a = ref_mask.detach().cpu().numpy().astype(bool)
            b = ros_mask.detach().cpu().numpy().astype(bool)
            only_ref = int((a & ~b).sum())
            only_ros = int((b & ~a).sum())
            print('mask', i, mode_ref, 'reference', int(a.sum()),
                  'ros', int(b.sum()), 'ref_only', only_ref,
                  'ros_only', only_ros, flush=True)
            outputs[f'ref_{mode_ref}_{i}'] = a
            outputs[f'ros_{mode_ref}_{i}'] = b
finally:
    node.destroy_node()
    rclpy.shutdown()
np.savez_compressed(sys.argv[2], **outputs)
