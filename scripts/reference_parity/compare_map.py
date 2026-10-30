"""Drive ROS camera strategy and standalone Mapper with identical bag frames."""
import os
import sys
sys.path.insert(0, '/ws/dynamic_scene/src')
sys.path.insert(0, '/ws/curobo_ros')

import numpy as np
import rclpy
import torch
from builtin_interfaces.msg import Time
from curobo.perception import FilterDepth, Mapper
from curobo.types import CameraObservation, Pose
from sensor_msgs.msg import Image
import mapper_config
from mapper_pipeline import build_mapper_cfg
from curobo_ros.cameras.camera_depth_map_strategy import DepthMapCameraStrategy

data = np.load(sys.argv[1], allow_pickle=False)
masks = np.load(sys.argv[2], allow_pickle=False)
os.chdir('/ws/dynamic_scene')
cfg, _ = mapper_config.load_config()
device = torch.device('cuda')
ref_mapper = Mapper(build_mapper_cfg(cfg, device))
ros_mapper = Mapper(build_mapper_cfg(cfg, device))
filter_ref = FilterDepth(image_shape=data['depth_0'].shape,
                         depth_minimum_distance=cfg.depth_filter.depth_min,
                         depth_maximum_distance=cfg.depth_filter.depth_max,
                         flying_pixel_threshold=cfg.depth_filter.flying_pixel_threshold,
                         device='cuda')
rclpy.init()
node = rclpy.create_node('reference_parity_camera')
node.declare_parameter('base_link', 'base_link')
node.mapper = ros_mapper
strategy = DepthMapCameraStrategy(
    node, 'wrist', topic='/parity/unused', frame_id=str(data['frame_id_0']),
    intrinsics=data['K'].reshape(-1).tolist(), extrinsics=None,
    depth_filter={'enabled': True, 'depth_min': cfg.depth_filter.depth_min,
                  'depth_max': cfg.depth_filter.depth_max,
                  'flying_pixel_threshold': cfg.depth_filter.flying_pixel_threshold})

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import SceneCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import DeviceCfg, JointState
from curobo_ros.robot.robot_description import load_robot_description
from curobo_ros.robot.locked_joints import LockedJoints
descriptor = '/opt/e2e/ur_station_seg_map.yaml'
robot_file = load_robot_description(descriptor).curobo_config_path
kin = Kinematics(KinematicsCfg.from_robot_yaml_file(robot_file))
locked = LockedJoints(kin, robot_file)
scenes = None

try:
    for i in range(int(data['n'])):
        raw = data[f'depth_{i}']
        mask_ref = masks[f'ref_combined_{i}']
        mask_ros = masks[f'ros_combined_{i}']
        assert np.array_equal(mask_ref, mask_ros)
        masked = np.where(mask_ref, 0.0, raw).astype(np.float32)
        wire_mm = np.rint(masked * 1000).astype(np.uint16)
        pose = Pose(
            position=torch.as_tensor(data[f'camera_xyz_{i}'][None], device=device),
            quaternion=torch.as_tensor(np.roll(data[f'camera_xyzw_{i}'], 1)[None].copy(), device=device))
        depth = torch.as_tensor(masked, device=device)
        filtered, _ = filter_ref(depth.unsqueeze(0))
        rgb = torch.zeros((*depth.shape, 3), dtype=torch.uint8, device=device)
        ref_obs = CameraObservation(
            name='wrist', depth_image=filtered,
            rgb_image=rgb.unsqueeze(0),
            intrinsics=torch.as_tensor(data['K'], device=device, dtype=torch.float32).unsqueeze(0),
            pose=pose)
        ref_mapper.integrate(ref_obs)
        torch.cuda.synchronize()

        ns = int(data[f'stamp_ns_{i}'])
        message = Image(height=depth.shape[0], width=depth.shape[1], encoding='16UC1',
                        step=2 * depth.shape[1], data=wire_mm.tobytes())
        message.header.frame_id = str(data[f'frame_id_{i}'])
        message.header.stamp = Time(sec=ns // 10**9, nanosec=ns % 10**9)
        strategy._integrate(message, pose)
        assert strategy.depth_map is not None
        filter_diff = int(torch.count_nonzero(strategy.depth_map != filtered[0]))
        a, b = ref_mapper.tsdf.data, ros_mapper.tsdf.data
        na, nb = int(a.num_allocated.item()), int(b.num_allocated.item())
        n = min(na, nb)
        coords_equal = torch.equal(a.block_coords[:3*n], b.block_coords[:3*n])
        coord_a = a.block_coords[:3*na].reshape(-1, 3).cpu().numpy()
        coord_b = b.block_coords[:3*nb].reshape(-1, 3).cpu().numpy()
        lookup_b = {tuple(x): j for j, x in enumerate(coord_b)}
        same_coords = na == nb and all(tuple(x) in lookup_b for x in coord_a)
        aligned_value_diff = None
        if same_coords:
            order_b = torch.as_tensor(
                [lookup_b[tuple(x)] for x in coord_a], device=device)
            aligned_value_diff = int(torch.count_nonzero(
                a.block_data[:na] != b.block_data[order_b]))
        print('frame', i, 'filter_diff', filter_diff,
              'blocks', na, nb, 'same_block_order', coords_equal,
              'same_block_coords', same_coords,
              'tsdf_value_diff_aligned', aligned_value_diff, flush=True)

        vg_ref_step = ref_mapper.compute_esdf()
        vg_ros_step = ros_mapper.compute_esdf()
        esdf_diff = int(torch.count_nonzero(
            vg_ref_step.feature_tensor != vg_ros_step.feature_tensor))
        scenes = [SceneCollision.from_config(SceneCollisionCfg(
            device_cfg=DeviceCfg(device='cuda'), scene_model=SceneCfg(voxel=[vg])))
            for vg in (vg_ref_step, vg_ros_step)]
        raw_q = data[f'joint_q_{i}']
        reported = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.689]
        model = [-0.074, 0.054, 0.174, 0.287, 0.407, 0.523, 0.617, 0.700]
        locked.update([float(np.interp(raw_q[6], reported, model))])
        state = kin.compute_kinematics(JointState(
            position=torch.as_tensor(raw_q[:6][None], device=device),
            joint_names=kin.joint_names))
        for activation in (0.0, 0.025):
            costs = []
            for scene in scenes:
                buffer = CollisionBuffer.from_shape(state.robot_spheres.shape,
                                                    DeviceCfg(device='cuda'))
                cost = scene.get_sphere_distance(
                    state, buffer, torch.ones(1, device=device),
                    activation_distance=torch.tensor([activation], device=device))
                costs.append(cost.reshape(-1))
            print('step_collision', i, 'activation', activation,
                  'esdf_diff', esdf_diff,
                  'max_ref', float(costs[0].max()),
                  'max_ros', float(costs[1].max()),
                  'positive_ref', int(torch.count_nonzero(costs[0] > 0)),
                  'positive_ros', int(torch.count_nonzero(costs[1] > 0)),
                  'cost_diff', float(torch.max(torch.abs(costs[0] - costs[1]))),
                  flush=True)

    vg_ref = ref_mapper.compute_esdf()
    vg_ros = ros_mapper.compute_esdf()
    x, y = vg_ref.feature_tensor, vg_ros.feature_tensor
    print('esdf_shape', tuple(x.shape), tuple(y.shape),
          'esdf_exact_diff', int(torch.count_nonzero(x != y)),
          'esdf_max_abs_diff', float(torch.max(torch.abs(x - y))), flush=True)

finally:
    node.destroy_node()
    rclpy.shutdown()
