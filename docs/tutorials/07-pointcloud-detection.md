# Tutorial 7: Camera-Based Obstacle Detection

**Difficulty**: Intermediate · **Time**: ~40 min · **Prerequisites**: [Tutorial 3](03-collision-objects.md), a depth camera (or a rosbag with depth topics)

curobo_ros integrates depth cameras directly into the collision world. Each depth frame is pushed into cuRobo v2's `Mapper` (a GPU TSDF/ESDF), and every solver — planning, MPC, IK — sees the result. No point-cloud processing on your side, no service calls: anything the camera sees becomes an obstacle.

## How it works

```
depth image topic ──► DepthMapCameraStrategy ──► mapper.integrate(CameraObservation)
                                                        │ (GPU TSDF, decays over time)
                                                        ▼
                                              ESDF shared by all solvers
```

Key properties:

- **Push-based**: integration happens in the camera callback, under a non-blocking GPU lock. Frames arriving while the GPU is busy (e.g. during CUDA-graph capture) are dropped — the next one catches up.
- **Voxels decay**: stale voxels fade with half-life `decay_half_life_s` (default 0.7 s), so a person walking through the scene doesn't leave a permanent ghost. The decay rate is derived from the cameras' declared `frame_rate_hz`.
- Only `type: depth_camera` is supported in v2 (the old pull-based point-cloud path was removed).

## 1. Configure your camera (`cameras.yaml`)

Pass a config file at launch: `cameras_config_file:=/path/to/cameras.yaml`. The shipped example is `config/cameras.yaml` (an Azure Kinect):

```yaml
cameras:
  - name: my_depth_cam
    type: depth_camera                      # the only supported type
    topic: /depth_to_rgb/image_raw          # sensor_msgs/Image, 16UC1 (mm) or 32FC1 (m)
    frame_id: rgb_camera_link               # required: header.frame_id of the depth images
    camera_info: /depth_to_rgb/camera_info  # used when intrinsics is empty
    intrinsics: []                          # [] = read once from camera_info (5 s wait)
    # or a 9-element row-major K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    extrinsics: [0.161, -0.428, 1.585, -0.3409296, 0.6840516, -0.6215661, 0.1717437]
    # [x, y, z, qw, qx, qy, qz] camera pose in the robot base frame;
    # [] = look it up from TF (base frame -> frame_id) at each image's stamp
    frame_rate_hz: 30.0                     # declared publication rate (drives decay)
    tf_wait_max_s: 0.05                     # TF mode: max wait for the TF at the image stamp
```

Two decisions per camera:

- **Intrinsics** — leave `[]` to read them once from the `camera_info` topic at startup, or hardcode the 9-element K matrix.
- **Extrinsics** — hardcode the 7-element pose (recommended once calibrated), or leave `[]` to resolve TF `base frame → frame_id` at each image's `header.stamp` (the acquisition time, so a moving camera is posed where it was when the image was taken). With TF, frames are **dropped** when the transform at that stamp is unavailable or the stamp is zero — there is no identity-pose or latest-transform fallback.

A depth image often arrives before the TF at its stamp (the next TF sample is up to one TF publish period away). Such a frame waits, at most `tf_wait_max_s` (default 0.05 s, enough for TF at 20 Hz or faster), and is integrated as soon as that TF arrives; only the newest frame waits, an older one still waiting when the next arrives is dropped. `0` drops it at once.

`frame_id` is required and must equal the `header.frame_id` of the depth images: frames from any other frame are dropped with an error, because they would be projected with the wrong pose.

Multiple cameras: add more list entries. Their rates sum for the decay computation, and all integrate into the same Mapper.

## 2. Launch with the camera

```bash
ros2 launch curobo_ros gen_traj.launch.py robot:=emulator \
  cameras_config_file:=$(ros2 pkg prefix curobo_ros)/share/curobo_ros/config/cameras.yaml
```

Startup log lines to look for:

```
Loading camera configuration from: ...
Added camera strategy 'my_depth_cam' of type 'depth_camera' ...
Camera intrinsics from topic: fx=...        # or: Using static extrinsics from config file
DepthMap camera initialized with depth topic: /depth_to_rgb/image_raw
```

Relevant perception parameters (see [Parameters](../concepts/parameters.md)): `mapper_extent_xyz` (perception volume), `voxel_size`, `decay_half_life_s`. Depth frames are integrated at their own size; cameras of different resolutions can feed the same Mapper.

## 3. Verify obstacles are seen

Wave a hand (or a box) in front of the camera, inside the mapper volume, then:

```bash
# Occupied voxels streamed as a sparse grid (default 7 Hz)
ros2 topic echo /unified_planner/voxel_grid_sparse --once

# Distances of the robot spheres to the nearest obstacle
ros2 service call /unified_planner/get_collision_distance curobo_msgs/srv/GetCollisionDistance

# Full voxel snapshot of a region
ros2 service call /unified_planner/get_voxel_grid curobo_msgs/srv/GetVoxelGrid \
  "{bbox_min_x: -1.0, bbox_min_y: -1.0, bbox_min_z: 0.0, bbox_max_x: 1.0, bbox_max_y: 1.0, bbox_max_z: 1.5}"
```

Then plan through the space the obstacle occupies ([Tutorial 1](01-first-trajectory.md)): the trajectory deflects around it. Remove the obstacle, wait a second (decay), and the same plan goes straight again. With MPC active ([Tutorial 5](05-mpc-planner.md)) the avoidance happens *during* motion.

## 4. Remove the robot from its own view (`robot_segmentation`)

If the camera sees the robot arm, the arm becomes an "obstacle" for itself. The second executable of the package subtracts the robot from the depth image before integration:

```bash
ros2 run curobo_ros robot_segmentation
```

It computes the robot geometry at the joint state of the depth image's stamp
and republishes the cleaned image. The compatibility default uses collision
spheres; projected visual meshes and the union of both are also available:

```bash
# Conservative self-mask: collision spheres plus projected link hulls
ros2 run curobo_ros robot_segmentation --ros-args \
  -p segmentation_mode:=combined \
  -p mesh_margin:=0.004
```

`mesh_margin` is calibration-dependent; zero is the neutral default. Select a
subset when only particular links need projected geometry:

```bash
ros2 run curobo_ros robot_segmentation --ros-args \
  -p segmentation_mode:=combined \
  -p mesh_link_names:="[gripper_base, left_finger, right_finger]"
```

| Interface | Name | Notes |
|---|---|---|
| Subscribes | `/depth_to_rgb/image_raw`, `/depth_to_rgb/camera_info` | Params `depth_image_topic` / `camera_info_topic` |
| Subscribes | the robot's joint states | The descriptor's `strategy_params.joint_states_topic` |
| Publishes | `/masked_depth_image` | The depth image with the robot removed, with the source image header (stamp and frame) |
| Publishes | `/collision_spheres`, `/robot_pointcloud_debug` | Debug visualization |
| Services | `/set_mask`, `/remove_mask` | Extra user-defined masks, optionally riding TF frames |

To use it, point the camera entry's `topic` at `/masked_depth_image` instead of
the raw depth topic. `segmentation_mode:=sphere` masks points within
`distance_threshold` (default 0.05 m) of collision spheres.
`segmentation_mode:=mesh` projects cached convex hulls of URDF visual meshes;
`combined` removes the union. Mesh links default to
`kinematics.mesh_link_names`, then `collision_link_names`; use
`mesh_link_names` to make the selection explicit. See the
[parameter reference](../concepts/parameters.md#robot_segmentation-node-parameters)
for asset roots, geometric margin, and raster stride.

A frame is segmented only with the TF and joint states at its stamp; it waits for them at most `stamp_wait_max_s` (default 0.05 s), and is dropped with a warning otherwise (see [ROS interfaces](../concepts/ros-interfaces.md#robot_segmentation-node)).

Mesh and sphere masks share that single timestamped FK result. The mesh
backend contains no ROS subscriptions or messages: initialization loads and
convexifies link-local geometry once, while each frame supplies camera
intrinsics, `base <- camera`, and link poses for GPU projection/rasterization.
Installed `package://` assets resolve through the ROS index; cuRobo-bundled
robot assets and uninstalled source packages can also be found automatically
or through `mesh_asset_roots`.

## Tuning

| Goal | Knob |
|---|---|
| Finer obstacles | `voxel_size` 0.02–0.03 (more VRAM; rebuild required) |
| Obstacles linger too long / flicker | `decay_half_life_s` up / down |
| Camera too far / too close clipped | `mapper_depth_min` / `mapper_depth_max` |
| Bigger workspace covered | `mapper_extent_xyz` (volume is centred on `mapper_grid_center`) |
| More safety margin | `collision_activation_distance` up |

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Failed to receive camera info from <topic>` at startup | `camera_info` topic wrong or not published within 5 s — check `ros2 topic list`, or hardcode `intrinsics` |
| `Could not transform <base> to <frame_id>` warnings | TF extrinsics chosen but the transform isn't published — publish a static TF or hardcode `extrinsics` |
| Obstacles appear shifted | Wrong extrinsics — re-check the calibration (position *and* quaternion order `[qw, qx, qy, qz]`) |
| Robot avoids itself / plans fail near the arm | Camera sees the arm — insert `robot_segmentation` (section 4) |
| Mesh mode skips a link | Its URDF visual mesh is absent/unresolvable. Install the description package, add its source root to `mesh_asset_roots`, or explicitly choose loadable `mesh_link_names` |
| Nothing appears in the voxel grid | Object outside `mapper_extent_xyz`, or depth encoding not `16UC1`/`32FC1`, or `blox` cache disabled |

## Next steps

- [Tutorial 5: MPC and Reactive Control](05-mpc-planner.md) — avoid moving obstacles while moving
- [Manager Architecture](../concepts/manager-architecture.md) — how perception integrates internally

[← Tutorial 6](06-ik-fk-services.md) | [Tutorials index](index.md)
