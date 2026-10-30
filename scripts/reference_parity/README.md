# Recorded station parity probe

Run inside `curobo_ros_dev` on ROS domain 94 with localhost discovery. These
scripts read `/opt/bags/live_moving_redcube_20261026` and the current mounted
`/ws/dynamic_scene` and `/ws/curobo_ros` sources. They publish no robot command.
Output arrays and logs belong in the container's `/tmp`, outside Git.

```bash
export ROS_DOMAIN_ID=94 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export PATH=/opt/venv/bin:$PATH
source /opt/deps_ws/install/setup.bash
source /opt/ros_ws/install/setup.bash
python3 -u /ws/curobo_ros/scripts/reference_parity/extract_frames.py /tmp/reference_frames_1hz.npz > /tmp/reference_extract.log
python3 -u /ws/curobo_ros/scripts/reference_parity/compare_masks.py /tmp/reference_frames_1hz.npz /tmp/reference_masks_1hz.npz > /tmp/reference_masks.log
python3 -u /ws/curobo_ros/scripts/reference_parity/compare_map.py /tmp/reference_frames_1hz.npz /tmp/reference_masks_1hz.npz > /tmp/reference_map.log
```

`extract_frames.py` selects the nearest recorded depth frame to each second
from 0.5 through 88.5 s. It preserves the depth stamp and uses the bag's TF,
camera intrinsics, and name-indexed joint history. `compare_masks.py` runs
the standalone segmenter and ROS segmenter on those same tensors. Its default
comparison aligns the 11 gripper mesh links, 4 mm margin, 8 px stride, and
zero gripper sphere padding. It adapts only the reference Robotiq asset root
to the mounted source path. Set `PARITY_REF_GRIPPER_BUFFER=0.015` to compare
the actual standalone padding, or `PARITY_AUTO_MESH=1` to use ROS automatic
link selection. `PARITY_MESH_ASSET_ROOT=/tmp/phase5_assets` includes the two
station camera STLs if they have been copied there.

`compare_map.py` uses the standalone Mapper config on both paths: 14 mm TSDF,
20 mm ESDF, 47 mm truncation, 0.15–2.0 m depth range, and no decay. It feeds
the saved combined masks through standalone FilterDepth and the ROS camera
strategy with equivalent filter settings. TSDF blocks are matched by their
voxel coordinates because pool allocation order may differ. ESDF and
same-time start-state collision costs are compared after each frame at both
zero and 25 mm collision activation distance. A fresh collision scene is
constructed per map snapshot; reusing one scene with `load_collision_model`
produced invalid ~100 m diagnostic costs in this probe.

This fixed 1 Hz subset is a deterministic component and collision parity
experiment. It does not reproduce ROS callback scheduling, the original
standalone live frame selection, or planner outcomes.
