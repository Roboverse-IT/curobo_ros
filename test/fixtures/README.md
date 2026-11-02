# Optional station integration assets

`test_dynamic_scene_lifecycle_integration.py` is a real-CUDA integration test
for the external UR10e + Robotiq station model. The robot XRDF and URDF are not
redistributed in this repository. Run it only where those assets are already
available:

```bash
CUROBO_ROS_STATION_ASSETS=/path/to/station-assets \
  python3 -m pytest -q test/test_dynamic_scene_lifecycle_integration.py
```

The supplied directory must contain `config/packing_station.xrdf` and
`config/ur_robotiq_camera.urdf`, plus every asset those files reference. The
test skips when the variable or assets are absent; generic CI remains covered
by the portable Mapper, attachment-lifecycle, and camera-to-planner tests.
