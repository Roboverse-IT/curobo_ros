# Tutorial 2: Adding Your Robot

**Difficulty**: Intermediate · **Time**: ~45 min · **Prerequisites**: [Tutorial 1](01-first-trajectory.md)

Integrating a robot into curobo_ros takes two files:

1. A **cuRobo robot configuration YAML** — kinematics (URDF), joint limits, and collision spheres. This is what the GPU solvers consume.
2. A **robot descriptor** — a small YAML in `robots/` that tells curobo_ros where the cuRobo config is and how to *command* the robot (topics, control strategy).

The Doosan M1013 shipped with the package is the working example throughout.

## 1. The robot descriptor (`robots/<name>.yaml`)

The `robot` launch argument selects a descriptor by name: `robot:=doosan_m1013` loads `robots/doosan_m1013.yaml`. The shipped one:

```yaml
name: doosan_m1013
display_name: "Doosan M1013"

# cuRobo robot config; package:// and relative paths are resolved automatically
curobo_config: package://curobo_ros/curobo_doosan/src/m1013/m1013.yml

# How to command the robot (see Tutorial 4): emulator | joint_speed | joint_pose
strategy: joint_speed
strategy_params:
  command_topic: /leeloo/execute_trajectory
  state_topic: /leeloo/trajectory_state
  joint_states_topic: /dsr01/joint_states
```

`base_link`, joint names, and DOF are **not** duplicated here — they come from the cuRobo config. `strategy_params` describes only the robot driver interface. The shipped `robots/emulator.yaml` shows the minimal hardware-free variant (`strategy: emulator`, one published topic).

To add *your* robot: drop `robots/my_robot.yaml` next to the existing ones, point `curobo_config` at your cuRobo YAML, pick a strategy, rebuild the workspace, and launch with `robot:=my_robot`.

### A robot described by an XRDF

If the robot is described by NVIDIA's [XRDF](https://nvidia-isaac-ros.github.io/concepts/manipulation/xrdf.html) instead of a cuRobo YAML, set `curobo_xrdf` and `urdf` in place of `curobo_config`:

```yaml
name: ur10e_xrdf
curobo_xrdf: ur10e.xrdf                    # package://, absolute, or relative
urdf: robot/ur_description/ur10e.urdf
strategy: emulator
strategy_params:
  joint_states_topic: /emulator/joint_states
```

The XRDF is converted by cuRobo's own `convert_xrdf_to_curobo` into a cuRobo YAML (written under `/tmp/curobo_ros_resolved/`), so the rest of curobo_ros sees a plain cuRobo config. A bare relative path that does not exist next to the descriptor resolves inside cuRobo's own content, as it does for cuRobo's bundled robots: the example above is `robots/ur10e_xrdf.yaml`, cuRobo's stock UR10e.

### One of cuRobo's own robots

A bare `curobo_config` name that does not exist next to the descriptor is one of cuRobo's bundled robots, as cuRobo itself reads it: `curobo_config: dual_ur10e.yml` loads cuRobo's stock dual UR10e (`robots/dual_ur10e.yaml`, the example dual-arm robot). Its URDF paths are made absolute in the resolved copy, so the launch file can hand the URDF to `robot_state_publisher`.

### Opt-in cuRobo fixes

`apply_curobo_patches: true` applies the fixes in `curobo_ros/core/curobo_patches.py` before any solver is built: disabling a *mesh* obstacle (which cuRobo v0.8.0 cannot do when a cuboid cache exists, and which attaching a mesh object relies on), a joint reindex that fails when locked joints precede the active ones in the chain, and `plan_grasp` start states that carry a trajectory `dt` (it makes the IK fail from then on). Off by default.

Attaching objects needs sphere slots on the attach link: in an XRDF, as many placeholder spheres (radius `-100`) on `attached_object` as the payload may use; in a cuRobo YAML, `extra_collision_spheres: {attached_object: N}`.

## 2. The cuRobo robot configuration

The reference example is `curobo_doosan/src/m1013/m1013.yml`. The essential structure:

```yaml
robot_cfg:
  kinematics:
    urdf_path: <path to your .urdf>
    base_link: "base_0"          # root of the kinematic chain
    ee_link: "link6"             # planning frame (tool)
    collision_spheres: { ... }   # per-link sphere approximation (see step 3)
    collision_link_names: [...]
    cspace:
      joint_names: [joint1, ..., joint6]
      # position/velocity/acceleration/jerk limits — this is also where
      # robot speed is scaled (see Tutorial 4)
```

Requirements for the URDF: a fixed, connected chain from `base_link` to `ee_link` with correct joint limits. Meshes are only needed for visualization — collision uses the spheres.

For the full schema, see the [cuRobo configuration documentation](https://curobo.org/tutorials/1_robot_configuration.html).

## 3. Collision spheres

cuRobo approximates the robot volume with spheres attached to links — this is what makes GPU collision checking fast. Guidelines:

- 2–5 spheres per link, more for long links; 10–20 % overlap between neighbours so no gap opens at joint bends.
- Cover the real volume conservatively: a missed wrist cover becomes a real collision.
- Include the tool/gripper if it is part of the URDF.

The [`curobo_robot_setup`](https://github.com/Lab-CORO/curobo_robot_setup) companion package provides an RViz-based editor for this step: load your URDF, place or auto-generate spheres interactively (backed by cuRobo's mesh sphere-fitting), and export the YAML. See its README for usage. You can also write the spheres by hand for a simple arm — the M1013 file is a good template.

To inspect the result at runtime: enable the sphere markers and look at them in RViz —

```bash
ros2 service call /unified_planner/set_collision_spheres_enabled std_srvs/srv/SetBool "{data: true}"
```

## 4. Test the integration

Start with the emulator strategy so nothing physical moves — either temporarily set `strategy: emulator` in your descriptor, or switch at runtime:

```bash
ros2 launch curobo_ros gen_traj.launch.py robot:=my_robot
ros2 service call /unified_planner/set_robot_strategy curobo_msgs/srv/SetRobotStrategy "{robot_strategy: 'emulator'}"
```

Checklist:

1. The node reaches `node_is_available: True` without errors (watch the launch log for YAML/URDF problems).
2. The robot model looks right in RViz and the collision spheres cover it.
3. A `generate_trajectory` call to a pose in the workspace succeeds ([Tutorial 1](01-first-trajectory.md)).
4. IK sanity check ([Tutorial 6](06-ik-fk-services.md)): FK of a known joint state returns the expected pose.

Then connect the real driver via the descriptor's `strategy_params` — see [Tutorial 4](04-robot-execution.md).

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Node crashes at startup parsing the config | Wrong indentation or missing `kinematics` keys in the cuRobo YAML |
| `Descriptor ... is missing required 'curobo_config'` | The descriptor YAML lacks the `curobo_config` key (or `curobo_xrdf` + `urdf`) |
| `Descriptor ...: 'curobo_xrdf' needs 'urdf'` | An XRDF descriptor without its URDF |
| Plans collide with the real robot body | Sphere coverage too sparse — add/enlarge spheres |
| IK always fails | `ee_link` name doesn't match the URDF, or joint limits are wrong |

## Next steps

- [Tutorial 4: Robot Execution](04-robot-execution.md) — wire up the real driver
- [Parameters Guide](../concepts/parameters.md) — solver parameters that interact with the robot config

[← Tutorial 1](01-first-trajectory.md) | [Tutorial 3 →](03-collision-objects.md)
