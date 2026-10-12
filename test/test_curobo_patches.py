"""Patches 5 and 6: goal buffer ownership, tool-pose kernels and their warp modules."""

import gc

import curobo._src.cost.cost_tool_pose as cost_tool_pose
from curobo._src.solver.manager_goal import GoalManager
from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solve_state import SolveState
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose
from curobo_ros.core.curobo_patches import apply
import pytest
import torch

apply()

DEVICE = DeviceCfg(device=torch.device('cpu'), dtype=torch.float32)
DOF = 7


def goal_poses(seed):
    g = torch.Generator().manual_seed(seed)
    quaternion = torch.nn.functional.normalize(torch.randn(1, 1, 1, 1, 4, generator=g), dim=-1)
    return GoalToolPose(tool_frames=['ee_link'],
                        position=torch.randn(1, 1, 1, 1, 3, generator=g), quaternion=quaternion)


def joint_state(seed):
    g = torch.Generator().manual_seed(seed)
    position, velocity, acceleration, jerk = (torch.randn(1, DOF, generator=g) for _ in range(4))
    return JointState(position=position, velocity=velocity, acceleration=acceleration, jerk=jerk,
                      joint_names=[f'j{i}' for i in range(DOF)], device_cfg=DEVICE)


# input name -> (update_goal_buffer keyword, factory, tensor components, buffer field)
CASES = {
    'goal_poses': ('goal_tool_poses', goal_poses, ('position', 'quaternion'), 'link_goal_poses'),
    'goal_js': ('goal_js', joint_state, ('position', 'velocity', 'acceleration', 'jerk'),
                'goal_js'),
    'current_js': ('current_js', joint_state, ('position', 'velocity', 'acceleration', 'jerk'),
                   'current_js'),
}


@pytest.mark.parametrize('name', CASES)
def test_goal_buffer_does_not_keep_caller_tensors(name):
    """The first input stays as given after the buffer is built and after it is updated."""
    keyword, factory, components, field = CASES[name]
    solve_state = SolveState(solve_type=SolveMode.SINGLE, batch_size=1, num_envs=1,
                             num_goalset=1, num_ik_seeds=4, tool_frames=['ee_link'])
    manager = GoalManager(DEVICE)
    first = factory(1)
    expected = {c: getattr(first, c).clone() for c in components}

    problems = []
    for step, value in (('build', first), ('update', factory(2))):
        buffer, _ = manager.update_goal_buffer(solve_state, **{keyword: value})
        stored = getattr(buffer, field)
        for c in components:
            if not torch.equal(getattr(first, c), expected[c]):
                problems.append(f'{step}: {name}.{c} changed')
            if getattr(stored, c).data_ptr() == getattr(first, c).data_ptr():
                problems.append(f'{step}: buffer shares {name}.{c} with the caller')
    assert not problems, '; '.join(problems)


@pytest.mark.parametrize('sequence', [(4, 1, 4, 1), (4, 1, 2, 4)],
                         ids=['switch_and_return', 'new_size'])
def test_tool_pose_kernel_switch_keeps_loaded_modules(sequence):
    """Goalset switches never change, so never reload, a module a kernel already used."""
    make = cost_tool_pose.create_goalset_pose_distance_kernel_with_constants
    modules = {}
    for size in (4, 1):  # like a warmup; ToolPoseCost keeps only its latest kernel
        modules[size] = make(size, 0).module
        gc.collect()
    prepared = {size: (module, module.get_module_hash()) for size, module in modules.items()}
    problems = []
    for step, size in enumerate(sequence):
        kernel = make(size, 0)
        gc.collect()
        if size in prepared and kernel.module is not prepared[size][0]:
            problems.append(f'step {step}: goalset {size} got another module')
        for prepared_size, (module, module_hash) in prepared.items():
            if module.get_module_hash() != module_hash:
                problems.append(
                    f'step {step} (goalset {size}): module of goalset {prepared_size} changed')
        del kernel
    assert not problems, '; '.join(problems)
