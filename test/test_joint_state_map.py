"""The descriptor's joint_states section: driver joint names and values in the model's terms."""

from curobo_ros.robot.joint_state_map import JointStateMap
from curobo_ros.robot.robot_description import load_robot_description
import numpy as np
import pytest
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# dynamic_scene's finger_joint_map (config/mapper.yaml): Robotiq driver opening -> URDF angle.
FINGER = {'reported': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.689],
          'model': [-0.074, 0.054, 0.174, 0.287, 0.407, 0.523, 0.617, 0.700]}


def joint_state(names, position, velocity=()):
    header = Header(frame_id='base')
    header.stamp.sec = 7
    return JointState(header=header, name=names, position=position, velocity=list(velocity))


def test_without_a_section_the_joint_states_are_the_model_s():
    msg = joint_state(['a', 'b'], [1.0, 2.0])
    assert JointStateMap(None).to_model(msg) is msg
    assert JointStateMap().driver_names(['a', 'b']) == ['a', 'b']
    assert JointStateMap().max_age_s == 0.0


def test_driver_names_are_renamed_to_the_model_s_and_back():
    joint_map = JointStateMap({'names': {'joint1': 'joint_1'}})
    msg = joint_state(['joint_1', 'joint2'], [1.0, 2.0], [0.1, 0.2])

    model = joint_map.to_model(msg)

    assert model.name == ['joint1', 'joint2']
    assert list(model.position) == [1.0, 2.0] and list(model.velocity) == [0.1, 0.2]
    assert model.header == msg.header
    assert list(msg.name) == ['joint_1', 'joint2']  # the message is not modified
    assert joint_map.driver_names(['joint1', 'joint2']) == ['joint_1', 'joint2']


def test_a_converted_joint_takes_the_table_value_held_at_the_ends():
    joint_map = JointStateMap({'conversions': {'finger_joint': FINGER}})
    for reported in (-0.5, 0.0, 0.25, 0.6674, 0.689, 2.0):
        model = joint_map.to_model(joint_state(['wrist', 'finger_joint'], [0.3, reported]))
        assert model.position[0] == 0.3
        assert model.position[1] == pytest.approx(
            np.interp(reported, FINGER['reported'], FINGER['model']))
    # The station bag's opening (0.6674) is about 0.680 rad in the URDF.
    assert joint_map.to_model(joint_state(['finger_joint'], [0.6674])).position[0] == \
        pytest.approx(0.680, abs=1e-3)


def test_a_converted_joint_velocity_is_scaled_by_the_table_slope():
    joint_map = JointStateMap({'conversions': {'f': {'reported': [0.0, 1.0, 2.0],
                                                     'model': [0.0, 2.0, 2.5]}}})
    velocity = [joint_map.to_model(joint_state(['f'], [q], [1.0])).velocity[0]
                for q in (0.5, 1.5, 3.0)]
    assert velocity == pytest.approx([2.0, 0.5, 0.0])  # held past the table's end


def test_a_conversion_applies_to_the_renamed_joint():
    joint_map = JointStateMap({'names': {'finger_joint': 'gripper'},
                               'conversions': {'finger_joint': FINGER}})
    model = joint_map.to_model(joint_state(['gripper'], [0.0]))
    assert model.name == ['finger_joint'] and model.position[0] == pytest.approx(-0.074)


@pytest.mark.parametrize('config', [
    ['names'],
    {'name': {}},
    {'names': ['joint1']},
    {'names': {'joint1': 7}},
    {'names': {'joint1': 'j', 'joint2': 'j'}},
    {'conversions': {'f': {'reported': [0.0], 'model': [0.0]}}},
    {'conversions': {'f': {'reported': [0.0, 1.0], 'model': [0.0]}}},
    {'conversions': {'f': {'reported': [1.0, 0.0], 'model': [0.0, 1.0]}}},
    {'conversions': {'f': {'reported': [0.0, 0.0], 'model': [0.0, 1.0]}}},
    {'conversions': {'f': {'reported': [0.0, float('nan')], 'model': [0.0, 1.0]}}},
    {'conversions': {'f': {'reported': [0.0, True], 'model': [0.0, 1.0]}}},
    {'conversions': {'f': {'reported': [0.0, 1.0], 'model': [0.0, 1.0], 'enabled': True}}},
    {'max_age_s': -1.0},
    {'max_age_s': '0.5'},
    {'max_age_s': float('nan')},
])
def test_an_invalid_section_is_refused(config):
    with pytest.raises(ValueError, match='joint_states'):
        JointStateMap(config)


def test_the_descriptor_carries_the_section(tmp_path):
    descriptor = tmp_path / 'franka_test.yaml'
    descriptor.write_text(
        'curobo_config: franka.yml\n'
        'joint_states:\n'
        '  names: {panda_joint1: joint_1}\n'
        '  max_age_s: 0.25\n')

    description = load_robot_description(str(descriptor))

    assert description.joint_state_map.driver_names(['panda_joint1']) == ['joint_1']
    assert description.joint_state_map.max_age_s == 0.25


def test_a_descriptor_with_an_invalid_section_is_refused(tmp_path):
    descriptor = tmp_path / 'franka_test.yaml'
    descriptor.write_text('curobo_config: franka.yml\njoint_states: {max_age: 0.25}\n')
    with pytest.raises(ValueError, match=f'{descriptor}.*unknown keys'):
        load_robot_description(str(descriptor))
