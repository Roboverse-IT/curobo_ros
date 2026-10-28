"""A robot driver's joint states, read by the model's joint names, and how old they may be."""

import math
import threading

from curobo_ros.robot.joint_feedback import JointFeedback
from curobo_ros.robot.joint_speed_strategy import JointSpeedStrategy
from curobo_ros.robot.joint_state_map import JointStateMap
from curobo_ros.robot.robot_context import RobotContext
from curobo_ros.robot.robot_description import RobotDescription
import pytest
import rclpy
from sensor_msgs.msg import JointState

MODEL = ['shoulder', 'elbow', 'wrist']


def description(joint_states=None, names=MODEL):
    return RobotDescription(
        name='test', display_name='test', curobo_config_path='', robot_cfg_dict={},
        base_link='base', strategy_key='joint_speed',
        strategy_params={'command_topic': '/test/command', 'joint_states_topic': '/test/js'},
        joint_state_map=JointStateMap(joint_states), _joint_names=list(names))


class Logger:
    def __init__(self):
        self.warnings = []

    def warning(self, text):
        self.warnings.append(text)


def test_joints_are_read_by_name_in_the_model_order():
    feedback = JointFeedback(description(), Logger())
    feedback.update(JointState(name=['finger', 'wrist', 'shoulder', 'elbow'],
                               position=[0.7, 3.0, 1.0, 2.0], velocity=[0.0, 0.3, 0.1, 0.2]))
    assert feedback.read() == ([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])


def test_the_model_joint_names_are_read_when_read():
    desc = description(names=MODEL + ['finger'])  # the cspace, before the kinematics
    feedback = JointFeedback(desc, Logger())
    feedback.update(JointState(name=['finger', 'wrist', 'shoulder', 'elbow'],
                               position=[0.7, 3.0, 1.0, 2.0]))
    assert feedback.read()[0] == [1.0, 2.0, 3.0, 0.7]

    desc._joint_names = list(MODEL)  # bind_kinematics: the finger is locked

    assert feedback.names == MODEL
    assert feedback.read() == ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])


def test_driver_names_and_values_are_mapped_to_the_model_s():
    feedback = JointFeedback(description({
        'names': {'shoulder': 'joint_1'},
        'conversions': {'wrist': {'reported': [0.0, 1.0], 'model': [0.0, 10.0]}}}), Logger())
    feedback.update(JointState(name=['joint_1', 'elbow', 'wrist'], position=[1.0, 2.0, 0.5]))
    assert feedback.read()[0] == [1.0, 2.0, 5.0]


def test_unresolved_names_are_read_in_publication_order_with_one_warning():
    logger = Logger()
    feedback = JointFeedback(description(), logger)
    for _ in range(2):
        feedback.update(JointState(name=['j1', 'j2', 'j3', 'j4'], position=[1.0, 2.0, 3.0, 4.0],
                                   velocity=[0.1, 0.2, 0.3, 0.4]))
        assert feedback.read() == ([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    assert len(logger.warnings) == 1
    assert "['shoulder', 'elbow', 'wrist']" in logger.warnings[0]


def test_before_the_first_joint_state_the_joints_are_zero_and_infinitely_old():
    feedback = JointFeedback(description(), Logger())
    assert feedback.read() == ([0.0] * 3, [0.0] * 3)
    assert feedback.age() == math.inf
    feedback.update(JointState(name=MODEL, position=[1.0, 2.0, 3.0]))
    assert 0.0 <= feedback.age() < 1.0


class Strategy:
    def __init__(self, age):
        self.age = age
        self.params = {'joint_states_topic': '/test/js'}

    def get_joint_state_age(self):
        return self.age

    def get_joint_pose(self):
        return [1.0, 2.0, 3.0]


def context(max_age_s, age):
    robot = RobotContext.__new__(RobotContext)
    robot.strategy_lock = threading.Lock()
    robot.description = description({'max_age_s': max_age_s})
    robot.robot_strategy = Strategy(age)
    return robot


@pytest.mark.parametrize('max_age_s,age', [(0.0, 10.0), (0.0, math.inf), (0.5, 0.1),
                                           (0.5, None)])
def test_joint_states_within_the_limit_are_used(max_age_s, age):
    assert context(max_age_s, age).get_fresh_joint_pose() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize('age,said', [(0.8, '0.800 s old'), (math.inf, 'missing')])
def test_joint_states_past_the_limit_are_refused(age, said):
    with pytest.raises(ValueError, match=f'/test/js within joint_states.max_age_s .*{said}'):
        context(0.5, age).get_fresh_joint_pose()


@pytest.fixture
def node():
    rclpy.init()
    node = rclpy.create_node('joint_feedback_test')
    yield node
    node.destroy_node()
    rclpy.shutdown()


def test_the_strategy_reads_by_name_and_commands_the_driver_names(node, monkeypatch):
    strategy = JointSpeedStrategy(node, 0.02, description({'names': {'shoulder': 'joint_1'}}))
    sent = []
    monkeypatch.setattr(strategy.pub_trajectory, 'publish', sent.append)

    strategy.callback_joint_pose(JointState(
        name=['elbow', 'joint_1', 'wrist'], position=[2.0, 1.0, 3.0], velocity=[0.2, 0.1, 0.3]))
    strategy.set_command(strategy.get_joint_name(), [[0.1, 0.2, 0.3]], [[0.0] * 3],
                         [[1.0, 2.0, 3.0]])
    strategy.send_trajectrory()

    assert strategy.get_joint_pose() == [1.0, 2.0, 3.0]
    assert strategy.get_joint_velocity() == [0.1, 0.2, 0.3]
    assert strategy.get_joint_name() == MODEL
    assert list(sent[0].joint_names) == ['joint_1', 'elbow', 'wrist']
    assert 0.0 <= strategy.get_joint_state_age() < 1.0
