"""JointStateHistory: joint positions by name, read back at a stamp."""

import sys
import threading

from builtin_interfaces.msg import Time
from curobo_ros.robot.joint_state_history import JointStateHistory
import numpy as np
import pytest
import rclpy.time
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

JOINTS = ['a', 'b', 'c']
T0 = 1_000_000_000_000  # ns


def state(t_ns, positions, names=JOINTS):
    return JointState(
        header=Header(stamp=Time(sec=t_ns // 10**9, nanosec=t_ns % 10**9)),
        name=list(names), position=[float(p) for p in positions])


def at(t_ns):
    return rclpy.time.Time(nanoseconds=t_ns)


def test_positions_are_mapped_by_name():
    history = JointStateHistory(JOINTS)
    # Another order and an extra joint (e.g. a gripper), as a real driver sends.
    assert history.add(state(T0, [3, 9, 1, 2], names=['c', 'finger', 'a', 'b'])) is None

    q, reason = history.at(at(T0))
    assert reason is None
    np.testing.assert_array_equal(q, [1, 2, 3])


def test_a_state_without_a_model_joint_is_rejected():
    history = JointStateHistory(JOINTS)
    reason = history.add(state(T0, [1, 2], names=['a', 'c']))

    assert "['b']" in reason
    assert history.at(at(T0)) == (None, 'no joint state received')


@pytest.mark.parametrize('msg, words', [
    (state(0, [1, 2, 3]), 'no header stamp'),
    (JointState(header=Header(stamp=Time(sec=1000)), name=JOINTS, position=[1.0]),
     '3 names and 1 positions'),
])
def test_malformed_states_are_rejected(msg, words):
    history = JointStateHistory(JOINTS)
    assert words in history.add(msg)
    assert not history.covers(at(T0))


def test_between_two_states_positions_are_interpolated():
    history = JointStateHistory(JOINTS)
    history.add(state(T0, [0, 0, 0]))
    history.add(state(T0 + 4_000_000, [4, -4, 8]))

    q, reason = history.at(at(T0 + 1_000_000))
    assert reason is None
    np.testing.assert_allclose(q, [1, -1, 2])


def test_a_stamp_in_a_gap_larger_than_the_max_is_not_covered():
    history = JointStateHistory(JOINTS, max_gap_s=0.05)
    history.add(state(T0, [0, 0, 0]))
    history.add(state(T0 + 60_000_000, [1, 1, 1]))

    q, reason = history.at(at(T0 + 30_000_000))
    assert q is None and '0.060 s apart (max 0.050 s)' in reason
    # Exactly at a stored state there is nothing to interpolate.
    assert history.at(at(T0 + 60_000_000))[1] is None


def test_a_stamp_after_the_newest_state_waits_for_the_next():
    history = JointStateHistory(JOINTS)
    history.add(state(T0, [0, 0, 0]))
    stamp = at(T0 + 2_000_000)

    assert not history.covers(stamp)
    q, reason = history.at(stamp)
    assert q is None and 'newest joint state is 0.002 s older' in reason

    history.add(state(T0 + 4_000_000, [2, 2, 2]))
    assert history.covers(stamp)
    np.testing.assert_allclose(history.at(stamp)[0], [1, 1, 1])


def test_states_older_than_the_window_are_dropped():
    history = JointStateHistory(JOINTS, window_s=1.0)
    for i in range(0, 1500, 10):  # 1.5 s at 100 Hz
        history.add(state(T0 + i * 1_000_000, [i, 0, 0]))

    assert history.at(at(T0 + 400_000_000))[0] is None
    np.testing.assert_allclose(history.at(at(T0 + 500_000_000))[0], [500, 0, 0])


def test_the_same_stamp_again_is_ignored():
    history = JointStateHistory(JOINTS)
    history.add(state(T0, [1, 1, 1]))
    assert history.add(state(T0, [5, 5, 5])) is None

    np.testing.assert_array_equal(history.at(at(T0))[0], [1, 1, 1])


def test_time_going_back_clears_the_history():
    history = JointStateHistory(JOINTS)
    history.add(state(T0, [1, 1, 1]))
    history.add(state(T0 + 4_000_000, [1, 1, 1]))
    history.add(state(T0 - 10**9, [7, 7, 7]))  # e.g. a replay restarted

    assert history.at(at(T0 + 2_000_000))[0] is None
    np.testing.assert_array_equal(history.at(at(T0 - 10**9))[0], [7, 7, 7])


def test_max_gap_is_validated():
    with pytest.raises(ValueError, match='max_gap_s'):
        JointStateHistory(JOINTS, max_gap_s=0.0)


def test_reads_while_another_thread_adds_are_consistent():
    # Joint states are added on their own thread while frames read them.
    # Every add prunes the oldest state. Positions are i**2 at state i, not
    # linear in time, so interpolating with the wrong pair of states (e.g.
    # shifted by a prune during the read) gives a wrong value.
    history = JointStateHistory(JOINTS, window_s=0.01)
    newest = [None]
    done = threading.Event()

    def writer():
        for i in range(20000):
            t = T0 + i * 2_000_000  # 500 Hz
            history.add(state(t, [float(i * i), 0, 0]))
            newest[0] = t
        done.set()

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads as often as possible
    try:
        thread = threading.Thread(target=writer)
        thread.start()
        wrong = []
        while not done.is_set():
            if newest[0] is None:
                continue
            t = newest[0] - 3_000_000  # halfway between two states, inside the window
            q, reason = history.at(at(t))
            i = (t - T0) // 2_000_000
            if q is not None and abs(q[0] - (i * i + (i + 1) ** 2) / 2) > 1e-6:
                wrong.append((t, q[0]))
        thread.join()
    finally:
        sys.setswitchinterval(interval)

    assert wrong == []
