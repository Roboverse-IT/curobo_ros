"""Recent joint states, read back at a given stamp."""

import bisect
import threading

import numpy as np
import rclpy.time

# Largest gap (s) between the two joint states around a stamp that is still
# interpolated. A driver publishing at 20 Hz or faster stays within it.
DEFAULT_JOINT_STATE_MAX_GAP_S = 0.05


class JointStateHistory:
    """
    The joint positions of the last `window_s` seconds, by joint name.

    Positions are stored in `joint_names` order, mapped by name from each
    sensor_msgs/JointState (its own order and extra joints, e.g. a gripper,
    do not matter). A message that lacks one of `joint_names` is rejected:
    reading positions by index from it would give another joint's value.

    Thread-safe: states are usually added on a thread of their own while
    frames read them on another.
    """

    def __init__(self, joint_names, max_gap_s=DEFAULT_JOINT_STATE_MAX_GAP_S, window_s=1.0):
        """
        Set up an empty history.

        joint_names are the joints to keep, in the order at() returns them.
        at() interpolates only between two states at most max_gap_s apart (s);
        a larger gap means the stamp is not covered. States older than the
        newest one minus window_s (s) are dropped.
        """
        if max_gap_s <= 0.0:
            raise ValueError(f'max_gap_s must be > 0, got {max_gap_s!r}')
        self._joint_names = list(joint_names)
        self._max_gap_ns = int(max_gap_s * 1e9)
        self._window_ns = int(window_s * 1e9)
        self._stamps = []  # ns, strictly increasing
        self._positions = []  # np arrays in joint_names order
        self._lock = threading.RLock()

    def add(self, msg):
        """Store a sensor_msgs/JointState. Returns None, or why it was rejected."""
        stamp = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds
        if stamp == 0:
            return 'the joint state has no header stamp'
        if len(msg.position) != len(msg.name):
            return (f'the joint state has {len(msg.name)} names and '
                    f'{len(msg.position)} positions')
        index = {name: i for i, name in enumerate(msg.name)}
        missing = [name for name in self._joint_names if name not in index]
        if missing:
            return f'the joint state has no {missing} (it has {list(msg.name)})'

        positions = np.array([msg.position[index[name]] for name in self._joint_names])
        with self._lock:
            if self._stamps:
                if stamp == self._stamps[-1]:
                    return None  # the same state again
                if stamp < self._stamps[-1]:
                    self.clear()  # time jumped back (e.g. a replay restarted)
            self._stamps.append(stamp)
            self._positions.append(positions)
            first = bisect.bisect_left(self._stamps, stamp - self._window_ns)
            del self._stamps[:first]
            del self._positions[:first]
        return None

    def clear(self):
        """Forget every stored state."""
        with self._lock:
            self._stamps.clear()
            self._positions.clear()

    def covers(self, stamp):
        """Return True once a state at or after `stamp` (rclpy Time) is stored."""
        with self._lock:
            return bool(self._stamps) and self._stamps[-1] >= stamp.nanoseconds

    def at(self, stamp):
        """
        Return the joint positions at `stamp` (rclpy Time), and why not when None.

        The stored state at `stamp`, or the linear interpolation of the two
        around it. None when `stamp` is outside the stored states or the two
        around it are more than max_gap_s apart.
        """
        with self._lock:
            return self._at(stamp.nanoseconds)

    def _at(self, t):
        if not self._stamps:
            return None, 'no joint state received'
        if t > self._stamps[-1]:
            return None, (f'the newest joint state is {(t - self._stamps[-1]) * 1e-9:.3f} s '
                          'older than the stamp')
        if t < self._stamps[0]:
            return None, (f'the oldest stored joint state is {(self._stamps[0] - t) * 1e-9:.3f} s '
                          'newer than the stamp')
        i = bisect.bisect_left(self._stamps, t)
        if self._stamps[i] == t:
            return self._positions[i].copy(), None
        t0, t1 = self._stamps[i - 1], self._stamps[i]
        if t1 - t0 > self._max_gap_ns:
            return None, (f'the joint states around the stamp are {(t1 - t0) * 1e-9:.3f} s '
                          f'apart (max {self._max_gap_ns * 1e-9:.3f} s)')
        w = (t - t0) / (t1 - t0)
        return (1.0 - w) * self._positions[i - 1] + w * self._positions[i], None
