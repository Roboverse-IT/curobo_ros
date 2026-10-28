"""The latest joint states of a robot driver, read in the model's joint order."""

import math
import time

from curobo_ros.robot.joint_state_map import JointStateMap


class JointFeedback:
    """
    The latest joint states from a robot driver, read by the model's joint names.

    The driver may publish its joints in any order, among other joints, under
    other names and in other units: the descriptor's ``joint_states`` section
    (see JointStateMap) maps them to the model's. A driver whose names still do
    not resolve is read positionally, in its publication order, with a single
    warning. Not thread-safe: the strategy reads and writes it under its
    buffer_lock.
    """

    def __init__(self, description, logger):
        """Read the joints of `description` (a RobotDescription, or None)."""
        self._description = description
        self._map = description.joint_state_map if description is not None else JointStateMap()
        self._logger = logger
        self._msg = None
        self._received = None
        self._warned = False

    @property
    def names(self):
        """Return the model's joint names, the order every read is in."""
        return self._description.joint_names if self._description is not None else []

    def update(self, msg):
        """Take the sensor_msgs/JointState `msg` as the latest joint states."""
        self._msg = self._map.to_model(msg)
        self._received = time.monotonic()

    def age(self):
        """Seconds since the latest joint states arrived (inf before the first)."""
        return math.inf if self._received is None else time.monotonic() - self._received

    def read(self):
        """Return (positions, velocities) of the model's joints (zeros before the first state)."""
        names, msg = self.names, self._msg
        if msg is None:
            return [0.0] * len(names), [0.0] * len(names)
        index = {name: i for i, name in enumerate(msg.name)}
        if names and len(msg.position) == len(msg.name) and all(n in index for n in names):
            position = [msg.position[index[n]] for n in names]
            velocity = ([msg.velocity[index[n]] for n in names]
                        if len(msg.velocity) == len(msg.name) else [0.0] * len(names))
            return position, velocity
        n = len(names) or len(msg.position)
        if not self._warned:
            self._warned = True
            missing = [name for name in names if name not in index]
            self._logger.warning(
                f'The joint states do not name the model joints {missing} (they name '
                f'{list(msg.name)}): reading the first {n} positions in publication order. '
                f'Map the driver names in the descriptor joint_states.names.')
        position = list(msg.position[:n])
        velocity = (list(msg.velocity[:n]) if len(msg.velocity) >= n
                    else [0.0] * len(position))
        return position, velocity

    def read_named(self, names):
        """
        Return the positions of the model joints `names`, any of the reported ones.

        None when the joint states do not name them all (or none has arrived):
        unlike read(), there is no positional fallback.
        """
        msg = self._msg
        if msg is None or len(msg.position) != len(msg.name):
            return None
        index = {name: i for i, name in enumerate(msg.name)}
        if not all(n in index for n in names):
            return None
        return [msg.position[index[n]] for n in names]
