"""A robot driver's joint states in the terms of its cuRobo model."""

import math

import numpy as np
from sensor_msgs.msg import JointState


class JointStateMap:
    """
    The descriptor's ``joint_states`` section: how the driver's joint states map to the model.

    names maps a model joint to the name the driver publishes it under, where
    the two differ. conversions maps a model joint to a piecewise-linear table
    from the value the driver reports to the model's (e.g. a gripper driver's
    opening to the URDF finger angle); outside the table the end values hold.
    max_age_s is how old the newest joint state may be when a plan or a
    collision query starts from it (0: no limit).

        joint_states:
          names: {joint1: joint_1}
          conversions:
            finger_joint: {reported: [0.0, 0.689], model: [-0.074, 0.700]}
          max_age_s: 0.5
    """

    def __init__(self, config=None):
        """Validate the descriptor's ``joint_states`` section (None: the identity)."""
        config = config or {}
        if not isinstance(config, dict):
            raise ValueError(f'joint_states: expected a mapping, got {config!r}')
        unknown = sorted(set(config) - {'names', 'conversions', 'max_age_s'})
        if unknown:
            raise ValueError(f'joint_states: unknown keys {unknown}')

        names = config.get('names') or {}
        if not isinstance(names, dict):
            raise ValueError(f'joint_states.names: expected a mapping, got {names!r}')
        self.names = dict(names)
        if not all(isinstance(k, str) and isinstance(v, str) and k and v
                   for k, v in self.names.items()):
            raise ValueError(f'joint_states.names: expected joint name pairs, got {self.names!r}')
        if len(set(self.names.values())) != len(self.names):
            raise ValueError(f'joint_states.names: two joints share a driver name: {self.names!r}')
        self._model_name = {driver: model for model, driver in self.names.items()}

        self._conversions = {}
        conversions = config.get('conversions') or {}
        if not isinstance(conversions, dict):
            raise ValueError(f'joint_states.conversions: expected a mapping, got {conversions!r}')
        for joint, table in conversions.items():
            table = table if isinstance(table, dict) else {}
            reported, model = table.get('reported'), table.get('model')
            if (set(table) != {'reported', 'model'}
                    or not isinstance(reported, list) or not isinstance(model, list)
                    or len(reported) < 2 or len(reported) != len(model)
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                               and math.isfinite(v) for v in reported + model)
                    or any(b <= a for a, b in zip(reported, reported[1:]))):
                raise ValueError(
                    f'joint_states.conversions.{joint}: expected only equal-length finite '
                    f'reported and model lists (2+ values, reported strictly increasing)')
            self._conversions[joint] = (np.array(reported, dtype=float),
                                        np.array(model, dtype=float))

        max_age_s = config.get('max_age_s', 0.0)
        if isinstance(max_age_s, bool) or not isinstance(max_age_s, (int, float)) \
                or not max_age_s >= 0.0:
            raise ValueError(f'joint_states.max_age_s: expected a number >= 0, got {max_age_s!r}')
        self.max_age_s = float(max_age_s)

    def driver_names(self, model_names):
        """Return the names the driver publishes `model_names` under."""
        return [self.names.get(name, name) for name in model_names]

    def to_model(self, msg):
        """
        Return the sensor_msgs/JointState `msg` with the model's joint names and values.

        Joints the map does not name keep their names. A converted joint's
        velocity is scaled by the table's slope at its position.
        """
        if not self.names and not self._conversions:
            return msg
        names = [self._model_name.get(name, name) for name in msg.name]
        position, velocity = list(msg.position), list(msg.velocity)
        for i, name in enumerate(names):
            table = self._conversions.get(name)
            if table is None or i >= len(position):
                continue
            reported, model = table
            value = position[i]
            position[i] = float(np.interp(value, reported, model))
            if i < len(velocity):
                k = int(np.searchsorted(reported, value, side='right')) - 1
                inside = 0 <= k < len(reported) - 1
                velocity[i] *= float((model[k + 1] - model[k]) / (reported[k + 1] - reported[k])
                                     if inside else 0.0)
        return JointState(header=msg.header, name=names, position=position,
                          velocity=velocity, effort=list(msg.effort))
