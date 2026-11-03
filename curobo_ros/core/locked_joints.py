"""Locked joints that follow the robot's measured joint state (e.g. gripper fingers).

cuRobo folds a locked joint into the kinematic chain: the fixed transform of
every link it moves (its own link and its mimic links) holds the link's pose at
the lock value. A model built from the XRDF default plans the fingers where the
XRDF puts them (fully open), however open the real gripper is. Here the locked
joints follow the measured values instead: their fixed transforms and lock
positions are swapped in place, in every model, before each plan.

For a revolute or prismatic joint the swap is a motion about a constant axis:

    F(q)[row] = F(q0)[row] * motion(axis_row, rate_row * (q - q0))

Axis and rate of each row come from one model build per joint at startup, and
are checked against one more build, so a model this does not describe fails at
startup. Same approach as the UR station's cuMotion (agfa_isaac_ros_cumotion
0891942), for cuRobo 0.8's [links, 3, 4] transforms.

No ROS imports: tested with synthetic models.
"""

import math
import threading
import xml.etree.ElementTree as ET

import torch


class LockedJointModelError(ValueError):
    """The model's locked joints are not a constant-axis motion."""


def urdf_joint_limits(urdf_text, names):
    """{name: (lower, upper)} of the named joints that have limits in the URDF."""
    limits = {}
    for joint in ET.fromstring(urdf_text).iter('joint'):
        limit = joint.find('limit')
        if joint.get('name') in names and limit is not None:
            limits[joint.get('name')] = (float(limit.get('lower', '-inf')),
                                         float(limit.get('upper', 'inf')))
    return limits


def _homogeneous(rows):
    """[..., 3, 4] -> [..., 4, 4]."""
    last = torch.zeros(rows.shape[:-2] + (1, 4), dtype=rows.dtype)
    last[..., 0, 3] = 1.0
    return torch.cat([rows, last], dim=-2)


def _skew(v):
    x, y, z = v.tolist()
    return torch.tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=v.dtype)


def _rotation(rotvec):
    """Rodrigues (float64)."""
    angle = float(torch.linalg.norm(rotvec))
    eye = torch.eye(3, dtype=rotvec.dtype)
    if angle < 1e-12:
        return eye
    k = _skew(rotvec / angle)
    return eye + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _rotvec(rotation):
    """Rotation vector of a rotation below pi (float64)."""
    cos_angle = max(-1.0, min(1.0, (float(torch.trace(rotation)) - 1.0) / 2.0))
    angle = math.acos(cos_angle)
    if angle < 1e-9:
        return torch.zeros(3, dtype=rotation.dtype)
    v = torch.stack([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]])
    return v * (angle / (2.0 * math.sin(angle)))


class LockedJointTransforms:
    """
    The fixed transforms of a model for any value of some of its locked joints.

    ``build_fn(values)`` returns the ``fixed_transforms`` ([links, 3, 4]) of a
    model built with ``values`` ({name: value}, the joints not named at their
    default). ``base`` is the loaded model's, at ``defaults``. Construction
    costs one build per joint plus one check build.
    """

    def __init__(self, defaults, build_fn, base, limits=None, probe_step=0.2, tolerance=1e-5):
        self.names = list(defaults)
        self.defaults = {n: float(v) for n, v in defaults.items()}
        limits = dict(limits or {})
        self._base = _homogeneous(base.detach().to('cpu', torch.float64))
        # (row, joint index, rotation per unit (3,), translation per unit (3,))
        self._rows = []
        for index, name in enumerate(self.names):
            probe = self._probe_value(name, probe_step, limits)
            delta = probe - self.defaults[name]
            built = self._build(build_fn, {name: probe})
            changed = (built - self._base).abs().flatten(1).max(dim=1).values > tolerance
            rows = [int(r) for r in torch.nonzero(changed).flatten()]
            if not rows:
                raise LockedJointModelError(
                    f'locked joint {name} moves no link between {self.defaults[name]} and {probe}')
            for row in rows:
                motion = torch.linalg.solve(self._base[row], built[row])
                rotvec, translation = _rotvec(motion[:3, :3]), motion[:3, 3]
                if float(torch.linalg.norm(rotvec)) > 1e-9:
                    if float(torch.linalg.norm(translation)) > tolerance:
                        raise LockedJointModelError(
                            f'locked joint {name} turns link row {row} about an axis that does '
                            'not pass through the link origin')
                    self._rows.append(
                        (row, index, rotvec / delta, torch.zeros(3, dtype=torch.float64)))
                else:
                    self._rows.append((row, index, torch.zeros(3, dtype=torch.float64),
                                       translation / delta))
        check = {n: self._check_value(n, probe_step, limits) for n in self.names}
        error = float((self._compute(check) - self._build(build_fn, check)).abs().max())
        if error > tolerance:
            raise LockedJointModelError(
                f'locked joint transforms at {check} differ from a model build by {error:.2e}')
        self.verification_error = error
        self.verified_at = check

    @property
    def moving_rows(self):
        return sorted({row for row, *_ in self._rows})

    def __call__(self, values):
        """[links, 3, 4] float64 for ``values`` (missing joints at their default)."""
        full = {n: float(values.get(n, self.defaults[n])) for n in self.names}
        return self._compute(full)[:, :3, :]

    def _compute(self, values):
        fixed = self._base.clone()
        for row, index, rotation_rate, translation_rate in self._rows:
            amount = values[self.names[index]] - self.defaults[self.names[index]]
            motion = torch.eye(4, dtype=torch.float64)
            motion[:3, :3] = _rotation(rotation_rate * amount)
            motion[:3, 3] = translation_rate * amount
            fixed[row] = self._base[row] @ motion
        return fixed

    def _build(self, build_fn, values):
        built = build_fn({**self.defaults, **values}).detach().to('cpu', torch.float64)
        built = _homogeneous(built)
        if built.shape != self._base.shape:
            raise LockedJointModelError(
                f'model build has {built.shape[0]} links, the loaded model {self._base.shape[0]}')
        return built

    def _probe_value(self, name, step, limits):
        lower, upper = limits.get(name, (-math.inf, math.inf))
        default = self.defaults[name]
        for value in (default + step, default - step):
            if lower <= value <= upper:
                return value
        return max((lower, upper), key=lambda v: abs(v - default))

    def _check_value(self, name, step, limits):
        """A value away from both the default and the probe, within the limits."""
        lower, upper = limits.get(name, (-math.inf, math.inf))
        default = self.defaults[name]
        probe = self._probe_value(name, step, limits)
        candidates = [v for v in (lower, upper) if math.isfinite(v)]
        candidates += [default - step if probe > default else default + step,
                       default + 2.5 * (probe - default)]
        inside = [v for v in candidates if lower <= v <= upper]
        return max(inside, key=lambda v: min(abs(v - default), abs(v - probe)))


class LockedJointFollower:
    """
    Keeps some locked joints of every planning model at their measured values.

    ``transforms`` is a LockedJointTransforms for those joints. ``observe`` takes
    joint states (any thread); ``apply`` writes the latest values into the given
    kinematics configs (cuRobo ``KinematicsParams``), in place: captured CUDA
    graphs keep reading the same memory. Values are clamped to ``limits`` and
    rounded to ``resolution``.
    """

    def __init__(self, transforms, limits=None, resolution=0.01):
        self._transforms = transforms
        self._limits = dict(limits or {})
        self._resolution = resolution
        self._measured = {}
        self._lock = threading.Lock()
        self._key = None
        self._fixed = None
        self._moving = transforms.moving_rows

    @property
    def names(self):
        return list(self._transforms.names)

    def observe(self, names, positions):
        """Record the measured positions of the followed joints."""
        with self._lock:
            for name, position in zip(names, positions):
                if name in self._transforms.defaults and math.isfinite(position):
                    self._measured[name] = float(position)

    def values(self):
        """({name: value}, {name: 'measured' | 'default'}), clamped and rounded."""
        with self._lock:
            measured = dict(self._measured)
        values, sources = {}, {}
        for name in self.names:
            value = measured.get(name, self._transforms.defaults[name])
            sources[name] = 'measured' if name in measured else 'default'
            lower, upper = self._limits.get(name, (-math.inf, math.inf))
            value = round(min(max(value, lower), upper) / self._resolution) * self._resolution
            values[name] = round(min(max(value, lower), upper), 6)
        return values, sources

    def apply(self, values, kinematics_configs):
        """
        Write ``values`` into every config of ``kinematics_configs``.

        Returns the configs refused: a different link layout from the model the
        transforms were measured on (only the followed joints' rows may differ).
        """
        key = tuple(values[n] for n in self.names)
        if key != self._key:
            self._fixed = self._transforms(values)
            self._key = key
        refused = []
        for kc in kinematics_configs:
            dst = kc.fixed_transforms
            fixed = self._fixed.to(device=dst.device, dtype=dst.dtype)
            if dst.shape != fixed.shape or not self._same_layout(dst, fixed):
                refused.append(kc)
                continue
            dst[self._moving] = fixed[self._moving]
            lock = kc.lock_jointstate
            if lock is not None:
                for name in self.names:
                    if name in lock.joint_names:
                        lock.position[..., lock.joint_names.index(name)] = values[name]
        return refused

    def _same_layout(self, dst, fixed):
        static = torch.ones(dst.shape[0], dtype=torch.bool, device=dst.device)
        static[self._moving] = False
        return bool(torch.allclose(dst[static], fixed[static], atol=1e-5))
