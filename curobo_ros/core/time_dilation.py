"""Slowing a sampled trajectory down without touching its path.

``time_dilation_factor`` f in (0, 1]: every time step is divided by f and every
velocity multiplied by f (accelerations scale by f^2), so limits that held at
f = 1 hold at any smaller f. 0 means "as planned".
"""


def check_factor(factor: float) -> None:
    """Raise ValueError unless the factor is 0 (unset) or in (0, 1]."""
    if factor != 0.0 and not 0.0 < factor <= 1.0:
        raise ValueError(
            f"time_dilation_factor must be 0 (as planned) or in (0, 1], got {factor}")


def dilate(waypoints, dt: float, factor: float) -> float:
    """Scale the waypoints' velocities in place by ``factor``; return the new dt."""
    check_factor(factor)
    if factor == 0.0 or factor == 1.0:
        return dt
    for waypoint in waypoints:
        waypoint.velocity = [v * factor for v in waypoint.velocity]
    return dt / factor
