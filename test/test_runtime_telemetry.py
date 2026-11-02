import pytest

from curobo_ros.core.runtime_telemetry import RuntimeTelemetry


def test_runtime_telemetry_is_disabled_by_default():
    telemetry = RuntimeTelemetry()
    telemetry.count("frames")
    telemetry.observe_ms("stage", 1.0)
    assert telemetry.snapshot() == {}


def test_runtime_telemetry_reports_bounded_latency_summary():
    telemetry = RuntimeTelemetry(enabled=True, max_samples=2)
    telemetry.count("frames", 3)
    telemetry.observe_ms("stage", 0.001)
    telemetry.observe_ms("stage", 0.003)
    telemetry.observe_ms("stage", 0.100)
    telemetry.set("stamp_ns", 7)

    snapshot = telemetry.snapshot()
    assert snapshot["counts"] == {"frames": 3}
    assert snapshot["fields"] == {"stamp_ns": 7}
    latency = snapshot["latency"]["stage"]
    assert latency["n"] == 2
    assert latency["median_ms"] == 51.5
    assert latency["p95_ms"] == pytest.approx(95.15)
    assert latency["max_ms"] == 100.0
