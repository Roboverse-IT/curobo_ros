"""Small opt-in runtime counters and latency summaries for perception nodes."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import threading

import numpy as np


class RuntimeTelemetry:
    """Thread-safe bounded statistics; inert unless the owner enables it."""

    def __init__(self, enabled=False, max_samples=2048):
        self.enabled = bool(enabled)
        self._counts = Counter()
        self._latencies = defaultdict(lambda: deque(maxlen=max_samples))
        self._fields = {}
        self._lock = threading.Lock()

    def count(self, name, value=1):
        if self.enabled:
            with self._lock:
                self._counts[name] += int(value)

    def observe_ms(self, name, seconds):
        if self.enabled:
            with self._lock:
                self._latencies[name].append(float(seconds) * 1e3)

    def set(self, name, value):
        if self.enabled:
            with self._lock:
                self._fields[name] = value

    def snapshot(self):
        if not self.enabled:
            return {}
        with self._lock:
            latency = {}
            for name, values in self._latencies.items():
                if values:
                    a = np.asarray(values, dtype=float)
                    latency[name] = {'n': int(a.size), 'median_ms': float(np.median(a)),
                                     'p95_ms': float(np.percentile(a, 95)),
                                     'max_ms': float(a.max())}
            return {'counts': dict(self._counts), 'latency': latency,
                    'fields': dict(self._fields)}
