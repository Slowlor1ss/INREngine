"""
Rolling buffer of MetricsSample, plus helpers to pull out per-metric arrays
for the table/graphs in MetricsWindow
"""
from collections import deque
from typing import Deque, List

import numpy as np

from metrics import MetricsSample

MAX_SAMPLES = 5000 # hard max for sanity reasons

class MetricsHistory:
    def __init__(self):
        self._samples: Deque[MetricsSample] = deque(maxlen=MAX_SAMPLES)

    def add(self, sample: MetricsSample):
        self._samples.append(sample)

    def __len__(self) -> int:
        return len(self._samples)

    def recent(self, count: int) -> List[MetricsSample]:
        """Most recent `count` samples, oldest first count <= 0 means "all" """
        if count <= 0 or count >= len(self._samples):
            return list(self._samples)
        return list(self._samples)[-count:]

    def field_array(self, field: str, count: int) -> np.ndarray:
        return np.array([getattr(s, field) for s in self.recent(count)], dtype=np.float64)

    def epoch_array(self, count: int) -> np.ndarray:
        return np.array([s.epoch for s in self.recent(count)], dtype=np.int64)
