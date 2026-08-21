"""
Periodic (NOT per frame windowed SSIM is too heavy for 30Hz) metrics computation on a background thread, 
owns the MetricsHistory + MetricsWindow lifecycle
"""
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, QTimer

from diff_utils import nearest_resize
from metrics_history import MetricsHistory
from metrics_worker import MetricsComputeThread
from metrics_window import MetricsWindow

COMPUTE_INTERVAL_MS = 1000  # Metrics are heavier than a frame update, 1Hz should be fine...


class MetricsController(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = MetricsHistory()
        self._window: Optional[MetricsWindow] = None
        self._worker: Optional[MetricsComputeThread] = None

        self._current_frame: Optional[np.ndarray] = None
        self._target_frame: Optional[np.ndarray] = None
        self._epoch = 0
        self._cost = 0.0
        self._frame_dirty = False  # True once a new frame has arrived since the last computation

        self._timer = QTimer(self)
        self._timer.setInterval(COMPUTE_INTERVAL_MS)
        self._timer.timeout.connect(self._maybe_compute)
        self._timer.start()  # runs continuously, independent of whether the window is open

    # Data in
    def set_current_frame(self, frame: np.ndarray):
        self._current_frame = frame
        self._frame_dirty = True  # set dirty flag see _maybe_compute()

    def set_target_frame(self, frame: np.ndarray):
        self._target_frame = frame

    def set_stats(self, epoch: int, cost: float):
        self._epoch = epoch
        self._cost = cost

    # Window
    def show_window(self, parent_widget=None):
        if self._window is None:
            # TODO: commented bc of some bug I dont want to fix rn
            #self._window = MetricsWindow(self.history, parent_widget) # To connect to movement of mian window 
            self._window = MetricsWindow(self.history, None)
            self._window.destroyed.connect(self._on_window_closed)
            self._timer.start()
        self._window.show()
        self._window.raise_()
        self._window.activateWindow()
        self._window.refresh()

    def _on_window_closed(self):
        self._window = None
        #self._timer.stop()

    # Computation
    def _maybe_compute(self):
        if self._worker is not None:
            return  # A previous computation is still running; skip this tick rather than queue up
        if self._current_frame is None or self._target_frame is None:
            return

        if not self._frame_dirty:
            return  # Skip if theres no new data
        self._frame_dirty = False

        current, target = self._current_frame, self._target_frame
        if current.shape != target.shape:
            target = nearest_resize(target, current.shape[1], current.shape[0])

        self._worker = MetricsComputeThread(current, target, self._epoch, self._cost)
        self._worker.finished_with_result.connect(self._on_metrics_ready)
        self._worker.finished.connect(self._cleanup_worker)
        self._worker.start()

    def _on_metrics_ready(self, sample):
        self.history.add(sample)
        if self._window is not None:
            self._window.refresh()

    def _cleanup_worker(self):
        self._worker = None
