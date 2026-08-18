"""
Bridges SharedMemoryBridge (plain numpy, no Qt knowledge) into Qt signals on
a timer. Kept as its own file so we can easily swap to a different viewer and reuse this
poll-and-emit pattern against a completely different data source without touching any shared-memory code, or vice versa
"""
from PySide6.QtCore import QObject, QTimer, Signal

from shared_memory_bridge import SharedMemoryBridge, TrainingStats


class FrameSource(QObject):
    frame_ready = Signal(object)     # np.ndarray (H, W, C) float32
    target_ready = Signal(object)    # np.ndarray, emitted once when first available
    stats_updated = Signal(object)   # TrainingStats
    producer_lost = Signal()         # C++ side exit'ed (header.running flipped to 0)

    def __init__(self, tag: str, poll_ms: int = 33, parent=None):
        super().__init__(parent)
        self.bridge = SharedMemoryBridge(tag)
        self._last_counter = -1
        self._target_emitted = False
        self._was_running = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(poll_ms)  # ~30Hz; plenty for a training preview, cheap on CPU

    def _poll(self):
        if not self.bridge.connected:
            self.bridge.connect()  # retries silently until the trainer creates the mappings
            return

        stats: TrainingStats = self.bridge.read_stats()
        self.stats_updated.emit(stats)

        if stats.running:
            self._was_running = True
        elif self._was_running:
            self.producer_lost.emit()
            self._was_running = False

        if not self._target_emitted:
            target = self.bridge.read_target_frame()
            if target is not None:
                self.target_ready.emit(target)
                self._target_emitted = True

        counter = self.bridge.last_frame_counter()
        if counter != self._last_counter:
            frame = self.bridge.read_current_frame()
            if frame is not None:
                self._last_counter = counter
                self.frame_ready.emit(frame)
