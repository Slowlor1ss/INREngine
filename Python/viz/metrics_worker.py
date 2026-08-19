"""
Runs compute_image_metrics() on a background QThread so a sliding-window SSIM pass on a large frame never cause the GUI to hnag
MetricsController owns making sure only one of these runs at a time
"""
from PySide6.QtCore import QThread, Signal

from metrics import compute_image_metrics, MetricsSample


class MetricsComputeThread(QThread):
    finished_with_result = Signal(object)  # MetricsSample

    def __init__(self, current, target, epoch: int, cost: float, parent=None):
        super().__init__(parent)
        self._current = current
        self._target = target
        self._epoch = epoch
        self._cost = cost

    def run(self):
        sample: MetricsSample = compute_image_metrics(self._current, self._target, self._epoch, self._cost)
        self.finished_with_result.emit(sample)
