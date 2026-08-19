"""
Main window for the INR live viewer. Specific for our 2D image reconstruction
so this can be replaced when building a different visualizer, while
shared_memory_bridge.py / frame_source.py / image_canvas.py / main.py's
can mostly stay the same
"""
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QPushButton, QButtonGroup, QRadioButton, QStatusBar
)

from frame_source import FrameSource
from image_canvas import ImageCanvas
from compare_view import CompareView

MODE_FULL, MODE_COMPARE = range(2)


class MainWindow(QMainWindow):
    def __init__(self, tag: str):
        super().__init__()
        self.setWindowTitle(f"INR Live Viewer -- {tag}")
        self.resize(1280, 900)

        self.source = FrameSource(tag)
        self.source.frame_ready.connect(self._on_frame)
        self.source.target_ready.connect(self._on_target)
        self.source.stats_updated.connect(self._on_stats)
        self.source.producer_lost.connect(self._on_producer_lost)

        self._build_ui()

    # UI
    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)

        top_row = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.radio_full = QRadioButton("Full")
        self.radio_compare = QRadioButton("Compare")
        self.radio_full.setChecked(True)
        for idx, rb in enumerate([self.radio_full, self.radio_compare]):
            self.mode_group.addButton(rb, idx)
            top_row.addWidget(rb)
        self.mode_group.idClicked.connect(self._on_mode_changed)

        top_row.addStretch()
        fit_btn = QPushButton("Fit to window")
        actual_btn = QPushButton("Actual size (100%)")
        fit_btn.clicked.connect(self._fit_all)
        actual_btn.clicked.connect(self._actual_all)
        top_row.addWidget(fit_btn)
        top_row.addWidget(actual_btn)
        root.addLayout(top_row)

        self.stack = QStackedWidget()

        self.canvas_full = ImageCanvas()
        self.stack.addWidget(self.canvas_full)          # index MODE_FULL

        self.compare_view = CompareView()
        self.stack.addWidget(self.compare_view)          # index MODE_COMPARE

        root.addWidget(self.stack, stretch=1)
        self.setCentralWidget(central)

        self.status = QStatusBar()
        self.status.showMessage("Waiting for trainer to attach...")
        self.setStatusBar(self.status)

    # Slots
    def _on_frame(self, frame: np.ndarray):
        self.canvas_full.set_frame(frame)
        self.compare_view.set_current_frame(frame)

    def _on_target(self, frame: np.ndarray):
        self.compare_view.set_target_frame(frame)

    def _on_stats(self, stats):
        state = "running" if stats.running else "PRODUCER STOPPED"
        self.status.showMessage(
            f"epoch={stats.epoch}   cost={stats.cost:.6f}   lr={stats.learning_rate:.6f}   [{state}]")

    def _on_producer_lost(self):
        self.status.showMessage("Training process disconnected, showing last frame received", 5000)

    def _on_mode_changed(self, idx: int):
        self.stack.setCurrentIndex(idx)

    def _fit_all(self):
        self.canvas_full.fit_to_window()
        self.compare_view.fit_all()

    def _actual_all(self):
        self.canvas_full.actual_size()
        self.compare_view.actual_size_all()
        