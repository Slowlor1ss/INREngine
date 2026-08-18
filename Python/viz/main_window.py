"""
Main window for the INR live viewer. Specific for our 2D image reconstruction
so this is the file to replace when building a different visualizer, while
shared_memory_bridge.py / frame_source.py / image_canvas.py / main.py's CLI
scaffolding can mostly stay as-is or be lightly adapted.
"""
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QLabel, QSlider, QPushButton, QButtonGroup, QRadioButton, QStatusBar
)

from frame_source import FrameSource
from image_canvas import ImageCanvas
from diff_utils import compute_diff_heatmap, nearest_resize

MODE_FULL, MODE_SIDE_BY_SIDE, MODE_DIFF = range(3)


class MainWindow(QMainWindow):
    def __init__(self, tag: str):
        super().__init__()
        self.setWindowTitle(f"INR Live Viewer -- {tag}")
        self.resize(1100, 800)

        self._latest_frame: Optional[np.ndarray] = None
        self._target_frame: Optional[np.ndarray] = None

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

        # Mode selector + size controls
        top_row = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.radio_full = QRadioButton("Full")
        self.radio_side = QRadioButton("Side by side")
        self.radio_diff = QRadioButton("Diff")
        self.radio_full.setChecked(True)
        for idx, rb in enumerate([self.radio_full, self.radio_side, self.radio_diff]):
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

        # Diff tolerance slider
        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Diff tolerance:"))
        self.tolerance_slider = QSlider(Qt.Horizontal)
        self.tolerance_slider.setRange(0, 100)
        self.tolerance_slider.setValue(5)
        self.tolerance_label = QLabel("5%")
        self.tolerance_slider.valueChanged.connect(lambda v: self.tolerance_label.setText(f"{v}%"))
        self.tolerance_slider.valueChanged.connect(self._refresh_diff_if_active)
        tol_row.addWidget(self.tolerance_slider)
        tol_row.addWidget(self.tolerance_label)
        root.addLayout(tol_row)

        # Stacked views (index must match MODE_* constants above)
        self.stack = QStackedWidget()

        self.canvas_full = ImageCanvas()
        self.stack.addWidget(self.canvas_full)

        side_widget = QWidget()
        side_layout = QHBoxLayout(side_widget)
        side_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_current = ImageCanvas()
        self.canvas_target = ImageCanvas()
        side_layout.addWidget(self.canvas_current)
        side_layout.addWidget(self.canvas_target)
        self.stack.addWidget(side_widget)

        self.canvas_diff = ImageCanvas()
        self.stack.addWidget(self.canvas_diff)

        root.addWidget(self.stack, stretch=1)
        self.setCentralWidget(central)

        self.status = QStatusBar()
        self.status.showMessage("Waiting for trainer to attach...")
        self.setStatusBar(self.status)

    # Slots
    def _on_frame(self, frame: np.ndarray):
        self._latest_frame = frame
        self.canvas_full.set_frame(frame)
        self.canvas_current.set_frame(frame)
        self._refresh_diff_if_active()

    def _on_target(self, frame: np.ndarray):
        self._target_frame = frame
        self.canvas_target.set_frame(frame)
        self._refresh_diff_if_active()

    def _on_stats(self, stats):
        state = "running" if stats.running else "PRODUCER STOPPED"
        self.status.showMessage(
            f"epoch={stats.epoch}   cost={stats.cost:.6f}   lr={stats.learning_rate:.6f}   [{state}]")

    def _on_producer_lost(self):
        self.status.showMessage("Training process disconnected -- showing last frame received.", 5000)

    def _on_mode_changed(self, idx: int):
        self.stack.setCurrentIndex(idx)
        if idx == MODE_DIFF:
            self._refresh_diff_if_active()

    def _refresh_diff_if_active(self):
        if self.stack.currentIndex() != MODE_DIFF:
            return
        if self._latest_frame is None or self._target_frame is None:
            return

        current = self._latest_frame
        target = self._target_frame
        if current.shape != target.shape:
            # Render resolution differs from the target's native resolution (e.g. upscaling) --
            # resize target to match rather than erroring out.
            target = nearest_resize(target, current.shape[1], current.shape[0])

        tolerance = self.tolerance_slider.value() / 100.0
        heatmap = compute_diff_heatmap(current, target, tolerance)
        self.canvas_diff.set_frame(heatmap)

    def _fit_all(self):
        for c in (self.canvas_full, self.canvas_current, self.canvas_target, self.canvas_diff):
            c.fit_to_window()

    def _actual_all(self):
        for c in (self.canvas_full, self.canvas_current, self.canvas_target, self.canvas_diff):
            c.actual_size()
