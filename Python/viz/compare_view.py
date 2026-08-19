from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QCheckBox

from image_canvas import ImageCanvas
from view_sync import ViewSyncGroup
from pixel_inspector import PixelInspector
from diff_utils import compute_diff_heatmap, apply_ghost_overlay, nearest_resize


class CompareView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_frame: Optional[np.ndarray] = None
        self._target_frame: Optional[np.ndarray] = None

        root = QVBoxLayout(self)

        # Top: left (current) / right (target)
        top_row = QHBoxLayout()
        self.canvas_left = ImageCanvas()
        self.canvas_right = ImageCanvas()
        top_row.addWidget(self.canvas_left)
        top_row.addWidget(self.canvas_right)
        root.addLayout(top_row, stretch=5) # 50/50 split with compare and diff

        # Controls: tolerance + ghost mode
        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Tolerance:"))
        self.tolerance_slider = QSlider(Qt.Horizontal)
        self.tolerance_slider.setRange(0, 100)
        self.tolerance_slider.setValue(5)
        self.tolerance_label = QLabel("5%")
        self.tolerance_slider.valueChanged.connect(lambda v: self.tolerance_label.setText(f"{v}%"))
        self.tolerance_slider.valueChanged.connect(self._refresh_diff)
        controls_row.addWidget(self.tolerance_slider)
        controls_row.addWidget(self.tolerance_label)

        controls_row.addSpacing(24)
        self.ghost_checkbox = QCheckBox("Ghost mode")
        self.ghost_checkbox.setChecked(True)
        self.ghost_checkbox.toggled.connect(self._refresh_diff)
        controls_row.addWidget(self.ghost_checkbox)

        controls_row.addWidget(QLabel("Ghost opacity:"))
        self.ghost_opacity_slider = QSlider(Qt.Horizontal)
        self.ghost_opacity_slider.setRange(0, 100)
        self.ghost_opacity_slider.setValue(35)
        self.ghost_opacity_label = QLabel("35%")
        self.ghost_opacity_slider.valueChanged.connect(lambda v: self.ghost_opacity_label.setText(f"{v}%"))
        self.ghost_opacity_slider.valueChanged.connect(self._refresh_diff)
        controls_row.addWidget(self.ghost_opacity_slider)
        controls_row.addWidget(self.ghost_opacity_label)
        controls_row.addStretch()
        root.addLayout(controls_row)

        # Bottom: pixel inspector (narrow) + diff/"tolerance" view (wide)
        bottom_row = QHBoxLayout()
        self.inspector = PixelInspector()
        bottom_row.addWidget(self.inspector)

        self.canvas_diff = ImageCanvas()
        bottom_row.addWidget(self.canvas_diff, stretch=1)
        # Diff view; largest stretch factor of the three rows
        root.addLayout(bottom_row, stretch=5) # 50/50 split with compare and diff

        # Keep zoom/pan identical across all three, assumes current/target share dimensions;
        # if they don't (e.g. output_image_scale upscaling), the sync is still applied
        # but "same location" becomes approximate since scene extents differ.
        self.sync_group = ViewSyncGroup([self.canvas_left, self.canvas_right, self.canvas_diff])

        # Continuous pixel inspection on hover from any of the three (they're synced,
        # so scene coordinates line up across all of them).
        self.canvas_left.pixel_hovered.connect(self._on_pixel_hovered)
        self.canvas_right.pixel_hovered.connect(self._on_pixel_hovered)
        self.canvas_diff.pixel_hovered.connect(self._on_pixel_hovered)


    # Data in
    def set_current_frame(self, frame: np.ndarray):
        self._current_frame = frame
        self.canvas_left.set_frame(frame)
        self._refresh_diff()

    def set_target_frame(self, frame: np.ndarray):
        self._target_frame = frame
        self.canvas_right.set_frame(frame)
        self._refresh_diff()

    def fit_all(self):
        for c in (self.canvas_left, self.canvas_right, self.canvas_diff):
            c.fit_to_window()

    def actual_size_all(self):
        for c in (self.canvas_left, self.canvas_right, self.canvas_diff):
            c.actual_size()


    # Internal
    def _refresh_diff(self):
        if self._current_frame is None or self._target_frame is None:
            return

        current = self._current_frame
        target = self._target_frame
        if current.shape != target.shape:
            target = nearest_resize(target, current.shape[1], current.shape[0])

        tolerance = self.tolerance_slider.value() / 100.0
        heatmap = compute_diff_heatmap(current, target, tolerance)

        if self.ghost_checkbox.isChecked():
            opacity = self.ghost_opacity_slider.value() / 100.0
            heatmap = apply_ghost_overlay(heatmap, target, opacity)

        self.canvas_diff.set_frame(heatmap)

    def _on_pixel_hovered(self, x: int, y: int):
        self.inspector.update_pixel(x, y, self._current_frame, self._target_frame)
