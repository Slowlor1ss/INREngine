"""
Pixel inspector: two 9x9 close-up grids (one per image) centered on whatever pixel the mouse is over in CompareView, 
plus X/Y and RGB readouts for both
"""
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel

GRID_N = 9
CELL_PX = 14


def extract_patch(arr: Optional[np.ndarray], x: int, y: int, n: int = GRID_N) -> np.ndarray:
    patch = np.zeros((n, n, 3), dtype=np.float32)
    if arr is None:
        return patch

    h, w = arr.shape[0], arr.shape[1]
    half = n // 2
    y0, y1 = y - half, y + half + 1
    x0, x1 = x - half, x + half + 1

    src_y0, src_y1 = max(0, y0), min(h, y1)
    src_x0, src_x1 = max(0, x0), min(w, x1)
    if src_y0 >= src_y1 or src_x0 >= src_x1:
        return patch  # entirely off-image

    dst_y0, dst_x0 = src_y0 - y0, src_x0 - x0
    patch[dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0)] = \
        arr[src_y0:src_y1, src_x0:src_x1]
    return patch


class PixelGrid(QWidget):
    """Renders one n x n patch as a grid of solid-colour cells, with the
    center cell (the actually-hovered pixel) outlined"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._patch = np.zeros((GRID_N, GRID_N, 3), dtype=np.float32)
        side = CELL_PX * GRID_N
        self.setFixedSize(QSize(side, side))

    def set_patch(self, patch: np.ndarray):
        self._patch = patch
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        for row in range(GRID_N):
            for col in range(GRID_N):
                r, g, b = np.clip(self._patch[row, col], 0.0, 1.0)
                painter.fillRect(col * CELL_PX, row * CELL_PX, CELL_PX, CELL_PX,
                                  QColor(int(r * 255), int(g * 255), int(b * 255)))

        center = GRID_N // 2
        painter.setPen(QPen(Qt.white, 2))
        painter.drawRect(center * CELL_PX + 1, center * CELL_PX + 1, CELL_PX - 2, CELL_PX - 2)


class PixelInspector(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.addWidget(QLabel("<b>Pixel Inspector</b>"))

        self.coord_label = QLabel("X=-   Y=-")
        root.addWidget(self.coord_label)

        grids_row = QHBoxLayout()

        left_col = QVBoxLayout()
        left_col.addWidget(QLabel("Left (current)"))
        self.grid_left = PixelGrid()
        left_col.addWidget(self.grid_left)
        self.rgb_left_label = QLabel("R=-  G=-  B=-")
        left_col.addWidget(self.rgb_left_label)
        grids_row.addLayout(left_col)

        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("Right (target)"))
        self.grid_right = PixelGrid()
        right_col.addWidget(self.grid_right)
        self.rgb_right_label = QLabel("R=-  G=-  B=-")
        right_col.addWidget(self.rgb_right_label)
        grids_row.addLayout(right_col)

        root.addLayout(grids_row)
        root.addStretch()

    def update_pixel(self, x: int, y: int, current: Optional[np.ndarray], target: Optional[np.ndarray]):
        """Update hovered pixel info for the 9x9 grid"""
        self.coord_label.setText(f"X={x}   Y={y}")
        self.grid_left.set_patch(extract_patch(current, x, y))
        self.grid_right.set_patch(extract_patch(target, x, y))
        self.rgb_left_label.setText(_rgb_text(current, x, y))
        self.rgb_right_label.setText(_rgb_text(target, x, y))


def _rgb_text(arr: Optional[np.ndarray], x: int, y: int) -> str:
    if arr is None or not (0 <= y < arr.shape[0] and 0 <= x < arr.shape[1]):
        return "R=-NaN  G=-NaN  B=-NaN"
    r, g, b = arr[y, x]
    return f"R={r:.3f}  G={g:.3f}  B={b:.3f}"
