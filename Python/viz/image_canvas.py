"""
Generic zoomable, pannable image display widget built on QGraphicsView.
"""
import numpy as np
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QImage, QPixmap, QPainter, QWheelEvent
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem


class ImageCanvas(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)  # click-drag to pan
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)  # zoom centers on cursor
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(Qt.darkGray)

        self._fit_mode = True  # True = "fit to window", False = free zoom / actual size

    def set_frame(self, rgb_float: np.ndarray):
        """rgb_float: (H, W, 3) float32, values expected roughly in [0,1]."""
        h, w, _ = rgb_float.shape
        clipped = np.clip(rgb_float, 0.0, 1.0)
        rgb8 = np.ascontiguousarray((clipped * 255.0).astype(np.uint8))
        qimg = QImage(rgb8.data, w, h, w * 3, QImage.Format_RGB888)
        # .copy() so the QImage/QPixmap owns its own buffer -- rgb8 is a local
        # temporary that would otherwise get garbage collected out from under it.
        self._pixmap_item.setPixmap(QPixmap.fromImage(qimg.copy()))
        self._scene.setSceneRect(QRectF(0, 0, w, h))
        if self._fit_mode:
            self.fit_to_window()

    def fit_to_window(self):
        self._fit_mode = True
        if not self._scene.sceneRect().isEmpty():
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def actual_size(self):
        self._fit_mode = False
        self.resetTransform()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit_to_window()

    def wheelEvent(self, event: QWheelEvent):
        self._fit_mode = False
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
