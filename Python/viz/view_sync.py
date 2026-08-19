"""
Keeps zoom and pan same over both images
"""
from PySide6.QtCore import QObject


class ViewSyncGroup(QObject):
    def __init__(self, canvases, parent=None):
        super().__init__(parent)
        self.canvases = list(canvases)
        self._syncing = False  # guards against reentering sync loops

        for c in self.canvases:
            c.view_changed.connect(self._on_view_changed)

    def _on_view_changed(self, source):
        if self._syncing:
            return
        self._syncing = True
        try:
            transform = source.transform()
            h_val = source.horizontalScrollBar().value()
            v_val = source.verticalScrollBar().value()
            fit_mode = source.is_fit_mode()

            for c in self.canvases:
                if c is source:
                    continue
                # Mirror fit_mode too: without this, a canvas still in "fit to window" mode would re-fit 
                # (and silently undo the sync) the next time it receives a new frame via set_frame()
                c.set_fit_mode(fit_mode)
                c.setTransform(transform)
                c.horizontalScrollBar().setValue(h_val)
                c.verticalScrollBar().setValue(v_val)
        finally:
            self._syncing = False
