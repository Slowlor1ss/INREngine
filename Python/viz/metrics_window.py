# Make sure this stays above "import pyqtgraph as pg" idk why but it tries to use PyQt5
import os
os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"  # Force pyqtgraph to use PySide6!

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QTableWidget, QTableWidgetItem, QTabWidget
)

from metrics import METRIC_FIELDS, METRIC_LABELS
from metrics_history import MetricsHistory, MAX_SAMPLES

pg.setConfigOptions(antialias=True)

# Numeric Sorting
class NumericTableWidgetItem(QTableWidgetItem):
    def __lt__(self, other):
        # Convert text to float for comparison
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return self.text() < other.text()

class MetricsWindow(QMainWindow):
    def __init__(self, history: MetricsHistory, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)  # so destroyed fires on close, not just hide
        self.setWindowTitle("Metrics")
        self.resize(950, 750)
        self.history = history

        central = QWidget()
        root = QVBoxLayout(central)

        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Show last:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(10, MAX_SAMPLES)
        self.count_spin.setValue(200)
        self.count_spin.setSuffix(" samples")
        self.count_spin.valueChanged.connect(self.refresh)
        controls_row.addWidget(self.count_spin)
        controls_row.addStretch()
        root.addLayout(controls_row)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, stretch=1)

        # History log tab
        self.table = QTableWidget()
        columns = ["Epoch", "Cost"] + [METRIC_LABELS[f] for f in METRIC_FIELDS]
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.tabs.addTab(self.table, "History Log")

        # Graphs tab: one PlotWidget per metric, stacked vertically
        graphs_widget = QWidget()
        graphs_layout = QVBoxLayout(graphs_widget)
        self.plot_curves = {}
        for field in METRIC_FIELDS:
            pw = pg.PlotWidget(title=METRIC_LABELS[field])
            pw.setLabel("bottom", "Epoch")
            pw.showGrid(x=True, y=True, alpha=0.3)
            self.plot_curves[field] = pw.plot(pen=pg.mkPen(width=2))
            graphs_layout.addWidget(pw)
        self.tabs.addTab(graphs_widget, "Graphs")

        self.setCentralWidget(central)

    def refresh(self):
        count = self.count_spin.value()
        samples = self.history.recent(count)

        self.table.setSortingEnabled(False)

        # Table: by deafault newest at the top, easiest to eyeball during a live run.
        self.table.setRowCount(len(samples))
        
        for row, s in enumerate(reversed(samples)):
            values = [s.epoch, s.cost] + [getattr(s, f) for f in METRIC_FIELDS]
            for col, v in enumerate(values):
                text = f"{v:.6f}" if isinstance(v, float) else str(v)
                self.table.setItem(row, col, NumericTableWidgetItem(text))

        self.table.resizeColumnsToContents()
        # Enable sorting (if clicked on a label)
        self.table.setSortingEnabled(True)

        # Graphs
        if samples:
            epochs = [s.epoch for s in samples]
            for field in METRIC_FIELDS:
                self.plot_curves[field].setData(epochs, [getattr(s, field) for s in samples])
