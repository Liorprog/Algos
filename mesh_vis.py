"""
algos/mesh_visualizer.py

Simple Qt-based visualizer for a 2D MeshND (pure-Python nested-list mesh).

Features:
- Shows the mesh as a heatmap (rectangles colored by amplitude).
- Hover to see index, center coordinate and value (tooltip).
- Optionally add points by clicking the heatmap.
- Normalize the mesh using configurable thresholds and divisor.
- Optional overlay of peaks (index tuples).
- Save view as PNG via File -> Save as PNG.

Dependencies:
- PyQt5 (pip install PyQt5) OR PySide6 (small import change shown below).

Usage:
    from algos.mesh import MeshND
    from algos.mesh_visualizer import MeshWindow, run_example_gui

    mesh = MeshND(bounds=[(0,1),(0,1)], cells=[50,50])
    ... add points ...
    app = MeshWindow(mesh, peaks=mesh.get_peaks(min_value=...))
    app.show()
    QApplication.exec_()

Notes:
- This visualizer expects a 2D mesh (ndim == 2). It raises ValueError otherwise.
- If you want PySide6 instead of PyQt5, replace the PyQt5 imports at top:
    from PySide6 import QtWidgets, QtGui, QtCore
  and adapt Q* names accordingly (they are mostly identical).
"""
from typing import List, Tuple, Optional
import math

# Use PyQt5 by default. If you prefer PySide6, replace imports accordingly.
from PyQt5 import QtWidgets, QtGui, QtCore


class MeshVisualizer(QtWidgets.QWidget):
    """
    QWidget that draws a 2D mesh (heatmap) from a MeshND instance.

    Important: expects mesh.ndim == 2.
    """

    point_added = QtCore.pyqtSignal(float, float)

    def __init__(
        self,
        mesh,
        show_values: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        if mesh.ndim != 2:
            raise ValueError("MeshVisualizer currently only supports 2D meshes.")
        self.mesh = mesh
        self.show_values = show_values
        self.add_points_enabled = False
        self.auto_color_scale = True
        self.fixed_color_max = 1.0
        # caches
        self._max_value = None
        self.setMouseTracking(True)  # track mouse to show tooltips
        self.setMinimumSize(300, 300)

    def set_mesh(self, mesh):
        if mesh.ndim != 2:
            raise ValueError("MeshVisualizer currently only supports 2D meshes.")
        self.mesh = mesh
        self._max_value = None
        self.update()

    def set_peaks(self, peaks: List[Tuple[Tuple[int, int], float]]):
        self.peaks = peaks or []
        self.update()

    def refresh(self):
        """Clear cached color scaling and repaint the mesh."""
        self._max_value = None
        self.update()

    def set_add_points_enabled(self, enabled: bool):
        self.add_points_enabled = enabled
        cursor = QtCore.Qt.CrossCursor if enabled else QtCore.Qt.ArrowCursor
        self.setCursor(cursor)

    def set_auto_color_scale(self, enabled: bool):
        self.auto_color_scale = enabled
        self.refresh()

    def set_fixed_color_max(self, value: float):
        self.fixed_color_max = max(float(value), 1e-12)
        if not self.auto_color_scale:
            self.update()

    def sizeHint(self):
        return QtCore.QSize(600, 600)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)

        counts = self.mesh.get_counts()
        nx = self.mesh.cells[0]
        ny = self.mesh.cells[1]

        w = self.width()
        h = self.height()

        # compute cell pixel sizes (we map ix -> x, iy -> y with iy increasing downward)
        cell_w = w / nx
        cell_h = h / ny

        # Auto scale highlights relative differences. A fixed scale keeps colors
        # comparable as points are added, clipping values above the limit to red.
        if self.auto_color_scale:
            if self._max_value is None:
                m = 0.0
                for i in range(nx):
                    for j in range(ny):
                        v = self.mesh.get(i,j)
                        if v > m:
                            m = v
                self._max_value = m if m > 0 else 1.0
            maxv = self._max_value
        else:
            maxv = self.fixed_color_max

        # draw cells
        for i in range(nx):
            x = int(i * cell_w)
            for j in range(ny):
                y = int(j * cell_h)
                v = self.mesh.get(i,j)
                color = self._value_to_color(v, maxv)
                painter.fillRect(QtCore.QRectF(x, y, cell_w + 0.5, cell_h + 0.5), color)

        # optionally draw grid lines for clarity (light)
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 40))
        pen.setWidth(0)
        painter.setPen(pen)
        # vertical lines
        for i in range(1, nx):
            xpos = i * cell_w
            painter.drawLine(QtCore.QLineF(xpos, 0, xpos, h))
        # horizontal lines
        for j in range(1, ny):
            ypos = j * cell_h
            painter.drawLine(QtCore.QLineF(0, ypos, w, ypos))

        # # draw peaks as small circles with labels
        # if self.peaks:
        #     pen = QtGui.QPen(QtGui.QColor(255, 255, 255))
        #     pen.setWidth(2)
        #     painter.setPen(pen)
        #     brush = QtGui.QBrush(QtGui.QColor(255, 0, 0, 200))
        #     painter.setBrush(brush)
        #     for idx, val in self.peaks:
        #         i, j = idx
        #         cx = (i + 0.5) * cell_w
        #         cy = (j + 0.5) * cell_h
        #         radius = min(cell_w, cell_h) * 0.3
        #         painter.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)
        #         # small label
        #         painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        #         painter.drawText(QtCore.QPointF(cx + radius + 2, cy), f"{round(val,3)}")
        #
        # # optionally draw numeric values (only if requested and scale allows)
        # if self.show_values:
        #     painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))
        #     font = painter.font()
        #     font.setPointSize(8)
        #     painter.setFont(font)
        #     for i in range(nx):
        #         x = i * cell_w
        #         for j in range(ny):
        #             y = j * cell_h
        #             v = counts[i][j]
        #             if v != 0:
        #                 painter.drawText(QtCore.QRectF(x, y, cell_w, cell_h), QtCore.Qt.AlignCenter, f"{v:.2f}")
        painter.end()

    def _value_to_color(self, v: float, maxv: float) -> QtGui.QColor:
        """
        Map value v (0..maxv) to a QColor.
        Uses a blue->cyan->yellow->red mapping by hue interpolation.
        """
        if v <= 0:
            return QtGui.QColor(30, 30, 30)  # dark background for zeros
        frac = max(0.0, min(1.0, v / float(maxv)))
        # hue from blue (0.66) to red (0) in HSV: interpolate hue
        hue_blue = 0.66  # blue
        hue_red = 0.0
        hue = hue_blue * (1.0 - frac) + hue_red * frac
        # convert HSV to RGB via QColor.fromHsvF
        qcol = QtGui.QColor()
        qcol.setHsvF(hue, 0.85, 0.9)
        return qcol

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        # Show tooltip with index, coordinates, and value for hovered cell
        pos = event.pos()
        nx = self.mesh.cells[0]
        ny = self.mesh.cells[1]
        cell_w = self.width() / nx
        cell_h = self.height() / ny
        ix = int(pos.x() // cell_w)
        iy = int(pos.y() // cell_h)
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            QtWidgets.QToolTip.hideText()
            return
        val = self.mesh.get(ix,iy)
        centers = self.mesh.get_centers()
        coord = (centers[0][ix], centers[1][iy])
        tip = f"idx=({ix},{iy}) val={val:.6g}\ncenter=({coord[0]:.4f},{coord[1]:.4f})"
        QtWidgets.QToolTip.showText(event.globalPos(), tip, self)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if not self.add_points_enabled or event.button() != QtCore.Qt.LeftButton:
            super().mousePressEvent(event)
            return

        width = self.width()
        height = self.height()
        if width <= 0 or height <= 0:
            return

        # The drawing maps axis 0 left-to-right and axis 1 top-to-bottom.
        x_fraction = max(0.0, min(1.0, event.pos().x() / width))
        y_fraction = max(0.0, min(1.0, event.pos().y() / height))
        x_min, x_max = self.mesh.bounds[0]
        y_min, y_max = self.mesh.bounds[1]
        x = x_min + x_fraction * (x_max - x_min)
        y = y_min + y_fraction * (y_max - y_min)

        self.mesh.add((x, y))
        self.refresh()
        self.point_added.emit(x, y)

    def leaveEvent(self, event):
        QtWidgets.QToolTip.hideText()

    def save_png(self, path: str):
        """Save current widget rendering to PNG."""
        pix = QtGui.QPixmap(self.size())
        self.render(pix)
        pix.save(path, "PNG")


class MeshWindow(QtWidgets.QMainWindow):
    """Simple QMainWindow wrapper providing a menu and the MeshVisualizer as central widget."""

    def __init__(self, mesh, peaks=None, show_values=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MeshND Visualizer")
        self.visualizer = MeshVisualizer(mesh, show_values=show_values)
        self.visualizer.set_peaks(peaks)
        self._create_central_widget()
        self._create_menu()

    def _create_central_widget(self):
        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(central)
        controls = QtWidgets.QHBoxLayout()

        self.add_points_checkbox = QtWidgets.QCheckBox("Add points")
        self.add_points_checkbox.toggled.connect(
            self.visualizer.set_add_points_enabled
        )
        controls.addWidget(self.add_points_checkbox)

        controls.addSpacing(12)
        self.auto_scale_checkbox = QtWidgets.QCheckBox("Auto colors")
        self.auto_scale_checkbox.setChecked(True)
        self.auto_scale_checkbox.toggled.connect(self._on_auto_scale_toggled)
        controls.addWidget(self.auto_scale_checkbox)

        self.color_max_input = self._make_number_input(1.0)
        self.color_max_input.setMinimum(0.000001)
        self.color_max_input.setEnabled(False)
        self.color_max_input.valueChanged.connect(
            self.visualizer.set_fixed_color_max
        )
        controls.addWidget(QtWidgets.QLabel("Color max:"))
        controls.addWidget(self.color_max_input)

        controls.addSpacing(12)
        self.low_threshold_input = self._make_number_input(0.0)
        self.high_threshold_input = self._make_number_input(1.0)
        self.divide_const_input = self._make_number_input(2.0)

        controls.addWidget(QtWidgets.QLabel("Low threshold:"))
        controls.addWidget(self.low_threshold_input)
        controls.addWidget(QtWidgets.QLabel("High threshold:"))
        controls.addWidget(self.high_threshold_input)
        controls.addWidget(QtWidgets.QLabel("Divide by:"))
        controls.addWidget(self.divide_const_input)

        normalize_button = QtWidgets.QPushButton("Normalize")
        normalize_button.clicked.connect(self._on_normalize)
        controls.addWidget(normalize_button)
        controls.addStretch()

        layout.addLayout(controls)
        layout.addWidget(self.visualizer, 1)
        self.setCentralWidget(central)

        self.visualizer.point_added.connect(
            lambda x, y: self.statusBar().showMessage(
                f"Added point ({x:.4f}, {y:.4f})", 2500
            )
        )

    def _on_auto_scale_toggled(self, enabled):
        self.color_max_input.setEnabled(not enabled)
        self.visualizer.set_auto_color_scale(enabled)

    @staticmethod
    def _make_number_input(value):
        field = QtWidgets.QDoubleSpinBox()
        field.setRange(0.0, 1_000_000_000.0)
        field.setDecimals(6)
        field.setValue(value)
        return field

    def _on_normalize(self):
        self.visualizer.mesh.normalize_counts(
            low_threshold=self.low_threshold_input.value(),
            high_threshold=self.high_threshold_input.value(),
            divide_const=self.divide_const_input.value(),
        )
        self.visualizer.refresh()
        self.statusBar().showMessage("Mesh normalized", 2500)

    def _create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        save_action = QtWidgets.QAction("Save as PNG...", self)
        save_action.triggered.connect(self._on_save_png)
        file_menu.addAction(save_action)

        quit_action = QtWidgets.QAction("Quit", self)
        quit_action.triggered.connect(QtWidgets.QApplication.instance().quit)
        file_menu.addAction(quit_action)

    def _on_save_png(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save as PNG", "", "PNG files (*.png)")
        if path:
            self.visualizer.save_png(path)


def run_gui(mesh, show_values=False):
    """Helper: start QApplication and open MeshWindow."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    win = MeshWindow(mesh, show_values=show_values)
    win.show()
    app.exec_()
