"""Animated experiment for recovering real points from noisy observations.

Each cycle adds ``SAMPLES_PER_POINT`` noisy observations around every fixed
real point, normalizes the mesh, recomputes the global peak range, and updates
the GUI. White circles show peak cells; red targets show the real points.
"""
import math
import random

from PyQt5 import QtCore, QtWidgets

from mesh import MeshND
from mesh_vis import MeshWindow


# Experiment parameters
RANDOM_SEED = 1234
NUM_REAL_POINTS = 3
NUM_CYCLES = 100
SAMPLES_PER_POINT = 200
INITIAL_NOISE_STD = 0.08
CYCLE_INTERVAL_MS = 150
INITIAL_PEAK_PARAM = 1.25

GRID_CELLS = [50, 50]
MESH_RADIUS = 0.06
LOW_THRESHOLD = 0.02
HIGH_THRESHOLD = 10.0


def generate_noisy_samples(real_points, samples_per_point, noise_std, bounds):
    """Yield Gaussian-noisy observations, clipped to the mesh bounds."""
    for real_x, real_y in real_points:
        for _ in range(samples_per_point):
            noisy_x = random.gauss(real_x, noise_std)
            noisy_y = random.gauss(real_y, noise_std)
            noisy_x = max(bounds[0][0], min(bounds[0][1], noisy_x))
            noisy_y = max(bounds[1][0], min(bounds[1][1], noisy_y))
            yield noisy_x, noisy_y


def peaks_as_coordinates(mesh, peaks):
    """Convert peak grid indices to their physical cell-center coordinates."""
    centers = mesh.get_centers()
    return [((centers[0][idx[0]], centers[1][idx[1]]), value)
            for idx, value in peaks]


def raster_expected_mesh(mesh, real_points, noise_std):
    """Build the expected mesh by weighted raster convolution, then normalize."""
    mesh.reset()
    nearest_real_indices = [mesh._nearest_index(point) for point in real_points]

    for idx in mesh._all_indices():
        center = tuple(mesh.centers[d][idx[d]] for d in range(mesh.ndim))
        if noise_std == 0.0:
            weight = sum(idx == real_idx for real_idx in nearest_real_indices)
        else:
            weight = 0.0
            for real_point in real_points:
                distance_squared = sum(
                    (center[d] - real_point[d]) ** 2
                    for d in range(mesh.ndim)
                )
                weight += math.exp(
                    -0.5 * distance_squared / (noise_std * noise_std)
                )

        if weight > 0.0:
            # add() applies the mesh kernel, so the result approximates the
            # convolution of sample density with the normal add operation.
            mesh.add(center, amount=weight)

    mesh.normalize_counts()


class ExperimentController(QtCore.QObject):
    finished = QtCore.pyqtSignal()

    def __init__(self, mesh, real_points, window, peak_param_input,
                 noise_std_input, parent=None):
        super().__init__(parent)
        self.mesh = mesh
        self.real_points = real_points
        self.window = window
        self.peak_param_input = peak_param_input
        self.noise_std_input = noise_std_input
        self.current_cycle = 0

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(CYCLE_INTERVAL_MS)
        self.timer.timeout.connect(self.step)

    def start(self):
        if self.real_points:
            self.timer.start()

    def pause(self):
        self.timer.stop()

    def reset(self):
        self.timer.stop()
        self.current_cycle = 0
        self.mesh.reset()
        self.window.visualizer.set_peaks([])
        self.window.visualizer.refresh()

    def step(self):
        if self.current_cycle >= NUM_CYCLES:
            self.timer.stop()
            self.finished.emit()
            peaks = self.mesh.get_peaks(self.peak_param_input.value())
            print("Finished.")
            print("Real points:", self.real_points)
            print("Final peak cells:", peaks_as_coordinates(self.mesh, peaks))
            self.window.statusBar().showMessage(
                f"Finished {NUM_CYCLES} cycles with {len(peaks)} peak cells"
            )
            return

        samples = generate_noisy_samples(
            self.real_points,
            SAMPLES_PER_POINT,
            self.noise_std_input.value(),
            self.mesh.bounds,
        )
        for sample in samples:
            self.mesh.add(sample)

        # Repeated scaling keeps the maximum at HIGH_THRESHOLD and discards
        # cells that fall below LOW_THRESHOLD after scaling.
        self.mesh.normalize_counts()
        peak_param = self.peak_param_input.value()
        peaks = self.mesh.get_peaks(peak_param)

        self.window.visualizer.set_peaks(peaks)
        self.window.visualizer.refresh()
        self.current_cycle += 1
        self.window.statusBar().showMessage(
            f"Cycle {self.current_cycle}/{NUM_CYCLES} | "
            f"peak cells: {len(peaks)} | peak range: {peak_param:.2f} | "
            f"noise std: {self.noise_std_input.value():.3f} | "
            f"max: {self.mesh.max_value:.3f}"
        )


def add_experiment_controls(window):
    """Add live controls used by the experiment on every cycle."""
    toolbar = QtWidgets.QToolBar("Experiment controls", window)
    toolbar.setMovable(False)
    window.addToolBar(QtCore.Qt.TopToolBarArea, toolbar)

    toolbar.addWidget(QtWidgets.QLabel("Peak range: "))
    peak_param_input = QtWidgets.QDoubleSpinBox()
    peak_param_input.setRange(1.0, 10.0)
    peak_param_input.setDecimals(2)
    peak_param_input.setSingleStep(0.05)
    peak_param_input.setValue(INITIAL_PEAK_PARAM)
    peak_param_input.setToolTip(
        "Select cells with value >= max_value / peak range"
    )
    toolbar.addWidget(peak_param_input)

    toolbar.addSeparator()
    toolbar.addWidget(QtWidgets.QLabel("Noise std: "))
    noise_std_input = QtWidgets.QDoubleSpinBox()
    noise_std_input.setRange(0.0, 0.5)
    noise_std_input.setDecimals(3)
    noise_std_input.setSingleStep(0.005)
    noise_std_input.setValue(INITIAL_NOISE_STD)
    noise_std_input.setToolTip(
        "Standard deviation of Gaussian noise around each real point"
    )
    toolbar.addWidget(noise_std_input)

    toolbar.addSeparator()
    choose_points = QtWidgets.QCheckBox("Choose real points")
    choose_points.setChecked(True)
    toolbar.addWidget(choose_points)

    clear_button = QtWidgets.QPushButton("Clear points")
    toolbar.addWidget(clear_button)
    random_button = QtWidgets.QPushButton("Random points")
    toolbar.addWidget(random_button)
    reset_button = QtWidgets.QPushButton("Reset run")
    toolbar.addWidget(reset_button)
    start_button = QtWidgets.QPushButton("Start")
    toolbar.addWidget(start_button)
    raster_button = QtWidgets.QPushButton("Raster expectation")
    raster_button.setToolTip(
        "Weight every grid center by all real points, convolve through add(), "
        "then normalize once"
    )
    toolbar.addWidget(raster_button)

    toolbar.addSeparator()
    legend = QtWidgets.QLabel("Red targets = real points | White circles = peaks")
    toolbar.addWidget(legend)
    return {
        "peak_param": peak_param_input,
        "noise_std": noise_std_input,
        "choose_points": choose_points,
        "clear": clear_button,
        "random": random_button,
        "reset": reset_button,
        "start": start_button,
        "raster": raster_button,
    }


def main():
    random.seed(RANDOM_SEED)
    real_points = [
        (random.uniform(0.15, 0.85), random.uniform(0.15, 0.85))
        for _ in range(NUM_REAL_POINTS)
    ]
    print("Real points:", real_points)

    mesh = MeshND(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        cells=GRID_CELLS,
        radius=MESH_RADIUS,
        low_thresh=LOW_THRESHOLD,
        high_thresh=HIGH_THRESHOLD,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MeshWindow(mesh, true_points=real_points)
    controls = add_experiment_controls(window)
    peak_param_input = controls["peak_param"]
    noise_std_input = controls["noise_std"]
    peak_param_input.valueChanged.connect(
        lambda value: window.visualizer.set_peaks(mesh.get_peaks(value))
    )
    window.add_points_checkbox.hide()
    window.visualizer.set_true_point_selection_enabled(True)
    window.resize(850, 750)
    window.show()

    # Keep a Python reference so the QObject and its timer remain alive.
    controller = ExperimentController(
        mesh,
        real_points,
        window,
        peak_param_input,
        noise_std_input,
    )

    def reset_run():
        controller.reset()
        controls["start"].setText("Start")

    def add_real_point(x, y):
        reset_run()
        real_points.append((x, y))
        window.visualizer.set_true_points(real_points)

    def remove_nearest_real_point(x, y):
        if not real_points:
            return
        reset_run()
        nearest = min(real_points, key=lambda point:
                      (point[0] - x) ** 2 + (point[1] - y) ** 2)
        real_points.remove(nearest)
        window.visualizer.set_true_points(real_points)

    def clear_real_points():
        reset_run()
        real_points.clear()
        window.visualizer.set_true_points(real_points)

    def randomize_real_points():
        reset_run()
        real_points[:] = [
            (random.uniform(0.15, 0.85), random.uniform(0.15, 0.85))
            for _ in range(NUM_REAL_POINTS)
        ]
        window.visualizer.set_true_points(real_points)

    def toggle_running():
        if controller.timer.isActive():
            controller.pause()
            controls["start"].setText("Start")
            return
        if not real_points:
            window.statusBar().showMessage("Choose at least one real point", 2500)
            return
        if controller.current_cycle >= NUM_CYCLES:
            controller.reset()
        controls["choose_points"].setChecked(False)
        controller.start()
        controls["start"].setText("Pause")

    def run_raster_expectation():
        if not real_points:
            window.statusBar().showMessage("Choose at least one real point", 2500)
            return
        controller.pause()
        controls["start"].setText("Start")
        controls["choose_points"].setChecked(False)
        raster_expected_mesh(mesh, real_points, noise_std_input.value())
        peaks = mesh.get_peaks(peak_param_input.value())
        window.visualizer.set_peaks(peaks)
        window.visualizer.refresh()
        window.statusBar().showMessage(
            f"Raster expectation | peak cells: {len(peaks)} | "
            f"noise std: {noise_std_input.value():.3f} | "
            f"max: {mesh.max_value:.3f}"
        )

    controls["choose_points"].toggled.connect(
        window.visualizer.set_true_point_selection_enabled
    )
    controls["clear"].clicked.connect(clear_real_points)
    controls["random"].clicked.connect(randomize_real_points)
    controls["reset"].clicked.connect(reset_run)
    controls["start"].clicked.connect(toggle_running)
    controls["raster"].clicked.connect(run_raster_expectation)
    controller.finished.connect(lambda: controls["start"].setText("Start"))
    window.visualizer.true_point_selected.connect(add_real_point)
    window.visualizer.true_point_remove_requested.connect(
        remove_nearest_real_point
    )

    window.experiment_controller = controller

    app.exec_()


if __name__ == "__main__":
    main()
