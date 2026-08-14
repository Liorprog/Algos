"""
algos/mesh.py

MeshND: N-dimensional mesh/grid implemented in pure Python (no numpy).

Added methods:
- normalize_counts(...) : zero small values, divide or cap large values
- get_peaks(...) : non-maximum suppression local maxima extraction

Pure-Python, easy to step through for debugging on a phone.
"""
from typing import Sequence, Tuple, List, Optional, Dict, Iterable
import math
import itertools


class MeshND:
    def __init__(self, bounds: Sequence[Tuple[float, float]],
                       cells: Sequence[int],
                       radius: float = 0.15,
                       low_thresh: float = 0.01,
                       high_thresh: float = 10):
        if len(bounds) != len(cells):
            raise ValueError("bounds and cells must have the same length (ndim).")
        self.ndim = len(bounds)
        # Validate and store bounds and cells
        self.bounds: List[Tuple[float, float]] = [(float(a), float(b)) for (a, b) in bounds]
        self.cells: List[int] = [int(c) for c in cells]
        self.radius = radius
        self.sigma = self.radius / 3.0
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.max_value = 0.0
        for (mn, mx) in self.bounds:
            if mx <= mn:
                raise ValueError("Each bound must have max > min.")
        for c in self.cells:
            if c <= 0:
                raise ValueError("Each cell count must be positive integer.")

        # Compute cell sizes and centers per axis
        self.cell_sizes: List[float] = []
        self.centers: List[List[float]] = []
        for d in range(self.ndim):
            mn, mx = self.bounds[d]
            n = self.cells[d]
            size = (mx - mn) / n
            self.cell_sizes.append(size)
            half = 0.5 * size
            centers_d = [mn + half + i * size for i in range(n)]
            self.centers.append(centers_d)

        # Create nested counts structure
        self.counts: Dict[Tuple[int, ...], float] = {}

    def _get_at(self, idx: Tuple[int, ...]) -> float:
        """Return the value at idx, or zero if the cell is empty."""
        return self.counts.get(idx, 0.0)

    def _set_at(self, idx: Tuple[int, ...], value: float) -> None:
        """Set a cell value. Remove it if the value is zero."""
        value = float(value)
        self.counts[idx] = value
        if value > self.max_value:
            self.max_value = value

    def _add_at(self, idx: Tuple[int, ...], delta: float) -> None:
        """Add delta to a cell. Do not keep zero-valued cells."""
        new_value = self.counts.get(idx, 0.0) + delta
        self._set_at(idx, new_value)

    def _all_indices(self) -> Iterable[Tuple[int, ...]]:
        """Yield all valid index tuples in the grid (product of ranges)."""
        ranges = [range(n) for n in self.cells]
        return itertools.product(*ranges)

    # -----------------------
    # Public API
    # -----------------------
    def reset(self) -> None:
        self.counts.clear()
        self.max_value = 0.0

    def get_centers(self) -> List[List[float]]:
        """Return per-axis centers lists."""
        return self.centers

    def _validate_point(self, point: Sequence[float]) -> List[float]:
        p = [float(x) for x in point]
        if len(p) != self.ndim:
            raise ValueError(f"Point must have dimension {self.ndim}.")
        for d in range(self.ndim):
            mn, mx = self.bounds[d]
            if p[d] < mn or p[d] > mx:
                raise ValueError(f"Point {p} outside bounds: axis {d} in [{mn}, {mx}].")
        return p

    def _nearest_index(self, p: Sequence[float]) -> Tuple[int, ...]:
        """Return nearest grid index tuple to point p by scanning centers per-axis."""
        idxs = []
        for d in range(self.ndim):
            centers_d = self.centers[d]
            best = 0
            best_dist = abs(centers_d[0] - p[d])
            for i in range(1, len(centers_d)):
                dist = abs(centers_d[i] - p[d])
                if dist < best_dist:
                    best_dist = dist
                    best = i
            idxs.append(best)
        return tuple(idxs)

    def _axis_index_range_within_radius(self, axis: int, coord: float) -> Tuple[int, int]:
        """
        Return inclusive range [min_i, max_i] of indices on axis whose centers are within radius.
        If none found return (nearest, nearest).
        """
        centers = self.centers[axis]
        min_i = None
        max_i = None
        for i, c in enumerate(centers):
            if abs(c - coord) <= self.radius:
                if min_i is None:
                    min_i = i
                max_i = i
        if min_i is None:
            return None, None
        return min_i, max_i

    def add(
        self,
        point: Sequence[float],
        amount: float = 1.0,
    ) -> None:
        """
        Add values distributed to the nearest cell and neighbors within `self.radius`.

        Args:
            point: coordinate sequence of length ndim.
            amount: total input weight before applying the mesh kernel.
        """
        p = self._validate_point(point)
        amount = float(amount)

        # Determine index ranges on each axis
        ranges = []
        for d in range(self.ndim):
            mn_i, mx_i = self._axis_index_range_within_radius(d, p[d])
            ranges.append(range(mn_i, mx_i + 1))

        for idx in itertools.product(*ranges):
            # compute Euclidean distance from point p to cell center at idx
            dist2 = 0.0
            for d, idd in enumerate(idx):
                c = self.centers[d][idd]
                diff = c - p[d]
                dist2 += diff * diff
            dist = math.sqrt(dist2)
            if dist > self.radius:
                continue  # outside influence

            val = amount * math.exp(-0.5 * (dist / self.sigma) ** 2)
            self._add_at(idx, val)

    def get(self, *idx):
        return self._get_at(idx)

    # -----------------------
    # Post-processing helpers
    # -----------------------
    def normalize_counts(self) -> None:
        """Apply low-value filtering and optional whole-mesh scaling.

        When the largest count exceeds ``high_threshold``, every retained
        count is divided by ``divide_const``. This preserves the relative
        shape of the mesh instead of changing only its largest cells.
        """
        if self.max_value < self.high_thresh:
            return

        divide_const = self.max_value / self.high_thresh

        old_counts = list(self.counts.items())
        self.reset()

        for idx, value in old_counts:
            value /= divide_const
            if value < self.low_thresh:
                continue
            self._set_at(idx, value)

    def get_peaks(
            self,
            param: float = 2.0,
    ) -> List[Tuple[Tuple[int, ...], float]]:
        """Return cells valued from ``max_value / param`` through ``max_value``."""
        if param <= 0:
            raise ValueError("param must be greater than zero.")
        if not self.counts:
            return []

        max_value = self.max_value
        min_value = max_value / param
        peaks = [
            (idx, value)
            for idx, value in self.counts.items()
            if value >= min_value
        ]

        return sorted(peaks, key=lambda peak: peak[1], reverse=True)

    # Utility repr
    def __repr__(self):
        return f"MeshND(ndim={self.ndim}, bounds={self.bounds}, cells={self.cells})"
