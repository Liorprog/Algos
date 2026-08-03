"""
examples/example_gui.py

Build the two-vector example, post-process, and open a Qt GUI visualizer.
Run:
    pip install PyQt5
    python3 examples/example_gui.py
"""
from mesh import MeshND
from mesh_vis import run_gui

def build_example_mesh():
    mesh = MeshND(bounds=[(0,1),(0,1)], cells=[25, 25])
    p1 = [0.48, 0.52]
    p2 = [0.51, 0.49]

    # accumulate without per-add normalization so weights are raw gaussian
    mesh.add(p1)
    mesh.add(p2)

    # post-process: zero tiny noise (<0.12), damp very large spikes (>1.0) by dividing by 2
    mesh.normalize_counts(low_threshold=0.12, zero_small=True,
                          high_threshold=1.0, divide_const=2.0)

    #peaks = mesh.get_peaks(min_value=0.5, neighborhood=1)
    return mesh

def main():
    mesh = build_example_mesh()
    #print("Peaks (index, value):", peaks)
    run_gui(mesh, show_values=False)

if __name__ == '__main__':
    main()
