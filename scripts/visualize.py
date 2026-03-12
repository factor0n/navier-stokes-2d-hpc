#!/usr/bin/env python3
"""
visualize.py - Post-processing and visualization for NS2D solver

Produces:
  1. Velocity magnitude + streamlines plot
  2. Vorticity contour plot
  3. Validation against Ghia et al. (1982) benchmark data
  4. Convergence history

Usage:
  python3 scripts/visualize.py results/
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import glob

# =============================================================================
#  Ghia et al. (1982) Benchmark Data for Lid-Driven Cavity
# =============================================================================

# u-velocity along vertical centerline (x=0.5) at Re=1000
GHIA_RE1000_Y = np.array([
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
    0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
    0.9531, 0.9609, 0.9688, 0.9766, 1.0000
])

GHIA_RE1000_U = np.array([
    0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289,
    -0.27805, -0.10648, -0.06080,  0.05702,  0.18719,  0.33304,
     0.46604,  0.51117,  0.57492,  0.65928,  1.00000
])

# v-velocity along horizontal centerline (y=0.5) at Re=1000
GHIA_RE1000_X = np.array([
    0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563,
    0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063,
    0.9453, 0.9531, 0.9609, 0.9688, 1.0000
])

GHIA_RE1000_V = np.array([
    0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
    0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
    -0.10313, -0.08864, -0.07391, -0.05906, 0.00000
])


def read_vtk(filename):
    """Read VTK structured points file."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse header
    nx, ny = 0, 0
    ox, oy = 0.0, 0.0
    dx, dy = 0.0, 0.0
    n_points = 0

    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith('DIMENSIONS'):
            parts = line.split()
            nx, ny = int(parts[1]), int(parts[2])
        elif line.startswith('ORIGIN'):
            parts = line.split()
            ox, oy = float(parts[1]), float(parts[2])
        elif line.startswith('SPACING'):
            parts = line.split()
            dx, dy = float(parts[1]), float(parts[2])
        elif line.startswith('POINT_DATA'):
            n_points = int(line.split()[1])
            idx += 1
            break
        idx += 1

    data = {}

    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith('VECTORS'):
            name = line.split()[1]
            vals = []
            for k in range(n_points):
                idx += 1
                parts = lines[idx].strip().split()
                vals.append([float(parts[0]), float(parts[1])])
            arr = np.array(vals)
            data[name + '_x'] = arr[:, 0].reshape(ny, nx)
            data[name + '_y'] = arr[:, 1].reshape(ny, nx)
        elif line.startswith('SCALARS'):
            name = line.split()[1]
            idx += 1  # skip LOOKUP_TABLE
            vals = []
            for k in range(n_points):
                idx += 1
                vals.append(float(lines[idx].strip()))
            data[name] = np.array(vals).reshape(ny, nx)
        idx += 1

    x = np.arange(nx) * dx + ox
    y = np.arange(ny) * dy + oy

    return x, y, data


def plot_flow_field(x, y, data, output_dir):
    """Plot velocity magnitude with streamlines."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    X, Y = np.meshgrid(x, y)
    u = data['velocity_x']
    v = data['velocity_y']
    mag = data['velocity_magnitude']

    # Velocity magnitude
    ax = axes[0]
    im = ax.contourf(X, Y, mag, levels=30, cmap='viridis')
    ax.streamplot(X, Y, u, v, color='white', linewidth=0.5,
                  density=2, arrowsize=0.8)
    plt.colorbar(im, ax=ax, label='|V|')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Velocity Magnitude & Streamlines')
    ax.set_aspect('equal')

    # Vorticity
    ax = axes[1]
    vort = data.get('vorticity', np.zeros_like(mag))
    vmax = np.percentile(np.abs(vort), 98)
    im = ax.contourf(X, Y, vort, levels=np.linspace(-vmax, vmax, 40),
                     cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='Vorticity')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Vorticity Field')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'flow_field.png'), dpi=200,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved flow_field.png")


def plot_pressure(x, y, data, output_dir):
    """Plot pressure field."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    X, Y = np.meshgrid(x, y)
    p = data['pressure']

    im = ax.contourf(X, Y, p, levels=30, cmap='coolwarm')
    plt.colorbar(im, ax=ax, label='Pressure')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Pressure Field')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pressure.png'), dpi=200,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved pressure.png")


def plot_validation(csv_file, output_dir):
    """Compare centerline profiles against Ghia et al. (1982)."""
    data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
    y_sim = data[:, 0]
    u_sim = data[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # u vs y at x=0.5
    ax = axes[0]
    ax.plot(u_sim, y_sim, 'b-', linewidth=1.5, label='This solver')
    ax.plot(GHIA_RE1000_U, GHIA_RE1000_Y, 'ro', markersize=6,
            label='Ghia et al. (1982)')
    ax.set_xlabel('u-velocity')
    ax.set_ylabel('y')
    ax.set_title('u-velocity along vertical centerline (x=0.5)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Info text
    ax = axes[1]
    ax.text(0.5, 0.5,
            'Lid-Driven Cavity\nRe = 1000\n\n'
            'Validation against\nGhia, Ghia & Shin (1982)\n'
            'J. Comp. Phys. 48, 387-411\n\n'
            f'Grid: {len(y_sim)}×{len(y_sim)}\n'
            'Method: Fractional Step\n'
            'Parallel: MPI + OpenMP',
            ha='center', va='center', fontsize=12,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_ghia.png'), dpi=200,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved validation_ghia.png")


def main():
    if len(sys.argv) < 2:
        result_dir = 'results'
    else:
        result_dir = sys.argv[1]

    print(f"Post-processing results from: {result_dir}")

    # Find the last VTK file
    vtk_files = sorted(glob.glob(os.path.join(result_dir, 'cavity_*.vtk')))
    csv_files = sorted(glob.glob(os.path.join(result_dir, 'centerline_*.csv')))

    if not vtk_files:
        print("No VTK files found!")
        return

    print(f"  Found {len(vtk_files)} VTK files, using: {vtk_files[-1]}")

    # Read and plot
    x, y, data = read_vtk(vtk_files[-1])
    plot_flow_field(x, y, data, result_dir)
    plot_pressure(x, y, data, result_dir)

    if csv_files:
        print(f"  Using centerline: {csv_files[-1]}")
        plot_validation(csv_files[-1], result_dir)

    print("\nDone! Visualizations saved in", result_dir)


if __name__ == '__main__':
    main()
