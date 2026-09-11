"""Plotting.

Matplotlib is an optional dependency: importing qthermo does not require it,
and these functions raise a clear error if it is missing rather than failing at
import time.

Every function returns the matplotlib ``Axes`` it drew on, so plots can be
composed into larger figures or restyled by the caller.
"""

from __future__ import annotations

import numpy as np

from .core import _as_matrix

__all__ = [
    "plot_cycle",
    "plot_sweep",
    "plot_scan",
    "plot_distribution",
    "plot_dashboard",
]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plotting requires matplotlib. Install it with "
            "`pip install matplotlib`, or use the numerical results directly."
        ) from exc
    return plt


def _gap_and_population(states, H_of_t, times):
    """Energy gap and excited-state population along a stroke.

    The gap is the spread of the Hamiltonian spectrum and the population is
    the weight on its highest eigenstate, so this works for any dimension
    rather than assuming a qubit.
    """
    gaps, populations = [], []
    for state, time in zip(states, times):
        H = _as_matrix(H_of_t(time))
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        gaps.append(eigenvalues[-1] - eigenvalues[0])
        excited = eigenvectors[:, -1]
        populations.append(float(np.real(excited.conj() @ state @ excited)))
    return np.array(gaps), np.array(populations)


def plot_cycle(result, cycle=None, ax=None, annotate: bool = True):
    """Energy gap against excited-state population -- the quantum P-V diagram.

    Isochores appear as vertical lines (the gap is fixed while population
    relaxes toward the bath) and driven strokes as horizontal ones (population
    is frozen while the gap is swept). The enclosed area is the cycle's work.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))

    strokes = cycle.strokes if cycle is not None else [None] * len(result.strokes)

    for stroke_result, stroke in zip(result.strokes, strokes):
        if stroke is not None:
            H = stroke.H
            H_of_t = H if callable(H) else (lambda t, _H=H: _H)
        else:
            H_of_t = lambda t: np.diag([0.0, 1.0])

        gaps, populations = _gap_and_population(
            stroke_result.states, H_of_t, stroke_result.times
        )
        dissipative = bool(stroke.c_ops) if stroke is not None else False
        ax.plot(
            populations, gaps,
            linewidth=2.4,
            label=stroke_result.name,
            linestyle="-" if dissipative else "--",
        )
        if annotate:
            ax.annotate(
                "", xy=(populations[-1], gaps[-1]),
                xytext=(populations[len(populations) // 2],
                        gaps[len(gaps) // 2]),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="0.35"),
            )

    ax.set_xlabel("excited-state population")
    ax.set_ylabel("energy gap")
    ax.set_title("Cycle diagram")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    return ax


def plot_sweep(sweep_result, ax=None, mark_optimum: bool = True):
    """Metric against a swept parameter, with the optimum marked."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.plot(sweep_result.values, sweep_result.metrics, "o-", linewidth=2,
            markersize=4)

    if mark_optimum:
        best_value, best_metric = sweep_result.optimum()
        ax.axvline(best_value, color="0.5", linestyle=":", linewidth=1.2)
        ax.plot([best_value], [best_metric], "*", markersize=15,
                color="crimson", zorder=5)
        ax.annotate(f"  optimum\n  {sweep_result.parameter} = {best_value:.3f}",
                    xy=(best_value, best_metric), fontsize=8,
                    va="top")

    ax.set_xlabel(sweep_result.parameter)
    ax.set_ylabel(sweep_result.metric_name)
    ax.set_title(f"{sweep_result.metric_name} against {sweep_result.parameter}")
    ax.grid(alpha=0.25)
    return ax


def plot_scan(scan_result, ax=None, log_x: bool = False, log_y: bool = False):
    """Heatmap of a two-parameter scan, with joint and sequential optima marked."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5))

    mesh = ax.pcolormesh(
        scan_result.x_values, scan_result.y_values, scan_result.grid,
        shading="nearest", cmap="viridis",
    )
    plt.colorbar(mesh, ax=ax, label=scan_result.metric_name)

    joint_x, joint_y, _ = scan_result.optimum()
    sequential_x, sequential_y, _ = scan_result.sequential_optimum()

    ax.plot([joint_x], [joint_y], "*", markersize=18, color="white",
            markeredgecolor="black", label="joint optimum", zorder=5)
    ax.plot([sequential_x], [sequential_y], "o", markersize=9, color="crimson",
            markeredgecolor="white", label="sequential optimum", zorder=5)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(scan_result.parameters[0])
    ax.set_ylabel(scan_result.parameters[1])
    ax.set_title("Joint parameter scan")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
    return ax


def plot_distribution(ensemble, which: str, deterministic: float | None = None,
                      ax=None, bins: int = 30):
    """Trajectory histogram with the master-equation mean marked.

    The point of the figure: the mean is one number, and the distribution
    around it is a different object entirely.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    data = ensemble.series(which)
    ax.hist(data, bins=bins, color="#4C72B0", alpha=0.8, edgecolor="white",
            linewidth=0.5)

    if deterministic is not None:
        ax.axvline(deterministic, color="crimson", linewidth=2.2,
                   label=f"master equation ({deterministic:+.4f})")
    ax.axvline(float(np.mean(data)), color="black", linestyle="--",
               linewidth=1.6, label=f"trajectory mean ({np.mean(data):+.4f})")
    ax.axvline(0.0, color="0.4", linewidth=1.0, linestyle=":")

    ax.set_xlabel(f"{which} per cycle (single trajectory)")
    ax.set_ylabel("trajectories")
    ax.set_title("Distribution over quantum-jump trajectories")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.2, axis="y")
    return ax


def plot_dashboard(result, cycle, sweep_result=None, ensemble=None,
                   deterministic=None, figsize=(12, 4.2)):
    """One figure summarising a machine: cycle diagram, sweep, distribution."""
    plt = _require_matplotlib()

    panels = 1 + (sweep_result is not None) + (ensemble is not None)
    fig, axes = plt.subplots(1, panels, figsize=figsize)
    if panels == 1:
        axes = [axes]

    index = 0
    plot_cycle(result, cycle, ax=axes[index])
    index += 1
    if sweep_result is not None:
        plot_sweep(sweep_result, ax=axes[index])
        index += 1
    if ensemble is not None:
        plot_distribution(ensemble, "cold_iso", deterministic, ax=axes[index])

    fig.tight_layout()
    return fig
