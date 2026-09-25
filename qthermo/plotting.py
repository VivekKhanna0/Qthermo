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
    "plot_heat_network",
    "plot_correlations",
    "plot_machine",
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
        dissipative = stroke.dissipative if stroke is not None else False
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


# --- multi-qubit machines ----------------------------------------------------

def _layout(n_sites, bonds):
    """Chain layout when the bonds form a path, polygon otherwise."""
    path = set(tuple(sorted(b)) for b in bonds if len(b) == 2)
    is_chain = n_sites <= 2 or path == {(i, i + 1) for i in range(n_sites - 1)}
    if is_chain:
        return {i: np.array([1.4 * i, 0.0]) for i in range(n_sites)}, True
    angles = np.pi / 2 + 2 * np.pi * np.arange(n_sites) / n_sites
    return {i: 1.2 * np.array([np.cos(a), np.sin(a)]) for i, a in enumerate(angles)}, False


def _arrow(ax, start, end, width, color, label=None, shrink=0.0, curve=0.0):
    from matplotlib.patches import FancyArrowPatch
    start, end = np.asarray(start, float), np.asarray(end, float)
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=8 + 2.5 * width,
        linewidth=width, color=color, shrinkA=shrink, shrinkB=shrink,
        connectionstyle=f"arc3,rad={curve}", zorder=2)
    ax.add_patch(patch)
    if label:
        direction = end - start
        length = np.linalg.norm(direction)
        normal = (np.array([-direction[1], direction[0]]) / length
                  if length > 0 else np.zeros(2))
        if normal[1] < 0 or (normal[1] == 0 and normal[0] < 0):
            normal = -normal
        mid = 0.5 * (start + end) + 0.2 * normal
        ax.annotate(label, mid, fontsize=7.5, ha="center", va="center",
                    color="0.15", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                              alpha=0.85))


def plot_heat_network(flow_map, ax=None, cmap="coolwarm", show_values=True,
                      title=None):
    """Draw a machine as a network: baths, sites, and the heat between them.

    * Squares are baths, coloured by their temperature.
    * Circles are sites, coloured by their **virtual temperature** on the
      same scale -- a qubit bluer than the bath it touches has been cooled
      below that bath's temperature.
    * Arrows point along the energy flow; width is proportional to the
      current. Bath arrows show heat entering each site; bond arrows the
      antisymmetrised transport current through each coupling.
    * Grey links show mutual information between sites (thicker = more).

    Takes a :class:`qthermo.network.HeatFlowMap`.
    """
    plt = _require_matplotlib()
    import matplotlib as mpl

    fm = flow_map
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    n = fm.n_sites
    bonds = sorted({key for (key, _) in fm.term_to_site})
    positions, chain = _layout(n, bonds or [(i, i + 1) for i in range(n - 1)])

    temps = [T for T in fm.bath_temperature.values() if T is not None]
    temps += [T for T in fm.virtual_temperature if np.isfinite(T) and T > 0]
    norm = mpl.colors.Normalize(vmin=min(temps), vmax=max(temps))
    colormap = mpl.colormaps[cmap]

    # bath positions: outward from their site
    centre = np.mean(list(positions.values()), axis=0)
    bath_pos = {}
    for k, (bath, sites) in enumerate(fm.bath_sites.items()):
        site_list = list(sites) or list(range(n))
        anchor = np.mean([positions[s] for s in site_list], axis=0)
        if chain:
            if site_list == [0] and n > 1:
                offset = np.array([-1.6, 0.0])
            elif site_list == [n - 1] and n > 1:
                offset = np.array([1.6, 0.0])
            else:
                offset = np.array([0.0, 1.5 if k % 2 == 0 else -1.5])
        else:
            direction = anchor - centre
            norm_d = np.linalg.norm(direction)
            direction = direction / norm_d if norm_d > 1e-9 else np.array([0, 1.0])
            offset = 1.5 * direction
        bath_pos[bath] = anchor + offset

    # Bath arrows carry the heat delivered to the sites, so each site's
    # arrows balance. Under a global ME that is identically zero (the heat
    # lands in the interaction energy), so show the total bath current.
    def bath_value(bath):
        if fm.extras.get("secular_blind"):
            return fm.bath_current(bath)
        return sum(J for (b, _), J in fm.bath_to_site.items() if b == bath)

    # current scale
    flows = [abs(bath_value(b)) for b in fm.bath_sites]
    flows += [abs(fm.bond_current(b)) for b in bonds if len(b) == 2]
    scale = max(flows) if flows and max(flows) > 0 else 1.0

    # mutual information links
    mi_max = np.max(fm.mutual_information) if n > 1 else 0.0
    if mi_max > 0:
        for a in range(n):
            for b in range(a + 1, n):
                w = fm.mutual_information[a, b] / mi_max
                if w > 0.05:
                    p, q = positions[a], positions[b]
                    ax.plot([p[0], q[0]], [p[1], q[1]], color="0.55",
                            linewidth=0.8 + 5 * w, alpha=0.35,
                            solid_capstyle="round", zorder=1)

    # bath -> site arrows
    for bath, sites in fm.bath_sites.items():
        J = bath_value(bath)
        site_list = list(sites) or list(range(n))
        target = np.mean([positions[s] for s in site_list], axis=0)
        width = 0.8 + 5.0 * abs(J) / scale
        start, end = (bath_pos[bath], target) if J >= 0 else (target, bath_pos[bath])
        _arrow(ax, start, end, width, "#B03A2E" if J >= 0 else "#2E86C1",
               f"{J:+.2e}" if show_values else None, shrink=24)

    # bond arrows (two-site terms); many-body terms get a hub
    for key in bonds:
        if len(key) == 2:
            J = fm.bond_current(key)
            if abs(J) < 1e-14 * scale:
                continue
            i, j = key if J >= 0 else key[::-1]
            width = 0.8 + 5.0 * abs(J) / scale
            _arrow(ax, positions[i], positions[j], width, "0.25",
                   f"{abs(J):.2e}" if show_values else None, shrink=20,
                   curve=0.0 if chain else 0.15)
        else:
            hub = np.mean([positions[s] for s in key], axis=0)
            ax.plot(*hub, marker="h", markersize=16, color="0.85",
                    markeredgecolor="0.3", zorder=3)
            for s in key:
                J = fm.term_to_site[(key, s)]
                if abs(J) < 1e-14 * scale:
                    continue
                width = 0.8 + 5.0 * abs(J) / scale
                start, end = (hub, positions[s]) if J >= 0 else (positions[s], hub)
                _arrow(ax, start, end, width, "0.25", None, shrink=14)

    # nodes
    for bath, pos in bath_pos.items():
        T = fm.bath_temperature[bath]
        colour = colormap(norm(T)) if T is not None else "0.8"
        ax.add_patch(mpl.patches.FancyBboxPatch(
            pos - 0.24, 0.48, 0.48, boxstyle="round,pad=0.02", fc=colour,
            ec="0.2", lw=1.2, zorder=4))
        T_text = "" if T is None else f"\nT={T:.3g}"
        ax.text(*pos, f"{bath}{T_text}", ha="center", va="center", fontsize=8,
                zorder=5)
    for i, pos in positions.items():
        T = fm.virtual_temperature[i]
        if np.isfinite(T) and T > 0:
            colour, T_text = colormap(norm(T)), f"T*={T:.3g}"
        else:
            colour, T_text = "white", ("T*<0" if T < 0 else "T*=inf")
        ax.add_patch(mpl.patches.Circle(pos, 0.27, fc=colour, ec="0.1", lw=1.5,
                                        zorder=4))
        ax.text(pos[0], pos[1] + 0.05, fm.site_names[i], ha="center",
                va="center", fontsize=9, weight="bold", zorder=5)
        ax.text(pos[0], pos[1] - 0.09, T_text, ha="center", va="center",
                fontsize=7, zorder=5)
        bath_temps = [t for t in fm.bath_temperature.values() if t is not None]
        if bath_temps and np.isfinite(T) and 0 < T < min(bath_temps) - 1e-9:
            # Colder than every reservoir: impossible without refrigeration.
            ax.text(pos[0], pos[1] - 0.42, "colder than every bath",
                    ha="center", fontsize=7.5, color="#1F618D", style="italic",
                    zorder=5)

    everything = np.array(list(positions.values()) + list(bath_pos.values()))
    lo, hi = everything.min(axis=0) - 0.6, everything.max(axis=0) + 0.6
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.axis("off")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(sm, ax=ax, shrink=0.7, label="T (baths),  T* (sites)")
    ax.set_title(title or "Heat flow", fontsize=10)
    if fm.extras.get("secular_blind"):
        ax.text(0.5, 0.01, "global ME: bond currents vanish identically "
                "(steady state diagonal in H)", transform=ax.transAxes,
                ha="center", fontsize=7, color="0.35")
    return ax


def plot_correlations(flow_map, ax=None, quantity="mutual_information"):
    """Matrix of pairwise correlations between sites.

    ``quantity`` is ``"mutual_information"``, ``"negativity"`` or
    ``"concurrence"``. Mutual information counts all correlations; negativity
    and concurrence only entanglement.
    """
    plt = _require_matplotlib()
    fm = flow_map
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    data = np.array(getattr(fm, quantity), dtype=float)
    shown = np.where(np.eye(fm.n_sites, dtype=bool), np.nan, data)
    image = ax.imshow(shown, cmap="Purples", vmin=0)
    plt.colorbar(image, ax=ax, shrink=0.8, label=quantity.replace("_", " "))
    ax.set_xticks(range(fm.n_sites), fm.site_names)
    ax.set_yticks(range(fm.n_sites), fm.site_names)
    for a in range(fm.n_sites):
        for b in range(fm.n_sites):
            if a != b and np.isfinite(data[a, b]):
                dark = data[a, b] > 0.6 * np.nanmax(shown)
                ax.text(b, a, f"{data[a, b]:.1e}", ha="center", va="center",
                        fontsize=7, color="white" if dark else "0.1")
    ax.set_title(quantity.replace("_", " "))
    return ax


def plot_machine(steady, figsize=(11, 8)):
    """One-figure summary of a continuous machine.

    Heat-flow network, per-bath currents with the entropy-production rate, and
    the pairwise mutual-information matrix. Takes a ``SteadyState`` returned
    by ``Model.analyze()``.
    """
    plt = _require_matplotlib()
    from .network import heat_flow_map

    fm = heat_flow_map(steady)
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
    plot_heat_network(fm, ax=fig.add_subplot(grid[0, :]),
                      title=getattr(steady.model, "description", None))

    ax = fig.add_subplot(grid[1, 0])
    names = list(steady.currents)
    values = [steady.currents[k] for k in names]
    ax.barh(names, values, color=["#B03A2E" if v >= 0 else "#2E86C1" for v in values])
    ax.axvline(0, color="0.3", linewidth=0.8)
    ax.set_xlabel("heat current into system")
    ax.set_title(f"sigma_dot = {steady.entropy_production_rate:.2e}", fontsize=10)
    ax.invert_yaxis()
    ax.grid(alpha=0.25, axis="x")

    if fm.n_sites > 1:
        plot_correlations(fm, ax=fig.add_subplot(grid[1, 1]))
    fig.tight_layout()
    return fig
