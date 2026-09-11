"""qthermo -- thermodynamic analysis for open quantum systems.

Mitiq took error mitigation techniques that every group was reimplementing and
turned them into a pluggable, one-call framework. qthermo does the same for
thermodynamic cycle analysis: define a cycle as a sequence of strokes, and get
heat, work, entropy production, COP and figure of merit -- deterministically
from the master equation, or trajectory-resolved with full distributions.

Units: hbar = k_B = 1.
"""

from .channels import (
    amplitude_damping,
    bit_flip,
    mean_occupation,
    pure_dephasing,
    qubit_hamiltonian,
    sigma_minus,
    sigma_plus,
    sigma_x,
    sigma_y,
    sigma_z,
    thermal_bath,
)
from .core import (
    entropy_production,
    free_energy,
    heat_work_increments,
    internal_energy,
    thermal_state,
    von_neumann_entropy,
)
from .analysis import (
    ScanResult,
    SweepResult,
    compare_channels,
    scan_2d,
    sweep,
)
from .cycle import Cycle, CycleResult, Stroke, StrokeResult
from .solver import evolve, lindbladian
from .validation import QThermoError
from .stochastic import TrajectoryEnsemble, jarzynski_tpm, unravel

__version__ = "0.2.0"


def __getattr__(name):
    """Lazily expose plotting so matplotlib stays an optional dependency."""
    if name in ("plot_cycle", "plot_sweep", "plot_scan",
                "plot_distribution", "plot_dashboard"):
        from . import plotting
        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Cycle", "CycleResult", "Stroke", "StrokeResult",
    "evolve", "lindbladian",
    "internal_energy", "von_neumann_entropy", "entropy_production",
    "thermal_state", "free_energy", "heat_work_increments",
    "qubit_hamiltonian", "thermal_bath", "amplitude_damping",
    "pure_dephasing", "bit_flip", "mean_occupation",
    "sigma_x", "sigma_y", "sigma_z", "sigma_plus", "sigma_minus",
    "unravel", "jarzynski_tpm", "TrajectoryEnsemble",
    "QThermoError",
    "sweep", "scan_2d", "compare_channels", "SweepResult", "ScanResult",
    "__version__",
]
