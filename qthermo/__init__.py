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
    ergotropy,
    passive_state,
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
from .baths import (
    Bath,
    bohr_decomposition,
    davies_bath,
    dissipator,
    flat_spectrum,
    local_bath,
    ohmic_spectrum,
)
from .steady import (
    MasterEquationComparison,
    SteadyState,
    analyze,
    compare_master_equations,
    liouvillian,
    relaxation_time,
    steady_state,
    trace_distance,
)
from .cycle import Cycle, CycleResult, Stroke, StrokeResult
from .solver import evolve, lindbladian
from .validation import QThermoError
from .stochastic import TrajectoryEnsemble, jarzynski_tpm, unravel
from .subsystems import (
    SiteResult,
    SubsystemResult,
    embed,
    mutual_information,
    partial_trace,
    resolve_cycle,
    resolve_stroke,
    total_correlation,
)

from .network import (
    HeatFlowMap,
    concurrence,
    correlation_matrices,
    heat_flow_map,
    negativity,
    site_dynamics,
    virtual_temperature,
)
from .fluctuations import (
    CurrentStatistics,
    current_statistics,
    jump_energy,
    scaled_cgf,
)
from .strong_coupling import (
    mean_force_state,
    rc_convergence,
    reaction_coordinate_model,
    ultrastrong_limit_state,
)
from .information import (
    ErasureResult,
    geodesic_schedule,
    landauer_bound,
    landauer_erasure,
    predicted_excess,
    thermodynamic_length,
)
from .baths import instantaneous_bath
from .engines import OttoLimit, adiabatic_pairing, ideal_otto, otto_cycle
from .audit import AuditReport, Finding, audit
from .counting import CycleCounting, cycle_counting
from .modes import MODES, classify, mode_map
from . import batteries  # noqa: E402  (quantum batteries: qthermo.batteries.*)
from .batteries import ergotropy_split, locked_ergotropy
from .geometry import OptimalSchedule, excess_work, friction, optimal_schedule
from .export import export, load, save
from . import models  # noqa: E402  (canonical machines: qthermo.models.*)

__version__ = "0.4.0"


def __getattr__(name):
    """Lazily expose plotting so matplotlib stays an optional dependency."""
    if name in ("plot_cycle", "plot_sweep", "plot_scan",
                "plot_distribution", "plot_dashboard",
                "plot_heat_network", "plot_correlations", "plot_machine",
                "plot_mode_map", "plot_site_dynamics", "plot_bloch_paths"):
        from . import plotting
        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Cycle", "CycleResult", "Stroke", "StrokeResult",
    "evolve", "lindbladian",
    "internal_energy", "von_neumann_entropy", "entropy_production",
    "thermal_state", "free_energy", "heat_work_increments",
    "ergotropy", "passive_state",
    "qubit_hamiltonian", "thermal_bath", "amplitude_damping",
    "pure_dephasing", "bit_flip", "mean_occupation",
    "sigma_x", "sigma_y", "sigma_z", "sigma_plus", "sigma_minus",
    "unravel", "jarzynski_tpm", "TrajectoryEnsemble",
    "QThermoError",
    "sweep", "scan_2d", "compare_channels", "SweepResult", "ScanResult",
    "partial_trace", "embed", "total_correlation", "mutual_information",
    "resolve_stroke", "resolve_cycle", "SiteResult", "SubsystemResult",
    "Bath", "davies_bath", "local_bath", "bohr_decomposition", "dissipator",
    "flat_spectrum", "ohmic_spectrum",
    "analyze", "steady_state", "liouvillian", "SteadyState", "relaxation_time",
    "compare_master_equations", "MasterEquationComparison", "trace_distance",
    "models",
    "heat_flow_map", "HeatFlowMap", "virtual_temperature", "concurrence",
    "negativity", "correlation_matrices", "site_dynamics",
    "current_statistics", "CurrentStatistics", "scaled_cgf", "jump_energy",
    "reaction_coordinate_model", "mean_force_state", "ultrastrong_limit_state",
    "rc_convergence",
    "landauer_erasure", "landauer_bound", "ErasureResult", "geodesic_schedule",
    "thermodynamic_length", "predicted_excess", "instantaneous_bath",
    "ideal_otto", "otto_cycle", "OttoLimit", "adiabatic_pairing",
    "audit", "AuditReport", "Finding",
    "cycle_counting", "CycleCounting",
    "classify", "mode_map", "MODES",
    "batteries", "ergotropy_split", "locked_ergotropy",
    "friction", "optimal_schedule", "OptimalSchedule", "excess_work",
    "export", "save", "load",
    "__version__",
]
