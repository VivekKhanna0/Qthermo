"""Canonical quantum thermal machines, built in one call.

Every model returns a :class:`Model`: the full Hamiltonian, the bare
(non-interacting) Hamiltonian, the per-site local Hamiltonians, and a list of
named baths, built either as *local* or *global* (Davies) master equations.
That is everything the rest of the package needs, so a model plugs straight
into :func:`qthermo.analyze`, the per-site resolution, the heat-flow plots and
the fluctuation tools.

Models are ordinary data. To explore a variant -- a different coupling, an
extra bath -- build one, then modify ``H`` or ``baths`` directly, or copy the
twenty lines of the builder. Nothing here is special-cased elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .baths import Bath, davies_bath
from .channels import qubit_hamiltonian, sigma_minus, sigma_plus, sigma_x, sigma_y, sigma_z
from .steady import SteadyState, analyze, compare_master_equations
from .subsystems import embed
from .validation import QThermoError

__all__ = [
    "Model",
    "absorption_refrigerator",
    "three_level_maser",
    "spin_chain",
    "two_qubit_heat_valve",
    "thermal_transistor",
    "build_model",
]

_MASTER_EQUATIONS = ("local", "global")


@dataclass
class Model:
    """A thermal machine: Hamiltonian, reference energies, baths, structure.

    Attributes
    ----------
    H : ndarray
        Full Hamiltonian (in the rotating frame for driven models).
    H0 : ndarray
        Bare Hamiltonian: the sum of local terms, no interactions or drive.
        The consistent energy operator for heat currents under local baths.
    dims : list of int
        Subsystem dimensions.
    local_H : list of ndarray
        Un-embedded Hamiltonian of each site (for per-site resolution).
    baths : list of Bath
    master_equation : str
        ``"local"`` or ``"global"``.
    roles : dict
        Machine-specific labels, e.g. ``{"cold": "cold", "source": "hot",
        "sink": "room"}`` for a refrigerator.
    site_names : list of str
    bonds : list of (int, int)
        Pairs of sites coupled by the interaction, for plotting.
    interaction_terms : dict
        ``(i, j) -> V_ij`` embedded operators, used for bond currents.
    energy : ndarray or None
        Energy operator the model's thermodynamics should be computed with,
        if it differs from ``H`` (rotating-frame models, local baths).
    rebuild : callable
        ``rebuild(master_equation)`` returns the same model under the other
        master equation. Used by :meth:`compare`.
    """

    H: np.ndarray
    H0: np.ndarray
    dims: list
    local_H: list
    baths: list
    master_equation: str
    roles: dict = field(default_factory=dict)
    site_names: list = field(default_factory=list)
    bonds: list = field(default_factory=list)
    interaction_terms: dict = field(default_factory=dict)
    energy: np.ndarray | None = None
    description: str = ""
    rebuild: object = None

    def analyze(self, energy="auto", initial_state=None) -> SteadyState:
        """Steady-state thermodynamics.

        ``energy="auto"`` uses the model's preferred energy operator: the bare
        Hamiltonian for local baths (so the local model's thermodynamics is
        consistent, with the coupling's boundary work reported explicitly),
        the full one for global baths.
        """
        if isinstance(energy, str) and energy == "auto":
            energy = self.energy
            if energy is None and self.master_equation == "local":
                energy = self.H0
        result = analyze(self.H, self.baths, energy=energy,
                         initial_state=initial_state)
        result.labels = dict(self.roles)
        result.model = self
        return result

    def compare(self):
        """Local against global master equation for this exact machine."""
        if self.rebuild is None:
            raise QThermoError("this model does not know how to rebuild itself")
        local = self if self.master_equation == "local" else self.rebuild("local")
        glob = self if self.master_equation == "global" else self.rebuild("global")
        return compare_master_equations(self.H, local.baths, glob.baths)

    def with_baths(self, baths) -> "Model":
        return replace(self, baths=list(baths))

    def bath(self, name: str) -> Bath:
        for b in self.baths:
            if b.name == name:
                return b
        raise KeyError(name)


def _check_me(master_equation: str) -> str:
    if master_equation not in _MASTER_EQUATIONS:
        raise QThermoError(
            f"master_equation must be one of {_MASTER_EQUATIONS}, "
            f"got {master_equation!r}")
    return master_equation


def _site_bath(H_full, H0, local_site_H, coupling_site, site, dims, T, gamma,
               name, master_equation, spectrum=None) -> Bath:
    """A bath touching one site, local or global."""
    coupling = embed(coupling_site, site, dims)
    if master_equation == "global":
        return davies_bath(H_full, coupling, T, gamma=gamma, spectrum=spectrum,
                           name=name, sites=(site,))
    # Local: Davies on the bare Hamiltonian restricted to this site. Using H0
    # rather than the single-site H and embedding gives the same operators
    # but keeps degenerate spectator levels grouped correctly.
    bath = davies_bath(H0, coupling, T, gamma=gamma, spectrum=spectrum,
                       name=name, sites=(site,))
    bath.kind = "local"
    return bath


def _basis_ket(bits) -> np.ndarray:
    index = int("".join(str(b) for b in bits), 2)
    v = np.zeros(2 ** len(bits), dtype=complex)
    v[index] = 1.0
    return v


def absorption_refrigerator(omega_c: float = 1.0, omega_h: float = 3.0,
                            T_c: float = 1.0, T_h: float = 4.0, T_r: float = 1.5,
                            g: float = 0.05, gamma: float = 0.01,
                            master_equation: str = "local") -> Model:
    """Three-qubit absorption refrigerator.

    Linden, Popescu & Skrzypczyk, PRL 105, 130401 (2010). Sites are
    ``[cold, hot, room]`` with ``omega_r = omega_c + omega_h``; the interaction
    ``g (|110><001| + h.c.)`` swaps one excitation of the room qubit for one
    each of the cold and hot qubits. Heat from the hot bath pays for pumping
    heat out of the cold bath into the room bath -- no work input at all.

    In the weak-coupling local model the three currents are locked:
    ``J_c : J_h : J_r = omega_c : omega_h : -omega_r``, so the COP is exactly
    ``omega_c / omega_h`` whenever the machine cools. Cooling happens when the
    cold qubit's *virtual temperature* drops below ``T_c``, i.e. when

        omega_c / omega_h  <  (1/T_r - 1/T_h) / (1/T_c - 1/T_r)   (roughly)
    """
    master_equation = _check_me(master_equation)
    omega_r = omega_c + omega_h
    dims = [2, 2, 2]
    local_H = [qubit_hamiltonian(w) for w in (omega_c, omega_h, omega_r)]
    H0 = sum(embed(h, i, dims) for i, h in enumerate(local_H))
    swap = np.outer(_basis_ket([1, 1, 0]), _basis_ket([0, 0, 1]).conj())
    V = g * (swap + swap.conj().T)
    H = H0 + V

    specs = [("cold", T_c), ("hot", T_h), ("room", T_r)]
    baths = [_site_bath(H, H0, local_H[i], sigma_x, i, dims, T, gamma, name,
                        master_equation)
             for i, (name, T) in enumerate(specs)]

    def rebuild(me):
        return absorption_refrigerator(omega_c, omega_h, T_c, T_h, T_r, g,
                                       gamma, master_equation=me)

    # The three-body term has no pairwise decomposition; attribute it to
    # every pair for plotting purposes only.
    return Model(
        H=H, H0=H0, dims=dims, local_H=local_H, baths=baths,
        master_equation=master_equation,
        roles={"cold": "cold", "source": "hot", "sink": "room"},
        site_names=["cold", "hot", "room"],
        bonds=[(0, 1), (1, 2), (0, 2)],
        interaction_terms={(0, 1, 2): V},
        description="three-qubit absorption refrigerator (Linden-Popescu-Skrzypczyk)",
        rebuild=rebuild,
    )


def three_level_maser(omega_c: float = 1.0, omega_h: float = 3.0,
                      T_c: float = 1.0, T_h: float = 10.0,
                      drive: float = 0.05, gamma_c: float = 0.02,
                      gamma_h: float = 0.02) -> Model:
    """Scovil--Schulz-DuBois three-level maser as a continuous heat engine.

    Scovil & Schulz-DuBois, PRL 2, 262 (1959). Levels ``|0>, |1>, |2>`` at
    energies ``0, omega_c, omega_h``. The hot bath drives ``0 <-> 2``, the cold
    bath ``0 <-> 1``, and a resonant classical field of frequency
    ``omega_h - omega_c`` couples ``1 <-> 2`` and extracts work.

    The model is written in the frame rotating with the drive, where the
    Hamiltonian is time-independent: ``H = drive (|1><2| + |2><1|)``. Heat
    currents use the bare energies ``H0``, and the power extracted by the field
    is ``J_h + J_c``. Because every quantum taken from the hot bath returns
    ``omega_c`` to the cold one, the efficiency is exactly
    ``1 - omega_c / omega_h`` -- the SSDB result -- whenever the machine runs
    as an engine, which requires it to be below Carnot.

    Baths are local to their transitions (the standard treatment); the
    rotating-frame model is valid for ``drive`` small against ``omega_h - omega_c``.
    """
    dims = [3]
    H0 = np.diag([0.0, omega_c, omega_h]).astype(complex)
    ket = np.eye(3, dtype=complex)
    X_h = np.outer(ket[0], ket[2]) + np.outer(ket[2], ket[0])
    X_c = np.outer(ket[0], ket[1]) + np.outer(ket[1], ket[0])
    H = drive * (np.outer(ket[1], ket[2]) + np.outer(ket[2], ket[1]))

    baths = [
        davies_bath(H0, X_h, T_h, gamma=gamma_h, name="hot", sites=(0,)),
        davies_bath(H0, X_c, T_c, gamma=gamma_c, name="cold", sites=(0,)),
    ]
    for b in baths:
        b.kind = "local"
    return Model(
        H=H, H0=H0, dims=dims, local_H=[H0], baths=baths,
        master_equation="local", roles={"hot": "hot", "cold": "cold"},
        site_names=["maser"], energy=H0,
        description="Scovil-Schulz-DuBois three-level maser (rotating frame)",
    )


def spin_chain(n: int = 3, omega=1.0, J: float = 0.2, delta: float = 1.0,
               T_left: float = 2.0, T_right: float = 1.0, gamma: float = 0.05,
               master_equation: str = "global", field_disorder=None) -> Model:
    """XXZ chain of ``n`` qubits between a left and a right thermal bath.

        H = sum_i (omega_i / 2) (-sigma_z^i)
            + J sum_i [ sigma_x^i sigma_x^{i+1} + sigma_y^i sigma_y^{i+1}
                        + delta sigma_z^i sigma_z^{i+1} ] / 2

    The left bath touches site 0, the right bath site ``n - 1``, both through
    ``sigma_x``. The standard minimal model of quantum heat transport;
    ``delta = 0`` is the XX chain, ``delta = 1`` the isotropic Heisenberg chain.
    ``omega`` may be a scalar or one value per site (to model a thermal diode
    or a gradient); ``field_disorder`` adds a fixed random offset per site.
    """
    master_equation = _check_me(master_equation)
    if n < 2:
        raise QThermoError("a chain needs at least two sites")
    omegas = np.broadcast_to(np.asarray(omega, dtype=float), (n,)).copy()
    if field_disorder is not None:
        omegas = omegas + np.asarray(field_disorder, dtype=float)
    dims = [2] * n
    local_H = [qubit_hamiltonian(w) for w in omegas]
    H0 = sum(embed(h, i, dims) for i, h in enumerate(local_H))

    interaction_terms = {}
    for i in range(n - 1):
        V = 0.5 * J * (
            embed(sigma_x, i, dims) @ embed(sigma_x, i + 1, dims)
            + embed(sigma_y, i, dims) @ embed(sigma_y, i + 1, dims)
            + delta * embed(sigma_z, i, dims) @ embed(sigma_z, i + 1, dims))
        interaction_terms[(i, i + 1)] = V
    H = H0 + sum(interaction_terms.values())

    baths = [
        _site_bath(H, H0, local_H[0], sigma_x, 0, dims, T_left, gamma, "left",
                   master_equation),
        _site_bath(H, H0, local_H[-1], sigma_x, n - 1, dims, T_right, gamma,
                   "right", master_equation),
    ]

    def rebuild(me):
        return spin_chain(n, omega, J, delta, T_left, T_right, gamma,
                          master_equation=me, field_disorder=field_disorder)

    return Model(
        H=H, H0=H0, dims=dims, local_H=local_H, baths=baths,
        master_equation=master_equation,
        roles={"hot": "left" if T_left >= T_right else "right",
               "cold": "right" if T_left >= T_right else "left"},
        site_names=[f"q{i}" for i in range(n)],
        bonds=[(i, i + 1) for i in range(n - 1)],
        interaction_terms=interaction_terms,
        description=f"{n}-site XXZ chain (delta={delta}) between two baths",
        rebuild=rebuild,
    )


def two_qubit_heat_valve(omega_1: float = 1.0, omega_2: float = 0.6,
                         g: float = 0.6, coupling: str = "xx",
                         T_hot: float = 2.0, T_cold: float = 1.0,
                         gamma: float = 0.1,
                         master_equation: str = "global") -> Model:
    """Two coupled qubits, hot bath on the first, cold bath on the second.

    The smallest system on which local and global master equations disagree
    qualitatively. With ``coupling="xx"`` (counter-rotating terms kept) and
    detuned qubits, the local master equation predicts heat flowing from the
    cold bath into the hot one -- a second-law violation first pointed out by
    Levy & Kosloff, EPL 107, 20004 (2014). ``coupling="exchange"`` keeps only
    the excitation-conserving part and does not show it.
    """
    master_equation = _check_me(master_equation)
    dims = [2, 2]
    local_H = [qubit_hamiltonian(omega_1), qubit_hamiltonian(omega_2)]
    H0 = embed(local_H[0], 0, dims) + embed(local_H[1], 1, dims)
    if coupling == "xx":
        V = g * embed(sigma_x, 0, dims) @ embed(sigma_x, 1, dims)
    elif coupling == "exchange":
        V = g * (embed(sigma_plus, 0, dims) @ embed(sigma_minus, 1, dims)
                 + embed(sigma_minus, 0, dims) @ embed(sigma_plus, 1, dims))
    else:
        raise QThermoError(f"coupling must be 'xx' or 'exchange', got {coupling!r}")
    H = H0 + V
    baths = [
        _site_bath(H, H0, local_H[0], sigma_x, 0, dims, T_hot, gamma, "hot",
                   master_equation),
        _site_bath(H, H0, local_H[1], sigma_x, 1, dims, T_cold, gamma, "cold",
                   master_equation),
    ]

    def rebuild(me):
        return two_qubit_heat_valve(omega_1, omega_2, g, coupling, T_hot, T_cold,
                                    gamma, master_equation=me)

    return Model(
        H=H, H0=H0, dims=dims, local_H=local_H, baths=baths,
        master_equation=master_equation, roles={"hot": "hot", "cold": "cold"},
        site_names=["q0", "q1"], bonds=[(0, 1)],
        interaction_terms={(0, 1): V},
        description=f"two qubits, {coupling} coupling, between two baths",
        rebuild=rebuild,
    )


def thermal_transistor(T_L: float = 1.0, T_M: float = 0.2, T_R: float = 0.2,
                       omega: float = 1.0, zz_left: float = 1.0, zz_right: float = 0.7,
                       gamma: float = 0.001) -> Model:
    """Three qubits with Ising (zz) couplings as a quantum thermal transistor.

    Joulain, Drevillon, Ezzahri & Ordonez-Miranda, PRL 116, 200601 (2016).
    Qubits L (emitter), M (base) and R (collector) each touch their own bath
    through ``sigma_x``; the zz couplings make every transition energy
    depend on the neighbours' states, so the base temperature steers the
    heat flowing from L to R. The gain is
    ``Model.analyze`` + :func:`qthermo.response`'s ``amplification("M", "R")``.
    Global (Davies) baths; the Hamiltonian is diagonal, so the model is exact
    in the secular sense. ``zz_left != zz_right`` avoids degenerate transitions.
    """
    dims = [2, 2, 2]
    local_H = [qubit_hamiltonian(omega) for _ in range(3)]
    H0 = sum(embed(h, i, dims) for i, h in enumerate(local_H))
    terms = {(0, 1): zz_left * embed(sigma_z, 0, dims) @ embed(sigma_z, 1, dims),
             (1, 2): zz_right * embed(sigma_z, 1, dims) @ embed(sigma_z, 2, dims)}
    H = H0 + sum(terms.values())
    baths = [davies_bath(H, embed(sigma_x, i, dims), T, gamma=gamma, name=name, sites=(i,))
             for i, (name, T) in enumerate((("L", T_L), ("M", T_M), ("R", T_R)))]
    return Model(H=H, H0=H0, dims=dims, local_H=local_H, baths=baths,
                 master_equation="global", roles={"hot": "L", "cold": "R"},
                 site_names=["L", "M", "R"], bonds=[(0, 1), (1, 2)],
                 interaction_terms=terms,
                 description="three-qubit quantum thermal transistor (zz-coupled)")


def build_model(local_H, interactions=None, baths=(), master_equation: str = "global",
                site_names=None, description: str = "custom machine") -> Model:
    """Build a :class:`Model` for your own multi-site machine.

    Parameters
    ----------
    local_H : list of arrays
        One Hamiltonian per site (unembedded); their dimensions define ``dims``.
    interactions : dict, optional
        ``{(i, j, ...): V}`` interaction terms. ``V`` is either already embedded
        (full dimension) or a product of single-site operators given as a tuple
        ``(op_i, op_j, ...)`` in the order of the key, which is embedded for you.
    baths : list of dict
        Each ``{"name", "site", "coupling", "T"}`` plus optional ``"gamma"`` or
        ``"spectrum"``. ``coupling`` acts on that site (unembedded).
    master_equation : "global" or "local"
        Global: Davies baths on the full Hamiltonian. Local: each bath acts
        through the jump operators of its isolated site.

    The model can be rebuilt under the other master equation (``compare()``),
    audited, reported and plotted like the built-in ones.

    Examples
    --------
    >>> m = qt.build_model(
    ...     [qt.qubit_hamiltonian(1.0), qt.qubit_hamiltonian(0.6)],
    ...     {(0, 1): (0.3 * qt.sigma_x, qt.sigma_x)},
    ...     [dict(name="hot", site=0, coupling=qt.sigma_x, T=2.0, gamma=0.1),
    ...      dict(name="cold", site=1, coupling=qt.sigma_x, T=1.0, gamma=0.1)])
    >>> print(qt.audit(m))
    """
    master_equation = _check_me(master_equation)
    local_H = [np.asarray(h, dtype=complex) for h in local_H]
    dims = [h.shape[0] for h in local_H]
    full = int(np.prod(dims))
    H0 = sum(embed(h, i, dims) for i, h in enumerate(local_H))
    terms = {}
    for key, V in (interactions or {}).items():
        key = tuple(int(k) for k in (key if isinstance(key, tuple) else (key,)))
        if isinstance(V, tuple):
            if len(V) != len(key):
                raise QThermoError(f"interaction {key}: {len(V)} factors for {len(key)} sites")
            op = np.eye(full, dtype=complex)
            for site, factor in zip(key, V):
                op = op @ embed(np.asarray(factor, dtype=complex), site, dims)
            V = op
        V = np.asarray(V, dtype=complex)
        if V.shape != (full, full):
            raise QThermoError(f"interaction {key} has shape {V.shape}, expected {(full, full)}")
        terms[key] = V
    H = H0 + sum(terms.values()) if terms else H0

    built = []
    for spec in baths:
        try:
            name, site, coupling, T = spec["name"], int(spec["site"]), spec["coupling"], spec["T"]
        except KeyError as exc:
            raise QThermoError(f"bath spec is missing {exc}: need name, site, coupling, T") from None
        built.append(_site_bath(H, H0, local_H[site], np.asarray(coupling, dtype=complex),
                                site, dims, T, spec.get("gamma", 0.01), name,
                                master_equation, spectrum=spec.get("spectrum")))
    temps = {b.name: b.temperature for b in built}
    roles = {}
    if len(temps) >= 2:
        ordered = sorted(temps, key=temps.get)
        roles = {"cold": ordered[0], "hot": ordered[-1]}

    def rebuild(me):
        return build_model(local_H, interactions, baths, me, site_names, description)

    return Model(H=H, H0=H0, dims=dims, local_H=local_H, baths=built,
                 master_equation=master_equation, roles=roles,
                 site_names=list(site_names or [f"q{i}" for i in range(len(dims))]),
                 bonds=[k for k in terms if len(k) == 2], interaction_terms=terms,
                 description=description, rebuild=rebuild)
