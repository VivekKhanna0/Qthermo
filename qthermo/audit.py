"""One call that checks a thermal-machine model for the known traps.

``audit(H, baths)`` (or ``audit(model)``) runs every diagnostic the package has
and reports, in plain language, what is wrong, suspicious, or fine:

* is the steady state unique, and how long does it take to reach it?
* is entropy production non-negative, and does heat flow the right way?
* with local baths: would a global master equation change the answer?
* with global baths: can internal (bond) currents be resolved at all?
* are degenerate levels connected only by a rate the spectrum leaves undefined?
* is the bath temperature consistent with the jump operators' detailed balance?
* do the current fluctuations respect the classical uncertainty relations?

Each finding carries a severity (``error``, ``warning``, ``info``, ``ok``), a
one-line summary, and an explanation of what it means for the physics. Nothing
is fixed silently; the point is to know before a referee does.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .baths import Bath
from .core import _as_matrix
from .validation import QThermoError, check_hermitian

__all__ = ["Finding", "AuditReport", "audit"]

_ORDER = {"error": 0, "warning": 1, "info": 2, "ok": 3}


@dataclass
class Finding:
    severity: str        # "error", "warning", "info", "ok"
    check: str
    summary: str
    detail: str = ""


@dataclass
class AuditReport:
    findings: list = field(default_factory=list)
    steady: object = None

    def add(self, severity, check, summary, detail=""):
        self.findings.append(Finding(severity, check, summary, detail))

    @property
    def ok(self) -> bool:
        """True when nothing rose above ``info``."""
        return not any(f.severity in ("error", "warning") for f in self.findings)

    def by_severity(self, severity: str) -> list:
        return [f for f in self.findings if f.severity == severity]

    def report(self, verbose: bool = False) -> str:
        marks = {"error": "[ERROR]", "warning": "[WARN] ", "info": "[info] ",
                 "ok": "[ok]   "}
        lines = []
        for f in sorted(self.findings, key=lambda f: _ORDER[f.severity]):
            lines.append(f"{marks[f.severity]} {f.check}: {f.summary}")
            if f.detail and (verbose or f.severity in ("error", "warning")):
                for chunk in _wrap(f.detail, 74):
                    lines.append(f"          {chunk}")
        n_err, n_warn = len(self.by_severity("error")), len(self.by_severity("warning"))
        lines.append("-" * 60)
        lines.append(f"{n_err} error(s), {n_warn} warning(s), "
                     f"{len(self.findings) - n_err - n_warn} other checks")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.report()


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


def _detailed_balance_temperature(bath: Bath, H) -> float | None:
    """Temperature implied by the ratio of up/down rates of the bath's jumps.

    For each pair of jump operators that are adjoint up to scale (``L_up ~
    L_down^dag``) with energy change ``w``, detailed balance fixes
    ``|L_up|^2 / |L_down|^2 = exp(-w / T)``. Returns the median implied T,
    or None when no such pairs exist.
    """
    ops = bath.c_ops
    energies = []
    for L in ops:
        comm = H @ L - L @ H
        norm = np.vdot(L, L).real
        if norm == 0:
            energies.append(None)
            continue
        e = float(np.real(np.vdot(L, comm) / norm))
        if np.linalg.norm(comm - e * L) > 1e-6 * max(np.sqrt(norm), 1e-300) * max(1, abs(e)):
            energies.append(None)
        else:
            energies.append(e)
    implied = []
    for i, (Li, ei) in enumerate(zip(ops, energies)):
        if ei is None or ei <= 1e-9:
            continue
        for j, (Lj, ej) in enumerate(zip(ops, energies)):
            if ej is None or abs(ei + ej) > 1e-6 * max(1, abs(ei)):
                continue
            # Li raises by ei, Lj lowers by ei: are they adjoint up to scale?
            a = Lj.conj().T
            na, ni = np.vdot(a, a).real, np.vdot(Li, Li).real
            if na == 0 or ni == 0:
                continue
            overlap = abs(np.vdot(a, Li)) / np.sqrt(na * ni)
            if overlap > 1 - 1e-6 and ni < na:
                implied.append(ei / np.log(na / ni))
    return float(np.median(implied)) if implied else None


def audit(source, baths=None, energy=None, fluctuations: bool = True,
          compare: bool = True) -> AuditReport:
    """Check a steady-state thermal machine for known modelling traps.

    Parameters
    ----------
    source : Model or Hamiltonian
        A :class:`~qthermo.models.Model`, or a Hamiltonian together with
        ``baths``.
    baths : list of Bath, optional
    energy : array, optional
        Energy operator for heat currents (defaults as in ``Model.analyze``).
    fluctuations : bool
        Also compute TUR/KUR ratios of every bath's heat current (small
        systems only).
    compare : bool
        For a ``Model``, rebuild it under the other master equation and
        compare currents.

    Returns
    -------
    AuditReport
        ``print(report)`` for the summary; ``report.findings`` for the data.
    """
    from .models import Model
    from .network import heat_flow_map
    from .steady import analyze, relaxation_time

    report = AuditReport()
    model = source if isinstance(source, Model) else None
    user_energy = energy
    if model is not None:
        H, baths = model.H, model.baths
        if energy is None:
            energy = model.energy if model.energy is not None else (
                model.H0 if model.master_equation == "local" else None)
    else:
        H = check_hermitian(_as_matrix(source))
        if baths is None:
            raise QThermoError("audit(H, baths): pass the baths")
    baths = list(baths)
    dim = H.shape[0]

    # 1. bath construction warnings (degeneracy etc.) are re-raised as findings
    kinds = {getattr(b, "kind", "custom") for b in baths}

    # 2. steady state: unique?
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            steady = (model.analyze(energy="auto" if user_energy is None else user_energy)
                      if model is not None else analyze(H, baths, energy=energy))
        except QThermoError as exc:
            report.add("error", "steady state", "not unique", str(exc))
            return report
    report.steady = steady
    report.add("ok", "steady state", "unique")
    for w in caught:
        if "negative entropy production" in str(w.message):
            continue   # reported below with more context
        report.add("warning", "numerics", str(w.message).split(".")[0], str(w.message))

    # 3. relaxation time
    if dim <= 64:
        t_pop = relaxation_time(H, baths, populations_only=True)
        detail = ""
        if dim <= 16:
            t_full = relaxation_time(H, baths)
            if t_full > 20 * t_pop:
                detail = (f"coherences relax much more slowly ({t_full:.3g}) than "
                          "populations: typical of degenerate levels; energetics "
                          "settle on the population time scale.")
        report.add("info", "relaxation time", f"{t_pop:.3g} (populations)", detail)

    # 4. second law and direction of heat flow
    sigma = steady.entropy_production_rate
    scale = max(abs(J) for J in steady.currents.values()) or 1.0
    thermal = [b for b in baths if b.temperature is not None]
    if sigma < -1e-10 * scale:
        report.add("error", "second law",
                   f"entropy production rate is negative ({sigma:.3e})",
                   "The model moves heat in a way no physical process can. With "
                   "local baths on coupled sites this is the Levy-Kosloff "
                   "inconsistency: use global baths, or measure heat with the "
                   "bare Hamiltonian (energy=H0), which adds boundary work.")
    else:
        report.add("ok", "second law", f"entropy production rate {sigma:.3e} >= 0")
    if len(thermal) == 2 and abs(steady.work_rate) < 1e-10 * scale:
        hot, cold = sorted(thermal, key=lambda b: -b.temperature)
        if hot.temperature > cold.temperature and steady.current(hot.name) < -1e-12 * scale:
            report.add("error", "heat direction",
                       f"heat flows from {cold.name!r} (T={cold.temperature:g}) into "
                       f"{hot.name!r} (T={hot.temperature:g}) with no work input")

    # 5. detailed balance vs declared temperature
    for b in thermal:
        try:
            T_implied = _detailed_balance_temperature(b, H if b.kind != "local" or model is None
                                                      else model.H0)
        except Exception:  # noqa: BLE001 -- diagnostic only
            T_implied = None
        if T_implied is not None and abs(T_implied - b.temperature) > 1e-3 * b.temperature:
            report.add("error", "bath temperature",
                       f"{b.name!r} is declared at T={b.temperature:g} but its "
                       f"rates satisfy detailed balance at T={T_implied:.4g}",
                       "Entropy production divides heat by the declared "
                       "temperature; if the jump operators encode another one, "
                       "sigma is meaningless.")

    # 6. local vs global
    can_rebuild = (model is not None and compare and model.rebuild is not None
                   and model.master_equation in ("local", "global"))
    if "local" in kinds or can_rebuild:
        if can_rebuild:
            try:
                comparison = model.compare()
                worst = 0.0
                flipped = []
                for name, j_loc in comparison.local.currents.items():
                    j_glob = comparison.global_.currents[name]
                    rel = abs(j_loc - j_glob) / max(abs(j_glob), 1e-300)
                    worst = max(worst, rel)
                    if np.sign(j_loc) != np.sign(j_glob) and abs(j_glob) > 1e-12 * scale:
                        flipped.append(name)
                if flipped:
                    report.add("error", "local vs global",
                               "the two master equations disagree on the direction "
                               f"of heat flow for {', '.join(flipped)}",
                               "Inter-site coupling is too strong for the local "
                               "approximation. Use the global model.")
                elif worst > 0.1:
                    report.add("warning", "local vs global",
                               f"currents differ by up to {worst:.0%}",
                               "Results depend on the master equation; check which "
                               "regime applies (local: coupling << bath rates; "
                               "global: coupling >> bath rates).")
                else:
                    report.add("ok", "local vs global",
                               f"currents agree within {worst:.1%}")
            except QThermoError as exc:
                report.add("info", "local vs global", "comparison not possible", str(exc))
        elif model is not None:
            report.add("info", "local vs global",
                       "local baths; this model has no global counterpart to compare")
        else:
            report.add("info", "local vs global",
                       "local baths used; build a Model to compare automatically")

    # 7. secular blindness of internal currents
    if model is not None and len(model.dims) > 1:
        try:
            fm = heat_flow_map(steady)
            if fm.extras.get("secular_blind"):
                report.add("warning", "internal currents",
                           "bond currents vanish identically in this global steady state",
                           "The secular approximation removes the coherences that "
                           "carry site-to-site currents. Totals per bath are fine; "
                           "per-bond transport is not resolvable in this model.")
            else:
                residual = np.max(np.abs(fm.site_balance()))
                report.add("ok", "internal currents",
                           f"site energy balances close to {residual:.1e}")
        except QThermoError:
            pass

    # 8. fluctuations
    if fluctuations and dim <= 32 and all(b.temperature is not None for b in baths):
        from .fluctuations import current_statistics
        for b in baths:
            try:
                stats = current_statistics(H, b.name, baths=baths,
                                           energy=energy if energy is not None else H)
            except QThermoError:
                continue
            if abs(stats.mean) < 1e-12:
                continue
            if stats.violates_tur:
                report.add("info", "uncertainty relation",
                           f"heat current of {b.name!r} beats the classical TUR "
                           f"(ratio {stats.tur_ratio:.4f} < 2)",
                           "A genuine signature of coherent transport -- or of "
                           "an inconsistent bath model if sigma is suspect.")
            else:
                report.add("ok", "uncertainty relation",
                           f"{b.name!r}: TUR ratio {stats.tur_ratio:.3f} >= 2")
    return report
