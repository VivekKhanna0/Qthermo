"""Published results, reproduced in one call.

Each function here rebuilds a known result from the literature with qthermo,
next to an exact or closed-form reference, so that a reader can check the
package against something they already trust -- and see what it adds.

    r = qt.papers.local_vs_global()
    print(r.report())
    r.plot()

Every reproduction is also a row in ``python -m qthermo.benchmarks``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .validation import QThermoError

__all__ = ["Reproduction", "local_vs_global", "exact_oscillator_current"]


@dataclass
class Reproduction:
    """A reproduced result: curves from each method next to the reference."""

    title: str
    reference: str
    parameter: str
    values: np.ndarray
    curves: dict                        # method -> array, same length as values
    exact: np.ndarray
    notes: list = field(default_factory=list)

    def errors(self) -> dict:
        """Largest relative deviation from the exact curve, per method."""
        return {m: float(np.max(np.abs(c - self.exact) / np.abs(self.exact)))
                for m, c in self.curves.items()}

    def valid_range(self, method: str, tolerance: float = 0.05) -> np.ndarray:
        """Parameter values where ``method`` is within ``tolerance`` of exact."""
        c = self.curves[method]
        return self.values[np.abs(c - self.exact) <= tolerance * np.abs(self.exact)]

    def report(self, tolerance: float = 0.05) -> str:
        methods = list(self.curves)
        lines = [self.title, f"reference: {self.reference}", "",
                 f"{self.parameter:>10}  {'exact':>11}" +
                 "".join(f"  {m:>11}" for m in methods)]
        for i, v in enumerate(self.values):
            row = f"{v:>10.4g}  {self.exact[i]:>11.4e}"
            for m in methods:
                c = self.curves[m][i]
                flag = " " if abs(c - self.exact[i]) <= tolerance * abs(self.exact[i]) else "x"
                row += f"  {c:>10.4e}{flag}"
            lines.append(row)
        lines.append(f"(x: more than {tolerance:.0%} from exact)")
        lines.append("")
        for m, e in self.errors().items():
            lines.append(f"{m:>10}: worst error {e:7.1%}")
        lines += [""] + self.notes
        return "\n".join(lines)

    def plot(self, axes=None):
        """Each method against the exact curve, and its relative error.

        ``axes``: None (new figure), one Axes (currents only), or a pair
        ``(currents_ax, error_ax)``.
        """
        import matplotlib.pyplot as plt

        if axes is None:
            fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(10, 3.8))
        elif isinstance(axes, (tuple, list, np.ndarray)):
            ax, ax_err = axes
            fig = ax.figure
        else:
            ax, ax_err, fig = axes, None, axes.figure
        styles = {"local": ("o", "#2F6FB0"), "global": ("s", "#C2412D"),
                  "redfield": ("^", "#0E7C78")}
        ax.plot(self.values, self.exact, "-", color="0.15", lw=2, label="exact")
        for m, c in self.curves.items():
            marker, color = styles.get(m, ("d", None))
            ax.plot(self.values, c, marker, color=color, ms=5, label=m)
            if ax_err is not None:
                ax_err.plot(self.values, np.abs(c - self.exact) / np.abs(self.exact),
                            marker + "-", color=color, ms=4, label=m)
        for a in [ax] + ([ax_err] if ax_err is not None else []):
            a.set_xscale("log")
            a.set_xlabel(self.parameter)
        ax.set_yscale("log")
        ax.set_ylabel("heat current")
        ax.legend(frameon=False, fontsize=8)
        if ax_err is not None:
            ax_err.set_yscale("log")
            ax_err.set_ylabel("relative error vs exact")
            ax_err.axhline(0.05, color="0.6", ls=":", lw=1)
        if axes is None:
            fig.tight_layout()
        return fig


def exact_oscillator_current(omega1: float, omega2: float, g: float, gamma: float,
                             T_hot: float, T_cold: float) -> float:
    """Exact heat current through two coupled oscillators between two baths.

    ``H = w1 a1^dag a1 + w2 a2^dag a2 + g (a1^dag a2 + a2^dag a1)``, oscillator
    1 coupled to the hot bath and 2 to the cold one, each through a flat
    (wide-band) spectral density ``gamma`` in the rotating-wave approximation.
    The model is quadratic, so the steady-state current is given exactly, at
    any coupling strength, by the Landauer formula

        J = (1/2 pi) int dw  w T(w) [n_hot(w) - n_cold(w)],
        T(w) = gamma^2 |G_12(w)|^2,   G(w) = [w - h + i gamma/2]^(-1).

    This is the reference against which approximate master equations are
    judged.
    """
    from scipy.integrate import quad

    from .channels import mean_occupation

    h = np.array([[omega1, g], [g, omega2]], dtype=float)
    damping = 0.5j * gamma * np.eye(2)

    def integrand(w):
        G = np.linalg.inv(w * np.eye(2) - h + damping)
        return (w * gamma ** 2 * abs(G[0, 1]) ** 2
                * (mean_occupation(w, T_hot) - mean_occupation(w, T_cold)))

    modes = np.linalg.eigvalsh(h)
    top = modes.max() + 60 * gamma + 30 * max(T_hot, T_cold)
    value = quad(integrand, 1e-6 * modes.min(), top, points=list(modes),
                 limit=1000, epsabs=0, epsrel=1e-10)[0]
    return value / (2 * np.pi)


def local_vs_global(couplings=None, detuning: float = 0.0, gamma: float = 0.02,
                    T_hot: float = 0.5, T_cold: float = 0.25, levels: int = 6,
                    methods=("local", "global", "redfield")) -> Reproduction:
    """Which master equation gives the right heat current? Checked against exact.

    Two oscillators, frequencies ``1`` and ``1 + detuning``, coupled with
    strength ``g``; the first touches a hot bath, the second a cold one. The
    heat current is computed with the local, global (Davies) and Redfield
    master equations and compared with :func:`exact_oscillator_current`.

    This is the standard test case for the local-versus-global question
    (Hofer et al., NJP 19, 123037 (2017); Gonzalez et al., Open Syst. Inf.
    Dyn. 24, 1740010 (2017)). What it shows:

    * resonant oscillators (``detuning=0``): the global equation is wrong at
      weak coupling -- it predicts a finite current as ``g -> 0`` because the
      secular approximation fails for nearly degenerate levels;
    * detuned oscillators: the local equation is off by ~10% at every
      coupling, because it assigns each bath the bare site frequency;
    * Redfield agrees with exact everywhere (to ~0.1% here).

    Parameters
    ----------
    couplings : array, optional
        Values of ``g``. Default: 12 points from ``gamma/10`` to ``5 gamma``.
    levels : int
        Fock states kept per oscillator. The default is converged to ~1e-4
        at the default temperatures; raise it for hotter baths.
    """
    from .baths import davies_bath, local_bath, redfield_bath
    from .channels import mean_occupation
    from .steady import analyze

    if couplings is None:
        couplings = gamma * np.geomspace(0.1, 5.0, 12)
    couplings = np.asarray(couplings, dtype=float)
    unknown = set(methods) - {"local", "global", "redfield"}
    if unknown:
        raise QThermoError(f"unknown method(s) {sorted(unknown)}; "
                           "choose from local, global, redfield")
    tail = mean_occupation(1.0, T_hot) / (1 + mean_occupation(1.0, T_hot))
    if tail ** levels > 1e-4:
        raise QThermoError(
            f"{levels} Fock levels are too few at T_hot={T_hot}: the hot "
            "oscillator's occupation of the top level is not negligible. "
            "Raise levels= or lower the temperature.")

    w1, w2 = 1.0, 1.0 + detuning
    a = np.diag(np.sqrt(np.arange(1, levels)), 1)
    eye = np.eye(levels)
    a1, a2 = np.kron(a, eye), np.kron(eye, a)
    X1, X2 = a1 + a1.T, a2 + a2.T
    H0 = w1 * a1.T @ a1 + w2 * a2.T @ a2
    hop = a1.T @ a2 + a2.T @ a1

    curves = {m: np.zeros(len(couplings)) for m in methods}
    exact = np.zeros(len(couplings))
    import warnings
    for i, g in enumerate(couplings):
        H = H0 + g * hop
        exact[i] = exact_oscillator_current(w1, w2, g, gamma, T_hot, T_cold)
        for m in methods:
            if m == "local":
                baths = [local_bath(w1 * a.T @ a, a + a.T, T_hot, 0, (levels, levels),
                                    gamma=gamma, name="hot"),
                         local_bath(w2 * a.T @ a, a + a.T, T_cold, 1, (levels, levels),
                                    gamma=gamma, name="cold")]
                energy = H0          # De Chiara et al.: heat counted with H0
            else:
                build = davies_bath if m == "global" else redfield_bath
                baths = [build(H, X1, T_hot, gamma=gamma, name="hot"),
                         build(H, X2, T_cold, gamma=gamma, name="cold")]
                energy = None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                curves[m][i] = analyze(H, baths, energy=energy).currents["hot"]

    kind = "resonant" if detuning == 0 else f"detuned by {detuning:g}"
    notes = [f"two oscillators ({kind}), gamma={gamma:g}, T_hot={T_hot:g}, "
             f"T_cold={T_cold:g}, {levels} Fock levels each",
             "exact: Landauer formula (rotating-wave, wide-band baths)"]
    return Reproduction(
        title="Heat current through two coupled oscillators: master equations vs exact",
        reference="Hofer et al., NJP 19, 123037 (2017); Gonzalez et al., "
                  "OSID 24, 1740010 (2017)",
        parameter="g", values=couplings, curves=curves, exact=exact, notes=notes)
