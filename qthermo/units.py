"""Converting between laboratory units and qthermo's dimensionless units.

qthermo works in units with hbar = k_B = 1 and an arbitrary energy unit E0.
Choosing E0 = h f0 for a reference frequency f0 (a qubit frequency, say) ties
everything to the lab:

    energy       E / E0
    frequency    omega = 2 pi f / (2 pi f0) = f / f0     (angular, dimensionless)
    temperature  k_B T / E0
    rate         Gamma hbar / E0 = Gamma / (2 pi f0)     (Gamma in 1/s)
    time         t E0 / hbar = 2 pi f0 t
    power        P hbar / E0^2

>>> lab = qt.LabUnits(5.0)                   # E0 = h x 5 GHz
>>> lab.temperature(20)                      # 20 mK in units of E0/k_B
0.0834...
>>> lab.frequency(5.0)                       # the qubit itself: omega = 1
1.0
>>> lab.rate(1 / 50e-6)                      # T1 = 50 us
6.37e-07...
>>> lab.to_watts(0.01)                       # a dimensionless heat current
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LabUnits", "PLANCK", "HBAR", "BOLTZMANN"]

PLANCK = 6.62607015e-34        # J s (exact, SI 2019)
HBAR = PLANCK / (2 * np.pi)
BOLTZMANN = 1.380649e-23       # J / K (exact, SI 2019)


@dataclass(frozen=True)
class LabUnits:
    """Unit system with energy unit ``E0 = h * reference_GHz * 1e9``."""

    reference_GHz: float

    @property
    def energy_joule(self) -> float:
        return PLANCK * self.reference_GHz * 1e9

    # lab -> dimensionless -----------------------------------------------
    def frequency(self, f_GHz):
        """Transition frequency in GHz -> dimensionless angular frequency."""
        return np.asarray(f_GHz, dtype=float) / self.reference_GHz

    def temperature(self, T_mK):
        """Temperature in mK -> k_B T / E0."""
        return BOLTZMANN * np.asarray(T_mK, dtype=float) * 1e-3 / self.energy_joule

    def rate(self, gamma_per_second):
        """Rate in 1/s (e.g. 1/T1) -> dimensionless rate."""
        return np.asarray(gamma_per_second, dtype=float) * HBAR / self.energy_joule

    def time(self, t_seconds):
        """Duration in seconds -> dimensionless time."""
        return np.asarray(t_seconds, dtype=float) * self.energy_joule / HBAR

    # dimensionless -> lab -----------------------------------------------
    def to_GHz(self, omega):
        return np.asarray(omega, dtype=float) * self.reference_GHz

    def to_mK(self, T):
        return np.asarray(T, dtype=float) * self.energy_joule / BOLTZMANN * 1e3

    def to_seconds(self, t):
        return np.asarray(t, dtype=float) * HBAR / self.energy_joule

    def to_per_second(self, gamma):
        return np.asarray(gamma, dtype=float) * self.energy_joule / HBAR

    def to_joules(self, energy):
        return np.asarray(energy, dtype=float) * self.energy_joule

    def to_watts(self, power):
        """Dimensionless power or heat current -> watts."""
        return np.asarray(power, dtype=float) * self.energy_joule ** 2 / HBAR

    def describe(self) -> str:
        return (f"E0 = h x {self.reference_GHz:g} GHz = {self.energy_joule:.4e} J; "
                f"T = 1 is {self.to_mK(1.0):.2f} mK; t = 1 is "
                f"{self.to_seconds(1.0) * 1e12:.2f} ps; power 1 is "
                f"{self.to_watts(1.0):.3e} W")
