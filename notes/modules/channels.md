---
title: channels.py
---

Collapse (jump) operators for thermal baths and common noise channels. Every
function returns a list of numpy arrays usable directly by `solver.py` or by
`qutip.mesolve`/`mcsolve` unchanged.

- `sigma_x, sigma_y, sigma_z, sigma_plus, sigma_minus` — module-level constants, standard 2x2 Pauli/ladder matrices
- `qubit_hamiltonian(omega) -> ndarray` — `H = -(omega/2) sigma_z`, sign chosen so |1> sits at +omega/2. See decisions/qubit-hamiltonian-sign.md.
- `mean_occupation(omega, temperature) -> float` — Bose-Einstein occupation; returns 0.0 for T<=0 or when overflow would occur (omega/T > 700)
- `thermal_bath(gamma, omega, temperature) -> [emission_op, absorption_op]` — rates satisfy detailed balance so the Lindbladian's unique steady state is the Gibbs state at `temperature`
- `amplitude_damping(gamma) -> [op]` — zero-temperature relaxation only (T1-type)
- `pure_dephasing(gamma) -> [op]` — T2-type, destroys coherence but exchanges no heat (commutes with diagonal H) — see modules/core.md re: entropy_production, this channel produces sigma > 0 with Q = 0
- `bit_flip(gamma) -> [op]` — symmetric population mixing

Depended on by: examples and tests build strokes from these directly.
