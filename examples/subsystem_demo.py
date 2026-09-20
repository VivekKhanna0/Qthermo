"""Spatially resolved thermodynamics: where does the heat actually go?

Two coupled qubits share a single hot bath that touches only qubit 0. The
totaled numbers say the system absorbed heat. This asks the question the
total cannot: how much reached qubit 1, and how much of the system's entropy
lives in the correlation between them rather than in either qubit.
"""

import numpy as np

from qthermo import (
    Cycle,
    Stroke,
    embed,
    qubit_hamiltonian,
    resolve_stroke,
    sigma_x,
    thermal_bath,
    thermal_state,
)

DIMS = [2, 2]
OMEGA_A, OMEGA_B = 1.0, 1.2
T_HOT = 3.0
COUPLING = 0.4

H_A = qubit_hamiltonian(OMEGA_A)
H_B = qubit_hamiltonian(OMEGA_B)

# Full Hamiltonian: two local terms plus an XX coupling.
H = (
    embed(H_A, 0, DIMS)
    + embed(H_B, 1, DIMS)
    + COUPLING * (embed(sigma_x, 0, DIMS) @ embed(sigma_x, 1, DIMS))
)

# The bath touches qubit 0 only. Qubit 1 can only heat up through the coupling.
c_ops = [embed(op, 0, DIMS) for op in thermal_bath(0.5, OMEGA_A, T_HOT)]

cycle = Cycle([Stroke("hot_contact", H, duration=4.0, c_ops=c_ops,
                      temperature=T_HOT, steps=400)])

rho0 = np.kron(thermal_state(H_A, 0.25), thermal_state(H_B, 0.25))
result = cycle.run(rho0)
stroke = result.strokes[0]

print("TOTALED (what Phase 1 gives you)")
print(f"  heat                  : {stroke.heat:+.6f}")
print(f"  work                  : {stroke.work:+.6f}")
print(f"  first-law residual    : {stroke.first_law_residual:.3e}")
print()

breakdown = resolve_stroke(stroke, DIMS, [H_A, H_B])
print(breakdown.report())
print()

heat_a = breakdown.local_heat[0]
heat_b = breakdown.local_heat[1]
print("READING THE BREAKDOWN")
print(f"  bath is on qubit 0; qubit 1 still absorbed {heat_b:+.6f}")
print(f"  indirect fraction     : {abs(heat_b) / (abs(heat_a) + abs(heat_b)):.1%}")
print(f"  dominant site         : {breakdown.dominant_site}")
print(f"  correlation built     : {breakdown.correlation_change:+.6f} nats")

local_entropy = sum(breakdown.local_entropy_change.values())
print(f"  sum of local dS       : {local_entropy:+.6f}")
print(f"  actual total dS       : {breakdown.total_delta_S:+.6f}")
print(
    f"  -> summing sites overcounts entropy by "
    f"{local_entropy - breakdown.total_delta_S:+.6f} nats, exactly the "
    "correlation built between them"
)
print(f"  balance residual      : {breakdown.entropy_balance_residual:.3e}")
