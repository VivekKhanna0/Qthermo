---
title: qubit-hamiltonian-sign
---

`channels.qubit_hamiltonian(omega)` returns `H = -(omega/2) * sigma_z`, not
the more naive `+(omega/2) * sigma_z`.

Reasoning documented directly in the function's docstring: with the plain
`+` convention, the excited state |1> (the second basis vector) sits at
energy `-omega/2`, which is backwards — the excited state should be higher
in energy. The sign is flipped so |1> sits at `+omega/2` and |0> at
`-omega/2`, giving a gap of `omega` in the physically expected direction.

If adding new Hamiltonians elsewhere in the package, match this convention
(|1> = excited = higher energy) rather than the textbook-default `+sigma_z`
form, or energy-gap-dependent code (ergotropy, thermal_state) will disagree
about which state is "hot."
