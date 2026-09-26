---
title: baths.py
---

`Bath` objects: a named set of jump operators plus a temperature. Every
multi-bath tool (steady states, per-bath stroke heat, flow maps, FCS) keys on
`Bath.name`.

## Functions
- `davies_bath(H, coupling, T, gamma|spectrum, name, sites)` — global (secular, detailed-balance) bath for any H. Vectorised Bohr decomposition; ops stored **sparse in the eigenbasis** (`eig_ops`, `eigenbasis`, `energies`); `.c_ops` is a lazy dense view.
- `local_bath(h_site, coupling_site, T, site, dims)` — local ME bath (Davies on the isolated site, embedded).
- `instantaneous_bath(H_of_t, ...)` — `t -> [Bath]`, adiabatic master equation; pass as a stroke's `c_ops`.
- `bohr_decomposition(H, A)` — `{omega: A(omega)}` dense, for inspection/tests.
- `flat_spectrum`, `ohmic_spectrum` — `J(w)` = zero-T emission rate. Flat has no finite w->0 limit.

## Bath methods
`.dissipator(rho)` (cached sparse superoperator for Davies baths), `.heat_current(rho, E)`, `.dim`, `.n_ops`, `.kind` ("global"/"local"/"custom").

See gotchas/degenerate-zero-frequency.md, decisions/eigenbasis-storage.md.
Tests: `tests/test_steady.py`.
