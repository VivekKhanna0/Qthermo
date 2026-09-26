---
title: stiff-steady-states
---

`steady_state` refuses when the trace-replaced Liouvillian has rcond < 1e-13.
Besides true non-uniqueness, this happens when transition energies are many
times a bath temperature: Boltzmann factors ~e^-200 underflow and disconnect
levels in floating point. Rescale (lower couplings or raise temperatures) or
pass initial_state=. Found building the zz thermal transistor with Delta=10,
T=0.1.
