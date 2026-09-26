---
title: degenerate-zero-frequency
---

With `flat_spectrum` the w=0 rate is undefined (n(w) ~ T/w), so it is set to 0.
If the coupling connects distinct degenerate eigenstates, that channel is off
and thermalisation detours through excited levels, possibly 1000x slower.
Found at B_c = 4J in the two-qubit Heisenberg Otto engine. `davies_bath` warns;
use `ohmic_spectrum` or `zero_frequency_rate=`. `relaxation_time(...,
populations_only=True)` is the relevant time scale for heat.
