---
title: local-master-equation
---

Local baths on coupled sites can move heat cold -> hot (sigma < 0), e.g.
`models.two_qubit_heat_valve(master_equation="local")` with XX coupling past
g ~ 0.55 (gamma=0.1). Two repairs: global baths (`davies_bath`), or measure heat
with the bare H0 (`Model.analyze()` does this automatically for local baths),
which restores sigma >= 0 with explicit boundary work (`work_rate`). The
second repair fixes the bookkeeping, not the currents.
