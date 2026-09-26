---
title: eigenbasis-storage
---

Davies jump operators are dense in the computational basis (d^2 operators of
d^2 entries: O(d^4) memory) but nearly 1-sparse in the eigenbasis of H. So
`davies_bath` stores them sparse in the eigenbasis and `steady_state` /
`analyze` work there when all baths share it. Result: dim-200 RC models in
~20 s instead of out-of-memory at dim 52. `Bath.c_ops` stays available as a
lazy dense view so older code paths (evolve, fluctuations) keep working.
