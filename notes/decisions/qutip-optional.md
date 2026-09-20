---
title: qutip-optional
---

qthermo has no hard dependency on QuTiP. `core._as_matrix` is the only place
the package touches QuTiP at all: if an object has a `.full()` method
(QuTiP's `Qobj` API), it's converted to a dense array; otherwise it's passed
through `np.asarray`. Every other module works on plain numpy arrays.

`pyproject.toml` confirms this: `dependencies = ["numpy>=1.23", "scipy>=1.9"]`,
with `qutip` only under `[project.optional-dependencies]`.

Why it matters: existing QuTiP-based research code can hand its `Qobj`s
straight to qthermo functions with no conversion step, but nobody has to
install QuTiP just to use the package.
