---
title: benchmarks.py
---

`python -m qthermo.benchmarks [--quick] [--group G]`. Registry + runner in
`benchmarks.py`; the suite itself lives in `_benchmark_suite.py`. Register with
`@benchmark(name, reference, tolerance, kind="equal"|"upper"|"lower",
relative, slow, group)`; the function returns `(computed, expected)`.
CI runs `--quick`. Every new physics feature should add at least one row.
