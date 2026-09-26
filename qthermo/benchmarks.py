"""Benchmarks against results fixed independently of this implementation.

    python -m qthermo.benchmarks          # everything
    python -m qthermo.benchmarks --quick  # skip the slow ones (CI)

Each benchmark computes one number with qthermo and compares it with a value
that comes from analysis or from the literature -- a closed-form limit, an
exact identity, a published bound. The table this prints is meant to be read
by someone deciding whether to trust the package: every row names what the
reference is and where it comes from, and a failure means the physics is
wrong, not that an interface changed.

Benchmarks register themselves with the ``@benchmark`` decorator. A benchmark
function returns ``(computed, reference)`` and passes when they agree within
its tolerance; for inequalities (bounds), it returns ``(computed, bound)`` and
declares ``kind="upper"`` or ``kind="lower"``.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Callable

__all__ = ["Benchmark", "BenchmarkOutcome", "benchmark", "run_benchmarks",
           "REGISTRY"]


@dataclass
class Benchmark:
    name: str
    reference: str        # where the expected value comes from
    function: Callable
    tolerance: float
    kind: str             # "equal", "upper" (computed <= bound), "lower"
    relative: bool
    slow: bool
    group: str


@dataclass
class BenchmarkOutcome:
    benchmark: Benchmark
    computed: float
    expected: float
    passed: bool
    seconds: float
    error: str | None = None

    @property
    def deviation(self) -> float:
        if self.benchmark.kind != "equal":
            return float("nan")
        scale = abs(self.expected) if self.benchmark.relative and self.expected else 1.0
        return abs(self.computed - self.expected) / scale


REGISTRY: list[Benchmark] = []


def benchmark(name: str, reference: str, tolerance: float = 1e-6,
              kind: str = "equal", relative: bool = False, slow: bool = False,
              group: str = "core"):
    """Register a function returning ``(computed, expected)``."""
    if kind not in ("equal", "upper", "lower"):
        raise ValueError(f"unknown benchmark kind {kind!r}")

    def decorate(function):
        REGISTRY.append(Benchmark(name, reference, function, tolerance, kind,
                                  relative, slow, group))
        return function
    return decorate


def _evaluate(bench: Benchmark) -> BenchmarkOutcome:
    start = time.perf_counter()
    try:
        computed, expected = bench.function()
        computed, expected = float(computed), float(expected)
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        return BenchmarkOutcome(bench, float("nan"), float("nan"), False,
                                time.perf_counter() - start,
                                f"{type(exc).__name__}: {exc}")
    if bench.kind == "equal":
        scale = abs(expected) if bench.relative and expected else 1.0
        passed = abs(computed - expected) <= bench.tolerance * scale
    elif bench.kind == "upper":
        passed = computed <= expected + bench.tolerance
    else:
        passed = computed >= expected - bench.tolerance
    return BenchmarkOutcome(bench, computed, expected, bool(passed),
                            time.perf_counter() - start)


def run_benchmarks(quick: bool = False, group: str | None = None,
                   stream=sys.stdout) -> list[BenchmarkOutcome]:
    """Run every registered benchmark and print a table of outcomes."""
    _load_all()
    selected = [b for b in REGISTRY
                if not (quick and b.slow) and (group is None or b.group == group)]
    first_seen = {}
    for b in REGISTRY:
        first_seen.setdefault(b.group, len(first_seen))
    selected.sort(key=lambda b: first_seen[b.group])      # stable within group
    outcomes = []
    current_group = None
    header = f"{'':4}{'benchmark':<60}{'qthermo':>13}{'reference':>13}  {'relation':<8}"
    print(header, file=stream)
    print("-" * len(header), file=stream)
    for bench in selected:
        if bench.group != current_group:
            current_group = bench.group
            print(f"[{current_group}]", file=stream)
        outcome = _evaluate(bench)
        outcomes.append(outcome)
        mark = "ok" if outcome.passed else "XX"
        relation = {"equal": "==", "upper": "<=", "lower": ">="}[bench.kind]
        if outcome.error:
            print(f"{mark:<4}{bench.name:<60}  ERROR {outcome.error}", file=stream)
        else:
            print(f"{mark:<4}{bench.name:<60}{outcome.computed:>13.6g}"
                  f"{outcome.expected:>13.6g}  {relation:<8}", file=stream)
        print(f"{'':6}{bench.reference}", file=stream)
    print("-" * len(header), file=stream)
    failed = [o for o in outcomes if not o.passed]
    total = sum(o.seconds for o in outcomes)
    print(f"{len(outcomes) - len(failed)}/{len(outcomes)} passed "
          f"in {total:.1f} s", file=stream)
    return outcomes


def _load_all():
    """Import every module that registers benchmarks."""
    from . import _benchmark_suite  # noqa: F401


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m qthermo.benchmarks",
        description="Check qthermo against analytic and published results.",
    )
    parser.add_argument("--quick", action="store_true",
                        help="skip benchmarks marked slow")
    parser.add_argument("--group", default=None,
                        help="run only one group of benchmarks")
    args = parser.parse_args(argv)
    outcomes = run_benchmarks(quick=args.quick, group=args.group)
    return 0 if all(o.passed for o in outcomes) else 1


if __name__ == "__main__":
    # Under ``python -m`` this file runs as ``__main__``, a different module
    # object from ``qthermo.benchmarks`` that the suite registers into.
    from qthermo.benchmarks import main as _main
    sys.exit(_main())
