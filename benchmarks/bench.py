"""Benchmark harness for the performance-sensitive sexpansion workloads.

Run from the repo root::

    python benchmarks/bench.py [--repeats N] [--only SUBSTRING] [--json PATH]

Each benchmark runs ``--repeats`` times (default 3) and the best wall-clock
time is reported, so numbers are comparable across runs and machines.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterator

import numpy as np

from sexpansion import (
    LieAlgebra,
    Resonance,
    Semigroup,
    find_all_resonances,
    load_semigroup,
    load_semigroups,
)

Bench = tuple[str, Callable[[], Callable[[], object]]]


def _resonance_scan(order: int, *, filtered: bool) -> Callable[[], Callable[[], object]]:
    def make() -> Callable[[], object]:
        commutative = [sg for sg in load_semigroups(order) if sg.is_commutative]

        def run() -> object:
            total = 0
            for sg in commutative:
                total += len(find_all_resonances(sg, filtered=filtered))
            return total

        return run

    return make


def _catalogue_props(order: int) -> Callable[[], Callable[[], object]]:
    def make() -> Callable[[], object]:
        # Fresh instances so per-instance caches from earlier repeats don't help.
        tables = [sg.table.copy() for sg in load_semigroups(order)]
        semigroups = [Semigroup(t) for t in tables]

        def run() -> object:
            zeros = sum(1 for sg in semigroups if sg.find_zero() is not None)
            commutative = sum(1 for sg in semigroups if sg.is_commutative)
            return zeros, commutative

        return run

    return make


def _isomorphism_test(order: int) -> Callable[[], Callable[[], object]]:
    def make() -> Callable[[], object]:
        # Catalogue entries are pairwise non-isomorphic, so this is the
        # worst case: the full permutation scan finds no match.
        a = load_semigroup(order, 1)
        b = load_semigroup(order, 2)

        def run() -> object:
            return a.isomorphism_test(b)

        return run

    return make


def _expansion_pipeline(loops: int) -> Callable[[], Callable[[], object]]:
    def make() -> Callable[[], object]:
        algebra = LieAlgebra.sl2()
        s991 = load_semigroup(5, 991)
        resonance = Resonance(s0=(0, 1, 2), s1=(0, 3, 4))
        v0, v1 = (0,), (1, 2)

        def run() -> object:
            result: object = None
            for _ in range(loops):
                expanded = algebra.s_expand(s991)
                resonant = expanded.resonant_subalgebra(resonance, v0, v1)
                both = resonant.zero_reduced()
                result = (both.det(), both.signature())
            return result

        return run

    return make


def _metric_reuse(loops: int) -> Callable[[], Callable[[], object]]:
    def make() -> Callable[[], object]:
        algebra = LieAlgebra.sl2()
        s991 = load_semigroup(5, 991)
        resonance = Resonance(s0=(0, 1, 2), s1=(0, 3, 4))
        expanded = algebra.s_expand(s991).resonant_subalgebra(resonance, (0,), (1, 2))

        def run() -> object:
            result: object = None
            for _ in range(loops):
                result = (expanded.det(), expanded.eigenvalues(), expanded.signature())
            return result

        return run

    return make


def benchmarks() -> Iterator[Bench]:
    yield "resonance_scan_order5", _resonance_scan(5, filtered=False)
    yield "resonance_scan_order5_filtered", _resonance_scan(5, filtered=True)
    yield "resonance_scan_order6", _resonance_scan(6, filtered=False)
    yield "resonance_scan_order6_filtered", _resonance_scan(6, filtered=True)
    yield "catalogue_props_order6", _catalogue_props(6)
    yield "isomorphism_test_order5", _isomorphism_test(5)
    yield "isomorphism_test_order6", _isomorphism_test(6)
    yield "expansion_pipeline_x1000", _expansion_pipeline(1000)
    yield "metric_reuse_x1000", _metric_reuse(1000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only", default="", help="run only benchmarks containing this substring")
    parser.add_argument("--json", default="", help="write results to this JSON file")
    args = parser.parse_args()

    results: dict[str, float] = {}
    for name, factory in benchmarks():
        if args.only and args.only not in name:
            continue
        best = float("inf")
        checksum: object = None
        for _ in range(max(1, args.repeats)):
            run = factory()
            start = time.perf_counter()
            checksum = run()
            best = min(best, time.perf_counter() - start)
        results[name] = best
        print(f"{name:35s} {best * 1000:12.2f} ms   ({_summary(checksum)})")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.json}")


def _summary(value: object) -> str:
    if isinstance(value, tuple):
        return ", ".join(_summary(v) for v in value)
    if isinstance(value, np.ndarray):
        return f"array[{value.size}]"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


if __name__ == "__main__":
    main()
