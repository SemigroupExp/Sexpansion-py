"""Resonant decompositions of semigroups (paper, Sections 2.5.1 and 3.4).

A resonant decomposition ``S = S0 U S1`` satisfies::

    S0 * S0 c S0,   S0 * S1 c S1,   S1 * S1 c S0

and matches the graded structure ``G = V0 + V1`` of the Lie algebra so
that ``(S0 x V0) + (S1 x V1)`` is the resonant subalgebra of the
S-expanded algebra.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass

from .semigroup import Semigroup


@dataclass(frozen=True, slots=True)
class Resonance:
    """A resonant decomposition ``(S0, S1)`` of a semigroup (0-based elements)."""

    s0: tuple[int, ...]
    s1: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "s0", tuple(sorted(self.s0)))
        object.__setattr__(self, "s1", tuple(sorted(self.s1)))


def is_resonant(semigroup: Semigroup, s0: Iterable[int], s1: Iterable[int]) -> bool:
    """True if ``(s0, s1)`` is a resonant decomposition of ``semigroup``.

    Port of the Java ``isResonant``: checks that the union covers the
    whole semigroup (Java ``fillTheSpace``) and the three closure
    conditions of Equation 26 of the paper.
    """
    set0, set1 = frozenset(s0), frozenset(s1)
    if set0 | set1 != frozenset(range(semigroup.order)):
        return False
    table = semigroup.table
    return (
        all(int(table[a, b]) in set0 for a in set0 for b in set0)
        and all(int(table[a, b]) in set1 for a in set0 for b in set1)
        and all(int(table[a, b]) in set0 for a in set1 for b in set1)
    )


def is_resonant_filtered(semigroup: Semigroup, s0: Iterable[int], s1: Iterable[int]) -> bool:
    """True if ``(s0, s1)`` is resonant and shares no non-zero element.

    Port of the Java ``isResonatF`` (typo fixed): as :func:`is_resonant`,
    but additionally the intersection of ``s0`` and ``s1`` may contain at
    most the semigroup's zero element.
    """
    set0, set1 = frozenset(s0), frozenset(s1)
    return not _shares_nonzero_element(semigroup, set0, set1) and is_resonant(semigroup, set0, set1)


def _shares_nonzero_element(semigroup: Semigroup, s0: frozenset[int], s1: frozenset[int]) -> bool:
    """Java ``hasNonZeroRepeatingElement``: shared elements other than the zero."""
    shared = s0 & s1
    zero = semigroup.find_zero()
    if zero is not None:
        shared = shared - {zero}
    return bool(shared)


def find_resonances(
    semigroup: Semigroup, n0: int, n1: int, *, filtered: bool = False
) -> list[Resonance]:
    """All resonant decompositions with ``|S0| = n0`` and ``|S1| = n1``.

    Port of the Java ``findResonances`` / ``findResonancesF``
    (``filtered=True`` applies the non-zero-overlap filter).
    """
    elements = range(semigroup.order)
    predicate = is_resonant_filtered if filtered else is_resonant
    return [
        Resonance(s0, s1)
        for s0 in itertools.combinations(elements, n0)
        for s1 in itertools.combinations(elements, n1)
        if predicate(semigroup, s0, s1)
    ]


def find_all_resonances(semigroup: Semigroup, *, filtered: bool = False) -> list[Resonance]:
    """All resonant decompositions with subset sizes ``1 .. order-1``.

    Port of the Java ``findAllResonances`` / ``findAllResonancesF`` /
    ``findAllResonancesF2`` (the three variants collapse into the
    ``filtered`` flag). Like the Java original, subset sizes run from 1
    to ``order - 1``.
    """
    return [
        resonance
        for n0 in range(1, semigroup.order)
        for n1 in range(1, semigroup.order)
        for resonance in find_resonances(semigroup, n0, n1, filtered=filtered)
    ]
