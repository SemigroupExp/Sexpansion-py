"""Resonant decompositions of semigroups (paper, Sections 2.5.1 and 3.4).

A resonant decomposition ``S = S0 U S1`` satisfies::

    S0 * S0 c S0,   S0 * S1 c S1,   S1 * S1 c S0

and matches the graded structure ``G = V0 + V1`` of the Lie algebra so
that ``(S0 x V0) + (S1 x V1)`` is the resonant subalgebra of the
S-expanded algebra.

The search functions replace the Java generate-and-test over all
``combinations x combinations`` pairs with a bitmask enumeration: every
subset of a semigroup of order ``n <= 6`` fits in an ``n``-bit integer,
products of subsets come from a precomputed lookup table, and each
closure condition of Equation 26 is a single integer comparison. Only
covering pairs are enumerated (``S1`` must contain the complement of
``S0``), and candidates whose ``S0`` is not closed are pruned before any
``S1`` is considered. The results are identical to the brute-force scan,
in the same order.
"""

from __future__ import annotations

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
    rows: list[list[int]] = semigroup.table.tolist()
    return (
        all(rows[a][b] in set0 for a in set0 for b in set0)
        and all(rows[a][b] in set1 for a in set0 for b in set1)
        and all(rows[a][b] in set0 for a in set1 for b in set1)
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
    return _sorted_resonances(
        (s0, s1)
        for s0, s1 in _resonant_mask_pairs(semigroup, filtered=filtered)
        if s0.bit_count() == n0 and s1.bit_count() == n1
    )


def find_all_resonances(semigroup: Semigroup, *, filtered: bool = False) -> list[Resonance]:
    """All resonant decompositions with subset sizes ``1 .. order-1``.

    Port of the Java ``findAllResonances`` / ``findAllResonancesF`` /
    ``findAllResonancesF2`` (the three variants collapse into the
    ``filtered`` flag). Like the Java original, subset sizes run from 1
    to ``order - 1``.
    """
    order = semigroup.order
    return _sorted_resonances(
        (s0, s1)
        for s0, s1 in _resonant_mask_pairs(semigroup, filtered=filtered)
        if 1 <= s0.bit_count() < order and 1 <= s1.bit_count() < order
    )


def _resonant_mask_pairs(semigroup: Semigroup, *, filtered: bool) -> list[tuple[int, int]]:
    """All covering pairs ``(S0, S1)`` satisfying Equation 26, as bitmasks.

    Covers every subset size including the degenerate ``S0`` empty/full
    cases; the public wrappers apply their size constraints on top.
    """
    n = semigroup.order
    full = (1 << n) - 1
    rows: list[list[int]] = semigroup.table.tolist()

    # prod_row[a][B] is the bitmask of {a*b : b in B}, filled in O(n 2^n)
    # by extending each subset B from B minus its lowest bit.
    prod_row = [[0] * (full + 1) for _ in range(n)]
    for a, row in enumerate(rows):
        pa = prod_row[a]
        for mask in range(1, full + 1):
            low = mask & -mask
            pa[mask] = pa[mask ^ low] | (1 << row[low.bit_length() - 1])

    def product(a_mask: int, b_mask: int) -> int:
        result = 0
        rest = a_mask
        while rest:
            low = rest & -rest
            result |= prod_row[low.bit_length() - 1][b_mask]
            rest ^= low
        return result

    zero = semigroup.find_zero()
    nonzero = full & ~(1 << zero) if zero is not None else full

    pairs: list[tuple[int, int]] = []
    for s0 in range(full + 1):
        if product(s0, s0) | s0 != s0:  # S0 * S0 not a subset of S0
            continue
        complement = full & ~s0
        overlap = s0
        while True:  # S1 = complement | overlap for every overlap subset of S0
            s1 = complement | overlap
            if (
                (not filtered or not (s0 & s1 & nonzero))
                and product(s0, s1) | s1 == s1
                and product(s1, s1) | s0 == s0
            ):
                pairs.append((s0, s1))
            if not overlap:
                break
            overlap = (overlap - 1) & s0
    return pairs


def _sorted_resonances(pairs: Iterable[tuple[int, int]]) -> list[Resonance]:
    """Masks to :class:`Resonance` objects in the historical scan order."""
    resonances = [Resonance(_bits(s0), _bits(s1)) for s0, s1 in pairs]
    resonances.sort(key=lambda r: (len(r.s0), len(r.s1), r.s0, r.s1))
    return resonances


def _bits(mask: int) -> tuple[int, ...]:
    return tuple(i for i in range(mask.bit_length()) if mask >> i & 1)
