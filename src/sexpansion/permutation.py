"""Permutations of semigroup elements (isomorphisms).

Replaces the permutation role of the Java ``SetS`` class. The group of
isomorphisms between semigroups of order *n* is the symmetric group on *n*
elements (paper, Section 2.1.1).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Permutation:
    """A permutation of ``{0, ..., n-1}`` given by its image tuple.

    ``image[i]`` is the element that ``i`` is mapped to, i.e. sigma(i).
    This is the 0-based version of the paper's ``(alpha_1 ... alpha_n)``
    notation (Equation 33).
    """

    image: tuple[int, ...]

    def __post_init__(self) -> None:
        if sorted(self.image) != list(range(len(self.image))):
            raise ValueError(f"not a permutation of 0..{len(self.image) - 1}: {self.image}")

    @property
    def degree(self) -> int:
        """Number of elements the permutation acts on."""
        return len(self.image)

    def __call__(self, i: int) -> int:
        """Return sigma(i)."""
        return self.image[i]

    def inverse(self) -> Permutation:
        """Return the inverse permutation (Java ``inversePermutation``)."""
        inv = [0] * len(self.image)
        for i, v in enumerate(self.image):
            inv[v] = i
        return Permutation(tuple(inv))

    @staticmethod
    def identity(n: int) -> Permutation:
        """The identity permutation on ``n`` elements."""
        return Permutation(tuple(range(n)))

    @staticmethod
    def all_of_degree(n: int) -> Iterator[Permutation]:
        """Iterate over all ``n!`` permutations (Java ``allPermutations``)."""
        for image in itertools.permutations(range(n)):
            yield Permutation(image)

    def __str__(self) -> str:
        return "(" + " ".join(str(v) for v in self.image) + ")"
