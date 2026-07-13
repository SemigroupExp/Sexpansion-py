"""Finite discrete semigroups given by their multiplication table.

Port of the Java ``Semigroup`` class (paper, Section 3). All element
indices are 0-based: a semigroup of order ``n`` has elements
``0, ..., n-1`` and ``table[i, j]`` is the product ``i * j``. The paper
and the bundled catalogue files label elements 1..n; the database loader
converts on read, and the report helpers can print 1-based labels.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from .permutation import Permutation

if TYPE_CHECKING:
    from ._typing import IntArray
    from .selector import Selector


class IsoResult(NamedTuple):
    """Result of :meth:`Semigroup.isomorphism_test` (Java ``isotest``)."""

    isomorphic: bool
    anti_isomorphic: bool


@dataclass(frozen=True, eq=False)
class Semigroup:
    """A finite magma; use :attr:`is_associative` to check it is a semigroup.

    Parameters
    ----------
    table:
        Square multiplication table with 0-based entries:
        ``table[i, j] = i * j``.
    sem_id:
        Optional catalogue identifier of the semigroup within its order
        (the ``a`` of the paper's ``S^a_(n)`` notation).
    """

    table: IntArray
    sem_id: int | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        table = np.asarray(self.table, dtype=np.int_)
        if table.ndim != 2 or table.shape[0] != table.shape[1]:
            raise ValueError(f"multiplication table must be square, got shape {table.shape}")
        n = table.shape[0]
        if table.size and (table.min() < 0 or table.max() >= n):
            raise ValueError(f"table entries must be 0-based elements in 0..{n - 1}")
        table.setflags(write=False)
        object.__setattr__(self, "table", table)

    @property
    def order(self) -> int:
        """Number of elements of the semigroup."""
        return int(self.table.shape[0])

    def multiply(self, i: int, j: int) -> int:
        """Return the product ``i * j`` (0-based)."""
        return int(self.table[i, j])

    @property
    def is_associative(self) -> bool:
        """True if ``(i*j)*k == i*(j*k)`` for all elements."""
        t = self.table
        return bool(np.array_equal(t[t, :], t[:, t]))

    @property
    def is_commutative(self) -> bool:
        """True if ``i*j == j*i`` for all elements."""
        return bool(np.array_equal(self.table, self.table.T))

    def find_zero(self) -> int | None:
        """Return the zero (absorbing) element, or ``None`` if there is none.

        The zero element ``z`` satisfies ``x * z == z`` for every ``x``
        (paper, Section 2.1; for the commutative semigroups used in
        S-expansions this is equivalent to the two-sided condition).
        """
        return self._zero

    @cached_property
    def _zero(self) -> int | None:
        # Column z is constant-z exactly when z is absorbing. The cache
        # relies on the class not using __slots__.
        zeros = np.flatnonzero((self.table == np.arange(self.order)).all(axis=0))
        return int(zeros[0]) if zeros.size else None

    def transpose(self) -> Semigroup:
        """Semigroup with the transposed multiplication table."""
        return Semigroup(self.table.T.copy())

    def permute(self, sigma: Permutation) -> Semigroup:
        """Apply the isomorphism ``sigma`` (Java ``permuteWith``).

        The new table ``B`` satisfies
        ``B[i, j] = sigma(A[sigma^-1(i), sigma^-1(j)])`` (Equation 5 of
        the paper).
        """
        if sigma.degree != self.order:
            raise ValueError(f"permutation degree {sigma.degree} != semigroup order {self.order}")
        inv = np.array(sigma.inverse().image, dtype=np.int_)
        image = np.array(sigma.image, dtype=np.int_)
        return Semigroup(image[self.table[np.ix_(inv, inv)]])

    def all_images(self) -> Iterator[Semigroup]:
        """All semigroups isomorphic to this one (Java ``permute``)."""
        for sigma in Permutation.all_of_degree(self.order):
            yield self.permute(sigma)

    def all_anti_images(self) -> Iterator[Semigroup]:
        """All semigroups anti-isomorphic to this one (Java ``antiPermute``)."""
        return self.transpose().all_images()

    def isomorphism_test(self, other: Semigroup) -> IsoResult:
        """Check whether ``self`` and ``other`` are (anti-)isomorphic.

        Port of the Java ``isotest``, returning a named tuple instead of a
        ``boolean[2]``. Two tables are isomorphic if some permutation maps
        one onto the other (Equation 5 of the paper), and anti-isomorphic
        if some permutation maps one onto the other's transpose
        (Equation 6). Unlike the Java method, the two flags are computed
        independently (the Java anti-isomorphism branch compared
        anti-images of both semigroups, which is equivalent to the plain
        isomorphism test and could never detect a pure anti-isomorphism).
        """
        if self.order != other.order:
            return IsoResult(isomorphic=False, anti_isomorphic=False)
        isomorphic = any(img == other for img in self.all_images())
        if self.is_commutative and other.is_commutative:
            anti = isomorphic
        else:
            anti = any(img == other for img in self.all_anti_images())
        return IsoResult(isomorphic=isomorphic, anti_isomorphic=anti)

    def selector(self) -> Selector:
        """Selector tensor ``K_ab^c`` of the semigroup (Java ``getSelector``)."""
        from .selector import Selector

        return Selector.from_semigroup(self)

    def _key(self) -> bytes:
        return self.table.tobytes()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Semigroup):
            return NotImplemented
        return self.order == other.order and bool(np.array_equal(self.table, other.table))

    def __hash__(self) -> int:
        return hash((self.order, self._key()))

    def __repr__(self) -> str:
        sid = f", sem_id={self.sem_id}" if self.sem_id is not None else ""
        return f"Semigroup(order={self.order}{sid})"

    def __str__(self) -> str:
        return "\n".join(" ".join(str(v) for v in row) for row in self.table)
