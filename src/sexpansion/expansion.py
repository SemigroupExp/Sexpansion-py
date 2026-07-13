"""The S-expansion of a Lie algebra and its derived algebras.

This module replaces the Java ``StructureConstantSetExpanded`` class and
its three subclasses (``...Reduced``, ``...Resonant``,
``...ResonantReduced``). Following the paper (Sections 4.3-4.6), the
derived algebras are obtained by chaining transforms::

    expanded = algebra.s_expand(semigroup)                    # G_S = S (x) G
    resonant = expanded.resonant_subalgebra(res, v0, v1)      # G_S,R
    reduced  = expanded.zero_reduced()                        # G_S,red
    both     = expanded.resonant_subalgebra(res, v0, v1).zero_reduced()

The structure constants of the expanded algebra are
``C_(i,a)(j,b)^(k,c) = K_ab^c C_ij^k`` (Equation 11), stored as a 6-d
tensor ``c[i, a, j, b, k, c]`` of shape ``(n, m, n, m, n, m)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from .resonance import Resonance, is_resonant

if TYPE_CHECKING:
    from ._typing import BoolArray, FloatArray
    from .algebra import LieAlgebra
    from .semigroup import Semigroup


def s_expand(algebra: LieAlgebra, semigroup: Semigroup) -> ExpandedAlgebra:
    """Perform the S-expansion ``G_S = S (x) G`` (paper, Theorem 3.1 of [19]).

    Port of the Java ``Semigroup.getExpandedStructureConstant``; the six
    nested loops collapse into a single einsum.
    """
    k = semigroup.selector().k.astype(np.float64)
    tensor: FloatArray = np.einsum("abg,ijk->iajbkg", k, algebra.constants)
    active = np.ones((algebra.n, semigroup.order), dtype=np.bool_)
    return ExpandedAlgebra(tensor=tensor, semigroup=semigroup, active=active)


@dataclass(frozen=True, eq=False)
class ExpandedAlgebra:
    """An S-expanded Lie algebra, possibly restricted to a smaller algebra.

    Attributes
    ----------
    tensor:
        Structure constants ``c[i, a, j, b, k, c] = C_(i,a)(j,b)^(k,c)``.
        Entries outside the active sector are zero.
    semigroup:
        The semigroup used for the expansion.
    active:
        Boolean mask of shape ``(n, m)``; ``active[i, a]`` tells whether
        the generator ``X_(i,a)`` belongs to the algebra. All generators
        are active for a plain expansion; the resonant/reduced transforms
        switch generators off.
    resonance:
        The resonant decomposition used, if any.
    zero:
        The semigroup zero element removed by 0_S-reduction, if any.
    """

    tensor: FloatArray
    semigroup: Semigroup
    active: BoolArray
    resonance: Resonance | None = None
    grading: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    zero: int | None = field(default=None)

    def __post_init__(self) -> None:
        tensor = np.asarray(self.tensor, dtype=np.float64)
        tensor.setflags(write=False)
        object.__setattr__(self, "tensor", tensor)
        active = np.asarray(self.active, dtype=np.bool_)
        active.setflags(write=False)
        object.__setattr__(self, "active", active)

    @property
    def n(self) -> int:
        """Dimension of the original Lie algebra."""
        return int(self.tensor.shape[0])

    @property
    def m(self) -> int:
        """Order of the semigroup."""
        return int(self.tensor.shape[1])

    @property
    def n_generators(self) -> int:
        """Number of generators of the (possibly restricted) algebra."""
        return int(np.count_nonzero(self.active))

    def generators(self) -> list[tuple[int, int]]:
        """Active generators ``(i, a)`` in generator-major order.

        The flattened position of ``(i, a)`` in the unrestricted algebra
        is ``i * m + a``, matching the row order of
        :meth:`cartan_killing_metric`.
        """
        return [(int(i), int(a)) for i, a in np.argwhere(self.active)]

    def resonant_subalgebra(
        self,
        resonance: Resonance,
        v0: Sequence[int],
        v1: Sequence[int],
    ) -> ExpandedAlgebra:
        """Extract the resonant subalgebra ``G_S,R`` (paper, Section 4.4).

        ``resonance`` gives the semigroup decomposition ``S = S0 u S1``
        and ``(v0, v1)`` the graded decomposition ``G = V0 (+) V1`` of the
        original algebra. A generator ``X_(i,a)`` survives when
        ``(i in V0 and a in S0)`` or ``(i in V1 and a in S1)``.
        """
        if not is_resonant(self.semigroup, resonance.s0, resonance.s1):
            raise ValueError(f"{resonance} is not a resonant decomposition of {self.semigroup!r}")
        new_active = np.zeros_like(self.active)
        new_active[
            np.ix_(np.asarray(v0, dtype=np.intp), np.asarray(resonance.s0, dtype=np.intp))
        ] = True
        new_active[
            np.ix_(np.asarray(v1, dtype=np.intp), np.asarray(resonance.s1, dtype=np.intp))
        ] = True
        return self._restricted(
            self.active & new_active, resonance=resonance, grading=(tuple(v0), tuple(v1))
        )

    def zero_reduced(self, zero: int | None = None) -> ExpandedAlgebra:
        """Perform the 0_S-reduction ``G_S,red`` (paper, Section 4.5).

        Removes the whole ``0_S (x) G`` sector. If ``zero`` is not given,
        the semigroup's zero element is looked up automatically.
        """
        if zero is None:
            zero = self.semigroup.find_zero()
            if zero is None:
                raise ValueError("semigroup has no zero element; 0_S-reduction is not defined")
        new_active = self.active.copy()
        new_active[:, zero] = False
        return self._restricted(new_active, zero=zero)

    def cartan_killing_metric(self, *, restricted: bool | None = None) -> FloatArray:
        """Cartan-Killing metric of the expanded algebra (Equation 23).

        ``g_(i,a)(j,b) = C_(i,a)(l,g)^(k,c) C_(j,b)(k,c)^(l,g)``, a square
        matrix in generator-major order (row ``i * m + a``).

        By default (``restricted=None``) the rows/columns of inactive
        generators are dropped whenever a transform has been applied,
        matching the Java ``cartanKillingMetricPretty``; pass
        ``restricted=False`` for the full ``(n*m)`` square matrix of the
        Java ``cartanKillingMetric``.
        """
        full = self._full_metric
        if restricted is None:
            restricted = not bool(self.active.all())
        if not restricted:
            return full
        keep = np.flatnonzero(self.active.reshape(-1))
        return full[np.ix_(keep, keep)]

    @cached_property
    def _full_metric(self) -> FloatArray:
        """The unrestricted metric, computed once per (frozen) instance.

        Read-only, like :attr:`tensor`; the cache relies on the class not
        using ``__slots__``.
        """
        t = self.tensor
        full: FloatArray = np.einsum("ialgkc,jbkclg->iajb", t, t).reshape(
            self.n * self.m, self.n * self.m
        )
        full.setflags(write=False)
        return full

    def det(self) -> float:
        """Determinant of the Cartan-Killing metric.

        Non-zero iff the (restricted) expanded algebra is semi-simple
        (paper, Section 2.3).
        """
        return float(np.linalg.det(self.cartan_killing_metric()))

    def eigenvalues(self) -> FloatArray:
        """Eigenvalues of the (symmetric) Cartan-Killing metric, ascending."""
        result: FloatArray = np.linalg.eigvalsh(self.cartan_killing_metric())
        return result

    def eigenvectors(self) -> tuple[FloatArray, FloatArray]:
        """Eigenvalues and orthonormal eigenvectors of the Cartan-Killing metric.

        Returns ``(eigenvalues, eigenvectors)`` with eigenvectors in
        columns, as from ``np.linalg.eigh``.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.cartan_killing_metric())
        return eigenvalues, eigenvectors

    def signature(self, tolerance: float = 1e-9) -> tuple[int, int, int]:
        """Counts of (positive, negative, zero) metric eigenvalues.

        Used to study compactness of the expanded algebras (paper,
        Sections 2.3 and 5.6).
        """
        eigenvalues = self.eigenvalues()
        positive = int(np.count_nonzero(eigenvalues > tolerance))
        negative = int(np.count_nonzero(eigenvalues < -tolerance))
        return positive, negative, len(eigenvalues) - positive - negative

    def jacobi_violation(self) -> float:
        """Maximum violation of the Jacobi identity by the structure constants.

        For a genuine Lie algebra this is zero up to rounding; useful as a
        mathematical sanity check of the expansion and its transforms.
        """
        c = self.tensor.reshape(self.n * self.m, self.n * self.m, self.n * self.m)
        cyclic = (
            np.einsum("abe,ecd->abcd", c, c)
            + np.einsum("bce,ead->abcd", c, c)
            + np.einsum("cae,ebd->abcd", c, c)
        )
        return float(np.abs(cyclic).max())

    def _restricted(
        self,
        new_active: BoolArray,
        *,
        resonance: Resonance | None = None,
        grading: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
        zero: int | None = None,
    ) -> ExpandedAlgebra:
        """New algebra keeping only entries whose three index pairs are active."""
        tensor = self.tensor.copy()
        inactive = ~new_active
        tensor[inactive] = 0.0
        tensor[:, :, inactive] = 0.0
        tensor[:, :, :, :, inactive] = 0.0
        return ExpandedAlgebra(
            tensor=tensor,
            semigroup=self.semigroup,
            active=new_active,
            resonance=resonance if resonance is not None else self.resonance,
            grading=grading if grading is not None else self.grading,
            zero=zero if zero is not None else self.zero,
        )

    def __repr__(self) -> str:
        parts = [f"n={self.n}", f"m={self.m}", f"generators={self.n_generators}"]
        if self.resonance is not None:
            parts.append(f"resonance={self.resonance}")
        if self.zero is not None:
            parts.append(f"zero_reduced={self.zero}")
        return f"ExpandedAlgebra({', '.join(parts)})"
