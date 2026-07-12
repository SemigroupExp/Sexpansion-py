"""Selector tensors ``K_ab^c`` and semigroup metrics (paper, Sections 2.1 and 4.2).

The selector of a semigroup is defined by ``lambda_a * lambda_b =
K_ab^c lambda_c`` with components 0 or 1 (Equation 2). The single
:class:`Selector` class replaces the Java ``Selector``,
``SelectorReduced``, ``SelectorResonant`` and ``SelectorResonantReduced``
hierarchy: the reduced/resonant variants are exposed as methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ._typing import BoolArray, FloatArray, IntArray
    from .resonance import Resonance
    from .semigroup import Semigroup


@dataclass(frozen=True, eq=False)
class Selector:
    """The 0/1 tensor ``k[a, b, c] = K_ab^c`` of a semigroup of order m."""

    k: IntArray

    def __post_init__(self) -> None:
        k = np.asarray(self.k, dtype=np.int_)
        if k.ndim != 3 or len(set(k.shape)) != 1:
            raise ValueError(f"selector must be a cubic 3-d array, got shape {k.shape}")
        k.setflags(write=False)
        object.__setattr__(self, "k", k)

    @classmethod
    def from_semigroup(cls, semigroup: Semigroup) -> Selector:
        """Build the selector of ``semigroup`` (Java ``Semigroup.getSelector``)."""
        m = semigroup.order
        k = np.zeros((m, m, m), dtype=np.int_)
        a, b = np.indices((m, m))
        k[a, b, semigroup.table] = 1
        return cls(k)

    @property
    def order(self) -> int:
        """Order of the underlying semigroup."""
        return int(self.k.shape[0])

    def metric(self) -> FloatArray:
        """Semigroup metric ``g_ab = K_ac^d K_bd^c`` (Equation 21 of the paper)."""
        kf = self.k.astype(np.float64)
        result: FloatArray = np.einsum("acd,bdc->ab", kf, kf)
        return result

    def reduced_metric(self, zero: int) -> FloatArray:
        """Metric of the 0_S-reduced selector (Java ``SelectorReduced``).

        Components touching the zero element are dropped before the
        contraction, and the zero row/column is removed from the result.
        """
        masked = self._masked(np.arange(self.order) != zero)
        full = masked.metric()
        keep = [i for i in range(self.order) if i != zero]
        return full[np.ix_(keep, keep)]

    def resonant_metric(self, resonance: Resonance) -> FloatArray:
        """Block metric of a resonant decomposition (Java ``SelectorResonant``).

        Returns a ``(|S0|+|S1|)`` square matrix, block-diagonal in the
        ``S0`` and ``S1`` sectors, where the contractions run only over
        the index combinations allowed by the resonance condition.
        """
        kf = self.k.astype(np.float64)
        i0 = np.array(resonance.s0, dtype=np.int_)
        i1 = np.array(resonance.s1, dtype=np.int_)
        # S0 block: contraction indices (c, d) in (S0 x S0) u (S1 x S1).
        g00: FloatArray = np.einsum(
            "acd,bdc->ab", kf[np.ix_(i0, i0, i0)], kf[np.ix_(i0, i0, i0)]
        ) + np.einsum("acd,bdc->ab", kf[np.ix_(i0, i1, i1)], kf[np.ix_(i0, i1, i1)])
        # S1 block: contraction indices (c, d) in (S0 x S1) u (S1 x S0).
        cross: FloatArray = np.einsum("akm,bmk->ab", kf[np.ix_(i1, i0, i1)], kf[np.ix_(i1, i1, i0)])
        g11 = cross + cross.T
        n0, n1 = len(i0), len(i1)
        result = np.zeros((n0 + n1, n0 + n1), dtype=np.float64)
        result[:n0, :n0] = g00
        result[n0:, n0:] = g11
        return result

    def resonant_reduced_metric(self, resonance: Resonance, zero: int) -> FloatArray:
        """Resonant metric after 0_S-reduction (Java ``SelectorResonantReduced``).

        Components touching the zero element are dropped before the
        contraction, and the rows/columns of the zero element are removed
        from both blocks.
        """
        masked = self._masked(np.arange(self.order) != zero)
        full = masked.resonant_metric(resonance)
        labels = list(resonance.s0) + list(resonance.s1)
        keep = [i for i, label in enumerate(labels) if label != zero]
        return full[np.ix_(keep, keep)]

    def expanded_metric(self, algebra_metric: FloatArray) -> FloatArray:
        """Killing metric of the S-expanded algebra, ``g_E = g (x) g_S``.

        Kronecker product of the algebra's Cartan-Killing metric with the
        semigroup metric (Equation 24 of the paper; Java
        ``Selector.expandedMetric``). Row order is generator-major:
        row ``i * m + a`` corresponds to ``X_(i,a)``.
        """
        kron = np.kron(np.asarray(algebra_metric, dtype=np.float64), self.metric())
        return np.asarray(kron, dtype=np.float64)

    def _masked(self, keep: BoolArray) -> Selector:
        """Selector with all components touching excluded elements zeroed."""
        mask3 = (
            keep[:, np.newaxis, np.newaxis]
            & keep[np.newaxis, :, np.newaxis]
            & keep[np.newaxis, np.newaxis, :]
        )
        return Selector(np.where(mask3, self.k, 0))
