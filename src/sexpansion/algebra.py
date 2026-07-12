"""Lie algebras represented by their structure constants (paper, Section 4.2).

Port of the Java ``StructureConstantSet`` class. With the convention
``[X_i, X_j] = C_ij^k X_k``, the constants are stored as a
``(n, n, n)`` tensor with ``constants[i, j, k] = C_ij^k`` (0-based
generator indices).
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ._typing import FloatArray
    from .expansion import ExpandedAlgebra
    from .semigroup import Semigroup

_TOLERANCE = 1e-12


class LieAlgebra:
    """A Lie algebra of dimension ``n`` given by its structure constants."""

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("dimension must be positive")
        self._constants = np.zeros((n, n, n), dtype=np.float64)

    @property
    def n(self) -> int:
        """Dimension (number of generators) of the algebra."""
        return self._constants.shape[0]

    @property
    def constants(self) -> FloatArray:
        """The structure constant tensor ``C_ij^k`` (read-only view)."""
        view = self._constants.view()
        view.setflags(write=False)
        return view

    def set_constant(self, i: int, j: int, k: int, value: float) -> None:
        """Set ``C_ij^k`` to ``value`` and ``C_ji^k`` to ``-value``.

        Port of the Java ``setStructureConstant``, which auto-fills the
        antisymmetric partner.
        """
        self._constants[i, j, k] = value
        self._constants[j, i, k] = -value

    def cartan_killing_metric(self) -> FloatArray:
        """Cartan-Killing metric ``g_ij = C_ik^l C_jl^k`` (Equation 22)."""
        c = self._constants
        result: FloatArray = np.einsum("ikl,jlk->ij", c, c)
        return result

    def adjoint_generators(self) -> FloatArray:
        """Adjoint representation matrices, shape ``(n, n, n)``.

        ``adjoint_generators()[i]`` is the matrix of ``ad(X_i)`` with
        entries ``M_jk = C_ik^j`` (as printed by the Java
        ``adjointGenerators`` / ``show`` methods).
        """
        result: FloatArray = self._constants.transpose(0, 2, 1).copy()
        return result

    def semisimple_adjoint_generators(self) -> tuple[FloatArray, FloatArray]:
        """Adjoint matrices with a non-degenerate eigenbasis.

        Port of the Java ``semisimpleAdjointGenerators`` and
        ``semisimpleAdjointGeneratorsReferences``: selects the generators
        whose adjoint matrix has an eigenvector matrix with
        ``|det| > 1e-10`` (i.e. it is diagonalizable over C with a
        well-conditioned eigenbasis). Returns ``(indices, matrices)``.
        """
        adjoints = self.adjoint_generators()
        indices = []
        for i, matrix in enumerate(adjoints):
            _, eigenvectors = np.linalg.eig(matrix)
            if abs(np.linalg.det(eigenvectors)) > 1e-10:
                indices.append(i)
        return np.array(indices, dtype=np.int_), adjoints[indices]

    def casimir(self) -> FloatArray:
        """Quadratic Casimir operator in the adjoint representation.

        ``M_il = C_ai^k C_bk^l g^ab`` with ``g^ab`` the inverse
        Cartan-Killing metric (Java ``casimir``). Requires a
        non-degenerate metric (semi-simple algebra).
        """
        inverse_metric = np.linalg.inv(self.cartan_killing_metric())
        c = self._constants
        result: FloatArray = np.einsum("aik,bkl,ab->il", c, c, inverse_metric)
        return result

    def casimir_eigenvalues(self) -> FloatArray:
        """Real parts of the eigenvalues of the Casimir operator."""
        result: FloatArray = np.real(np.linalg.eigvals(self.casimir()))
        return result

    def generators_commute(self, i: int, j: int) -> bool:
        """True if ``[X_i, X_j] = 0``."""
        return bool(np.all(np.abs(self._constants[i, j]) < _TOLERANCE))

    def is_abelian_subset(self, generators: tuple[int, ...]) -> bool:
        """True if all given generators commute pairwise (Java
        ``generatorsSelfCommute``)."""
        return all(self.generators_commute(a, b) for a, b in itertools.combinations(generators, 2))

    def maximal_abelian_subalgebra(self) -> tuple[int, ...]:
        """A maximal abelian subset of the generators.

        The Java ``maximalAbelianSubalgebra`` had a non-terminating loop
        and an out-of-range index; this is the corrected algorithm: try
        subset sizes from ``n`` down to 1 and return the first pairwise
        commuting subset found.
        """
        for size in range(self.n, 0, -1):
            for subset in itertools.combinations(range(self.n), size):
                if self.is_abelian_subset(subset):
                    return subset
        return ()

    def s_expand(self, semigroup: Semigroup) -> ExpandedAlgebra:
        """S-expand this algebra with ``semigroup`` (paper, Section 4.3)."""
        from .expansion import s_expand

        return s_expand(self, semigroup)

    @classmethod
    def sl2(cls) -> LieAlgebra:
        """The sl(2, R) algebra in the basis used throughout the paper.

        ``[X_1, X_2] = -2 X_3``, ``[X_1, X_3] = 2 X_2``,
        ``[X_2, X_3] = 2 X_1`` (Equation 34; 0-based indices here).
        """
        algebra = cls(3)
        algebra.set_constant(0, 1, 2, -2.0)
        algebra.set_constant(0, 2, 1, 2.0)
        algebra.set_constant(1, 2, 0, 2.0)
        return algebra

    def __repr__(self) -> str:
        nonzero = int(np.count_nonzero(self._constants))
        return f"LieAlgebra(n={self.n}, nonzero_constants={nonzero})"
