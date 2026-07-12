"""Human-readable reports (replacements for the Java ``show*`` methods).

All functions return strings instead of printing. Indices are shown
0-based by default; pass ``one_based=True`` to reproduce the labelling of
the paper and the Java library outputs (lambda_1 .. lambda_n).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ._typing import FloatArray
    from .algebra import LieAlgebra
    from .expansion import ExpandedAlgebra
    from .selector import Selector
    from .semigroup import Semigroup


def semigroup_report(semigroup: Semigroup, *, one_based: bool = False) -> str:
    """Multiplication table of a semigroup (Java ``toElegantReport``)."""
    shift = 1 if one_based else 0
    return "\n".join(" ".join(str(int(v) + shift) for v in row) for row in semigroup.table)


def selector_report(selector: Selector, *, one_based: bool = False) -> str:
    """The m adjoint matrices ``Adj[lambda_a] = (K_ab^c)`` (Java ``Selector.show``)."""
    shift = 1 if one_based else 0
    blocks = []
    for a in range(selector.order):
        rows = "\n".join(
            " ".join(str(int(v)) for v in selector.k[a, b]) for b in range(selector.order)
        )
        blocks.append(f"Adj[lambda_{a + shift}] = (K_{{{a + shift},b}}^c) =\n{rows}")
    return "\n\n".join(blocks)


def adjoint_report(algebra: LieAlgebra, *, one_based: bool = False) -> str:
    """The n adjoint matrices of a Lie algebra (Java ``StructureConstantSet.show``)."""
    shift = 1 if one_based else 0
    blocks = []
    for i, matrix in enumerate(algebra.adjoint_generators()):
        rows = "\n".join(" ".join(_fmt(v) for v in row) for row in matrix)
        blocks.append(f"Adj[X_{i + shift}] = (C_{{{i + shift},j}}^k) =\n{rows}")
    return "\n\n".join(blocks)


def generators_report(expanded: ExpandedAlgebra, *, one_based: bool = False) -> str:
    """The generators ``Y_l = X_(i,a)`` of a (restricted) expanded algebra."""
    shift = 1 if one_based else 0
    lines = [
        f"Y_{index + shift} = X_({i + shift},{a + shift})"
        for index, (i, a) in enumerate(expanded.generators())
    ]
    return "\n".join(lines)


def commutator_table(expanded: ExpandedAlgebra, *, one_based: bool = False) -> str:
    """Non-vanishing commutators ``[X_(i,a), X_(j,b)] = C X_(k,c)``.

    Port of the Java ``showCommut`` / ``showCommutRes`` / ``showCommutRed``
    / ``showCommutResRed``: only pairs with ``i < j`` are listed.
    """
    shift = 1 if one_based else 0
    lines = []
    for (i, a), (j, b) in _generator_pairs(expanded):
        terms = [
            f"{_fmt(expanded.tensor[i, a, j, b, k, c])} X_({k + shift},{c + shift})"
            for k, c in np.argwhere(expanded.tensor[i, a, j, b] != 0.0)
        ]
        if terms:
            lines.append(
                f"[X_({i + shift},{a + shift}), X_({j + shift},{b + shift})] = " + " + ".join(terms)
            )
    return "\n".join(lines)


def structure_constants_report(expanded: ExpandedAlgebra, *, one_based: bool = False) -> str:
    """Non-vanishing structure constants ``C_(i,a)(j,b)^(k,c)``.

    Port of the Java ``showSC*`` family: only pairs with ``i < j`` are
    listed.
    """
    shift = 1 if one_based else 0
    lines = []
    for (i, a), (j, b) in _generator_pairs(expanded):
        for k, c in np.argwhere(expanded.tensor[i, a, j, b] != 0.0):
            value = expanded.tensor[i, a, j, b, k, c]
            lines.append(
                f"C_({i + shift},{a + shift})({j + shift},{b + shift})"
                f"^({k + shift},{c + shift}) = {_fmt(value)}"
            )
    return "\n".join(lines)


def metric_report(metric: FloatArray, *, precision: int = 2) -> str:
    """Fixed-point rendering of a metric matrix (replaces Jama ``Matrix.print``)."""
    matrix = np.asarray(metric, dtype=np.float64)
    width = max(
        (len(f"{v:.{precision}f}") for v in matrix.reshape(-1)),
        default=1,
    )
    return "\n".join(" ".join(f"{v:{width}.{precision}f}" for v in row) for row in matrix)


def _generator_pairs(
    expanded: ExpandedAlgebra,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Ordered pairs of active generators with ``i < j`` (Java convention)."""
    generators = expanded.generators()
    return [((i, a), (j, b)) for i, a in generators for j, b in generators if i < j]


def _fmt(value: float) -> str:
    """Render floats that are whole numbers without a trailing ``.0``-noise."""
    if float(value).is_integer():
        return str(int(value))
    return str(float(value))
