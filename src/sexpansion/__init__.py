"""S-expansions of Lie algebras with finite abelian semigroups.

Python port of the Sexpansion Java library described in:

    C. Inostroza, I. Kondrashuk, N. Merino, F. Nadal,
    "A Java Library to Perform S-Expansions of Lie Algebras",
    Axioms (2025).

Quickstart::

    from sexpansion import LieAlgebra, Semigroup, find_all_resonances

    sl2 = LieAlgebra.sl2()
    s = Semigroup([[0, 1, 2, 3], [1, 2, 3, 3], [2, 3, 3, 3], [3, 3, 3, 3]])
    expanded = sl2.s_expand(s)
    print(expanded.cartan_killing_metric())
"""

from .algebra import LieAlgebra
from .database import (
    SEMIGROUP_COUNTS,
    load_all_semigroups,
    load_semigroup,
    load_semigroups,
)
from .expansion import ExpandedAlgebra, s_expand
from .permutation import Permutation
from .resonance import (
    Resonance,
    find_all_resonances,
    find_resonances,
    is_resonant,
    is_resonant_filtered,
)
from .selector import Selector
from .semigroup import IsoResult, Semigroup

__version__ = "1.0.0"

__all__ = [
    "SEMIGROUP_COUNTS",
    "ExpandedAlgebra",
    "IsoResult",
    "LieAlgebra",
    "Permutation",
    "Resonance",
    "Selector",
    "Semigroup",
    "__version__",
    "find_all_resonances",
    "find_resonances",
    "is_resonant",
    "is_resonant_filtered",
    "load_all_semigroups",
    "load_semigroup",
    "load_semigroups",
    "s_expand",
]
