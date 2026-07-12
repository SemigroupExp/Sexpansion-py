"""Check associativity of hand-entered multiplication tables.

Port of the Java example ``I_associative_checking`` (paper, Section 5.1).
Tables are entered with 0-based element labels.
"""

import numpy as np

from sexpansion import Semigroup

TABLES = {
    # Paper Figure 8 (converted to 0-based labels).
    "S_ex1": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
    "S_ex2": [[0, 1, 2], [1, 0, 1], [2, 1, 2]],
    # Paper Figure 5: the semigroup S_E^(2).
    "S_E2": [[0, 1, 2, 3], [1, 2, 3, 3], [2, 3, 3, 3], [3, 3, 3, 3]],
}

for name, table in TABLES.items():
    semigroup = Semigroup(np.array(table))
    verdict = "is" if semigroup.is_associative else "is not"
    print(f"The semigroup {name} {verdict} associative")
