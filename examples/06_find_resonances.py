"""Find all resonant decompositions of specific semigroups from the literature.

Port of the Java example ``II_findresonances_ex_console`` (paper,
Section 5.2). The semigroups S_E^(2), S_K^(3), S_N1..S_N3 were used to
obtain Bianchi algebras as S-expansions.
"""

import numpy as np

from sexpansion import Semigroup, find_all_resonances

# Tables from paper Figures 5 and 14, 0-based labels.
SEMIGROUPS = {
    "S_E2": [[0, 1, 2, 3], [1, 2, 3, 3], [2, 3, 3, 3], [3, 3, 3, 3]],
    "S_N3": [[3, 0, 3, 3], [0, 1, 2, 3], [3, 2, 3, 3], [3, 3, 3, 3]],
}

for name, table in SEMIGROUPS.items():
    resonances = find_all_resonances(Semigroup(np.array(table)))
    print(f"The semigroup {name} has {len(resonances)} resonances:")
    for k, resonance in enumerate(resonances, start=1):
        print(f"  Resonance #{k}: S0 = {resonance.s0}, S1 = {resonance.s1}")
