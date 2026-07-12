"""Find the catalogued semigroup isomorphic to SN3 and the permutation relating them.

Port of the Java example ``I_isomorphisms_ex2`` (paper, Section 5.4).
The answer (paper, Table 4): S^42_(4), related by P#19 = (4 1 3 2) among
others.
"""

import numpy as np

from sexpansion import Permutation, Semigroup, load_semigroups

# SN3 from paper Figure 14 (0-based labels).
SN3 = Semigroup(np.array([[3, 0, 3, 3], [0, 1, 2, 3], [3, 2, 3, 3], [3, 3, 3, 3]]))

for candidate in load_semigroups(4):
    if candidate.isomorphism_test(SN3).isomorphic:
        print(f"The semigroup #{candidate.sem_id}")
        print(candidate)
        print("is isomorphic to SN3.")
        for k, sigma in enumerate(Permutation.all_of_degree(4)):
            if candidate.permute(sigma) == SN3:
                print(f"A permutation that brings #{candidate.sem_id} to SN3 is P#{k}: {sigma}")
                print(f"  its inverse is {sigma.inverse()}")
