"""List all permutations of n elements with their inverses.

Port of the Java example ``I_Permutations_and_inverses_console_n4``
(paper, Section 5.4).
"""

from sexpansion import Permutation

N = 4

permutations = list(Permutation.all_of_degree(N))
index = {p: i for i, p in enumerate(permutations)}
for i, p in enumerate(permutations):
    inverse = p.inverse()
    print(f"Permutation P#{i}: {p}")
    print(f"  inverse (P#{i})^-1 = P#{index[inverse]}: {inverse}")
