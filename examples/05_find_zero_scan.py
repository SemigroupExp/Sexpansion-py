"""List the commutative semigroups of a given order that have a zero element.

Port of the Java example ``II_findzero_console_ord4`` (paper, Section 5.2).
Expected count for order 4 (paper, Table 5): 39.
"""

from sexpansion import load_semigroups
from sexpansion.reports import semigroup_report

ORDER = 4

count = 0
for semigroup in load_semigroups(ORDER):
    zero = semigroup.find_zero()
    if semigroup.is_commutative and zero is not None:
        count += 1
        print(f"#{semigroup.sem_id}")
        print(semigroup_report(semigroup, one_based=True))
        print(f"The zero element is {zero + 1} (1-based label)\n")
print(f"Number of semigroups with zero element: {count}")
