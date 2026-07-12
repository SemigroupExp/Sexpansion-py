"""Count the commutative semigroups of each catalogued order.

Port of the Java example ``I_commutative_ord2_to_6`` (paper, Section 5.1).
Expected counts (paper, Table 5): 3, 12, 58, 325, 2143.
"""

from sexpansion import SEMIGROUP_COUNTS, load_semigroups

for order in SEMIGROUP_COUNTS:
    commutative = [sg for sg in load_semigroups(order) if sg.is_commutative]
    print(
        f"Order {order}: {len(commutative)} of {SEMIGROUP_COUNTS[order]} "
        "non-isomorphic semigroups are commutative"
    )
