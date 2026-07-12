"""Find every resonant decomposition over the whole catalogue of one order.

Port of the Java example ``II_findAllResonances_console_ord4`` (paper,
Section 5.3). Expected result for order 4 (paper, Table 5): 48 semigroups
with at least one resonance, 124 resonances in total.
"""

from sexpansion import find_all_resonances, load_semigroups

ORDER = 4

n_semigroups = 0
n_resonances = 0
for semigroup in load_semigroups(ORDER):
    if not semigroup.is_commutative:
        continue
    resonances = find_all_resonances(semigroup)
    if resonances:
        n_semigroups += 1
        n_resonances += len(resonances)
        print(f"The semigroup #{semigroup.sem_id} has {len(resonances)} resonances")

print(
    f"\nThere are {n_semigroups} semigroups with at least one resonance "
    f"and in total {n_resonances} different resonances."
)
