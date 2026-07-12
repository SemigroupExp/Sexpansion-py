"""Full S-expansion pipeline: sl(2) expanded with the semigroup S^991_(5).

Port of the Java example ``II_SExp_sl2_S991`` (paper, Section 5.5).
Computes the expanded algebra, the resonant subalgebra, the 0_S-reduced
algebra and the 0_S-reduction of the resonant subalgebra, with their
commutators, structure constants, Cartan-Killing metrics and
determinants. All four levels preserve semi-simplicity (det != 0); the
resonant-reduced algebra is su(2) (+) sl(2,R) (paper, Section 5.6).
"""

from sexpansion import LieAlgebra, Resonance, load_semigroup
from sexpansion.reports import commutator_table, metric_report, structure_constants_report

sl2 = LieAlgebra.sl2()
s991 = load_semigroup(5, 991)

print("Semigroup S991 (0-based table):")
print(s991)
print("Zero element:", s991.find_zero())

# Resonant decomposition (paper: S0 = {l1,l2,l3}, S1 = {l1,l4,l5}) and the
# sl(2) grading V0 = {X1}, V1 = {X2, X3}, all converted to 0-based.
resonance = Resonance(s0=(0, 1, 2), s1=(0, 3, 4))
v0, v1 = (0,), (1, 2)

expanded = sl2.s_expand(s991)
resonant = expanded.resonant_subalgebra(resonance, v0, v1)
reduced = expanded.zero_reduced()
resonant_reduced = resonant.zero_reduced()

for title, algebra in [
    ("Expanded algebra G_S", expanded),
    ("Resonant subalgebra G_S,R", resonant),
    ("Reduced algebra G_S,red", reduced),
    ("Reduction of the resonant subalgebra G_S,R,red", resonant_reduced),
]:
    print(f"\n{'=' * 60}\n{title}: {algebra.n_generators} generators")
    print("\nNon-vanishing commutators:")
    print(commutator_table(algebra, one_based=True))
    print("\nNon-vanishing structure constants:")
    print(structure_constants_report(algebra, one_based=True))
    print("\nCartan-Killing metric:")
    print(metric_report(algebra.cartan_killing_metric(), precision=0))
    print(f"det = {algebra.det():.6g}   signature (+,-,0) = {algebra.signature()}")
