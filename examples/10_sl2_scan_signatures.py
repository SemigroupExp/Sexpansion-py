"""Scan all expansions of sl(2) of one order for semi-simplicity and signature.

Port of the Java examples ``III_sl2_SExp_ord2_to_6_console`` and
``III_Signature_Sem_ord*`` (paper, Section 5.6). For each commutative
semigroup of the chosen order, the S-expanded algebra G_S is computed and
classified by the determinant and eigenvalue signature of its
Cartan-Killing metric. Expected for order 3 (paper, Table 5): 12
commutative semigroups, 5 expansions preserving semi-simplicity.
"""

from sexpansion import LieAlgebra, load_semigroups

ORDER = 3

sl2 = LieAlgebra.sl2()
n_commutative = 0
n_semisimple = 0
for semigroup in load_semigroups(ORDER):
    if not semigroup.is_commutative:
        continue
    n_commutative += 1
    expanded = sl2.s_expand(semigroup)
    determinant = expanded.det()
    if abs(determinant) > 1e-6:
        n_semisimple += 1
        print(
            f"Semigroup #{semigroup.sem_id}: semi-simple expansion, "
            f"det = {determinant:.6g}, signature (+,-,0) = {expanded.signature()}"
        )

print(
    f"\nThere are {n_commutative} commutative semigroups of order {ORDER} "
    f"and {n_semisimple} expansions that give a semi-simple algebra."
)
