"""Adjoint representation, Casimir operator and abelian subalgebras of sl(2).

Port of the Java example ``II_AdjointRep_Casimirs_ex`` (paper, Section 5.5).
"""

from sexpansion import LieAlgebra
from sexpansion.reports import adjoint_report, metric_report

sl2 = LieAlgebra.sl2()

print("Adjoint representation of sl(2):")
print(adjoint_report(sl2, one_based=True))

print("\nCartan-Killing metric:")
print(metric_report(sl2.cartan_killing_metric()))

print("\nQuadratic Casimir operator (adjoint representation):")
print(metric_report(sl2.casimir(), precision=3))
print("Casimir eigenvalues:", sl2.casimir_eigenvalues())

print("\nMaximal abelian subalgebra (generator indices):", sl2.maximal_abelian_subalgebra())
