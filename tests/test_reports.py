import numpy as np

from sexpansion import LieAlgebra, Semigroup
from sexpansion.reports import (
    adjoint_report,
    commutator_table,
    generators_report,
    metric_report,
    selector_report,
    semigroup_report,
    structure_constants_report,
)

S_E2 = Semigroup(np.array([[0, 1, 2, 3], [1, 2, 3, 3], [2, 3, 3, 3], [3, 3, 3, 3]]))


def test_semigroup_report_one_based_matches_paper():
    assert semigroup_report(S_E2, one_based=True).splitlines()[0] == "1 2 3 4"
    assert semigroup_report(S_E2).splitlines()[0] == "0 1 2 3"


def test_selector_report_mentions_all_elements():
    report = selector_report(S_E2.selector(), one_based=True)
    for a in range(1, 5):
        assert f"Adj[lambda_{a}]" in report


def test_adjoint_report():
    report = adjoint_report(LieAlgebra.sl2(), one_based=True)
    assert "Adj[X_1]" in report
    assert "Adj[X_3]" in report


def test_commutator_table_sl2_expansion():
    expanded = LieAlgebra.sl2().s_expand(S_E2)
    table = commutator_table(expanded, one_based=True)
    # The (1,1)-(2,1) commutator is [X_11, X_21] = -2 X_31 (paper Figure 23).
    assert "[X_(1,1), X_(2,1)] = -2 X_(3,1)" in table
    # Only i < j pairs are reported.
    assert "[X_(2," not in table.replace("X_(2,", "", 0) or all(
        line.startswith("[X_(1,") or line.startswith("[X_(2,") for line in table.splitlines()
    )


def test_structure_constants_report_nonempty():
    expanded = LieAlgebra.sl2().s_expand(S_E2)
    report = structure_constants_report(expanded, one_based=True)
    assert "C_(1,1)(2,1)^(3,1) = -2" in report


def test_generators_report_counts():
    expanded = LieAlgebra.sl2().s_expand(S_E2)
    lines = generators_report(expanded).splitlines()
    assert len(lines) == 12
    reduced = expanded.zero_reduced()
    assert len(generators_report(reduced).splitlines()) == 9


def test_metric_report_shape():
    metric = LieAlgebra.sl2().cartan_killing_metric()
    lines = metric_report(metric).splitlines()
    assert len(lines) == 3
    assert "-8.00" in lines[0]
