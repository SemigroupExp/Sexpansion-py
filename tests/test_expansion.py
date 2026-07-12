import numpy as np
import pytest

from sexpansion import LieAlgebra, Resonance, Semigroup, load_semigroup, s_expand

S_E2 = Semigroup(np.array([[0, 1, 2, 3], [1, 2, 3, 3], [2, 3, 3, 3], [3, 3, 3, 3]]))

# sl(2) grading of the paper (Section 4.4): V0 = {X1}, V1 = {X2, X3} (0-based).
V0 = (0,)
V1 = (1, 2)

# The four order-5 semigroups whose expansions of sl(2) preserve
# semi-simplicity at every level (paper, Sections 5.5-5.6), with the
# resonances used in the Java example programs (converted to 0-based).
PRESERVING_ORDER5 = {
    770: Resonance(s0=(0, 1, 2), s1=(0, 3, 4)),
    968: Resonance(s0=(0, 1, 2), s1=(0, 3, 4)),
    990: Resonance(s0=(0, 1, 2), s1=(0, 3, 4)),
    991: Resonance(s0=(0, 1, 2), s1=(0, 3, 4)),
}


def test_einsum_matches_loop_reference():
    """The einsum core must equal the Java 6-nested-loop definition."""
    sl2 = LieAlgebra.sl2()
    expanded = sl2.s_expand(S_E2)
    k = S_E2.selector().k
    c = sl2.constants
    n, m = 3, 4
    for i in range(n):
        for a in range(m):
            for j in range(n):
                for b in range(m):
                    for kk in range(n):
                        for g in range(m):
                            assert expanded.tensor[i, a, j, b, kk, g] == pytest.approx(
                                k[a, b, g] * c[i, j, kk]
                            )


def test_expanded_dimensions():
    expanded = LieAlgebra.sl2().s_expand(S_E2)
    assert expanded.n == 3
    assert expanded.m == 4
    assert expanded.n_generators == 12
    assert expanded.generators() == [(i, a) for i in range(3) for a in range(4)]


def test_jacobi_identity_all_levels():
    """The expansion and all transforms must yield genuine Lie algebras."""
    expanded = LieAlgebra.sl2().s_expand(S_E2)
    resonance = Resonance(s0=(0, 2, 3), s1=(1, 3))
    assert expanded.jacobi_violation() == pytest.approx(0.0, abs=1e-12)
    resonant = expanded.resonant_subalgebra(resonance, V0, V1)
    assert resonant.jacobi_violation() == pytest.approx(0.0, abs=1e-12)
    reduced = expanded.zero_reduced()
    assert reduced.jacobi_violation() == pytest.approx(0.0, abs=1e-12)
    both = resonant.zero_reduced()
    assert both.jacobi_violation() == pytest.approx(0.0, abs=1e-12)


def test_resonant_subalgebra_rejects_non_resonance():
    expanded = LieAlgebra.sl2().s_expand(S_E2)
    with pytest.raises(ValueError):
        expanded.resonant_subalgebra(Resonance(s0=(0, 1), s1=(2, 3)), V0, V1)


def test_zero_reduced_requires_zero():
    z2 = Semigroup(np.array([[0, 1], [1, 0]]))
    expanded = LieAlgebra.sl2().s_expand(z2)
    with pytest.raises(ValueError):
        expanded.zero_reduced()


def test_zero_reduced_generator_count():
    expanded = LieAlgebra.sl2().s_expand(S_E2)
    reduced = expanded.zero_reduced()
    assert reduced.zero == 3
    assert reduced.n_generators == 9  # n*m - n = 12 - 3
    # The whole 0_S sector is gone.
    np.testing.assert_allclose(reduced.tensor[:, 3], 0.0)
    np.testing.assert_allclose(reduced.tensor[:, :, :, 3], 0.0)
    np.testing.assert_allclose(reduced.tensor[..., 3], 0.0)


def test_resonant_generator_count():
    expanded = LieAlgebra.sl2().s_expand(S_E2)
    resonance = Resonance(s0=(0, 2, 3), s1=(1, 3))
    resonant = expanded.resonant_subalgebra(resonance, V0, V1)
    # |V0|*|S0| + |V1|*|S1| = 1*3 + 2*2 = 7 (V0/V1 disjoint).
    assert resonant.n_generators == 7


def test_metric_restriction_default():
    expanded = LieAlgebra.sl2().s_expand(S_E2)
    assert expanded.cartan_killing_metric().shape == (12, 12)
    reduced = expanded.zero_reduced()
    assert reduced.cartan_killing_metric().shape == (9, 9)
    assert reduced.cartan_killing_metric(restricted=False).shape == (12, 12)
    # The restricted metric is the corresponding submatrix of the full one.
    full = reduced.cartan_killing_metric(restricted=False)
    keep = [i * 4 + a for i in range(3) for a in range(4) if a != 3]
    np.testing.assert_allclose(reduced.cartan_killing_metric(), full[np.ix_(keep, keep)])


@pytest.mark.parametrize("sem_id", sorted(PRESERVING_ORDER5))
def test_order5_expansions_preserve_semisimplicity(sem_id):
    """Paper Section 5.6: S770, S968, S990, S991 preserve det(g) != 0 at all levels."""
    sl2 = LieAlgebra.sl2()
    semigroup = load_semigroup(5, sem_id)
    resonance = PRESERVING_ORDER5[sem_id]
    expanded = sl2.s_expand(semigroup)

    assert abs(expanded.det()) > 1e-6
    resonant = expanded.resonant_subalgebra(resonance, V0, V1)
    assert abs(resonant.det()) > 1e-6
    reduced = expanded.zero_reduced()
    assert abs(reduced.det()) > 1e-6
    both = resonant.zero_reduced()
    assert abs(both.det()) > 1e-6
    # The resonant-reduced algebra has 3 + 3 generators...
    assert both.n_generators == 6
    # ...and every level satisfies the Jacobi identity.
    for algebra in (expanded, resonant, reduced, both):
        assert algebra.jacobi_violation() == pytest.approx(0.0, abs=1e-12)


def test_s991_resonant_reduced_signature():
    """S991's resonant-reduced expansion is su(2) (+) sl(2,R): signature (2, 4)."""
    sl2 = LieAlgebra.sl2()
    s991 = load_semigroup(5, 991)
    both = sl2.s_expand(s991).resonant_subalgebra(PRESERVING_ORDER5[991], V0, V1).zero_reduced()
    positive, negative, null = both.signature()
    assert (positive, negative, null) == (2, 4, 0)


def test_transform_order_is_irrelevant():
    """Resonant-then-reduced equals reduced-then-resonant."""
    expanded = LieAlgebra.sl2().s_expand(S_E2)
    resonance = Resonance(s0=(0, 2, 3), s1=(1, 3))
    a = expanded.resonant_subalgebra(resonance, V0, V1).zero_reduced()
    b = expanded.zero_reduced().resonant_subalgebra(resonance, V0, V1)
    np.testing.assert_allclose(a.tensor, b.tensor)
    np.testing.assert_array_equal(a.active, b.active)


def test_expansion_of_abelian_stays_abelian():
    """Paper Table 1: abelian algebras stay abelian under any expansion."""
    abelian = LieAlgebra(2)
    expanded = abelian.s_expand(S_E2)
    np.testing.assert_allclose(expanded.tensor, 0.0)


def test_s_expand_free_function_equals_method():
    sl2 = LieAlgebra.sl2()
    np.testing.assert_allclose(s_expand(sl2, S_E2).tensor, sl2.s_expand(S_E2).tensor)


def test_eigenvalues_and_eigenvectors_consistent():
    expanded = LieAlgebra.sl2().s_expand(S_E2)
    eigenvalues, eigenvectors = expanded.eigenvectors()
    metric = expanded.cartan_killing_metric()
    np.testing.assert_allclose(
        metric @ eigenvectors, eigenvectors @ np.diag(eigenvalues), atol=1e-9
    )
    np.testing.assert_allclose(expanded.eigenvalues(), eigenvalues)
