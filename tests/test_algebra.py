import numpy as np
import pytest

from sexpansion import LieAlgebra


def test_set_constant_antisymmetrizes():
    algebra = LieAlgebra(3)
    algebra.set_constant(0, 1, 2, -2.0)
    assert algebra.constants[0, 1, 2] == -2.0
    assert algebra.constants[1, 0, 2] == 2.0


def test_constants_readonly():
    algebra = LieAlgebra.sl2()
    with pytest.raises(ValueError):
        algebra.constants[0, 0, 0] = 1.0


def test_sl2_cartan_killing_metric():
    """sl(2,R) in the paper's basis has KC metric diag(-8, 8, 8)."""
    metric = LieAlgebra.sl2().cartan_killing_metric()
    np.testing.assert_allclose(metric, np.diag([-8.0, 8.0, 8.0]))
    assert np.linalg.det(metric) != 0  # semi-simple


def test_cartan_killing_metric_is_symmetric():
    algebra = LieAlgebra(4)
    algebra.set_constant(0, 1, 2, 1.5)
    algebra.set_constant(2, 3, 0, -0.5)
    metric = algebra.cartan_killing_metric()
    np.testing.assert_allclose(metric, metric.T)


def test_adjoint_generators_shape_and_bracket():
    """ad matrices must represent the algebra: [ad_i, ad_j] = C_ij^k ad_k."""
    sl2 = LieAlgebra.sl2()
    adj = sl2.adjoint_generators()
    assert adj.shape == (3, 3, 3)
    c = sl2.constants
    for i in range(3):
        for j in range(3):
            bracket = adj[i] @ adj[j] - adj[j] @ adj[i]
            expected = np.einsum("k,kab->ab", c[i, j], adj)
            np.testing.assert_allclose(bracket, expected, atol=1e-12)


def test_casimir_of_sl2_is_identity_multiple():
    """Schur: the quadratic Casimir of a simple algebra is c * Id."""
    casimir = LieAlgebra.sl2().casimir()
    scale = casimir[0, 0]
    np.testing.assert_allclose(casimir, scale * np.eye(3), atol=1e-12)
    assert abs(scale) > 1e-12
    np.testing.assert_allclose(
        LieAlgebra.sl2().casimir_eigenvalues(), [scale, scale, scale], atol=1e-12
    )


def test_generators_commute():
    algebra = LieAlgebra(3)
    algebra.set_constant(0, 1, 2, 1.0)
    assert not algebra.generators_commute(0, 1)
    assert algebra.generators_commute(0, 2)
    assert algebra.generators_commute(1, 2)


def test_maximal_abelian_subalgebra_abelian_algebra():
    """An abelian algebra's maximal abelian subalgebra is the whole algebra."""
    algebra = LieAlgebra(4)
    assert algebra.maximal_abelian_subalgebra() == (0, 1, 2, 3)


def test_maximal_abelian_subalgebra_sl2():
    """No two sl(2) generators commute: maximal abelian subsets have size 1."""
    subset = LieAlgebra.sl2().maximal_abelian_subalgebra()
    assert len(subset) == 1


def test_maximal_abelian_subalgebra_heisenberg():
    """Heisenberg algebra [X0, X1] = X2: {X0, X2} (or {X1, X2}) is abelian."""
    heisenberg = LieAlgebra(3)
    heisenberg.set_constant(0, 1, 2, 1.0)
    subset = heisenberg.maximal_abelian_subalgebra()
    assert len(subset) == 2
    assert heisenberg.is_abelian_subset(subset)


def test_semisimple_adjoint_generators_sl2():
    indices, matrices = LieAlgebra.sl2().semisimple_adjoint_generators()
    assert len(indices) == len(matrices)
    adjoints = LieAlgebra.sl2().adjoint_generators()
    for idx, matrix in zip(indices, matrices, strict=True):
        np.testing.assert_allclose(matrix, adjoints[idx])
