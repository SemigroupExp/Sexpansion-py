import numpy as np

from sexpansion import LieAlgebra, Resonance, Semigroup, load_semigroup

S_E2 = Semigroup(np.array([[0, 1, 2, 3], [1, 2, 3, 3], [2, 3, 3, 3], [3, 3, 3, 3]]))


def test_from_semigroup_is_one_hot():
    selector = S_E2.selector()
    assert selector.order == 4
    k = selector.k
    # Exactly one 1 per (a, b) pair, at position table[a, b].
    np.testing.assert_array_equal(k.sum(axis=2), np.ones((4, 4)))
    for a in range(4):
        for b in range(4):
            assert k[a, b, S_E2.multiply(a, b)] == 1


def test_metric_matches_loop_reference():
    selector = S_E2.selector()
    k = selector.k
    m = selector.order
    reference = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            reference[a, b] = sum(k[a, c, d] * k[b, d, c] for c in range(m) for d in range(m))
    np.testing.assert_allclose(selector.metric(), reference)


def test_expanded_metric_equals_expansion_metric():
    """g_E = g (x) g_S must match the metric computed from the expanded tensor.

    This is Equation 24 of the paper and the check performed by hand in the
    Java II_SExp_CheckingByHand example.
    """
    sl2 = LieAlgebra.sl2()
    selector = S_E2.selector()
    via_kron = selector.expanded_metric(sl2.cartan_killing_metric())
    via_tensor = sl2.s_expand(S_E2).cartan_killing_metric(restricted=False)
    np.testing.assert_allclose(via_kron, via_tensor, atol=1e-12)


def test_reduced_metric_shape_and_content():
    selector = S_E2.selector()
    zero = S_E2.find_zero()
    assert zero == 3
    reduced = selector.reduced_metric(zero)
    assert reduced.shape == (3, 3)
    # Must equal the metric of the selector with zero-slices removed.
    masked = selector.k.copy()
    masked[zero, :, :] = 0
    masked[:, zero, :] = 0
    masked[:, :, zero] = 0
    kf = masked.astype(float)
    full = np.einsum("acd,bdc->ab", kf, kf)
    np.testing.assert_allclose(reduced, np.delete(np.delete(full, zero, 0), zero, 1))


def test_resonant_metric_is_block_diagonal():
    s991 = load_semigroup(5, 991)
    resonance = Resonance(s0=(0, 1, 2), s1=(0, 3, 4))
    metric = s991.selector().resonant_metric(resonance)
    assert metric.shape == (6, 6)
    np.testing.assert_allclose(metric[:3, 3:], 0.0)
    np.testing.assert_allclose(metric[3:, :3], 0.0)
    np.testing.assert_allclose(metric, metric.T)


def test_resonant_reduced_metric_drops_zero_rows():
    s991 = load_semigroup(5, 991)
    resonance = Resonance(s0=(0, 1, 2), s1=(0, 3, 4))
    zero = s991.find_zero()
    assert zero == 0
    metric = s991.selector().resonant_reduced_metric(resonance, zero)
    # zero appears once in S0 and once in S1: 6 - 2 = 4.
    assert metric.shape == (4, 4)
    np.testing.assert_allclose(metric, metric.T)
