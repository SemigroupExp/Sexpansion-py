import numpy as np
import pytest

from sexpansion import Permutation, Semigroup
from sexpansion.database import load_semigroup

# Semigroups from the paper (converted to 0-based labels).
# Figure 8, first table: commutative but NOT associative ((2*2)*3 != 2*(2*3)).
S_EX1 = Semigroup(np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]]))
# S_E^(2) (Figure 5): the semigroup used in the paper's Section 4 examples.
S_E2 = Semigroup(np.array([[0, 1, 2, 3], [1, 2, 3, 3], [2, 3, 3, 3], [3, 3, 3, 3]]))
# SN3 (Figure 14 / program 5 in the paper).
SN3 = Semigroup(np.array([[3, 0, 3, 3], [0, 1, 2, 3], [3, 2, 3, 3], [3, 3, 3, 3]]))


def test_table_validation():
    with pytest.raises(ValueError):
        Semigroup(np.array([[0, 1], [1, 2]]))  # entry out of range
    with pytest.raises(ValueError):
        Semigroup(np.array([[0, 1, 0]]))  # not square


def test_order_and_multiply():
    assert S_E2.order == 4
    assert S_E2.multiply(1, 1) == 2
    assert S_E2.multiply(3, 0) == 3


def test_is_associative():
    assert S_E2.is_associative
    z3 = Semigroup(np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]]))
    assert z3.is_associative
    assert not S_EX1.is_associative
    # Non-associative magma: 0*(0*1) = 0*0 = 1 but (0*0)*1 = 1*1 = 0.
    bad = Semigroup(np.array([[1, 0], [0, 0]]))
    assert not bad.is_associative


def test_is_commutative():
    assert S_E2.is_commutative
    non_comm = Semigroup(np.array([[0, 0], [1, 1]]))  # left-zero band
    assert non_comm.is_associative
    assert not non_comm.is_commutative


def test_find_zero():
    assert S_E2.find_zero() == 3
    s991 = load_semigroup(5, 991)
    assert s991.find_zero() == 0
    no_zero = Semigroup(np.array([[0, 1], [1, 0]]))  # Z_2 group
    assert no_zero.find_zero() is None


def test_transpose():
    non_comm = Semigroup(np.array([[0, 0], [1, 1]]))
    assert non_comm.transpose() == Semigroup(np.array([[0, 1], [0, 1]]))
    assert S_E2.transpose() == S_E2


def test_permute_matches_paper_table4():
    """Paper Table 4: P#19 = (4 1 3 2) maps S^42_(4) onto SN3."""
    s42 = load_semigroup(4, 42)
    sigma = Permutation((3, 0, 2, 1))  # 0-based version of (4 1 3 2)
    assert s42.permute(sigma) == SN3
    # And the inverse permutation brings SN3 back to S^42_(4).
    assert SN3.permute(sigma.inverse()) == s42


def test_permute_identity_is_noop():
    assert S_E2.permute(Permutation.identity(4)) == S_E2


def test_all_images_count_and_membership():
    images = list(SN3.all_images())
    assert len(images) == 24
    assert load_semigroup(4, 42) in images


def test_isomorphism_test():
    s42 = load_semigroup(4, 42)
    result = s42.isomorphism_test(SN3)
    assert result.isomorphic
    # Both are commutative, so anti-isomorphic too.
    assert result.anti_isomorphic
    # Different orders are never isomorphic.
    assert not S_EX1.isomorphism_test(S_E2).isomorphic


def test_isomorphism_test_negative():
    # S^1_(3) (all products = 0) vs Z_3: not isomorphic.
    trivial = Semigroup(np.zeros((3, 3), dtype=np.int_))
    z3 = Semigroup(np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]]))
    result = trivial.isomorphism_test(z3)
    assert not result.isomorphic
    assert not result.anti_isomorphic


def test_anti_isomorphism_of_bands():
    """The left-zero and right-zero bands are anti- but not isomorphic."""
    left = Semigroup(np.array([[0, 0], [1, 1]]))
    right = left.transpose()
    result = left.isomorphism_test(right)
    assert not result.isomorphic
    assert result.anti_isomorphic


def test_canonical_key():
    s42 = load_semigroup(4, 42)
    # Isomorphic semigroups share the key...
    assert s42.canonical_key() == SN3.canonical_key()
    # ...non-isomorphic ones do not.
    trivial = Semigroup(np.zeros((3, 3), dtype=np.int_))
    z3 = Semigroup(np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]]))
    assert trivial.canonical_key() != z3.canonical_key()
    # The key is invariant under any relabeling.
    assert all(img.canonical_key() == SN3.canonical_key() for img in SN3.all_images())


def test_equality_and_hash():
    a = Semigroup(np.array([[0, 1], [1, 0]]))
    b = Semigroup(np.array([[0, 1], [1, 0]]), sem_id=4)
    assert a == b  # sem_id does not affect equality
    assert hash(a) == hash(b)
    assert a != Semigroup(np.zeros((2, 2), dtype=np.int_))


def test_table_is_readonly():
    with pytest.raises(ValueError):
        S_E2.table[0, 0] = 1
