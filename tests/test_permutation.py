import math

import pytest

from sexpansion import Permutation


def test_identity():
    p = Permutation.identity(4)
    assert p.image == (0, 1, 2, 3)
    assert all(p(i) == i for i in range(4))


def test_invalid_permutation_rejected():
    with pytest.raises(ValueError):
        Permutation((0, 0, 1))


def test_inverse_roundtrip():
    p = Permutation((3, 0, 2, 1))
    inv = p.inverse()
    assert all(inv(p(i)) == i for i in range(4))
    assert all(p(inv(i)) == i for i in range(4))
    assert p.inverse().inverse() == p


def test_all_of_degree_counts():
    for n in range(1, 6):
        perms = list(Permutation.all_of_degree(n))
        assert len(perms) == math.factorial(n)
        assert len(set(perms)) == math.factorial(n)


def test_degree():
    assert Permutation((1, 0, 2)).degree == 3
