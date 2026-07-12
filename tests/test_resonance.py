import numpy as np
import pytest

from sexpansion import (
    Resonance,
    Semigroup,
    find_all_resonances,
    find_resonances,
    is_resonant,
    is_resonant_filtered,
    load_semigroup,
    load_semigroups,
)

# Expected scan results over the commutative catalogue (paper, Table 5:
# rows #G_S,R and #r): {order: (semigroups_with_resonance, total_resonances)}.
RESONANCE_COUNTS = {2: (1, 1), 3: (8, 9), 4: (48, 124), 5: (299, 1653), 6: (2059, 25512)}


def test_s991_resonance():
    """Paper Section 4.4/5.5: S0={l1,l2,l3}, S1={l1,l4,l5} is resonant for S991."""
    s991 = load_semigroup(5, 991)
    assert is_resonant(s991, (0, 1, 2), (0, 3, 4))
    # Shared element 0 is the zero, so the filtered variant accepts it too.
    assert is_resonant_filtered(s991, (0, 1, 2), (0, 3, 4))
    assert not is_resonant(s991, (0, 1, 3), (0, 2, 4))


def test_se2_resonance_from_paper_section_4_4():
    """Figure 5 semigroup: S0={l1,l3,l4}, S1={l2,l4} is resonant (0-based below)."""
    s_e2 = Semigroup(np.array([[0, 1, 2, 3], [1, 2, 3, 3], [2, 3, 3, 3], [3, 3, 3, 3]]))
    assert is_resonant(s_e2, (0, 2, 3), (1, 3))
    # The shared element 3 IS the zero of this semigroup.
    assert is_resonant_filtered(s_e2, (0, 2, 3), (1, 3))


def test_is_resonant_requires_coverage():
    trivial = Semigroup(np.zeros((3, 3), dtype=np.int_))
    assert not is_resonant(trivial, (0,), (1,))  # element 2 uncovered


def test_filtered_rejects_nonzero_overlap():
    z2 = Semigroup(np.array([[0, 1], [1, 0]]))  # the group Z_2: no zero element
    # (S0, S1) = ({0,1}, {0,1}) satisfies the closure conditions...
    assert is_resonant(z2, (0, 1), (0, 1))
    # ...but shares the non-zero elements 0 and 1, so the filter rejects it.
    assert not is_resonant_filtered(z2, (0, 1), (0, 1))
    # The disjoint decomposition ({0}, {1}) passes both.
    assert is_resonant(z2, (0,), (1,))
    assert is_resonant_filtered(z2, (0,), (1,))


def test_find_resonances_sizes():
    s991 = load_semigroup(5, 991)
    found = find_resonances(s991, 3, 3)
    assert Resonance(s0=(0, 1, 2), s1=(0, 3, 4)) in found
    assert all(len(r.s0) == 3 and len(r.s1) == 3 for r in found)


def test_resonance_normalizes_order():
    assert Resonance(s0=(2, 0, 1), s1=(4, 3)) == Resonance(s0=(0, 1, 2), s1=(3, 4))


def test_all_found_resonances_are_resonant():
    for sg in load_semigroups(3):
        if sg.is_commutative:
            for r in find_all_resonances(sg):
                assert is_resonant(sg, r.s0, r.s1)
            for r in find_all_resonances(sg, filtered=True):
                assert is_resonant_filtered(sg, r.s0, r.s1)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_resonance_scan_counts_small(order):
    _assert_scan_counts(order)


@pytest.mark.slow
@pytest.mark.parametrize("order", [5, 6])
def test_resonance_scan_counts_large(order):
    _assert_scan_counts(order)


def _assert_scan_counts(order: int) -> None:
    n_semigroups = 0
    n_resonances = 0
    for sg in load_semigroups(order):
        if sg.is_commutative:
            resonances = find_all_resonances(sg)
            if resonances:
                n_semigroups += 1
                n_resonances += len(resonances)
    assert (n_semigroups, n_resonances) == RESONANCE_COUNTS[order]
