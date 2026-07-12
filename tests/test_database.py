import numpy as np
import pytest

from sexpansion import SEMIGROUP_COUNTS, load_all_semigroups, load_semigroup, load_semigroups


def test_catalogue_counts():
    for order, count in SEMIGROUP_COUNTS.items():
        semigroups = load_semigroups(order)
        assert len(semigroups) == count
        assert all(sg.order == order for sg in semigroups)


def test_catalogue_ids_are_sequential():
    for order in SEMIGROUP_COUNTS:
        ids = [sg.sem_id for sg in load_semigroups(order)]
        assert ids == list(range(1, SEMIGROUP_COUNTS[order] + 1))


def test_load_semigroup_by_id():
    sg = load_semigroup(4, 42)
    assert sg.sem_id == 42
    assert sg.order == 4
    with pytest.raises(ValueError):
        load_semigroup(4, 0)
    with pytest.raises(ValueError):
        load_semigroup(7, 1)


def test_s991_table_matches_java_example():
    """The table hard-coded in the Java II_SExp_sl2_S991 example (0-based)."""
    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 2, 3, 4],
            [0, 2, 1, 4, 3],
            [0, 3, 4, 2, 1],
            [0, 4, 3, 1, 2],
        ]
    )
    s991 = load_semigroup(5, 991)
    assert s991.is_associative
    assert s991.is_commutative
    assert s991.find_zero() == 0
    assert np.array_equal(s991.table, expected)


def test_small_orders_all_associative():
    for order in (2, 3, 4):
        assert all(sg.is_associative for sg in load_semigroups(order))


@pytest.mark.slow
def test_large_orders_all_associative():
    for order in (5, 6):
        assert all(sg.is_associative for sg in load_semigroups(order))


def test_load_all_semigroups():
    all_sgs = load_all_semigroups()
    assert len(all_sgs) == sum(SEMIGROUP_COUNTS.values())
