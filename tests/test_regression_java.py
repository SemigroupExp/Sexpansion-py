"""Regression tests against the captured output of the Java library.

These tests parse the plain-text files in the Java repository's
``Output_examples`` folder (numbers, not raw text, are compared) and are
skipped when that folder is not available. The headline counts are also
asserted as hard-coded values in ``test_resonance.py`` /
``test_regression_headline_counts`` so CI still validates them without
the Java repository.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sexpansion import (
    Permutation,
    Resonance,
    find_all_resonances,
    load_semigroups,
)

# Paper Table 5 headline numbers (independent of the Java repo).
COMMUTATIVE_COUNTS = {2: 3, 3: 12, 4: 58, 5: 325, 6: 2143}
ZERO_ELEMENT_COUNTS = {2: 2, 3: 8, 4: 39, 5: 226, 6: 1538}


def _assert_headline_counts(order: int) -> None:
    n_commutative = 0
    n_zero = 0
    for sg in load_semigroups(order):
        if sg.is_commutative:
            n_commutative += 1
            if sg.find_zero() is not None:
                n_zero += 1
    assert n_commutative == COMMUTATIVE_COUNTS[order]
    assert n_zero == ZERO_ELEMENT_COUNTS[order]


@pytest.mark.parametrize("order", [2, 3, 4])
def test_regression_headline_counts(order):
    _assert_headline_counts(order)


@pytest.mark.slow
@pytest.mark.parametrize("order", [5, 6])
def test_regression_headline_counts_large(order):
    _assert_headline_counts(order)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_commutative_count_vs_java(java_outputs: Path, order: int):
    text = (java_outputs / f"Output_I_commutative_ord{order}.txt").read_text()
    match = re.search(r"Number of commutative semigroups[^:]*:\s*(\d+)", text)
    assert match is not None
    java_count = int(match.group(1))
    ours = sum(1 for sg in load_semigroups(order) if sg.is_commutative)
    assert ours == java_count


@pytest.mark.parametrize("order", [2, 3, 4])
def test_find_zero_vs_java(java_outputs: Path, order: int):
    """Compare the (semigroup ID, zero element) pairs with the Java scan."""
    text = (java_outputs / f"Output_II_findzero_ord{order}.txt").read_text()
    java_pairs = {
        (int(sem_id), int(zero))
        for sem_id, zero in re.findall(r"#(\d+)\n(?:[\d ]+\n)+The zero element is (\d+)", text)
    }
    ours = {
        (sg.sem_id, zero + 1)  # Java reports 1-based zero labels
        for sg in load_semigroups(order)
        if sg.is_commutative and (zero := sg.find_zero()) is not None
    }
    assert ours == java_pairs


@pytest.mark.parametrize("order", [2, 3, 4])
def test_resonances_vs_java(java_outputs: Path, order: int):
    """Compare every resonant decomposition per semigroup with the Java scan."""
    text = (java_outputs / f"Output_II_findAllResonances_ord{order}.txt").read_text()
    java_resonances: dict[int, set[Resonance]] = {}
    for block in re.split(r"\*+\n", text):
        header = re.search(r"The semigroup #(\d+)", block)
        if header is None:
            continue
        sem_id = int(header.group(1))
        found = {
            Resonance(
                s0=tuple(int(v) - 1 for v in s0.split()),
                s1=tuple(int(v) - 1 for v in s1.split()),
            )
            for s0, s1 in re.findall(r"S0: ([\d ]+?)\s*\nS1: ([\d ]+?)\s*\n", block)
        }
        assert found, f"no resonances parsed for semigroup #{sem_id}"
        java_resonances[sem_id] = found

    summary = re.search(
        r"There are (\d+) semigroups with at least one resonance and there are\s*"
        r"in total (\d+) different resonances",
        text,
    )
    assert summary is not None
    assert len(java_resonances) == int(summary.group(1))
    assert sum(len(v) for v in java_resonances.values()) == int(summary.group(2))

    ours: dict[int, set[Resonance]] = {}
    for sg in load_semigroups(order):
        if sg.is_commutative:
            resonances = set(find_all_resonances(sg))
            if resonances:
                assert sg.sem_id is not None
                ours[sg.sem_id] = resonances
    assert ours == java_resonances


@pytest.mark.parametrize("n", [2, 3, 4])
def test_permutation_inverses_vs_java(java_outputs: Path, n: int):
    """Compare the full permutation list and inverses with the Java output."""
    text = (java_outputs / f"Output_I_Permutations_and_inverses_n{n}.txt").read_text()
    java_entries = re.findall(
        r"Permutation P#(\d+)\n([\d ]+?)\s*\n"
        r"The inverse permutation \(P#\d+\)\^\(-1\) = P#(\d+)\n([\d ]+?)\s*\n",
        text,
    )
    assert java_entries
    permutations = list(Permutation.all_of_degree(n))
    assert len(java_entries) == len(permutations)
    for index_str, image_str, inv_index_str, inv_image_str in java_entries:
        ours = permutations[int(index_str)]
        java_image = tuple(int(v) - 1 for v in image_str.split())
        assert ours.image == java_image
        inverse = ours.inverse()
        assert permutations[int(inv_index_str)] == inverse
        assert inverse.image == tuple(int(v) - 1 for v in inv_image_str.split())
