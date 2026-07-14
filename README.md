# sexpansion

**S-expansions of Lie algebras with finite abelian semigroups, in Python.**

This package is a full port of the [Sexpansion Java library](https://github.com/cinostroza/Sexpansion)
described in:

> C. Inostroza, I. Kondrashuk, N. Merino, F. Nadal,
> *A Java Library to Perform S-Expansions of Lie Algebras*, Axioms (2025).

The S-expansion method combines a Lie algebra **G** with a finite abelian
semigroup **S** to produce new, generally non-isomorphic Lie algebras
`G_S = S ⊗ G`, from which smaller algebras can be extracted: resonant
subalgebras, 0_S-reduced algebras, and 0_S-reductions of resonant
subalgebras.

## Installation

```bash
pip install sexpansion
```

Requires Python ≥ 3.10 and NumPy. The catalogue of all non-isomorphic
semigroups of orders 2–6 (17,281 tables) ships with the package.

## Quickstart

Expand `sl(2, R)` with the semigroup `S⁹⁹¹₍₅₎`, extract the resonant
subalgebra, and 0_S-reduce it (the paper's Section 5.5 example):

```python
from sexpansion import LieAlgebra, Resonance, load_semigroup

sl2 = LieAlgebra.sl2()                 # [X1,X2]=-2X3, [X1,X3]=2X2, [X2,X3]=2X1
s991 = load_semigroup(5, 991)          # from the bundled catalogue

expanded = sl2.s_expand(s991)          # G_S = S (x) G, 15 generators
resonant = expanded.resonant_subalgebra(
    Resonance(s0=(0, 1, 2), s1=(0, 3, 4)),   # S = S0 u S1
    v0=(0,), v1=(1, 2),                       # G = V0 (+) V1
)
final = resonant.zero_reduced()        # 6 generators: su(2) (+) sl(2,R)

print(final.det())                     # != 0 -> semi-simplicity preserved
print(final.signature())               # (2, 4, 0)
```

Work with the semigroup catalogue:

```python
from sexpansion import load_semigroups, find_all_resonances

for sg in load_semigroups(4):
    if sg.is_commutative and (zero := sg.find_zero()) is not None:
        print(sg.sem_id, zero, len(find_all_resonances(sg)))
```

Readable reports for the physicist's eye (pass `one_based=True` to match
the λ₁…λₙ labelling of the paper):

```python
from sexpansion.reports import commutator_table
print(commutator_table(final, one_based=True))
# [X_(1,2), X_(2,4)] = -2 X_(3,5)
# ...
```

## Conventions

- **Everything is 0-based**: semigroup elements are `0..n-1` and Lie
  algebra generators `0..n-1`. The catalogue files (`sem.2`…`sem.6`) use
  1-based labels and are converted on load; the paper and the Java
  library print 1-based labels, which the report functions reproduce
  with `one_based=True`.
- `Semigroup.table[i, j]` is the product `i * j`.
- `LieAlgebra.constants[i, j, k]` is `C_ij^k` in `[X_i, X_j] = C_ij^k X_k`;
  `set_constant` fills the antisymmetric partner automatically.
- `ExpandedAlgebra.tensor[i, a, j, b, k, c]` is `C_(i,a)(j,b)^(k,c) = K_ab^c C_ij^k`.
- Metrics are in generator-major order: row `i * m + a` ↔ generator `X_(i,a)`.

## Mapping from the Java library

| Java | Python |
|---|---|
| `Semigroup.isAssociative()` / `isCommutative()` | `Semigroup.is_associative` / `is_commutative` |
| `Semigroup.findZero()` → `-1` | `Semigroup.find_zero()` → `None` |
| `Semigroup.loadFromFile(...)` | `load_semigroups(order)` / `load_all_semigroups()` |
| `Semigroup.isResonant(S0, S1)` | `is_resonant(sg, s0, s1)` |
| `isResonatF` / `findAllResonancesF/F2` | `filtered=True` flag |
| `Semigroup.permuteWith(SetS)` / `permute()` | `Semigroup.permute(sigma)` / `all_images()` |
| `Semigroup.isotest(B)` | `Semigroup.isomorphism_test(other)` |
| `SetS` | `tuple[int, ...]`, `itertools`, `Permutation` |
| `StructureConstantSet` | `LieAlgebra` |
| `getExpandedStructureConstant(s)` | `LieAlgebra.s_expand(semigroup)` |
| `StructureConstantSetExpanded{,Reduced,Resonant,ResonantReduced}` | `ExpandedAlgebra` + `.resonant_subalgebra(...)` / `.zero_reduced()` |
| `cartanKillingMetric()` / `...Pretty()` | `cartan_killing_metric(restricted=False)` / default |
| `show*` methods | `sexpansion.reports` functions (return strings) |
| Jama `Matrix` | NumPy arrays (`np.linalg.det/eigh/inv`) |

Deliberate deviations from the Java code: the anti-isomorphism branch of
`isotest` (which could never trigger) is fixed; the non-terminating
`maximalAbelianSubalgebra` is replaced by a correct algorithm; the
`isResonatF` typo is renamed.

## Examples and tests

The `examples/` folder contains ten Jupyter notebooks porting
representative programs from the paper (Appendix A), from associativity
checks up to the full S-expansion pipeline. They are committed with
executed outputs, so the results can be read directly on GitHub without
installing anything; to run them yourself, `pip install sexpansion
jupyterlab` and open the notebooks (re-execute in order — later cells
reuse earlier ones).

The test suite reproduces the paper's Table 5 catalogue
statistics (e.g. order 6: 2,059 semigroups with 25,512 resonances) and,
when the Java repository is present, cross-checks against its captured
outputs (`SEXPANSION_JAVA_OUTPUTS` environment variable; run the slow
catalogue scans with `pytest -m slow`).

## License

GPL-3.0-only, matching the original Java library. If you use this
package in academic work, please cite the paper above.
