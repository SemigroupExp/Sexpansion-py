Semigroup catalogue files sem.2 .. sem.6
=========================================

These files contain all non-isomorphic finite semigroups of orders 2 to 6,
generated with the Fortran program gen.f of Jurgensen & Wick (lexicographic
ordering). They were distributed with the original Sexpansion Java library
(Inostroza, Kondrashuk, Merino, Nadal — "A Java Library to Perform
S-Expansions of Lie Algebras", Axioms 2025).

Record format (repeated):
    <ID> <order>          one header line
    <entry>               order^2 lines, one integer per line, row-major

Table entries are 1-based element labels (1..order). The Python loader in
sexpansion.database converts them to 0-based indices on read.

Counts per order: 2 -> 4, 3 -> 18, 4 -> 126, 5 -> 1160, 6 -> 15973.
