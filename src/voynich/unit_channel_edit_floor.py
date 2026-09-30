"""Evaluator-only best possible edit error under a fixed unit dictionary.

The true text is an explicit input. This is an oracle diagnostic, never a
ciphertext-only decoder or a way to choose a dictionary without answers.
"""
from collections.abc import Sequence


def minimum_unit_edit_distance(alphabet: Sequence[str], units: Sequence[str],
                               record: str, truth: str, *, max_cells: int = 5_000_000) -> int | None:
    """Shortest path in the product of the parse DAG and edit-distance graph.

    An edge either consumes one true character (deletion), emits one compatible
    source character (insertion), or does both (match/substitution). Every edge
    increases glyph offset plus true-character offset, so dynamic programming
    is exact. No source probability is used; zero-probability source constraints
    would make this only a relaxed lower bound. None means no compatible parse.
    """
    alphabet, units = tuple(alphabet), tuple(units)
    if (not alphabet or len(set(alphabet)) != len(alphabet)
            or any(not isinstance(char, str) or len(char) != 1 for char in alphabet)
            or len(alphabet) != len(units)
            or any(not isinstance(unit, str) or not unit for unit in units)):
        raise ValueError("Need unique source characters and corresponding nonempty units")
    if not isinstance(record, str) or not isinstance(truth, str):
        raise ValueError("Record and truth must be strings")
    if type(max_cells) is not int or max_cells < 1 or (len(record) + 1) * (len(truth) + 1) > max_cells:
        raise ValueError("Edit product exceeds cell cap")
    n, m = len(record), len(truth)
    unreachable = n + m + 1
    costs = [[unreachable] * (m + 1) for _ in range(n + 1)]
    costs[0][0] = 0
    for offset in range(n + 1):
        matches = [(char, offset + len(unit)) for char, unit in zip(alphabet, units, strict=True)
                   if record.startswith(unit, offset)]
        row = costs[offset]
        for j in range(m + 1):
            value = row[j]
            if value == unreachable:
                continue
            if j < m and value + 1 < row[j + 1]:
                row[j + 1] = value + 1
            for char, end in matches:
                if value + 1 < costs[end][j]:
                    costs[end][j] = value + 1
                if j < m:
                    changed = value + (char != truth[j])
                    if changed < costs[end][j + 1]:
                        costs[end][j + 1] = changed
    return costs[n][m] if costs[n][m] < unreachable else None
