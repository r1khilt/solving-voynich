"""Bounded exact non-erasing word-equation oracle, with explicit incompletion.

Only known source and unsegmented cipher records enter the solver. No gold
dictionary or boundary offsets are accepted. Distinct rows may share units.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import math


def solve_known_plaintext(source, cipher, *, rows=23, glyphs=6,
                          node_cap=1_000_000, solution_cap=4096):
    if any(type(v) is not int or v < 1 for v in (rows, glyphs, node_cap, solution_cap)):
        raise ValueError("Positive integer dimensions and search caps required")
    source, cipher = tuple(map(tuple, source)), tuple(map(tuple, cipher))
    if (not source or len(source) != len(cipher)
            or any(not r or any(type(v) is not int or not 0 <= v < rows for v in r) for r in source)
            or any(not r or any(type(v) is not int or not 0 <= v < glyphs for v in r) for r in cipher)):
        raise ValueError("Paired nonempty in-alphabet integer records required")
    # Suffix multiplicities give safe length bounds, including repeated rows.
    suffix = []
    for record in source:
        counts = [0]*rows
        here = [tuple(counts)]
        for symbol in reversed(record):
            counts = counts.copy()
            counts[symbol] += 1
            here.append(tuple(counts))
        suffix.append(tuple(reversed(here)))
    used = tuple(sorted({r for record in source for r in record}))
    solutions, nodes, cutoff = [], 0, None

    def visit(record, position, offset, key):
        nonlocal nodes, cutoff
        if cutoff is not None:
            return
        if nodes >= node_cap:
            cutoff = "node_cap"
            return
        nodes += 1
        while record < len(source):
            # Reject this or any later record if no legal lengths can fit.
            for later in range(record, len(source)):
                counts = suffix[later][position if later == record else 0]
                remaining = len(cipher[later])-(offset if later == record else 0)
                low = sum(n*(len(k) if k is not None else 1) for n, k in zip(counts, key, strict=True))
                high = sum(n*(len(k) if k is not None else 2) for n, k in zip(counts, key, strict=True))
                if not low <= remaining <= high:
                    return
            if position == len(source[record]):
                if offset != len(cipher[record]):
                    return
                record, position, offset = record+1, 0, 0
                continue
            row = source[record][position]
            unit = key[row]
            if unit is not None:
                if cipher[record][offset:offset+len(unit)] != unit:
                    return
                position, offset = position+1, offset+len(unit)
                continue
            # A new row's value must be the next observed substring: two
            # legal lengths exhaust the possibilities without guessing glyphs.
            for length in (1, 2):
                if offset+length <= len(cipher[record]):
                    candidate = list(key)
                    candidate[row] = cipher[record][offset:offset+length]
                    visit(record, position+1, offset+length, tuple(candidate))
                    if cutoff is not None:
                        return
            return
        solutions.append(key)
        if len(solutions) >= solution_cap:
            # Conservative even if this was the last possible solution: a
            # cap hit cannot be presented as an exhaustive count.
            cutoff = "solution_cap"

    visit(0, 0, 0, (None,)*rows)
    if len(solutions) != len(set(solutions)):
        raise AssertionError("Each used-row dictionary must be enumerated once")
    return {"complete": cutoff is None, "cutoff": cutoff, "nodes": nodes,
            "used_rows": used, "solutions": tuple(solutions),
            "solution_count_lower_bound": len(solutions), "rows": rows, "glyphs": glyphs}


def oracle_statistics(result):
    """Uniform iid-row oracle posterior, only for exhaustive nonempty support."""
    if not result["complete"] or not result["solutions"]:
        return None
    solutions, used = result["solutions"], result["used_rows"]
    unused = result["rows"]-len(used)
    units = result["glyphs"]+result["glyphs"]**2
    support = len(solutions)*units**unused
    marginals = {r: Counter(k[r] for k in solutions) for r in used}
    certain = [r for r in used if len(marginals[r]) == 1]
    used_matches = sum((Fraction(max(marginals[r].values()), len(solutions)) for r in used), Fraction(0))
    return {"used_solution_count": len(solutions), "unused_rows": unused,
            "full_key_support_size": support, "determined_used_rows": certain,
            "used_key_map_probability": str(Fraction(1, len(solutions))),
            "full_key_map_probability": str(Fraction(1, support)),
            "bayes_expected_used_row_matches": str(used_matches),
            "oracle_entropy_nats": math.log(len(solutions))+unused*math.log(units),
            "scope": "Known plaintext, iid uniform literal row prior; not ciphertext-only or training-exclusion posterior."}
