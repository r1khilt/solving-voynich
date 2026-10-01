"""Bounded post-outcome exact mathematics; no corpus, search or model calls."""

from __future__ import annotations

import argparse
import itertools
import json
from fractions import Fraction as F

from scripts.check_shared_key_moments001 import ROWS, UNITS
from scripts.run_source_bellman001 import OUT, save_new

PARTIALS = ((-1, -1), (-1, 0), (0, -1), (-1, 2), (2, -1), (0, 1), (1, 1))
GEOMETRY = (F(1, 4) * F(3, 4)) ** 3


def moments(probabilities, partial, glyphs):
    free = [r for r, k in enumerate(partial) if k < 0]
    means = [
        sum((p[r] for r, k in enumerate(partial) if k == u), F(0))
        + sum((p[r] for r in free), F(0)) / UNITS
        for p, u in zip(probabilities, glyphs, strict=True)
    ]
    pair_correction = F(0)
    for i, j in itertools.combinations(range(3), 2):
        covariance = sum(
            (probabilities[i][r] * probabilities[j][r] for r in free), F(0)
        ) * (F(int(glyphs[i] == glyphs[j]), UNITS) - F(1, UNITS**2))
        pair_correction += means[3 - i - j] * covariance
    all_equal = int(glyphs[0] == glyphs[1] == glyphs[2])
    equal_pairs = sum(glyphs[i] == glyphs[j] for i, j in itertools.combinations(range(3), 2))
    third = sum((probabilities[0][r] * probabilities[1][r] * probabilities[2][r] for r in free), F(0)) * (
        F(all_equal, UNITS) - F(equal_pairs, UNITS**2) + F(2, UNITS**3)
    )
    mean_product = means[0] * means[1] * means[2]
    return GEOMETRY * (mean_product + pair_correction + third), GEOMETRY * (mean_product + pair_correction), GEOMETRY * third


def enumerate_full_keys(probabilities, partial, glyphs):
    free = [r for r, k in enumerate(partial) if k < 0]
    result = F(0)
    for codes in itertools.product(range(UNITS), repeat=len(free)):
        key = list(partial)
        for r, code in zip(free, codes, strict=True):
            key[r] = code
        for letters in itertools.product(range(2), repeat=3):
            if all(key[r] == u for r, u in zip(letters, glyphs, strict=True)):
                weight = GEOMETRY / UNITS ** len(free)
                for p, r in zip(probabilities, letters, strict=True):
                    weight *= p[r]
                result += weight
    return result


def check():
    checked = 0
    mismatches = 0
    for contexts in itertools.product(range(3), repeat=3):
        probabilities = [ROWS[c] for c in contexts]
        for partial in PARTIALS:
            for glyphs in itertools.product(range(2), repeat=3):
                exact, pair_only, third = moments(probabilities, partial, glyphs)
                assert exact == enumerate_full_keys(probabilities, partial, glyphs)
                assert exact >= 0 and exact - pair_only == third
                mismatches += pair_only != exact
                checked += 1
    exact, pair_only, third = moments([ROWS[0]] * 3, (-1, -1), (0, 0, 0))
    assert third > 0 and pair_only < exact
    return {
        "status": "PASS_exact_rational_third_moment",
        "context_partial_key_observation_cases": checked,
        "pair_only_not_exact_cases": mismatches,
        "same_glyph_witness": {"exact": str(exact), "pair_only": str(pair_only), "third_correction": str(third), "exact_over_pair_only": str(exact / pair_only)},
        "scope": "Three unfinished one-glyph records; independent uniform remaining dictionary rows; seven stated partial bindings and supplied two-row contexts. No whole-text surrogate qualification.",
        "post_outcome_analysis": True,
        "empirical_panel_calls": 0,
        "paid_spend_usd": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    result = check()
    if args.save:
        save_new(OUT / "shared-key-third-moments-post-outcome.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))
