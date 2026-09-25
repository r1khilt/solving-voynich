"""Metadata-only folio-order challenger for HERBAL-CONTROL-0003.

No drawing pixels, feature distances or Voynich material enter this module.
The six affine manuscript maps are fitted from development-known classes only.
"""

from __future__ import annotations

from itertools import combinations
from math import isfinite
import re
from statistics import median

from .herbal_open_set import MANUSCRIPTS


FOLIO_RE = re.compile(r"f\.(\d+)([rv])\.jpg$", re.IGNORECASE)


def folio_side(title: str) -> int:
    match = FOLIO_RE.search(title)
    if match is None:
        raise ValueError(f"missing unqualified folio-side JPEG title: {title}")
    return 2 * int(match.group(1)) + (match.group(2).lower() == "v")


def fit_development_maps(panel: dict) -> dict[str, dict[str, float]]:
    """Theil–Sen slope and median intercept from 24 known correspondences."""
    known = panel["roles"]["development_known"]
    if len(known) != 24 or len({item["chapter"] for item in known}) != 24:
        raise ValueError("development-known class list differs from 24 distinct chapters")
    coordinates = {manuscript: [folio_side(item["pages"][manuscript]["title"])
                                for item in known] for manuscript in MANUSCRIPTS}
    if any(len(set(values)) != 24 for values in coordinates.values()):
        raise ValueError("duplicate folio-side coordinate in development-known split")
    maps = {}
    for query in MANUSCRIPTS:
        x = coordinates[query]
        for source in MANUSCRIPTS:
            if source == query:
                continue
            y = coordinates[source]
            slopes = [(y[j] - y[i]) / (x[j] - x[i]) for i, j in combinations(range(24), 2)]
            slope = median(slopes)
            intercept = median(y[i] - slope * x[i] for i in range(24))
            if not isfinite(slope) or slope <= 0 or not isfinite(intercept):
                raise ValueError(f"invalid positive folio fit: {query}->{source}")
            maps[f"{query}->{source}"] = {"slope": slope, "intercept": intercept}
    return maps


def folio_distance_matrix(panel: dict, split: str,
                          maps: dict[str, dict[str, float]]) -> tuple[list[list[float]], list[dict]]:
    """Directed raw folio-distance matrix with canonical role/class/manuscript order."""
    if split not in ("development", "evaluation"):
        raise ValueError("split must be development or evaluation")
    expected_pairs = {f"{q}->{s}" for q in MANUSCRIPTS for s in MANUSCRIPTS if q != s}
    if set(maps) != expected_pairs:
        raise ValueError("not all six frozen ordered manuscript maps are present")
    rows = []
    for kind in ("known", "unknown"):
        role = f"{split}_{kind}"
        for item in panel["roles"][role]:
            for manuscript in MANUSCRIPTS:
                rows.append({"role": role, "chapter_class": item["chapter"],
                             "manuscript": manuscript,
                             "folio_side": folio_side(item["pages"][manuscript]["title"])})
    if len(rows) != 108:
        raise ValueError("folio panel is not 108 images")
    matrix = []
    for query in rows:
        costs = []
        for reference in rows:
            if query["manuscript"] == reference["manuscript"]:
                costs.append(0.0)
                continue
            fit = maps[f"{query['manuscript']}->{reference['manuscript']}"]
            cost = abs(fit["slope"] * query["folio_side"]
                       + fit["intercept"] - reference["folio_side"])
            if not isfinite(cost):
                raise ValueError("nonfinite folio distance")
            costs.append(cost)
        matrix.append(costs)
    return matrix, rows
