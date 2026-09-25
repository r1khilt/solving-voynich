"""Development-fitted folio-order costs for HERBAL-CONTROL-0003.

No image pixels, crop positions, chapter spelling or evaluation labels fit the
ordered-manuscript maps. The caller controls the full-input and split gates.
"""

from __future__ import annotations

from math import isfinite
import re
from statistics import median

from voynich.herbal_open_set import MANUSCRIPTS


FOLIO = re.compile(r"f\.(\d+)([rv])\b")
FEATURE_METHOD = "Folio-side Theil-Sen affine distance, development-known fit"


def folio_side(title: str) -> int:
    matches = FOLIO.findall(title)
    if len(matches) != 1:
        raise ValueError(f"expected one folio coordinate: {title}")
    number, side = matches[0]
    return 2 * int(number) + (side == "v")


def _positions(chapters: list[dict], manuscript: str) -> list[int]:
    values = [folio_side(item["pages"][manuscript]["title"]) for item in chapters]
    if len(values) != len(set(values)):
        raise ValueError(f"folio sides are not unique for {manuscript}")
    return values


def fit_development_maps(panel: dict) -> dict[str, dict[str, float]]:
    """Six positive Theil–Sen maps from 24 known development correspondences."""
    known = panel["roles"]["development_known"]
    if len(known) != 24:
        raise ValueError("expected 24 development-known chapters")
    positions = {manuscript: _positions(known, manuscript)
                 for manuscript in MANUSCRIPTS}
    fits = {}
    for query in MANUSCRIPTS:
        for source in MANUSCRIPTS:
            if query == source:
                continue
            x, y = positions[query], positions[source]
            slopes = [(y[j] - y[i]) / (x[j] - x[i])
                      for i in range(24) for j in range(i + 1, 24)]
            if len(slopes) != 276:
                raise AssertionError("unexpected pair-slope count")
            a = float(median(slopes))
            b = float(median(y_value - a * x_value
                             for x_value, y_value in zip(x, y, strict=True)))
            if not isfinite(a) or a <= 0 or not isfinite(b):
                raise ValueError(f"invalid folio map {query}->{source}")
            fits[f"{query}->{source}"] = {"slope": a, "intercept": b}
    return fits


def make_direction_panels(
    panel: dict, split: str, fits: dict[str, dict[str, float]],
    normalizers: dict[str, float] | None = None,
) -> tuple[dict[str, list[list[float]]], dict[str, list[list[float]]],
           dict[str, list[int | None]], dict[str, float]]:
    """Return the same cost-family inputs as the image scorer, from folios only."""
    if split not in ("development", "evaluation"):
        raise ValueError("unknown folio split")
    known = panel["roles"][f"{split}_known"]
    unknown = panel["roles"][f"{split}_unknown"]
    if (len(known) != 24 or len(unknown) != 12
            or len({item["chapter"] for item in known + unknown}) != 36):
        raise ValueError("expected 24 known and 12 unknown chapters")
    if split == "evaluation" and normalizers is None:
        raise ValueError("evaluation requires frozen development normalizers")
    queries = known + unknown
    positions = {manuscript: _positions(queries, manuscript)
                 for manuscript in MANUSCRIPTS}
    pairs = {f"{query}->{source}" for query in MANUSCRIPTS
             for source in MANUSCRIPTS if query != source}
    if set(fits) != pairs:
        raise ValueError("six ordered folio maps required")
    if normalizers is not None and (
            set(normalizers) != pairs or any(not isfinite(value) or value <= 0
                                             for value in normalizers.values())):
        raise ValueError("six positive frozen normalizers required")
    raw = {}
    medians = dict(normalizers) if normalizers is not None else {}
    for query in MANUSCRIPTS:
        for source in MANUSCRIPTS:
            if query == source:
                continue
            key = f"{query}->{source}"
            fit = fits[key]
            if (set(fit) != {"slope", "intercept"}
                    or not isfinite(fit["slope"]) or fit["slope"] <= 0
                    or not isfinite(fit["intercept"])):
                raise ValueError(f"invalid frozen folio map {key}")
            matrix = [[abs(fit["slope"] * x + fit["intercept"] - y)
                       for y in positions[source][:24]]
                      for x in positions[query]]
            raw[key] = matrix
            if normalizers is None:
                scale = float(median(value for row in matrix for value in row))
                if not isfinite(scale) or scale <= 0:
                    raise ValueError(f"invalid development distance median {key}")
                medians[key] = scale
    first, second = {}, {}
    for query in MANUSCRIPTS:
        sources = [source for source in MANUSCRIPTS if source != query]
        matrices = [[[value / medians[f"{query}->{source}"] for value in row]
                     for row in raw[f"{query}->{source}"]] for source in sources]
        first[query], second[query] = matrices
    truths = {query: list(range(24)) + [None] * 12 for query in MANUSCRIPTS}
    return first, second, truths, medians


def directed_matrix(panel: dict, split: str, row_order: list[dict],
                    fits: dict[str, dict[str, float]]) -> list[list[float]]:
    """Make a 108-square *directed* distance matrix for shared open-set scoring.

    Same-manuscript entries are unused by the scorer and set to zero. This
    matrix is not symmetric because the fitted query→source maps differ.
    """
    if split not in ("development", "evaluation"):
        raise ValueError("unknown folio split")
    expected = {}
    for role in (f"{split}_known", f"{split}_unknown"):
        for chapter in panel["roles"][role]:
            for manuscript in MANUSCRIPTS:
                key = (role, chapter["chapter"], manuscript)
                expected[key] = folio_side(chapter["pages"][manuscript]["title"])
    keys = [(row["role"], row["chapter_class"], row["manuscript"])
            for row in row_order]
    if len(expected) != 108 or len(keys) != 108 or len(set(keys)) != 108 or set(keys) != set(expected):
        raise ValueError("folio matrix rows differ from frozen split")
    matrix = []
    for query in keys:
        row = []
        for source in keys:
            if query[2] == source[2]:
                row.append(0.0)
            else:
                fit = fits[f"{query[2]}->{source[2]}"]
                row.append(abs(fit["slope"] * expected[query] +
                               fit["intercept"] - expected[source]))
        matrix.append(row)
    return matrix
