"""Train-only comparison of proxy box gaps with archived blind ink gaps.

The six folios in the paper's 300-boundary direct-pixel sample all belong to
our training split. The script matches their accepted measurements to our
separately built box alignment and records only aggregate statistics.
"""

from __future__ import annotations

import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
INK_SHA = "3c15627dc3ddfca4f331a1fed1e1a290a9b8412ebf9247fe4c9b2778412aedee"
BOX_SHA = "9effc6e34f3f8ab4951d07e762070ca65fa88f912f96f75cfecf9c54b22c4a32"
SEED = 430061
BOOTSTRAPS = 2000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    output = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        output[order[start:end]] = (start + end - 1) / 2
        start = end
    return output


def auc(values: np.ndarray, uncertain: np.ndarray) -> float:
    certain = values[~uncertain]
    small = values[uncertain]
    if not len(certain) or not len(small):
        raise ValueError("Both boundary classes required")
    return float(np.mean((certain[:, None] > small) + 0.5 * (certain[:, None] == small)))


def run() -> dict:
    ink_path = ROOT / "data/raw/external/voynich-units/data/direct_pixel/results_unblinded.csv"
    box_path = ROOT / "data/processed/boundary_geometry/train_pairs.jsonl"
    if sha(ink_path) != INK_SHA or sha(box_path) != BOX_SHA:
        raise AssertionError("Frozen ink/box data changed")
    boxes = [json.loads(line) for line in box_path.read_text().splitlines()]
    lookup = defaultdict(list)
    for box in boxes:
        key = (box["folio"], str(box["visual_line"]), box["left_word"], box["right_word"])
        lookup[key].append(box)
    rows = list(csv.DictReader(ink_path.open()))
    if len(rows) != 300:
        raise AssertionError("Direct-ink archive sample size changed")
    split = json.loads((ROOT / "data/manifests/zl3b_split.json").read_text())["leaf_assignments"]
    matched = []
    accepted = []
    for row in rows:
        if split.get(row["folio"].split("r")[0].split("v")[0]) != "train":
            raise AssertionError("Direct-ink archive includes nontraining folio")
        key = (row["folio"], row["line"], row["left_word"], row["right_word"])
        if len(lookup[key]) != 1:
            raise AssertionError("Ink boundary lacks a unique box/text match")
        matched.append(key)
        if row["include"] == "True":
            if not row["direct_gap_norm_local_boxheight"]:
                raise AssertionError("Accepted ink boundary has no measurement")
            accepted.append((row, lookup[key][0]))
    if len(set(matched)) != len(matched):
        raise AssertionError("Duplicate direct-ink boundary")
    x = np.array([box["gap_over_median_word_width"] for _, box in accepted])
    y = np.array([float(row["direct_gap_norm_local_boxheight"]) for row, _ in accepted])
    ink_px = np.array([float(row["gap_px"]) for row, _ in accepted])
    uncertain = np.array([row["label"] == "," for row, _ in accepted])
    folios = np.array([row["folio"] for row, _ in accepted])
    all_folios = sorted(set(folios))
    box_auc, ink_auc = auc(x, uncertain), auc(y, uncertain)
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rank(x), rank(y))[0, 1])
    by_folio = {}
    for folio in all_folios:
        chosen = folios == folio
        c, u = ink_px[chosen & ~uncertain], ink_px[chosen & uncertain]
        by_folio[folio] = {"certain": len(c), "uncertain": len(u),
                           "ink_mean_certain_minus_uncertain_px":
                           float(c.mean() - u.mean()) if len(c) and len(u) else None}
    rng = np.random.default_rng(SEED)
    boot_box, boot_ink, boot_diff = [], [], []
    members = {folio: np.flatnonzero(folios == folio) for folio in all_folios}
    for _ in range(BOOTSTRAPS):
        selected = rng.choice(all_folios, len(all_folios), replace=True)
        indices = np.concatenate([members[folio] for folio in selected])
        if not np.any(uncertain[indices]) or np.all(uncertain[indices]):
            continue
        a, b = auc(x[indices], uncertain[indices]), auc(y[indices], uncertain[indices])
        boot_box.append(a)
        boot_ink.append(b)
        boot_diff.append(a - b)
    report = {
        "status": "train_only_proxy_sensitivity", "source_ink_sha256": INK_SHA,
        "source_box_rows_sha256": BOX_SHA, "sampled": len(rows), "exact_matches": len(matched),
        "accepted": len(accepted), "accepted_certain": int((~uncertain).sum()),
        "accepted_uncertain": int(uncertain.sum()), "folios": len(all_folios),
        "box_ink_pearson": pearson, "box_ink_spearman": spearman,
        "box_label_auc": box_auc, "ink_label_auc": ink_auc,
        "box_minus_ink_auc": box_auc - ink_auc,
        "ink_mean_certain_px": float(ink_px[~uncertain].mean()),
        "ink_mean_uncertain_px": float(ink_px[uncertain].mean()),
        "ink_mean_difference_px": float(ink_px[~uncertain].mean() - ink_px[uncertain].mean()),
        "folio_ink_means": by_folio, "bootstrap_seed": SEED,
        "bootstrap_draws": len(boot_box),
        "box_auc_folio_bootstrap95": [float(np.quantile(boot_box, .025)),
                                       float(np.quantile(boot_box, .975))],
        "ink_auc_folio_bootstrap95": [float(np.quantile(boot_ink, .025)),
                                       float(np.quantile(boot_ink, .975))],
        "box_minus_ink_auc_folio_bootstrap95": [float(np.quantile(boot_diff, .025)),
                                                 float(np.quantile(boot_diff, .975))],
    }
    path = ROOT / "results/BOUNDARY-CHANNEL-0001/ink_overlap.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("accepted", "box_ink_pearson", "box_ink_spearman",
                       "box_label_auc", "ink_label_auc", "ink_auc_folio_bootstrap95")},
                     indent=2))
    return report


if __name__ == "__main__":
    run()
