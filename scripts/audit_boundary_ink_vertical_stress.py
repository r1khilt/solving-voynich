"""Independent rank/cluster-bootstrap audit of the six-folio ink stress table."""

from __future__ import annotations

import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data/raw/external/voynich-units/data/direct_pixel/results_unblinded.csv"
ROWS = ROOT / "data/processed/boundary_ink_0002/vertical_stress_rows.jsonl"
REPORT = ROOT / "results/BOUNDARY-CHANNEL-0002/vertical_stress.json"
SOURCE_SHA = "3c15627dc3ddfca4f331a1fed1e1a290a9b8412ebf9247fe4c9b2778412aedee"
NAMES = ["primary", "overlap_projection", "same_scanline", "shift_-20", "shift_+20",
         "xshift_-20", "xshift_+20"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc_rank(values: list[float], labels: list[bool]) -> float:
    certain = [value for value, uncertain in zip(values, labels, strict=True) if not uncertain]
    weak = [value for value, uncertain in zip(values, labels, strict=True) if uncertain]
    if not certain or not weak:
        raise AssertionError("AUC class empty")
    wins = sum((a > b) + .5 * (a == b) for a in certain for b in weak)
    return wins / (len(certain) * len(weak))


def verify_metric(rows: list[dict], metric: str, recorded: dict, seed: int, draws: int) -> float:
    eligible = [row for row in rows if row[metric] is not None]
    values = [float(row[metric]) for row in eligible]
    labels = [row["label"] == "," for row in eligible]
    folios = [row["folio"] for row in eligible]
    counts = {"n": len(eligible), "certain": labels.count(False),
              "uncertain": labels.count(True)}
    for key, actual in counts.items():
        if actual != recorded[key]:
            raise AssertionError(f"{metric} {key}: {actual} != {recorded[key]}")
    errors = []
    for label, key in ((False, "certain"), (True, "uncertain")):
        mean = float(np.mean([value for value, weak in zip(values, labels, strict=True)
                              if weak == label]))
        errors.append(abs(mean - recorded["mean_px"][key]))
    value = auc_rank(values, labels)
    errors.append(abs(value - recorded["auc"]))
    by_folio = {}
    for folio in sorted(set(folios)):
        certain = [v for v, lab, f in zip(values, labels, folios, strict=True)
                   if f == folio and not lab]
        uncertain = [v for v, lab, f in zip(values, labels, folios, strict=True)
                     if f == folio and lab]
        difference = float(np.mean(certain) - np.mean(uncertain)) if certain and uncertain else None
        saved = recorded["by_folio"][folio]
        if (len(certain), len(uncertain)) != (saved["certain"], saved["uncertain"]):
            raise AssertionError("Folio group counts differ")
        if difference is None:
            if saved["difference"] is not None:
                raise AssertionError("Folio difference unexpectedly defined")
        else:
            errors.append(abs(difference - saved["difference"]))
        by_folio[folio] = [i for i, f in enumerate(folios) if f == folio]
    unique = sorted(by_folio)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = [index for folio in sampled for index in by_folio[folio]]
        bs_labels = [labels[i] for i in indices]
        if any(bs_labels) and not all(bs_labels):
            boot.append(auc_rank([values[i] for i in indices], bs_labels))
    interval = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
    errors.extend(abs(a - b) for a, b in zip(interval, recorded["folio_bootstrap95_auc"], strict=True))
    certain_by_pair = defaultdict(list)
    certain_by_folio_pair = defaultdict(list)
    for row in eligible:
        if row["label"] == ".":
            certain_by_pair[row["edge_pair"]].append(row[metric])
            certain_by_folio_pair[(row["folio"], row["edge_pair"])].append(row[metric])
    matched = [float(np.mean(certain_by_pair[row["edge_pair"]]) - row[metric])
               for row in eligible if row["label"] == "," and certain_by_pair[row["edge_pair"]]]
    local = [float(np.mean(certain_by_folio_pair[(row["folio"], row["edge_pair"])])
                   - row[metric]) for row in eligible if row["label"] == ","
             and certain_by_folio_pair[(row["folio"], row["edge_pair"])]]
    if len(matched) != recorded["edge_pair_matched_uncertain"] or len(local) != recorded[
        "within_folio_edge_pair_matched_uncertain"]:
        raise AssertionError("Edge-pair matching counts differ")
    if matched:
        errors.append(abs(float(np.mean(matched)) - recorded["edge_pair_matched_mean_difference_px"]))
    if local:
        errors.append(abs(float(np.mean(local)) - recorded[
            "within_folio_edge_pair_matched_mean_difference_px"]))
    if max(errors, default=0) > 1e-12:
        raise AssertionError(f"{metric} numerical replay differs: {max(errors)}")
    return max(errors, default=0)


def run() -> dict:
    if sha(SOURCE) != SOURCE_SHA:
        raise AssertionError("Archived pilot source changed")
    report = json.loads(REPORT.read_text())
    if sha(ROWS) != report["rows_sha256"]:
        raise AssertionError("Derived row digest changed")
    rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
    original = {row["blind_id"]: row for row in csv.DictReader(SOURCE.open())
                if row["include"] == "True"}
    if len(rows) != len(original) or len(set(row["blind_id"] for row in rows)) != len(rows):
        raise AssertionError("Accepted ID coverage differs")
    for row in rows:
        source = original[row["blind_id"]]
        if (row["folio"], row["label"], row["primary"]) != (
            source["folio"], source["label"], float(source["gap_px"])):
            raise AssertionError("Primary source row differs")
    errors = {}
    for i, name in enumerate(NAMES):
        errors[name] = verify_metric(rows, name, report["summaries"][name],
                                     report["seed"] + i, report["bootstrap_draws"])
        if name != "primary":
            same_rows = [row for row in rows if row[name] is not None]
            errors[f"primary_on_{name}"] = verify_metric(
                same_rows, "primary", report["primary_on_same_rows"][name],
                report["seed"] + 100 + i, report["bootstrap_draws"])
    audit = {"status": "pass", "rows_sha256": report["rows_sha256"],
             "report_sha256": sha(REPORT), "archived_source_sha256": SOURCE_SHA,
             "accepted_rows": len(rows), "metrics_replayed": len(errors),
             "max_abs_numerical_error": max(errors.values())}
    path = ROOT / "results/BOUNDARY-CHANNEL-0002/vertical_stress_audit.json"
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2))
    return audit


if __name__ == "__main__":
    run()
