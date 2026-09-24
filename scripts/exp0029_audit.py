"""Independent pair-construction, arithmetic and decision audit for EXP-0029."""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re


ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r"[a-z']+\Z")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distance(a: str, b: str) -> int:
    costs = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        next_costs = [i]
        for j, cb in enumerate(b, 1):
            next_costs.append(min(next_costs[-1] + 1, costs[j] + 1,
                                  costs[j - 1] + (ca != cb)))
        costs = next_costs
    return costs[-1]


def logloss(margin: float) -> float:
    return (max(0, -margin) + math.log1p(math.exp(-abs(margin)))) / math.log(2)


def close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-10


def main() -> None:
    result_path = ROOT / "results/EXP-0029/rows.json"
    compact_path = ROOT / "results/EXP-0029/results.json"
    result = json.loads(result_path.read_text())
    compact = json.loads(compact_path.read_text())
    assert compact["raw_archive_sha256"] == digest(result_path)
    assert compact["experiment"] == result["experiment"]
    assert compact["decision"] == result["decision"]
    assert compact["source_sha256"] == result["source_sha256"]
    assert compact["validation"]["observed"] == result["validation"]["observed"]
    assert compact["calibration"] == result["calibration"]
    assert compact["null"] == result["null"]
    assert compact["validation"]["selection"] == result["validation"]["selection"]
    assert compact["validation"]["leaf_bootstrap_95_gain"] == result["validation"]["leaf_bootstrap_95_gain"]
    assert "pairs" not in compact["validation"]
    assert result["experiment"] == "EXP-0029" and result["status"] == "complete"
    assert digest(ROOT / "src/voynich/locus_association.py") == result["source_sha256"]
    train_path = ROOT / "data/processed/zl3b/train.jsonl"
    val_path = ROOT / "data/processed/zl3b/validation.jsonl"
    assert digest(train_path) == result["train_sha256"]
    assert digest(val_path) == result["validation_sha256"]
    train = [json.loads(line) for line in train_path.read_text().splitlines()]
    validation = [json.loads(line) for line in val_path.read_text().splitlines()]
    counts = Counter(word for page in train for locus in page["loci"]
                     for word in locus["text"].split() if WORD.fullmatch(word))
    expected = {}
    stats = Counter()
    for page in validation:
        loci = [(locus["locus_id"], [w for w in locus["text"].split() if WORD.fullmatch(w)])
                for locus in page["loci"] if locus["locus_type"] == "P0"]
        for locus_index, (locus_id, words) in enumerate(loci):
            if len(words) < 4:
                stats["short_locus"] += 1
                continue
            elsewhere = set().union(*(set(other) for j, (_, other) in enumerate(loci)
                                       if j != locus_index))
            own = Counter(words)
            for slot, target in enumerate(words):
                if own[target] != 1:
                    stats["repeated_target"] += 1
                    continue
                if counts[target] < 3:
                    stats["rare_target"] += 1
                    continue
                context = [w for j, w in enumerate(words) if j != slot]
                nearest = min(distance(target, x) for x in context)
                choices = []
                for candidate in elsewhere:
                    if candidate in own or counts[candidate] < 3 or len(candidate) != len(target):
                        continue
                    if not 0.5 <= counts[candidate] / counts[target] <= 2:
                        continue
                    if candidate[0] != target[0] and candidate[-1] != target[-1]:
                        continue
                    if min(distance(candidate, x) for x in context) != nearest:
                        continue
                    choices.append(candidate)
                if not choices:
                    stats["no_matched_negative"] += 1
                    continue
                selected = min(choices, key=lambda candidate: hashlib.sha256(
                    f"290029|{page['page_id']}|{locus_id}|{slot}|{candidate}".encode()).hexdigest())
                key = (page["page_id"], locus_id, slot)
                expected[key] = (page["leaf_id"], target, selected, context)
                stats["selected"] += 1
    rows = result["validation"]["pairs"]
    assert len(rows) == len(expected) == result["validation"]["observed"]["n"]
    assert dict(sorted(stats.items())) == result["validation"]["selection"]
    # f73 has only label/circular loci in this source and no eligible P0 pair.
    assert len({r["leaf"] for r in rows}) == 9
    assert {r["leaf"] for r in rows} == {p["leaf_id"] for p in validation
                                            if any(locus["locus_type"] == "P0" for locus in p["loci"])}
    assert {(r["page"], r["locus_id"], r["slot"]) for r in rows} == set(expected)
    for row in rows:
        key = (row["page"], row["locus_id"], row["slot"])
        assert (row["leaf"], row["target"], row["negative"], row["context"]) == expected[key]
        assert math.isfinite(row["form_margin"]) and math.isfinite(row["full_margin"])
    leaf_totals = defaultdict(lambda: {"pairs": 0, "form_bits_sum": 0.0,
                                      "full_bits_sum": 0.0, "form_correct_sum": 0.0,
                                      "full_correct_sum": 0.0})
    for row in rows:
        leaf = leaf_totals[row["leaf"]]
        leaf["pairs"] += 1
        for name in ("form", "full"):
            margin = row[f"{name}_margin"]
            leaf[f"{name}_bits_sum"] += logloss(margin)
            leaf[f"{name}_correct_sum"] += (margin > 0) + 0.5 * (margin == 0)
    assert set(compact["validation"]["per_leaf"]) == set(leaf_totals)
    for leaf_id, metrics in leaf_totals.items():
        for key, value in metrics.items():
            assert close(value, compact["validation"]["per_leaf"][leaf_id][key])
    form_losses = [logloss(r["form_margin"]) for r in rows]
    full_losses = [logloss(r["full_margin"]) for r in rows]
    observed = result["validation"]["observed"]
    assert close(sum(form_losses) / len(rows), observed["form_bits"])
    assert close(sum(full_losses) / len(rows), observed["full_bits"])
    assert close(sum(a - b for a, b in zip(form_losses, full_losses)) / len(rows),
                 observed["gain_bits"])
    for name, field in (("form_accuracy", "form_margin"), ("full_accuracy", "full_margin")):
        accuracy = sum((r[field] > 0) + 0.5 * (r[field] == 0) for r in rows) / len(rows)
        assert close(accuracy, observed[name])
    by_leaf = defaultdict(list)
    for index, row in enumerate(rows):
        by_leaf[row["leaf"]].append(index)
    leaves = sorted(by_leaf)
    rng = random.Random(290129)
    delta = [a - b for a, b in zip(form_losses, full_losses)]
    draws = []
    for _ in range(2000):
        indices = [i for _ in leaves for i in by_leaf[rng.choice(leaves)]]
        draws.append(sum(delta[i] for i in indices) / len(indices))
    draws.sort()
    ci = result["validation"]["leaf_bootstrap_95_gain"]
    assert close(draws[50], ci[0]) and close(draws[1950], ci[1])
    null = result["null"]
    assert null["n"] == len(null["gains"]) == 100
    passed = (observed["gain_bits"] >= 0.02 and ci[0] > 0
              and observed["gain_bits"] > sorted(null["gains"])[95]
              and observed["full_accuracy"] >= 0.55)
    assert result["decision"] == ("locus_association_support" if passed
                                  else "no_support_under_registered_controls")
    print(json.dumps({"audit": "pass", "pairs": len(rows), "leaves": len(leaves),
                      "decision": result["decision"], "raw_sha256": digest(result_path),
                      "compact_sha256": digest(compact_path)}, indent=2))


if __name__ == "__main__":
    main()
