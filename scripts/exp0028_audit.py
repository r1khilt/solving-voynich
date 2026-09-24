"""Independent page-arithmetic and registered-decision audit for EXP-0028."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a: float, b: float) -> bool:
    return abs(a - b) < 1e-10


def main() -> None:
    result_path = ROOT / "results/EXP-0028/results.json"
    result = json.loads(result_path.read_text())
    assert result["experiment"] == "EXP-0028" and result["status"] == "complete"
    assert sha(ROOT / "src/voynich/word_copy_followup.py") == result["source_sha256"]
    assert sha(ROOT / "results/EXP-0027/results.json") == result["exp0027_sha256"]
    assert sha(ROOT / "data/processed/zl3b/train.jsonl") == result["train_sha256"]
    assert sha(ROOT / "data/processed/zl3b/validation.jsonl") == result["validation_sha256"]
    pages = [json.loads(line) for line in (ROOT / "data/processed/zl3b/validation.jsonl").read_text().splitlines()]
    expected = {page["page_id"]: (page["leaf_id"], sum(
        bool(re.fullmatch(r"[a-z']+", w)) for locus in page["loci"]
        for w in locus["text"].split())) for page in pages}
    rows = result["validation"]["pages"]
    assert len(rows) == len(expected) == 24
    assert len({r["leaf"] for r in rows}) == 10
    assert {r["page"] for r in rows} == set(expected)
    for row in rows:
        assert (row["leaf"], row["count"]) == expected[row["page"]]
        assert set(row["bits"]) == {"base", "exact", "edit"}
        assert all(b > 0 for b in row["bits"].values())
    n = sum(r["count"] for r in rows)
    observed = result["validation"]["observed"]
    assert n == observed["count"] == 3700
    b = {key: sum(r["bits"][key] for r in rows) / n for key in ("base", "exact", "edit")}
    for key in b:
        assert same(b[key], observed["bits_per_word"][key])
    for name, left, right in (("exact_minus_edit", "exact", "edit"),
                              ("base_minus_edit", "base", "edit"),
                              ("base_minus_exact", "base", "exact")):
        assert same(b[left] - b[right], observed[name])
    by_leaf = defaultdict(list)
    for row in rows:
        by_leaf[row["leaf"]].append(row)
    rng = random.Random(270027)
    leaves = sorted(by_leaf)
    bootstrap = []
    for _ in range(2000):
        groups = [by_leaf[rng.choice(leaves)] for _ in leaves]
        diff = sum(r["bits"]["exact"] - r["bits"]["edit"] for g in groups for r in g)
        count = sum(r["count"] for g in groups for r in g)
        bootstrap.append(diff / count)
    bootstrap.sort()
    interval = result["validation"]["leaf_bootstrap_95_exact_minus_edit"]
    assert same(interval[0], bootstrap[50]) and same(interval[1], bootstrap[1950])
    controls = result["order_controls"]
    assert set(controls) == {"type", "locus"}
    for control in controls.values():
        assert control["n"] == 100
        assert len(control["exact_minus_edit"]) == len(control["base_minus_edit"]) == 100
    passed = (observed["exact_minus_edit"] >= 0.03 and interval[0] > 0
              and all(observed["exact_minus_edit"] > sorted(c["exact_minus_edit"])[95]
                      for c in controls.values())
              and result["positive_control"]["exact_minus_edit"] > 0)
    assert result["decision"] == ("local_edit_order_support" if passed
                                  else "not_supported_under_stronger_controls")
    print(json.dumps({"audit": "pass", "result_sha256": sha(result_path),
                      "decision": result["decision"], "pages": len(rows), "leaves": len(leaves),
                      "words": n}, indent=2))


if __name__ == "__main__":
    main()
