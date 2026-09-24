"""Independent aggregate and registered-decision audit for EXP-0027."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-10


def main() -> None:
    result = json.loads((ROOT / "results/EXP-0027/results.json").read_text())
    assert result["experiment"] == "EXP-0027" and result["status"] == "complete"
    assert digest(ROOT / "src/voynich/word_copy_channel.py") == result["source_sha256"]
    assert digest(ROOT / "data/processed/zl3b/train.jsonl") == result["train_sha256"]
    assert digest(ROOT / "data/processed/zl3b/validation.jsonl") == result["validation_sha256"]
    raw_pages = [json.loads(line) for line in (ROOT / "data/processed/zl3b/validation.jsonl").read_text().splitlines()]
    expected = {page["page_id"]: (
        page["leaf_id"],
        sum(bool(re.fullmatch(r"[a-z']+", word)) for locus in page["loci"] for word in locus["text"].split()),
    ) for page in raw_pages}
    rows = result["validation"]["pages"]
    assert len(rows) == len(expected) == 24
    assert len({row["leaf"] for row in rows}) == 10
    assert {row["page"] for row in rows} == set(expected)
    for row in rows:
        assert (row["leaf"], row["count"]) == expected[row["page"]]
        assert 0 <= row["p0_count"] <= row["count"]
        assert set(row["bits"]) == {"base", "exact", "edit"}
        assert all(v > 0 for v in row["bits"].values())

    n = sum(row["count"] for row in rows)
    observed = result["validation"]["observed"]
    assert n == observed["count"]
    bits = {arm: sum(row["bits"][arm] for row in rows) / n
            for arm in ("base", "exact", "edit")}
    for arm, value in bits.items():
        assert close(value, observed["bits_per_word"][arm])
    assert close(bits["exact"] - bits["edit"], observed["exact_minus_edit"])
    assert close(bits["base"] - bits["edit"], observed["base_minus_edit"])
    assert close(bits["base"] - bits["exact"], observed["base_minus_exact"])

    # Repeat the specified leaf bootstrap using an independent aggregation path.
    groups = defaultdict(list)
    for row in rows:
        groups[row["leaf"]].append(row)
    leaves = sorted(groups)
    rng = random.Random(270027)
    draws = []
    for _ in range(2000):
        sampled = [groups[rng.choice(leaves)] for _ in leaves]
        numerator = sum(r["bits"]["exact"] - r["bits"]["edit"]
                        for group in sampled for r in group)
        denominator = sum(r["count"] for group in sampled for r in group)
        draws.append(numerator / denominator)
    draws.sort()
    ci = result["validation"]["leaf_bootstrap_95_exact_minus_edit"]
    assert close(draws[50], ci[0]) and close(draws[1950], ci[1])

    order = result["order_control"]
    assert order["n"] == 100
    assert len(order["base_minus_edit"]) == len(order["exact_minus_edit"]) == 100
    null95 = sorted(order["base_minus_edit"])[95]
    positive = result["positive_control"]["metrics"]["exact_minus_edit"]
    passed = (observed["exact_minus_edit"] >= 0.05 and ci[0] > 0
              and observed["base_minus_edit"] > null95 and positive > 0)
    assert result["decision"] == ("support_copy_channel_research" if passed
                                  else "no_support_under_registered_gate")
    print(json.dumps({"audit": "pass", "pages": len(rows), "leaves": len(leaves),
                      "words": n, "decision": result["decision"],
                      "result_sha256": digest(ROOT / "results/EXP-0027/results.json")}, indent=2))


if __name__ == "__main__":
    main()
