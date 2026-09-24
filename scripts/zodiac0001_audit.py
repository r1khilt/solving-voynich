"""Independent corpus-to-decision audit of the frozen ZODIAC-0001 result."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ("f72r3", "f72v3", "f72v2", "f72v1")
VALID = ("f73r", "f73v")
CERTAIN = re.compile(r"[a-z']+\Z")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def almost(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=1e-10, rel_tol=0)


def grams(word: str) -> frozenset[str]:
    symbols = "^" + word + "$"
    return frozenset(symbols[k:k + n] for n in (2, 3)
                     for k in range(len(symbols) - n + 1))


def rank(scores: list[float], index: int) -> tuple[float, float]:
    value = scores[index]
    greater = len([x for x in scores if x > value + 1e-12])
    equal = len([x for x in scores if abs(x - value) <= 1e-12])
    position = 1.0 + greater + (equal - 1) / 2
    return position, (30 - position) / 29


def main() -> None:
    result_path = ROOT / "results/ZODIAC-0001/results.json"
    result = json.loads(result_path.read_text())
    assert result["experiment"] == "ZODIAC-0001" and result["status"] == "complete"
    assert result["source_sha256"] == sha(ROOT / "src/voynich/zodiac_ordinal.py")
    assert result["registration_sha256"] == sha(ROOT / "docs/experiments/ZODIAC-0001.md")
    assert result["split_manifest_sha256"] == sha(ROOT / "data/manifests/zl3b_split.json")
    assert result["train_diagrams"] == list(TRAIN) and result["validation_diagrams"] == list(VALID)
    split = json.loads((ROOT / "data/manifests/zl3b_split.json").read_text())
    assert split["leaf_assignments"]["f72"] == "train"
    assert split["leaf_assignments"]["f73"] == "validation"
    diagrams = {}
    for corpus_split in ("train", "validation"):
        path = ROOT / f"data/processed/zl3b/{corpus_split}.jsonl"
        assert result["corpus_sha256"][corpus_split] == sha(path)
        for raw_line in path.read_text().splitlines():
            page = json.loads(raw_line)
            if page["page_id"] not in set(TRAIN + VALID):
                continue
            assert page["page_id"] not in diagrams and page["split"] == corpus_split
            loci = [locus for locus in page["loci"] if locus["locus_type"] == "Lz"]
            assert len(loci) == len({locus["locus_id"] for locus in loci}) == 30
            diagrams[page["page_id"]] = loci
    assert set(diagrams) == set(TRAIN + VALID)

    training = [(position, locus["text"])
                for page in TRAIN for position, locus in enumerate(diagrams[page])
                if CERTAIN.fullmatch(locus["text"])]
    assert result["train_eligible_count"] == len(training) == 95
    frequencies = Counter(feature for _, word in training for feature in grams(word))
    weights = {feature: 1 + math.log((len(training) + 1) / (count + 1))
               for feature, count in frequencies.items()}
    prototypes = {position: [] for position in range(30)}
    for position, word in training:
        prototypes[position].append(grams(word) & weights.keys())
    assert all(prototypes.values())

    def similarity(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
        if not a or not b:
            return 0.0
        numerator = sum(weights[k] ** 2 for k in a.intersection(b))
        denominator = math.sqrt(sum(weights[k] ** 2 for k in a)
                                * sum(weights[k] ** 2 for k in b))
        return min(1.0, numerator / denominator)

    expected = []
    for page in VALID:
        for position, locus in enumerate(diagrams[page]):
            label = locus["text"]
            if not CERTAIN.fullmatch(label):
                continue
            observed = grams(label) & weights.keys()
            scores = [sum(similarity(observed, other) for other in prototypes[j])
                      / len(prototypes[j]) for j in range(30)]
            ordinal_rank, fraction = rank(scores, position)
            expected.append((page, position, locus["locus_id"], label,
                             scores, ordinal_rank, fraction))
    assert len(result["rows"]) == len(expected) == 55
    for row, reference in zip(result["rows"], expected, strict=True):
        page, position, locus_id, label, scores, ordinal_rank, fraction = reference
        assert row["page"] == page and row["ordinal"] == position
        assert row["locus_id"] == locus_id and row["label"] == label
        assert len(row["scores"]) == len(scores) == 30
        assert all(almost(x, y) for x, y in zip(row["scores"], scores, strict=True))
        assert almost(row["rank"], ordinal_rank) and almost(row["rank_fraction"], fraction)

    fractions = [item[6] for item in expected]
    observed_mean = sum(fractions) / len(fractions)
    assert almost(result["observed_mean_rank_fraction"], observed_mean)
    per_page = {}
    shifted_sums = {}
    for page in VALID:
        items = [item for item in expected if item[0] == page]
        assert result["validation_eligible_counts"][page] == len(items)
        per_page[page] = sum(item[6] for item in items) / len(items)
        assert almost(result["page_means"][page], per_page[page])
        shifted_sums[page] = [sum(rank(item[4], (item[1] + offset) % 30)[1]
                                  for item in items) for offset in range(30)]
    all_null = sorted((x + y) / len(expected)
                      for x in shifted_sums[VALID[0]] for y in shifted_sums[VALID[1]])
    assert len(all_null) == result["null"]["n"] == 900
    p = sum(x >= observed_mean - 1e-12 for x in all_null) / len(all_null)
    for key, value in (("min", all_null[0]), ("max", all_null[-1]),
                       ("mean", sum(all_null) / 900), ("p_one_sided", p)):
        assert almost(result["null"][key], value)
    assert result["top1_count"] == sum(item[5] == 1.0 for item in expected)
    support = (all(result["validation_eligible_counts"][page] >= 25 for page in VALID)
               and observed_mean >= 0.60 and p <= 0.05
               and all(value >= 0.55 for value in per_page.values()))
    assert result["decision"] == ("positional_form_candidate" if support
                                  else "no_support_for_fixed_ordinal_proxy")
    print(json.dumps({"audit": "pass", "validation_labels": len(expected),
                      "rotations": len(all_null), "decision": result["decision"],
                      "result_sha256": sha(result_path)}, indent=2))


if __name__ == "__main__":
    main()
