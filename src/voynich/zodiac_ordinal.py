"""Registered position-transfer assay for Voynich zodiac figure labels."""

from collections import Counter, defaultdict
from itertools import product
import hashlib
import json
import math
from pathlib import Path
import re
import time


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PAGES = ("f72r3", "f72v3", "f72v2", "f72v1")
VALIDATION_PAGES = ("f73r", "f73v")
EXPECTED_SHA = {
    "train": "49618c7be69cef573fe9ad8ae3627ad9f7b495601af1897aafb6082837fceca4",
    "validation": "9bee4149f26fc49b49e4a826b73598dfb79a1e785731c15a1763489315a2efd3",
}
WORD = re.compile(r"[a-z']+\Z")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ngrams(word: str) -> set[str]:
    padded = "^" + word + "$"
    return {padded[i:i + width] for width in (2, 3)
            for i in range(len(padded) - width + 1)}


def make_idf(words: list[str]) -> dict[str, float]:
    document_frequency = Counter(g for word in words for g in ngrams(word))
    return {g: 1.0 + math.log((len(words) + 1) / (frequency + 1))
            for g, frequency in document_frequency.items()}


def vector(word: str, idf: dict[str, float]) -> dict[str, float]:
    return {g: idf[g] for g in ngrams(word) if g in idf}


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    left_norm = math.sqrt(sum(x * x for x in left.values()))
    right_norm = math.sqrt(sum(x * x for x in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot = sum(x * right.get(g, 0.0) for g, x in left.items())
    return min(1.0, max(0.0, dot / (left_norm * right_norm)))


def rank_fraction(scores: list[float], target: int) -> tuple[float, float]:
    if len(scores) != 30 or not 0 <= target < 30:
        raise ValueError("Expected 30 candidate positions and a valid target")
    value = scores[target]
    better = sum(x > value + 1e-12 for j, x in enumerate(scores) if j != target)
    tied = sum(abs(x - value) <= 1e-12 for j, x in enumerate(scores) if j != target)
    rank = 1.0 + better + 0.5 * tied
    return (30.0 - rank) / 29.0, rank


def load_diagrams() -> tuple[dict[str, list[dict]], dict[str, str]]:
    paths = {split: ROOT / f"data/processed/zl3b/{split}.jsonl"
             for split in ("train", "validation")}
    hashes = {split: digest(path) for split, path in paths.items()}
    if hashes != EXPECTED_SHA:
        raise ValueError("Pinned ZL3b derived corpus changed")
    assignments = json.loads((ROOT / "data/manifests/zl3b_split.json").read_text())["leaf_assignments"]
    assert assignments["f72"] == "train" and assignments["f73"] == "validation"
    wanted = set(TRAIN_PAGES + VALIDATION_PAGES)
    diagrams: dict[str, list[dict]] = {}
    for split, path in paths.items():
        for line in path.read_text().splitlines():
            page = json.loads(line)
            page_id = page["page_id"]
            if page_id not in wanted:
                continue
            assert page_id not in diagrams and page["split"] == split
            labels = [locus for locus in page["loci"] if locus["locus_type"] == "Lz"]
            if len(labels) != 30 or len({locus["locus_id"] for locus in labels}) != 30:
                raise ValueError(f"Expected 30 distinct Lz loci on {page_id}")
            diagrams[page_id] = labels
    if set(diagrams) != wanted:
        raise ValueError("Missing a registered zodiac diagram")
    return diagrams, hashes


def score_diagrams(diagrams: dict[str, list[dict]]) -> tuple[list[dict], dict[str, float], list[float]]:
    train = [(page_id, index, locus["text"])
             for page_id in TRAIN_PAGES for index, locus in enumerate(diagrams[page_id])
             if WORD.fullmatch(locus["text"])]
    if len(train) != 95:
        raise ValueError("Unexpected training-label eligibility")
    idf = make_idf([word for _, _, word in train])
    vectors = defaultdict(list)
    for _, index, word in train:
        vectors[index].append(vector(word, idf))
    if len(vectors) != 30:
        raise ValueError("Missing train prototype for an ordinal")
    rows = []
    for page_id in VALIDATION_PAGES:
        for index, locus in enumerate(diagrams[page_id]):
            word = locus["text"]
            if not WORD.fullmatch(word):
                continue
            query = vector(word, idf)
            scores = [sum(cosine(query, candidate) for candidate in vectors[j]) / len(vectors[j])
                      for j in range(30)]
            fraction, rank = rank_fraction(scores, index)
            rows.append({"page": page_id, "ordinal": index, "locus_id": locus["locus_id"],
                         "label": word, "scores": scores, "rank": rank, "rank_fraction": fraction})
    if len(rows) != 55:
        raise ValueError("Unexpected validation-label eligibility")
    page_means = {page: sum(row["rank_fraction"] for row in rows if row["page"] == page)
                  / sum(row["page"] == page for row in rows) for page in VALIDATION_PAGES}
    rotation_totals = {}
    for page in VALIDATION_PAGES:
        page_rows = [row for row in rows if row["page"] == page]
        rotation_totals[page] = [sum(rank_fraction(row["scores"], (row["ordinal"] + shift) % 30)[0]
                                     for row in page_rows) for shift in range(30)]
    null = sorted((left + right) / len(rows)
                  for left, right in product(rotation_totals[VALIDATION_PAGES[0]],
                                             rotation_totals[VALIDATION_PAGES[1]]))
    return rows, page_means, null


def run() -> dict:
    started = time.perf_counter()
    diagrams, hashes = load_diagrams()
    rows, page_means, null = score_diagrams(diagrams)
    observed = sum(row["rank_fraction"] for row in rows) / len(rows)
    p_value = sum(value >= observed - 1e-12 for value in null) / len(null)
    support = (sum(row["page"] == "f73r" for row in rows) >= 25
               and sum(row["page"] == "f73v" for row in rows) >= 25
               and observed >= 0.60 and p_value <= 0.05
               and all(value >= 0.55 for value in page_means.values()))
    result = {
        "experiment": "ZODIAC-0001", "status": "complete",
        "decision": "positional_form_candidate" if support else "no_support_for_fixed_ordinal_proxy",
        "source_sha256": digest(Path(__file__)),
        "registration_sha256": digest(ROOT / "docs/experiments/ZODIAC-0001.md"),
        "corpus_sha256": hashes,
        "split_manifest_sha256": digest(ROOT / "data/manifests/zl3b_split.json"),
        "train_diagrams": list(TRAIN_PAGES), "validation_diagrams": list(VALIDATION_PAGES),
        "train_eligible_count": 95,
        "validation_eligible_counts": {page: sum(row["page"] == page for row in rows)
                                       for page in VALIDATION_PAGES},
        "rows": rows,
        "observed_mean_rank_fraction": observed,
        "page_means": page_means,
        "top1_count": sum(row["rank"] == 1.0 for row in rows),
        "null": {"n": len(null), "mean": sum(null) / len(null), "min": min(null),
                 "max": max(null), "p_one_sided": p_value},
        "wall_seconds": time.perf_counter() - started,
        "limits": "IVTFF-order proxy; two adjacent leaves only; orthographic position, not plaintext.",
    }
    output = ROOT / "results/ZODIAC-0001/results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
