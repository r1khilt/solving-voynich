"""Predeclared lexical-transfer readout for matched manuscript plant drawings."""

from collections import defaultdict
from itertools import permutations, product
import hashlib
import json
from pathlib import Path
import re
import time


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "ANCHOR-0001"
CERTAIN = re.compile(r"[a-z']+\Z")
TRAIN_SHA = "49618c7be69cef573fe9ad8ae3627ad9f7b495601af1897aafb6082837fceca4"
VALIDATION_SHA = "9bee4149f26fc49b49e4a826b73598dfb79a1e785731c15a1763489315a2efd3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edit_distance(left: str, right: str) -> int:
    costs = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        next_costs = [i]
        for j, b in enumerate(right, 1):
            next_costs.append(min(next_costs[-1] + 1, costs[j] + 1,
                                  costs[j - 1] + (a != b)))
        costs = next_costs
    return costs[-1]


def similarity(label: str, words: set[str]) -> tuple[float, str]:
    if not words:
        raise ValueError("Herbal page has no certain P0 words")
    word = min(words, key=lambda candidate: (
        edit_distance(label, candidate) / max(len(label), len(candidate)), candidate))
    score = 1 - edit_distance(label, word) / max(len(label), len(word))
    return score, word


def rank_fraction(scores: list[float], target_index: int) -> tuple[float, float]:
    if len(scores) < 2:
        raise ValueError("At least two same-page labels are required")
    target = scores[target_index]
    greater = sum(score > target + 1e-12 for i, score in enumerate(scores)
                  if i != target_index)
    tied = sum(abs(score - target) <= 1e-12 for i, score in enumerate(scores)
               if i != target_index)
    rank = 1 + greater + 0.5 * tied
    fraction = (len(scores) - rank) / (len(scores) - 1)
    return fraction, rank


def load_pages() -> tuple[dict[str, dict], dict[str, str]]:
    paths = {"train": ROOT / "data/processed/zl3b/train.jsonl",
             "validation": ROOT / "data/processed/zl3b/validation.jsonl"}
    hashes = {name: sha256(path) for name, path in paths.items()}
    if hashes != {"train": TRAIN_SHA, "validation": VALIDATION_SHA}:
        raise ValueError("Pinned derived train/validation corpus hashes changed")
    pages = {}
    for name, path in paths.items():
        for line in path.read_text().splitlines():
            page = json.loads(line)
            assert page["split"] == name and page["page_id"] not in pages
            pages[page["page_id"]] = page
    return pages, hashes


def fragment_labels(page: dict) -> tuple[dict[str, str], list[str]]:
    labels: dict[str, str] = {}
    ambiguous = []
    for locus in page["loci"]:
        if locus["locus_type"] != "Lf":
            continue
        locus_id = locus["locus_id"]
        if locus_id in labels or locus_id in ambiguous:
            raise ValueError(f"Duplicate Lf locus: {locus_id}")
        if CERTAIN.fullmatch(locus["text"]):
            labels[locus_id] = locus["text"]
        else:
            ambiguous.append(locus_id)
    return labels, sorted(ambiguous)


def herbal_words(page: dict) -> set[str]:
    return {word for locus in page["loci"] if locus["locus_type"] == "P0"
            for word in locus["text"].split() if CERTAIN.fullmatch(word)}


def run() -> dict:
    start = time.perf_counter()
    manifest_path = ROOT / "data/manifests/anchor0001_pairs.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["id"] == EXPERIMENT and len(manifest["source_match_claims"]) == 5
    split = json.loads((ROOT / "data/manifests/zl3b_split.json").read_text())
    pages, hashes = load_pages()
    groups = defaultdict(list)
    for pair in manifest["source_match_claims"]:
        source, target = pair["pharma_page"], pair["herbal_page"]
        source_leaf = re.match(r"f\d+", source).group()
        target_leaf = re.match(r"f\d+", target).group()
        assert split["leaf_assignments"][source_leaf] == pair["transcription_split"]
        assert split["leaf_assignments"][target_leaf] == pages[target]["split"]
        assert pages[source]["split"] == pair["transcription_split"]
        groups[source].append(pair)
    rows = []
    assignment_options = []
    all_eligible = True
    for source in sorted(groups):
        labels, ambiguous = fragment_labels(pages[source])
        candidates = sorted(labels, key=lambda locus_id: int(locus_id.rsplit(".", 1)[1]))
        pairs = groups[source]
        for pair in pairs:
            locus_id = pair["label_locus_id"]
            if locus_id not in labels:
                all_eligible = False
            else:
                locus = next(item for item in pages[source]["loci"]
                             if item["locus_id"] == locus_id)
                if pair["label_alignment"].startswith("explicit"):
                    assert f"<!{pair['fragment']}>" in locus["raw"]
                else:
                    assert source == "f89v2" and locus_id == "f89v2.6"
        if len(candidates) < max(2, len(pairs)):
            all_eligible = False
        if not all_eligible:
            rows.extend({"pharma_page": source, "fragment": pair["fragment"],
                         "label_locus_id": pair["label_locus_id"],
                         "herbal_page": pair["herbal_page"], "eligible": False,
                         "ambiguous_fragments": ambiguous} for pair in pairs)
            continue
        per_pair_fractions = []
        for pair in pairs:
            words = herbal_words(pages[pair["herbal_page"]])
            scored = [similarity(labels[locus_id], words) for locus_id in candidates]
            fractions = [rank_fraction([score for score, _ in scored], i)[0]
                         for i in range(len(candidates))]
            true_index = candidates.index(pair["label_locus_id"])
            fraction, rank = rank_fraction([score for score, _ in scored], true_index)
            score, nearest = scored[true_index]
            rows.append({"pharma_page": source, "fragment": pair["fragment"],
                         "label_locus_id": pair["label_locus_id"],
                         "herbal_page": pair["herbal_page"], "eligible": True,
                         "label": labels[pair["label_locus_id"]], "score": score,
                         "nearest_word": nearest,
                         "exact_match": labels[pair["label_locus_id"]] in words,
                         "rank": rank, "rank_fraction": fraction,
                         "candidate_count": len(candidates), "ambiguous_fragments": ambiguous})
            per_pair_fractions.append(fractions)
        assignment_options.append([sum(per_pair_fractions[i][candidate]
                                       for i, candidate in enumerate(indices))
                                   for indices in permutations(range(len(candidates)), len(pairs))])
    observed = sum(row["rank_fraction"] for row in rows if row["eligible"]) / 5 if all_eligible else None
    direct_only = (sum(row["rank_fraction"] for row in rows
                       if row["eligible"] and row["fragment"] != 54) / 4 if all_eligible else None)
    null = (sorted(sum(option for option in combination) / 5
                   for combination in product(*assignment_options)) if all_eligible else [])
    p_value = sum(value >= observed - 1e-12 for value in null) / len(null) if null else None
    support = (all_eligible and observed >= 0.75 and p_value <= 0.05
               and sum(row["rank_fraction"] >= 0.5 for row in rows) >= 4)
    result = {"experiment": EXPERIMENT, "status": "complete",
              "decision": "lexical_transfer_candidate" if support
                          else "not_supported_in_this_small_panel",
              "source_sha256": sha256(Path(__file__)),
              "pair_manifest_sha256": sha256(manifest_path),
              "corpus_sha256": hashes,
              "split_manifest_sha256": sha256(ROOT / "data/manifests/zl3b_split.json"),
              "rows": rows,
              "observed_mean_rank_fraction": observed,
              "observed_four_direct_mean_rank_fraction": direct_only,
              "null": {"n": len(null), "mean": sum(null) / len(null) if null else None,
                       "min": min(null) if null else None,
                       "max": max(null) if null else None,
                       "p_one_sided": p_value},
              "exact_matches": sum(row["exact_match"] for row in rows if row["eligible"]),
              "wall_seconds": time.perf_counter() - start,
              "limits": "Five published visual matches on three source pages; orthography only; no plaintext."}
    output = ROOT / "results/ANCHOR-0001/results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
