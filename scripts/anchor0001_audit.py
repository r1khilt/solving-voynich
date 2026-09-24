"""Independent corpus, label-assignment, score and decision audit for ANCHOR-0001."""

from itertools import permutations, product
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r"[a-z']+\Z")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        next_row = [i]
        for j, cb in enumerate(b, 1):
            next_row.append(min(row[j] + 1, next_row[-1] + 1,
                                row[j - 1] + (ca != cb)))
        row = next_row
    return row[-1]


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=1e-12, rel_tol=0)


def main() -> None:
    result_path = ROOT / "results/ANCHOR-0001/results.json"
    result = json.loads(result_path.read_text())
    manifest_path = ROOT / "data/manifests/anchor0001_pairs.json"
    manifest = json.loads(manifest_path.read_text())
    assert result["experiment"] == manifest["id"] == "ANCHOR-0001"
    assert result["status"] == "complete"
    assert result["source_sha256"] == digest(ROOT / "src/voynich/anchor_transfer.py")
    assert result["pair_manifest_sha256"] == digest(manifest_path)
    assert result["split_manifest_sha256"] == digest(ROOT / "data/manifests/zl3b_split.json")
    split = json.loads((ROOT / "data/manifests/zl3b_split.json").read_text())
    paths = {name: ROOT / f"data/processed/zl3b/{name}.jsonl"
             for name in ("train", "validation")}
    assert result["corpus_sha256"] == {name: digest(path) for name, path in paths.items()}
    pages = {}
    for name, path in paths.items():
        for line in path.read_text().splitlines():
            page = json.loads(line)
            assert page["split"] == name
            pages[page["page_id"]] = page
    groups = defaultdict(list)
    for item in manifest["source_match_claims"]:
        source, herb = item["pharma_page"], item["herbal_page"]
        assert pages[source]["split"] == item["transcription_split"]
        assert split["leaf_assignments"][re.match(r"f\d+", herb).group()] == pages[herb]["split"]
        groups[source].append(item)
    assert len(groups) == 3 and len(manifest["source_match_claims"]) == 5
    expected = {}
    null_options = []
    for source in sorted(groups):
        candidate_labels = {}
        ambiguous = []
        for locus in pages[source]["loci"]:
            if locus["locus_type"] != "Lf":
                continue
            locus_id = locus["locus_id"]
            if WORD.fullmatch(locus["text"]):
                candidate_labels[locus_id] = locus["text"]
            else:
                ambiguous.append(locus_id)
        candidates = sorted(candidate_labels,
                            key=lambda locus_id: int(locus_id.rsplit(".", 1)[1]))
        group_option_rows = []
        for item in groups[source]:
            herb = item["herbal_page"]
            words = {word for locus in pages[herb]["loci"] if locus["locus_type"] == "P0"
                     for word in locus["text"].split() if WORD.fullmatch(word)}
            locus_id = item["label_locus_id"]
            assert words and locus_id in candidate_labels
            original_locus = next(locus for locus in pages[source]["loci"]
                                  if locus["locus_id"] == locus_id)
            if item["label_alignment"].startswith("explicit"):
                assert f"<!{item['fragment']}>" in original_locus["raw"]
            else:
                assert source == "f89v2" and locus_id == "f89v2.6"
            scores = []
            best_words = []
            for candidate_id in candidates:
                label = candidate_labels[candidate_id]
                best = min(words, key=lambda word: (distance(label, word) /
                           max(len(label), len(word)), word))
                best_words.append(best)
                scores.append(1 - distance(label, best) / max(len(label), len(best)))
            fractions = []
            ranks = []
            for index, score in enumerate(scores):
                greater = sum(other > score + 1e-12 for j, other in enumerate(scores)
                              if j != index)
                equal = sum(abs(other - score) <= 1e-12 for j, other in enumerate(scores)
                            if j != index)
                rank = 1 + greater + 0.5 * equal
                ranks.append(rank)
                fractions.append((len(candidates) - rank) / (len(candidates) - 1))
            index = candidates.index(locus_id)
            expected[(source, item["fragment"], herb)] = {
                "pharma_page": source, "fragment": item["fragment"],
                "label_locus_id": locus_id,
                "herbal_page": herb, "eligible": True,
                "label": candidate_labels[locus_id],
                "score": scores[index], "nearest_word": best_words[index],
                "exact_match": candidate_labels[locus_id] in words,
                "rank": ranks[index], "rank_fraction": fractions[index],
                "candidate_count": len(candidates),
                "ambiguous_fragments": sorted(ambiguous)}
            group_option_rows.append(fractions)
        null_options.append([sum(group_option_rows[i][j] for i, j in enumerate(indices))
                             for indices in permutations(range(len(candidates)), len(groups[source]))])
    observed_rows = result["rows"]
    assert len(observed_rows) == len(expected) == 5
    for row in observed_rows:
        reference = expected[(row["pharma_page"], row["fragment"], row["herbal_page"])]
        assert set(row) == set(reference)
        for key, value in reference.items():
            if isinstance(value, float):
                assert close(row[key], value)
            else:
                assert row[key] == value
    observed = sum(item["rank_fraction"] for item in expected.values()) / 5
    assert close(result["observed_mean_rank_fraction"], observed)
    direct = sum(item["rank_fraction"] for item in expected.values()
                 if item["fragment"] != 54) / 4
    assert close(result["observed_four_direct_mean_rank_fraction"], direct)
    null = sorted(sum(choice for choice in selected) / 5
                  for selected in product(*null_options))
    p = sum(value >= observed - 1e-12 for value in null) / len(null)
    assert result["null"]["n"] == len(null)
    for key, value in (("mean", sum(null) / len(null)), ("min", min(null)),
                       ("max", max(null)), ("p_one_sided", p)):
        assert close(result["null"][key], value)
    assert result["exact_matches"] == sum(item["exact_match"] for item in expected.values())
    support = (observed >= 0.75 and p <= 0.05
               and sum(item["rank_fraction"] >= 0.5 for item in expected.values()) >= 4)
    assert result["decision"] == ("lexical_transfer_candidate" if support
                                  else "not_supported_in_this_small_panel")
    print(json.dumps({"audit": "pass", "pairs": len(expected), "source_pages": len(groups),
                      "assignments": len(null), "decision": result["decision"],
                      "result_sha256": digest(result_path)}, indent=2))


if __name__ == "__main__":
    main()
