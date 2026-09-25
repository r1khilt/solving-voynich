"""Independent source/stream/scoring replay for EXP-0040.

This uses EXP-0039's separately written auditor for its own source parser and
probability arithmetic, never the EXP-0040 scoring runner.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

from scripts import audit_exp0039 as independent


EXTERNAL_COMMIT = "da5320c47910b9eee54d372e3a68be23bc9639fb"
EXTERNAL_SHA = "d43cf570df996fc5dcddddf39bf39b109ca054b8ef77dede5777ce326d72c80c"
RESULT_REFERENCE_SHA = "e1cb00a0f1ab964622650a30f93a025d712ac7aead82b647a7f413cfbe985329"
FAMILIES = ("stock_flow", "stroke", "timm_faithful", "cardan_grille", "iid_train_words")
N_SEEDS = 16
N_PERM = 200
CPU_CAP = 300.0
COMPOUND_PATTERN = re.compile(
    "|".join(re.escape(x) for x in sorted(independent.COMPOUNDS, key=lambda v: (-len(v), v)))
    + "|[a-z']"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_close(found, saved):
    if not math.isclose(found, saved, abs_tol=1e-10, rel_tol=0):
        raise AssertionError((found, saved))


def replay_external(path):
    code = path / "voynich_mechanism/generators.py"
    assert sha(code) == EXTERNAL_SHA
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip() == EXTERNAL_COMMIT
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=path, text=True).strip()
    sys.path.insert(0, str(path))
    return importlib.import_module("voynich_mechanism.generators")


def regenerate(name, n, profile, training, seed, external):
    if name == "stock_flow":
        result = external.generate_stock_flow(n, profile, training, seed=seed).split()
    elif name == "stroke":
        result = external.generate_stroke(n, profile, training, seed=seed).split()
    elif name == "timm_faithful":
        result = external.generate_timm(n + 1, profile, seed=seed).split()
    elif name == "cardan_grille":
        result = external.generate_cardan_grille(n, profile, seed=seed).split()
    elif name == "iid_train_words":
        result = random.Random(seed).choices(profile.seed_words, weights=profile.seed_w, k=n)
    else:
        raise AssertionError(name)
    assert len(result) == n
    return result


def token_stats(words, train_vocabulary):
    counts = Counter(words)
    return {"novel_token_fraction": sum(w not in train_vocabulary for w in words) / len(words),
            "hapax_type_fraction": sum(count == 1 for count in counts.values()) / len(counts),
            "type_token_ratio": len(counts) / len(words)}


def audit(root: Path, external_path: Path):
    started = time.monotonic()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("EXP-0040 audit requires PYTHONHASHSEED=0 before Python starts")
    external = replay_external(external_path)
    result_path = root / "results/EXP-0040/results.json"
    report = json.loads(result_path.read_text())
    assert report["experiment"] == "EXP-0040" and report["status"] == "complete"
    assert sha(root / "src/voynich/external_generator_transfer.py") == report["source_sha256"]
    assert report["external_commit"] == EXTERNAL_COMMIT
    assert report["external_generator_file_sha256"] == EXTERNAL_SHA
    assert report["pythonhashseed"] == "0"
    paths = [root / "data/raw/v101/GC2a-n.txt",
             root / "data/manifests/zl3b_split.json",
             root / "data/processed/zl3b/train.jsonl",
             root / "data/processed/zl3b/validation.jsonl",
             root / "results/EXP-0039/results.json"]
    expected = [independent.GC_HASH, independent.SPLIT_HASH, independent.TRAIN_HASH,
                independent.VALID_HASH, RESULT_REFERENCE_SHA]
    assert [sha(path) for path in paths] == expected
    split = json.loads(paths[1].read_text())["leaf_assignments"]
    _, _, ids = independent.gc_source(paths[0], split)
    training, _ = independent.zl_source(paths[2], ids)
    skeleton, _ = independent.zl_source(paths[3], ids)
    assert len(training) == report["train_groups"] == 2697
    assert len(skeleton) == report["validation_groups"] == 350
    cleaned = "\n".join(" ".join("".join(word) for word in words)
                        for _, _, _, words in training)
    profile = external.profile_real_corpus(cleaned)
    train_vocab = set(cleaned.split())
    n = sum(len(words) for _, _, _, words in skeleton)
    assert n == report["validation_words"] == 2685
    assert len(cleaned.split()) == report["train_words"] == 19071
    assert sum(len(words) - 1 for _, _, _, words in skeleton) == report["validation_pairs"] == 2335
    model = independent.model_fit(training)
    previous = json.loads(paths[4].read_text())["views"]["zl_basic_matched"]
    real_words = ["".join(word) for _, _, _, words in skeleton for word in words]
    reference = report["real_reference"]
    require_close(reference["last_gain"], previous["real_mean"]["last"])
    require_close(reference["last_minus_shuffle"], previous["contrasts"]["last_minus_null"])
    for key, value in token_stats(real_words, train_vocab).items():
        require_close(value, reference["lexical"][key])
    assert reference["bootstrap95"] == previous["bootstrap95"]
    assert set(report["families"]) == set(FAMILIES)
    n_streams = n_permutations = 0
    for family_index, family in enumerate(FAMILIES):
        records = report["families"][family]["seeds"]
        assert len(records) == N_SEEDS
        for rep, saved in enumerate(records):
            if time.monotonic() - started > CPU_CAP:
                raise TimeoutError("EXP-0040 auditor CPU cap exceeded")
            seed = 400000 + 1000 * family_index + rep
            assert saved["seed"] == seed
            words = regenerate(family, n, profile, cleaned, seed, external)
            stream = "\n".join(words) + "\n"
            saved_text = (root / f"outputs/EXP-0040/{family}-seed{seed}.txt").read_text()
            assert stream == saved_text
            assert hashlib.sha256(stream.encode("ascii")).hexdigest() == saved["stream_sha256"]
            synthetic = []
            cursor = 0
            for leaf, locus, context, row in skeleton:
                length = len(row)
                segment = words[cursor:cursor + length]
                atoms = [tuple(COMPOUND_PATTERN.findall(word)) for word in segment]
                assert all("".join(parts) == word for parts, word in zip(atoms, segment))
                synthetic.append((leaf, locus, context, atoms))
                cursor += length
            assert cursor == n
            scored = independent.score(synthetic, model)
            last = independent.aggregate(scored, 0)
            first = independent.aggregate(scored, 1)
            shuffles = []
            for perm in range(N_PERM):
                control_seed = 410000 + 100000 * family_index + 1000 * rep + perm
                shuffles.append(independent.aggregate(
                    independent.score(independent.scramble(synthetic, control_seed), model), 0))
            null_mean = sum(shuffles) / N_PERM
            require_close(last, saved["last_gain"])
            require_close(null_mean, saved["shuffle_last_mean"])
            require_close(last - null_mean, saved["last_minus_shuffle"])
            require_close(last - first, saved["last_minus_first"])
            for key, value in token_stats(words, train_vocab).items():
                require_close(value, saved[key])
            n_streams += 1
            n_permutations += N_PERM
        summary = report["families"][family]["summary"]
        for key in ("last_gain", "shuffle_last_mean", "last_minus_shuffle",
                    "last_minus_first", "novel_token_fraction", "hapax_type_fraction",
                    "type_token_ratio"):
            sorted_values = sorted(row[key] for row in records)
            assert len(summary[key]["values_sorted"]) == len(sorted_values)
            for value, saved in zip(sorted_values, summary[key]["values_sorted"]):
                require_close(value, saved)
            require_close(sorted_values[0], summary[key]["min"])
            require_close(sorted_values[-1], summary[key]["max"])
            require_close(sum(sorted_values) / N_SEEDS, summary[key]["mean"])
        low = previous["bootstrap95"]["last_minus_null"][0] - .02
        high = previous["bootstrap95"]["last_minus_null"][1] + .02
        for value, saved in zip((low, high), summary["reference_margin_band"]):
            require_close(value, saved)
        lo = summary["last_minus_shuffle"]["min"]
        hi = summary["last_minus_shuffle"]["max"]
        decision = ("edge_below_reference" if hi < low else
                    "edge_above_reference" if lo > high else
                    "not_falsified_by_this_edge_assay")
        assert summary["edge_decision"] == decision
        assert summary["novelty_gap_over_005"] == (
            abs(summary["novel_token_fraction"]["mean"] -
                reference["lexical"]["novel_token_fraction"]) > .05)
        assert summary["hapax_gap_over_005"] == (
            abs(summary["hapax_type_fraction"]["mean"] -
                reference["lexical"]["hapax_type_fraction"]) > .05)
    assert n_streams == 80 and n_permutations == 16000
    return {"experiment": "EXP-0040", "audit": "pass", "result_sha256": sha(result_path),
            "external_commit": EXTERNAL_COMMIT, "external_generator_file_sha256": EXTERNAL_SHA,
            "generated_streams": n_streams, "shuffles_recomputed": n_permutations,
            "wall_seconds": time.monotonic() - started,
            "scope": "Independent input parsing, generator replay, scorer, shuffles, summaries and gates."}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--external", required=True, type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    record = audit(root, args.external.resolve())
    target = root / "results/EXP-0040/audit.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
