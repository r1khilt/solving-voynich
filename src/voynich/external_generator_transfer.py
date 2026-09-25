"""EXP-0040: train-only external generators challenged on EXP-0039's edge assay.

The third-party generator checkout is imported at runtime after its commit and
file digest are checked. Neither its code nor generated manuscript-like text is
tracked in this repository.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
import platform
from pathlib import Path
import random
import subprocess
import sys
import time

from . import boundary_cross_transcription as edge


EXPERIMENT = "EXP-0040"
EXTERNAL_COMMIT = "da5320c47910b9eee54d372e3a68be23bc9639fb"
EXTERNAL_FILE_SHA = "d43cf570df996fc5dcddddf39bf39b109ca054b8ef77dede5777ce326d72c80c"
REFERENCE_SHA = "e1cb00a0f1ab964622650a30f93a025d712ac7aead82b647a7f413cfbe985329"
FAMILIES = ("stock_flow", "stroke", "timm_faithful", "cardan_grille", "iid_train_words")
N_SEEDS = 16
N_PERM = 200
CPU_CAP = 300.0


def check_external(directory: Path):
    code = directory / "voynich_mechanism/generators.py"
    if edge.sha(code) != EXTERNAL_FILE_SHA:
        raise ValueError("External generator file SHA changed")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=directory, text=True,
    ).strip()
    if revision != EXTERNAL_COMMIT:
        raise ValueError("External generator commit changed")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=directory, text=True,
    ).strip()
    if dirty:
        raise ValueError("External generator checkout is dirty")
    sys.path.insert(0, str(directory))
    return importlib.import_module("voynich_mechanism.generators")


def make_stream(name: str, n_words: int, profile, cleaned: str, seed: int, external) -> list[str]:
    if name == "stock_flow":
        text = external.generate_stock_flow(n_words, profile, cleaned, seed=seed)
        words = text.split()
    elif name == "stroke":
        text = external.generate_stroke(n_words, profile, cleaned, seed=seed)
        words = text.split()
    elif name == "timm_faithful":
        # The external implementation omits its first internally sampled word.
        text = external.generate_timm(n_words + 1, profile, seed=seed)
        words = text.split()
    elif name == "cardan_grille":
        text = external.generate_cardan_grille(n_words, profile, seed=seed)
        words = text.split()
    elif name == "iid_train_words":
        rng = random.Random(seed)
        words = rng.choices(profile.seed_words, weights=profile.seed_w, k=n_words)
    else:
        raise ValueError(f"Unknown generator family {name}")
    if len(words) != n_words or any(not edge.ZL_WORD.fullmatch(w) for w in words):
        raise ValueError(f"External generator {name} produced an invalid word stream")
    return words


def wrap_stream(words: list[str], skeleton: list[dict]) -> list[dict]:
    expected = sum(len(group["words"]) for group in skeleton)
    if len(words) != expected:
        raise ValueError("Generated token count differs from validation skeleton")
    generated = []
    pos = 0
    for group in skeleton:
        n = len(group["words"])
        generated.append({"leaf": group["leaf"], "locus": group["locus"],
                          "context": group["context"],
                          "words": [edge.zl_symbols(w) for w in words[pos:pos + n]]})
        pos += n
    assert pos == expected
    return generated


def lexical_stats(words: list[str], train_vocab: set[str]) -> dict[str, float]:
    counts = Counter(words)
    return {
        "novel_token_fraction": sum(w not in train_vocab for w in words) / len(words),
        "hapax_type_fraction": sum(n == 1 for n in counts.values()) / len(counts),
        "type_token_ratio": len(counts) / len(words),
    }


def stream_sha(words: list[str]) -> str:
    return hashlib.sha256(("\n".join(words) + "\n").encode("ascii")).hexdigest()


def summarize(records: list[dict], real_reference: dict) -> dict:
    if len(records) != N_SEEDS:
        raise ValueError("Incomplete seed panel")
    fields = ("last_gain", "shuffle_last_mean", "last_minus_shuffle",
              "last_minus_first", "novel_token_fraction", "hapax_type_fraction",
              "type_token_ratio")
    summary = {}
    for field in fields:
        values = sorted(row[field] for row in records)
        summary[field] = {"mean": sum(values) / len(values),
                          "min": values[0], "max": values[-1],
                          "values_sorted": values}
    low = real_reference["bootstrap95"]["last_minus_null"][0] - 0.02
    high = real_reference["bootstrap95"]["last_minus_null"][1] + 0.02
    contrasts = summary["last_minus_shuffle"]
    if contrasts["max"] < low:
        decision = "edge_below_reference"
    elif contrasts["min"] > high:
        decision = "edge_above_reference"
    else:
        decision = "not_falsified_by_this_edge_assay"
    summary["edge_decision"] = decision
    summary["reference_margin_band"] = [low, high]
    summary["novelty_gap_over_005"] = (
        abs(summary["novel_token_fraction"]["mean"] -
            real_reference["lexical"]["novel_token_fraction"]) > 0.05)
    summary["hapax_gap_over_005"] = (
        abs(summary["hapax_type_fraction"]["mean"] -
            real_reference["lexical"]["hapax_type_fraction"]) > 0.05)
    return summary


def run(root: Path, external_dir: Path) -> dict:
    started = time.monotonic()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("EXP-0040 requires PYTHONHASHSEED=0 before Python starts")
    external = check_external(external_dir)
    inputs = [root / "data/raw/v101/GC2a-n.txt",
              root / "data/manifests/zl3b_split.json",
              root / "data/processed/zl3b/train.jsonl",
              root / "data/processed/zl3b/validation.jsonl",
              root / "results/EXP-0039/results.json"]
    actual = [edge.sha(p) for p in inputs]
    expected = [edge.GC_SHA, edge.SPLIT_SHA, edge.ZL_TRAIN_SHA,
                edge.ZL_VALID_SHA, REFERENCE_SHA]
    if actual != expected:
        raise ValueError("EXP-0040 pinned input changed")
    split = json.loads(inputs[1].read_text())["leaf_assignments"]
    _, _, ids = edge.load_gc(inputs[0], split)
    train, _ = edge.load_zl(inputs[2], ids)
    skeleton, _ = edge.load_zl(inputs[3], ids)
    if len(train) != 2697 or len(skeleton) != 350:
        raise ValueError("Matched input coverage changed")
    clean_train = "\n".join(" ".join("".join(word) for word in group["words"])
                            for group in train)
    profile = external.profile_real_corpus(clean_train)
    train_vocab = set(clean_train.split())
    n_words = sum(len(group["words"]) for group in skeleton)
    reference = json.loads(inputs[4].read_text())["views"]["zl_basic_matched"]
    real_words = ["".join(word) for group in skeleton for word in group["words"]]
    if reference["validation_pairs"] != 2335 or n_words != 2685:
        raise ValueError("Unexpected EXP-0039 validation denominator")
    real_reference = {"last_gain": reference["real_mean"]["last"],
                      "last_minus_shuffle": reference["contrasts"]["last_minus_null"],
                      "bootstrap95": reference["bootstrap95"],
                      "lexical": lexical_stats(real_words, train_vocab)}
    scorer = edge.fit(train)
    out_dir = root / "outputs/EXP-0040"
    out_dir.mkdir(parents=True, exist_ok=False)
    panel = {}
    for family_index, family in enumerate(FAMILIES):
        records = []
        for rep in range(N_SEEDS):
            if time.monotonic() - started > CPU_CAP:
                raise TimeoutError("EXP-0040 runner CPU cap exceeded")
            seed = 400000 + 1000 * family_index + rep
            words = make_stream(family, n_words, profile, clean_train, seed, external)
            generated = wrap_stream(words, skeleton)
            stream_path = out_dir / f"{family}-seed{seed}.txt"
            stream_path.write_text("\n".join(words) + "\n", encoding="ascii")
            real_score = edge.score(generated, scorer)
            last = edge.average(real_score, "last")
            first = edge.average(real_score, "first")
            shuffled = []
            for permutation in range(N_PERM):
                shuffle_seed = 410000 + 100000 * family_index + 1000 * rep + permutation
                shuffled.append(edge.average(edge.score(edge.shuffle(generated, shuffle_seed), scorer),
                                             "last"))
            null_mean = sum(shuffled) / N_PERM
            records.append({"seed": seed, "stream_sha256": stream_sha(words),
                            "last_gain": last, "shuffle_last_mean": null_mean,
                            "last_minus_shuffle": last - null_mean,
                            "last_minus_first": last - first,
                            **lexical_stats(words, train_vocab)})
        panel[family] = {"seeds": records, "summary": summarize(records, real_reference)}
    if time.monotonic() - started > CPU_CAP:
        raise TimeoutError("EXP-0040 runner CPU cap exceeded")
    return {"experiment": EXPERIMENT, "status": "complete",
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "input_sha256": dict(zip(("gc", "split", "zl_train", "zl_validation", "exp0039"), actual)),
            "external_commit": EXTERNAL_COMMIT, "external_generator_file_sha256": EXTERNAL_FILE_SHA,
            "source_sha256": edge.sha(Path(__file__)), "python": platform.python_version(),
            "pythonhashseed": os.environ["PYTHONHASHSEED"],
            "train_groups": len(train), "train_words": len(clean_train.split()),
            "validation_groups": len(skeleton), "validation_words": n_words,
            "validation_pairs": reference["validation_pairs"],
            "real_reference": real_reference, "families": panel,
            "wall_seconds": time.monotonic() - started,
            "limits": "Exposed ZL validation and fixed external implementation; surface generator check only."}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--external", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    result = run(root, args.external.resolve())
    destination = root / "results/EXP-0040/results.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"experiment": EXPERIMENT, "reference": result["real_reference"],
                      "families": {name: value["summary"] for name, value in
                                   result["families"].items()},
                      "wall_seconds": result["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
