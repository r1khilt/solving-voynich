"""Prepare anonymous linked-codebook inputs, with evaluator secrets separate.

This is a restricted calibration: codebook equivalence classes and the v1
role grammar are supplied. No historical unknown-script claim is made.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "data/raw/external/voynich-units"
RAW = ARCHIVE / "data/controls/naibbe"
CORPORA = ARCHIVE / "voynich_decipherment_repro_bundle/decipherment_attack_v6/lm_corpora"
OUT = ROOT / "data/processed/naibbe001"
MANIFEST = ROOT / "data/manifests/naibbe001_data.json"
REVISION = "956a7c4fc39981f4d116fa3f4edfccce6d065571"
ALPHABET = "abcdefghilmnopqrstuvxyz"
SEED = 6069101
SPLITS = {"fit": (0, 8192), "dev": (8192, 10240), "transfer": (10240, 18432)}
PINNED = {
    "greshko_naibbe_tables.csv": "4e7cfd54b7ec66515d39a51e11ec97e8e19b643b0b189124eebc3982e707dcec",
    "greshko_nathist_output_ciphertext.txt": "9cdf2de12f371ac7efdb2e78713f229ada508286c1717758184238a59cd64326",
    "greshko_nathist_pre_encryption_respaced_plaintext.txt": "4979b6826c75dd47b90d6c95ac212a34cd3735b1151ca2a524e9d13b4112e93b",
    "caesar_la.txt": "84ac8411841a4d8f5f4a49b6a2cd1f466917c6a5af72916d5e0b2b1ecb2f659c",
    "caesar_la2.txt": "8bae4d88747b318b58ea193f981766fed337ddf613df452c1a06d72c9af75ffd",
    "lm_english.txt": "7e1a58776d28d7385752c4be7931f80ad385d6aad28309f56e28f4bcd38f4789",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_lm(raw: str, opening: str | None = None) -> str:
    start = re.search(r"^\*\*\* START[^\n]*\n", raw, flags=re.M)
    end = re.search(r"^\*\*\* END[^\n]*", raw, flags=re.M)
    if start is None or end is None or end.start() <= start.end():
        raise ValueError("Both ordered Gutenberg body markers required")
    body = raw[start.end():end.start()]
    if opening is not None:
        if body.count(opening) != 1:
            raise ValueError("Expected unique pinned body opening")
        body = body[body.index(opening):]
    body = body.lower().replace("æ", "ae").replace("œ", "oe")
    body = "".join(c for c in unicodedata.normalize("NFKD", body)
                   if not unicodedata.combining(c))
    body = body.replace("j", "i").replace("k", "c").replace("w", "uu").replace("ß", "ss")
    return "".join(c for c in body if c in ALPHABET)


def parse_tables(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            role, table, letter = row["code"].split("_")
            if role not in ("unigram", "prefix", "suffix") or letter not in ALPHABET:
                raise ValueError("Invalid table row")
            records.append({"role": role, "table": table, "letter": letter,
                            "glyphs": row["glyphs"]})
    if len(records) != 414 or len({(r["role"], r["table"], r["letter"]) for r in records}) != 414:
        raise AssertionError("Expected 23 letters by 6 tables by 3 roles")
    return records


def inverse_lattice(records: list[dict]) -> dict[str, set[tuple[str, ...]]]:
    """Uni precedence is part of Naibbe v1; duplicate table paths collapse."""
    inverse: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    unigram_codes = {row["glyphs"] for row in records if row["role"] == "unigram"}
    for row in records:
        if row["role"] == "unigram":
            inverse[row["glyphs"]].add((row["letter"],))
    prefixes = [row for row in records if row["role"] == "prefix"]
    suffixes = [row for row in records if row["role"] == "suffix"]
    for prefix in prefixes:
        for suffix in suffixes:
            glyphs = prefix["glyphs"] + suffix["glyphs"]
            if glyphs not in unigram_codes:
                inverse[glyphs].add((prefix["letter"], suffix["letter"]))
    return dict(inverse)


def anonymous_classes(seed: int = SEED) -> tuple[dict[str, str], list[str]]:
    rng = random.Random(seed)
    identifiers = [f"c_{rng.getrandbits(80):020x}" for _ in ALPHABET]
    rng.shuffle(identifiers)
    mapping = dict(zip(ALPHABET, identifiers, strict=True))
    presentation_order = identifiers.copy()
    rng.shuffle(presentation_order)
    return mapping, presentation_order


def build() -> dict:
    revision = subprocess.check_output(["git", "-C", str(ARCHIVE), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != REVISION:
        raise AssertionError("External archive revision changed")
    sources = {}
    for filename, expected in PINNED.items():
        path = (RAW if filename.startswith("greshko_") else CORPORA) / filename
        if sha(path) != expected:
            raise AssertionError(f"Pinned source changed: {filename}")
        sources[filename] = {"path": str(path.relative_to(ROOT)), "sha256": expected}
    records = parse_tables(RAW / "greshko_naibbe_tables.csv")
    inverse = inverse_lattice(records)
    ciphertext = (RAW / "greshko_nathist_output_ciphertext.txt").read_text().split()
    plaintext = (RAW / "greshko_nathist_pre_encryption_respaced_plaintext.txt").read_text().split()
    if len(ciphertext) != 34764 or len(plaintext) != len(ciphertext):
        raise AssertionError("Published token alignment changed")
    if any(tuple(plain) not in inverse.get(cipher, set())
           for cipher, plain in zip(ciphertext, plaintext, strict=True)):
        raise AssertionError("Gold plaintext outside allowed v1 grammar")
    classes, order = anonymous_classes()
    OUT.mkdir(parents=True, exist_ok=True)
    latin = "".join(normalize_lm((CORPORA / name).read_text(encoding="utf-8"), opening)
                    for name, opening in (("caesar_la.txt", "GALLIA est omnis"),
                                          ("caesar_la2.txt", "L. Domitio Ap. Claudio consulibus")))
    english = normalize_lm((CORPORA / "lm_english.txt").read_text(encoding="utf-8"),
                           "I.--All Gaul is divided into three parts")
    english = english[:len(latin)]
    lms = {}
    for name, content in (("latin", latin), ("english", english)):
        path = OUT / f"{name}_lm.txt"
        path.write_text(content + "\n")
        lms[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                     "characters": len(content)}
    solver = {"schema_version": 1, "experiment": "NAIBBE-001", "class_ids": order,
              "alphabet": list(ALPHABET), "lms": lms, "splits": {}}
    answers = {"schema_version": 1, "class_to_letter": {value: key for key, value in classes.items()},
               "splits": {}}
    counts = {}
    for name, (start, stop) in SPLITS.items():
        tokens = ciphertext[start:stop]
        candidates = [sorted({tuple(classes[c] for c in path) for path in inverse[token]})
                      for token in tokens]
        solver["splits"][name] = {"token_range": [start, stop], "tokens": tokens,
                                   "candidates": candidates}
        answers["splits"][name] = {"token_range": [start, stop],
                                    "plaintext_chunks": plaintext[start:stop]}
        counts[name] = {"tokens": len(tokens), "candidate_count_histogram": dict(Counter(
            len(paths) for paths in candidates)), "gold_characters": sum(map(len, plaintext[start:stop]))}
    for name, value in (("solver_input.json", solver), ("answer_key.json", answers)):
        (OUT / name).write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    split_inputs = {}
    for name, split in solver["splits"].items():
        isolated = {key: value for key, value in solver.items() if key != "splits"}
        isolated["split"] = split
        isolated["split_name"] = name
        path = OUT / f"{name}_input.json"
        path.write_text(json.dumps(isolated, sort_keys=True, separators=(",", ":")) + "\n")
        split_inputs[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
    manifest = {
        "experiment": "NAIBBE-001", "stage": "data_preflight_only_no_optimized_scores",
        "archive_revision": REVISION, "anonymization_seed": SEED,
        "sources": sources, "lms": lms, "splits": counts, "split_inputs": split_inputs,
        "full_published_tokens": len(ciphertext),
        "full_candidate_count_histogram": dict(Counter(len(inverse[token]) for token in ciphertext)),
        "solver_input_sha256": sha(OUT / "solver_input.json"),
        "answer_key_sha256": sha(OUT / "answer_key.json"),
        "oracle_gifts": ["Known 23-letter Latin/Italian alphabet", "Exact codeword role grammar",
                          "True same-letter links across six code tables and three roles",
                          "Observed ciphertext token boundaries", "Known source language candidate Latin"],
        "hidden": ["Global class-to-plaintext-letter permutation", "Correct ambiguous-token parses"],
        "normalization": "Gutenberg body markers and pinned narrative openings; English truncated to Latin character budget; lowercase/NFKD accent fold; ae/oe ligatures; j->i,k->c,w->uu,ss for sharp s; retain allowed23 letters; erase spaces; preserve u/v distinction. Caesar parts concatenate at one artificial boundary.",
        "license": "Raw and derived third-party text/codebook remain ignored. Greshko copyright/license stays beside source; cite Cryptologia DOI10.1080/01611194.2025.2566408. No third-party code copied.",
        "limit": "Codebook-anonymized calibration; supplied grouping removes most verbose-cipher inference. Not blind cipher-family discovery or Voynich decipherment.",
    }
    MANIFEST.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return {key: manifest[key] for key in ("stage", "splits", "full_candidate_count_histogram",
                                          "solver_input_sha256", "answer_key_sha256")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(build(), indent=2))
