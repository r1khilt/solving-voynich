"""Prepare six anonymous Naibbe codebooks without cross-table letter links.

The six table identities and within-table unigram/prefix/suffix linkage remain
oracle inputs. Equal glyph strings across different tables retain distinct
candidate classes. This builder may check gold support mechanically; solvers
must consume only isolated solver inputs and must never import either builder.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from scripts.build_naibbe001_data import ALPHABET, ARCHIVE, REVISION, parse_tables, sha


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/processed/naibbe002"
MANIFEST = ROOT / "data/manifests/naibbe002_data.json"
PARENT_MANIFEST = ROOT / "data/manifests/naibbe001_data.json"
PARENT_MANIFEST_SHA = "1783ec695407a77eaac903a0620291c1dcfa42bec3469738b5015122c421651f"
SEED = 6069202
SPLITS = {"fit": (0, 8192), "transfer": (18432, 26624)}
TABLES = ("alpha", "beta1", "beta2", "beta3", "gamma1", "gamma2")


def anonymous_classes(tables: list[str] | tuple[str, ...], alphabet: str,
                      seed: int = SEED) -> tuple[dict[tuple[str, str], str], list[str],
                                                 list[list[str]], list[str]]:
    """Independent per-table assignments, group order and display orders.

    The actual same-letter relationship between tables is never included in
    the returned solver-facing groups. ``mapping`` is builder/evaluator-only.
    """
    if len(set(tables)) != len(tables) or len(set(alphabet)) != len(alphabet):
        raise ValueError("Unique table and alphabet names required")
    rng = random.Random(seed)
    mapping = {}
    displayed = []
    for table in sorted(tables):
        table_id = f"t_{rng.getrandbits(80):020x}"
        identifiers = [f"c_{rng.getrandbits(80):020x}" for _ in alphabet]
        rng.shuffle(identifiers)
        mapping.update({(table, letter): identifier for letter, identifier in
                        zip(alphabet, identifiers, strict=True)})
        within_group = identifiers.copy()
        rng.shuffle(within_group)
        displayed.append((table_id, within_group))
    rng.shuffle(displayed)
    group_ids = [identifier for identifier, _ in displayed]
    groups = [members for _, members in displayed]
    class_ids = list(mapping.values())
    rng.shuffle(class_ids)
    if len(set(class_ids)) != len(class_ids) or len(set(group_ids)) != len(group_ids):
        raise AssertionError("Opaque identifier collision")
    return mapping, class_ids, groups, group_ids


def inverse_lattice(records: list[dict], mapping: dict[tuple[str, str], str]) -> dict[str, set[tuple[str, ...]]]:
    """Support-only v1 grammar; never collapse different local classes."""
    inverse: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    unigram_glyphs = {row["glyphs"] for row in records if row["role"] == "unigram"}
    for row in records:
        if row["role"] == "unigram":
            inverse[row["glyphs"]].add((mapping[(row["table"], row["letter"])],))
    prefixes = [row for row in records if row["role"] == "prefix"]
    suffixes = [row for row in records if row["role"] == "suffix"]
    for prefix in prefixes:
        for suffix in suffixes:
            glyphs = prefix["glyphs"] + suffix["glyphs"]
            if glyphs not in unigram_glyphs:
                inverse[glyphs].add((mapping[(prefix["table"], prefix["letter"])],
                                    mapping[(suffix["table"], suffix["letter"]) ]))
    return dict(inverse)


def write_immutable_json(path: Path, value: dict) -> None:
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if path.exists() and path.read_bytes() != encoded:
        raise FileExistsError(f"Preserve previous prepared data: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def build() -> dict:
    if sha(PARENT_MANIFEST) != PARENT_MANIFEST_SHA:
        raise AssertionError("NAIBBE-001 source/LM manifest changed")
    parent = json.loads(PARENT_MANIFEST.read_text())
    revision = subprocess.check_output(["git", "-C", str(ARCHIVE), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != REVISION or parent["archive_revision"] != REVISION:
        raise AssertionError("External source revision changed")
    for source in (*parent["sources"].values(), *parent["lms"].values()):
        if sha(ROOT / source["path"]) != source["sha256"]:
            raise AssertionError("Pinned inherited source/LM hash mismatch")
    records = parse_tables(ROOT / parent["sources"]["greshko_naibbe_tables.csv"]["path"])
    if set(row["table"] for row in records) != set(TABLES):
        raise AssertionError("Expected six source tables")
    expected = {(table, letter, role) for table in TABLES for letter in ALPHABET
                for role in ("unigram", "prefix", "suffix")}
    if {(row["table"], row["letter"], row["role"]) for row in records} != expected:
        raise AssertionError("Within-table role inventory changed")
    mapping, class_ids, groups, group_ids = anonymous_classes(TABLES, ALPHABET)
    if len(class_ids) != 138 or len(groups) != 6 or any(len(group) != 23 for group in groups):
        raise AssertionError("Expected six independent 23-class groups")
    inverse = inverse_lattice(records, mapping)
    ciphertext_path = ROOT / parent["sources"]["greshko_nathist_output_ciphertext.txt"]["path"]
    plaintext_path = ROOT / parent["sources"]["greshko_nathist_pre_encryption_respaced_plaintext.txt"]["path"]
    ciphertext, plaintext = ciphertext_path.read_text().split(), plaintext_path.read_text().split()
    if len(ciphertext) != 34764 or len(plaintext) != len(ciphertext):
        raise AssertionError("Published token alignment changed")
    answers = {"schema_version": 1, "experiment": "NAIBBE-002",
               "class_to_letter": {identifier: letter for (_, letter), identifier in mapping.items()},
               "class_to_table": {identifier: table for (table, _), identifier in mapping.items()},
               "splits": {}}
    metadata = {"schema_version": 1, "experiment": "NAIBBE-002", "class_ids": class_ids,
                "alphabet": list(ALPHABET), "groups": groups, "group_ids": group_ids,
                "lms": parent["lms"]}
    split_inputs, counts = {}, {}
    for name, (start, stop) in SPLITS.items():
        tokens, gold = ciphertext[start:stop], plaintext[start:stop]
        candidates = [sorted(inverse[token]) for token in tokens]
        for paths, reference in zip(candidates, gold, strict=True):
            decoded = {"".join(answers["class_to_letter"][identifier] for identifier in path)
                       for path in paths}
            if reference not in decoded:
                raise AssertionError(f"Gold outside allowed local-class grammar in {name}")
        data = {**metadata, "split_name": name,
                "split": {"token_range": [start, stop], "tokens": tokens, "candidates": candidates}}
        path = OUT / f"{name}_input.json"
        write_immutable_json(path, data)
        split_inputs[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
        answers["splits"][name] = {"token_range": [start, stop], "plaintext_chunks": gold}
        counts[name] = {"token_range": [start, stop], "tokens": len(tokens),
                        "candidate_count_histogram": dict(Counter(map(len, candidates))),
                        "gold_characters": sum(map(len, gold)),
                        "unique_candidate_tokens": sum(len(paths) == 1 for paths in candidates)}
    answer_path = OUT / "answer_key.json"
    write_immutable_json(answer_path, answers)
    if Counter(answers["class_to_letter"].values()) != Counter({letter: 6 for letter in ALPHABET}):
        raise AssertionError("Expected sixfold many-to-one evaluator key")
    manifest = {
        "experiment": "NAIBBE-002", "stage": "data_preflight_only_no_optimized_scores",
        "status": "Exploratory architecture followup after the exposed NAIBBE-001 result",
        "archive_revision": REVISION, "anonymization_seed": SEED,
        "parent_manifest": {"path": str(PARENT_MANIFEST.relative_to(ROOT)), "sha256": PARENT_MANIFEST_SHA},
        "shared_parser_source": {"path": "scripts/build_naibbe001_data.py",
                                 "sha256": sha(ROOT / "scripts/build_naibbe001_data.py")},
        "sources": parent["sources"], "lms": parent["lms"],
        "split_inputs": split_inputs, "splits": counts,
        "answer_key": {"path": str(answer_path.relative_to(ROOT)), "sha256": sha(answer_path)},
        "answer_key_sha256": sha(answer_path), "classes": 138, "groups": 6,
        "classes_per_group": 23, "group_constraint": "Each group has an independent bijection to the 23-letter alphabet",
        "collision_handling": "Identical glyph strings from different tables retain separate local class candidates, including dar",
        "oracle_gifts": ["Known 23-letter Latin/Italian alphabet",
                         "Six known code tables and exact v1 unigram/prefix/suffix grammar",
                         "True same-letter linkage between the three roles WITHIN each table",
                         "Observed ciphertext token boundaries",
                         "One unknown bijective letter permutation per supplied table group",
                         "Known source-language candidate Latin and the previously used independent Caesar LMs"],
        "hidden": ["Six class-to-letter permutations and all true cross-table letter equivalences",
                   "Correct ambiguous-token readings; published data does not provide table draw traces"],
        "split_policy": "Fit tokens [0,8192) were exposed in NAIBBE-001; transfer [18432,26624) is a new unscored block from the same Pliny publication and may only be scored after key freeze. Prior001 transfer [10240,18432) is excluded.",
        "normalization": parent["normalization"],
        "license": parent["license"],
        "limit": "Restricted six-codebook calibration, not blind family/role discovery. Fresh transfer is same-work, not source-disjoint. Support lattice discards table draw probabilities.",
    }
    encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    if MANIFEST.exists() and MANIFEST.read_bytes() != encoded:
        raise FileExistsError("Preserve previous NAIBBE-002 data manifest")
    MANIFEST.write_bytes(encoded)
    return {name: manifest[name] for name in ("experiment", "stage", "classes", "groups", "splits",
                                              "split_inputs", "answer_key_sha256")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(build(), indent=2))
