"""Reuse NAIBBE-002 identities and fit data; prepare its unscored final block.

Only this builder/evaluators may open the answer key. Solver-facing files retain
the exact previous class/group/fit payloads. The source model is selected in a
separate source-only experiment; this module never fits or scores any model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/processed/naibbe003"
MANIFEST = ROOT / "data/manifests/naibbe003_data.json"
PARENT_MANIFEST = ROOT / "data/manifests/naibbe002_data.json"
REVISION = "956a7c4fc39981f4d116fa3f4edfccce6d065571"
ALPHABET = "abcdefghilmnopqrstuvxyz"
TABLES = ("alpha", "beta1", "beta2", "beta3", "gamma1", "gamma2")
SPLITS = {"fit": (0, 8192), "transfer": (26624, 34764)}
PINNED = {
    "data/manifests/naibbe002_data.json": "63ca1c779b52176ce7ca181d64471687781f2ae918e7ad711c91cce1a58f77a4",
    "scripts/build_naibbe001_data.py": "2d042527882ba0673fdbe222a3d0941aa4b2f3bfe3320b9233d162cc16250e1c",
    "scripts/build_naibbe002_data.py": "098dcd707b0de49f91a899ffed4caf4771fb9700a344a90d0dae148ccfe4c100",
    "data/processed/naibbe002/fit_input.json": "02da4631aab90fdf3c1e8c2bf13c7bba2a5bb41416e10a04b2885efb73390ccf",
    "data/processed/naibbe002/answer_key.json": "410e0e85a26a059d2e48b2863c68859df8ed3da191a610784a1bbb033b24086d",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded_json(value: dict, *, pretty: bool = False) -> bytes:
    options = {"indent": 2} if pretty else {"separators": (",", ":")}
    return (json.dumps(value, sort_keys=True, **options) + "\n").encode()


def write_immutable_bundle(files: dict[Path, bytes]) -> None:
    """Validate every collision before writing any member of the bundle."""
    for path, encoded in files.items():
        if path.exists() and path.read_bytes() != encoded:
            raise FileExistsError(f"Preserve previous prepared data: {path.name}")
    for path, encoded in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            # Exclusive creation also rejects a conflicting concurrent writer.
            with path.open("xb") as stream:
                stream.write(encoded)


def validate_identities(fit: dict, answers: dict, *, alphabet: str = ALPHABET,
                        tables: tuple[str, ...] = TABLES) -> dict[tuple[str, str], str]:
    """Check an existing opaque partition; never make new random identities."""
    ids = fit["class_ids"]
    group_ids, groups = fit["group_ids"], fit["groups"]
    letters, table_map = answers["class_to_letter"], answers["class_to_table"]
    if fit["alphabet"] != list(alphabet):
        raise ValueError("Parent alphabet changed")
    if len(ids) != len(set(ids)) or len(ids) != len(alphabet) * len(tables):
        raise ValueError("Parent classes do not form the required unique inventory")
    if any(re.fullmatch(r"c_[0-9a-f]{20}", value) is None for value in ids):
        raise ValueError("Parent class identifiers are not opaque")
    if set(letters) != set(ids) or set(table_map) != set(ids):
        raise ValueError("Answer identities differ from solver identities")
    if len(groups) != len(tables) or len(group_ids) != len(set(group_ids)) or len(group_ids) != len(groups):
        raise ValueError("Parent group inventory changed")
    if any(re.fullmatch(r"t_[0-9a-f]{20}", value) is None for value in group_ids):
        raise ValueError("Parent group identifiers are not opaque")
    if Counter(member for group in groups for member in group) != Counter(ids):
        raise ValueError("Parent groups must partition all classes exactly once")
    observed_tables = []
    for group in groups:
        if len(group) != len(alphabet) or {letters[member] for member in group} != set(alphabet):
            raise ValueError("Each parent group must contain the full alphabet once")
        group_tables = {table_map[member] for member in group}
        if len(group_tables) != 1:
            raise ValueError("Parent group mixes different source tables")
        observed_tables.extend(group_tables)
    if Counter(observed_tables) != Counter(tables):
        raise ValueError("Each original table must have exactly one parent group")
    mapping = {(table_map[value], letters[value]): value for value in ids}
    if len(mapping) != len(ids):
        raise ValueError("Duplicate source-table/letter identities")
    return mapping


def read_table_rows(path: Path) -> list[dict[str, str]]:
    """Independent CSV parser, with the entire source inventory checked."""
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["code", "glyphs"]:
            raise ValueError("Unexpected original codebook columns")
        for row in reader:
            role, table, letter = row["code"].split("_")
            if not row["glyphs"] or any(char.isspace() for char in row["glyphs"]):
                raise ValueError("Empty or space-bearing codeword")
            rows.append({"role": role, "table": table, "letter": letter, "glyphs": row["glyphs"]})
    expected = {(role, table, letter) for role in ("unigram", "prefix", "suffix")
                for table in TABLES for letter in ALPHABET}
    observed = [(row["role"], row["table"], row["letter"]) for row in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Original codebook must contain all 414 unique source rows")
    return rows


def reconstruct_support(rows: list[dict[str, str]], mapping: dict[tuple[str, str], str]
                        ) -> dict[str, list[list[str]]]:
    """Build local-class paths directly; preserve glyph collisions and v1 precedence."""
    unigrams: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    prefixes: dict[str, set[str]] = defaultdict(set)
    suffixes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        identity = mapping[(row["table"], row["letter"])]
        if row["role"] == "unigram":
            unigrams[row["glyphs"]].add((identity,))
        elif row["role"] == "prefix":
            prefixes[row["glyphs"]].add(identity)
        elif row["role"] == "suffix":
            suffixes[row["glyphs"]].add(identity)
        else:
            raise ValueError("Unrecognized codeword role")
    paths = {glyphs: set(candidates) for glyphs, candidates in unigrams.items()}
    for prefix, first_ids in prefixes.items():
        for suffix, second_ids in suffixes.items():
            word = prefix + suffix
            if word in unigrams:
                continue
            paths.setdefault(word, set()).update((first, second) for first in first_ids for second in second_ids)
    return {word: [list(candidate) for candidate in sorted(candidates)] for word, candidates in paths.items()}


def prepare_split(tokens: list[str], gold: list[str], token_range: tuple[int, int],
                  support: dict[str, list[list[str]]], letters: dict[str, str]) -> tuple[dict, dict, dict]:
    start, stop = token_range
    if not 0 <= start < stop <= len(tokens) or len(tokens) != len(gold):
        raise ValueError("Invalid token-aligned source interval")
    selected, chunks = tokens[start:stop], gold[start:stop]
    candidates = []
    for token, reference in zip(selected, chunks, strict=True):
        paths = support.get(token)
        if not paths or not any("".join(letters[identity] for identity in path) == reference for path in paths):
            raise ValueError("Gold chunk outside exact local-class support")
        candidates.append(paths)
    split = {"token_range": [start, stop], "tokens": selected, "candidates": candidates}
    answer = {"token_range": [start, stop], "plaintext_chunks": chunks}
    counts = {"token_range": [start, stop], "tokens": len(selected),
              "candidate_count_histogram": dict(Counter(map(len, candidates))),
              "gold_characters": sum(map(len, chunks)),
              "unique_candidate_tokens": sum(len(paths) == 1 for paths in candidates)}
    return split, answer, counts


def build() -> dict:
    for relative, expected in PINNED.items():
        if sha(ROOT / relative) != expected:
            raise AssertionError(f"Pinned parent input/source changed: {relative}")
    parent = json.loads(PARENT_MANIFEST.read_text())
    fit_reference = parent["split_inputs"]["fit"]
    answer_reference = parent["answer_key"]
    for reference in (fit_reference, answer_reference, *parent["sources"].values(), *parent["lms"].values()):
        if sha(ROOT / reference["path"]) != reference["sha256"]:
            raise AssertionError("Parent source/LM/input hash mismatch")
    for reference in (fit_reference, answer_reference):
        if PINNED.get(reference["path"]) != reference["sha256"]:
            raise AssertionError("Parent references differ from explicitly pinned identities")
    archive = ROOT / "data/raw/external/voynich-units"
    revision = subprocess.check_output(["git", "-C", str(archive), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION or parent["archive_revision"] != REVISION:
        raise AssertionError("External archive revision changed")
    old_fit = json.loads((ROOT / fit_reference["path"]).read_text())
    old_answers = json.loads((ROOT / answer_reference["path"]).read_text())
    if old_fit["experiment"] != "NAIBBE-002" or old_answers["experiment"] != "NAIBBE-002":
        raise AssertionError("Expected original NAIBBE-002 experiment")
    if old_fit["lms"] != parent["lms"] or old_fit["split_name"] != "fit":
        raise AssertionError("Parent fit metadata changed")
    mapping = validate_identities(old_fit, old_answers)
    sources = parent["sources"]
    rows = read_table_rows(ROOT / sources["greshko_naibbe_tables.csv"]["path"])
    support = reconstruct_support(rows, mapping)
    tokens = (ROOT / sources["greshko_nathist_output_ciphertext.txt"]["path"]).read_text().split()
    gold = (ROOT / sources["greshko_nathist_pre_encryption_respaced_plaintext.txt"]["path"]).read_text().split()
    if len(tokens) != 34764 or len(gold) != len(tokens):
        raise AssertionError("Published source token alignment changed")
    metadata = {key: value for key, value in old_fit.items() if key not in ("split", "split_name")}
    metadata["experiment"] = "NAIBBE-003"
    answers = {"schema_version": 1, "experiment": "NAIBBE-003", "splits": {},
               "class_to_letter": old_answers["class_to_letter"], "class_to_table": old_answers["class_to_table"]}
    files, inputs, counts = {}, {}, {}
    for name, interval in SPLITS.items():
        split, answer, count = prepare_split(tokens, gold, interval, support, answers["class_to_letter"])
        data = {**metadata, "split_name": name, "split": split}
        if name == "fit":
            expected_fit = {**old_fit, "experiment": "NAIBBE-003"}
            if data != expected_fit or answer != old_answers["splits"]["fit"]:
                raise AssertionError("Rebuilt fit payload differs from immutable NAIBBE-002 fit")
        path = OUT / f"{name}_input.json"
        files[path] = encoded_json(data)
        inputs[name] = {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(files[path]).hexdigest()}
        answers["splits"][name] = answer
        counts[name] = count
    answer_path = OUT / "answer_key.json"
    files[answer_path] = encoded_json(answers)
    answer_reference = {"path": str(answer_path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(files[answer_path]).hexdigest()}
    manifest = {
        "experiment": "NAIBBE-003", "stage": "data_preflight_only_no_optimized_scores",
        "status": "Exploratory source-model followup after exposed NAIBBE-002 results",
        "archive_revision": REVISION, "anonymization_seed": parent["anonymization_seed"],
        "identity_policy": "Reuse all 138 NAIBBE-002 class IDs, six group IDs, orders and hidden identities; no new anonymization",
        "parent_manifest": {"path": str(PARENT_MANIFEST.relative_to(ROOT)),
                            "sha256": PINNED[str(PARENT_MANIFEST.relative_to(ROOT))]},
        "parent_fit_input": fit_reference, "parent_answer_key": parent["answer_key"],
        "builder_sources": [{"path": path, "sha256": PINNED[path]} for path in PINNED if path.endswith(".py")]
                           + [{"path": "scripts/build_naibbe003_data.py", "sha256": sha(Path(__file__))}],
        "fit_payload_equality": "Exact JSON equality with NAIBBE-002 fit input except experiment=NAIBBE-003; independently reconstructed support and source slice agree",
        "sources": sources, "lms": parent["lms"], "split_inputs": inputs, "splits": counts,
        "answer_key": answer_reference, "answer_key_sha256": answer_reference["sha256"],
        "classes": 138, "groups": 6, "classes_per_group": 23,
        "group_constraint": parent["group_constraint"], "collision_handling": parent["collision_handling"],
        "oracle_gifts": parent["oracle_gifts"], "hidden": parent["hidden"],
        "split_policy": "Reuse exposed fit [0,8192) unchanged. Transfer [26624,34764) is the final previously unscored block from the same Pliny publication, not source-disjoint. Earlier transfer blocks [10240,18432) and [18432,26624) excluded. Neither new model nor old baseline keys may score new transfer before new-key freeze.",
        "intervention": "Only source-model choice changes in a separately registered source-only calibration; no corpus, key identities, codebook clues, fit data, preprocessing or support grammar intervention here",
        "preflight_access": "Builder reads gold only for source-slice and support checks; reports aggregate counts/hashes, never plaintext or hidden mappings; no model/key evaluation",
        "normalization": parent["normalization"], "license": parent["license"],
        "limit": parent["limit"],
    }
    files[MANIFEST] = encoded_json(manifest, pretty=True)
    write_immutable_bundle(files)
    return {key: manifest[key] for key in ("experiment", "stage", "classes", "groups", "splits", "split_inputs",
                                           "answer_key_sha256", "fit_payload_equality")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(build(), indent=2))
