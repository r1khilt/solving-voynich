"""Independent, explicitly post-evaluation audit for six-table NAIBBE-002.

No production search, metrics, language model, Viterbi routine, or data builder
is imported. The independently implemented dictionary LM and Wagner--Fischer
distance from audit_naibbe001 are reused. ``--self-test`` opens no experiment
data; ``--after-evaluation`` checks the completed report and key freezes before
opening answers. Neither command prints recovered text, keys, or gold letters.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from scripts.audit_naibbe001 import (
    IndependentLM,
    committed,
    edit_distance,
    pinned,
    require,
    same_number,
    self_test as primitive_self_test,
    sha,
)


ROOT = Path(__file__).resolve().parent.parent
ARMS = ("latin_joint", "latin_local", "english_joint")
ALPHABET = "abcdefghilmnopqrstuvxyz"
TABLES = {"alpha", "beta1", "beta2", "beta3", "gamma1", "gamma2"}
RANGES = {"fit": [0, 8192], "transfer": [18432, 26624]}
SEARCH_SOURCES = {"scripts/run_naibbe002.py", "src/voynich/naibbe_homophone_search.py",
                  "src/voynich/naibbe_key_search.py"}


def validate_metadata(data: dict) -> None:
    alphabet, identifiers, groups = data["alphabet"], data["class_ids"], data["groups"]
    require(bool(alphabet) and len(set(alphabet)) == len(alphabet), "Invalid alphabet")
    require(all(isinstance(letter, str) and len(letter) == 1 for letter in alphabet),
            "Invalid alphabet symbol")
    require(len(identifiers) == len(set(identifiers)), "Duplicate class identity")
    require(bool(groups) and all(len(group) == len(alphabet) for group in groups),
            "Wrong table group size")
    require(Counter(identifier for group in groups for identifier in group) == Counter(identifiers),
            "Groups do not partition class identities")
    if "group_ids" in data:
        require(len(data["group_ids"]) == len(groups) == len(set(data["group_ids"])),
                "Invalid group identifiers")


def key_mapping(data: dict, key: list[int]) -> dict[str, str]:
    validate_metadata(data)
    alphabet, identifiers = data["alphabet"], data["class_ids"]
    require(len(key) == len(identifiers) and all(type(value) is int for value in key),
            "Invalid key size or value type")
    values = dict(zip(identifiers, key, strict=True))
    for group in data["groups"]:
        require(sorted(values[identifier] for identifier in group) == list(range(len(alphabet))),
                "Key violates within-table permutation")
    return {identifier: alphabet[value] for identifier, value in values.items()}


def decoded_candidates(data: dict, key: list[int]) -> list[list[str]]:
    mapping = key_mapping(data, key)
    candidates = data["split"]["candidates"]
    require(bool(candidates), "Empty token lattice")
    result = []
    for paths in candidates:
        require(bool(paths), "Empty token candidate set")
        require(len({tuple(path) for path in paths}) == len(paths), "Duplicate local-class path")
        require(len({len(path) for path in paths}) == 1 and 1 <= len(paths[0]) <= 2,
                "Candidates must have equal one- or two-character lengths")
        require(all(identifier in mapping for path in paths for identifier in path),
                "Unknown class in lattice")
        result.append(["".join(mapping[identifier] for identifier in path) for path in paths])
    return result


def verify_prediction(prediction: dict, candidates: list[list[str]], label: str) -> str:
    choices, chunks = prediction["choices"], prediction["chunks"]
    require(len(choices) == len(chunks) == len(candidates), f"Prediction length mismatch: {label}")
    for choice, chunk, alternatives in zip(choices, chunks, candidates, strict=True):
        require(type(choice) is int and 0 <= choice < len(alternatives),
                f"Illegal candidate index: {label}")
        require(chunk == alternatives[choice], f"Chunk/key/candidate inconsistency: {label}")
    return "".join(chunks)


def independent_metrics(chunks: list[str], gold: list[str], data: dict,
                        key: list[int], gold_key: list[int]) -> tuple[dict, list[dict]]:
    """Weight only positions whose local class is identical in every path.

    A table-draw trace is not present in the published answers. Frequencies of
    plaintext letters or of a gold-compatible chosen path therefore cannot be
    treated as gold local-class frequencies. This metric uses ciphertext support
    alone to identify its restricted denominator.
    """
    mapping, gold_mapping = key_mapping(data, key), key_mapping(data, gold_key)
    require(len(chunks) == len(gold) == len(data["split"]["candidates"]) > 0,
            "Invalid metric lengths")
    answer, predicted = "".join(gold), "".join(chunks)
    require(bool(answer), "Empty metric reference")
    counts: Counter[str] = Counter()
    ambiguous = []
    for token, paths in enumerate(data["split"]["candidates"]):
        require(bool(paths) and len({len(path) for path in paths}) == 1,
                "Unique-class weighting requires equally long candidates")
        if len(paths) > 1:
            ambiguous.append(token)
        for alternatives_at_position in zip(*paths, strict=True):
            classes = set(alternatives_at_position)
            if len(classes) == 1:
                counts[next(iter(classes))] += 1
    require(set(counts) <= set(mapping), "Unknown weighted class")
    total_weight = sum(counts.values())
    require(total_weight > 0, "Undefined unique-class weighted accuracy")
    correct = {identifier for identifier in mapping if mapping[identifier] == gold_mapping[identifier]}
    distance = edit_distance(answer, predicted)
    identifiers = data["class_ids"]
    metrics = {
        "characters": len(answer), "predicted_characters": len(predicted), "edit_distance": distance,
        "cer": distance / len(answer), "chunks": len(gold),
        "exact_chunks": sum(left == right for left, right in zip(chunks, gold, strict=True)),
        "ambiguous_chunks": len(ambiguous),
        "ambiguous_correct": sum(chunks[i] == gold[i] for i in ambiguous),
        "key_types": len(identifiers), "key_types_correct": len(correct),
        "key_macro_accuracy": len(correct) / len(identifiers),
        "unique_class_characters": total_weight,
        "unique_class_weighted_key_accuracy": sum(counts[c] for c in correct) / total_weight,
        "no_unique_evidence_classes": sum(counts[c] == 0 for c in identifiers),
        "per_group_key_correct": [sum(c in correct for c in group) for group in data["groups"]],
        "key_errors": [{"class_id": c, "true": gold_mapping[c], "predicted": mapping[c],
                        "unique_class_count": counts[c]} for c in identifiers if c not in correct],
    }
    group_statistics = []
    for index, group in enumerate(data["groups"]):
        total = sum(counts[c] for c in group)
        count_correct = sum(c in correct for c in group)
        group_statistics.append({"group_index": index, "classes": len(group),
                                 "correct": count_correct, "errors": len(group) - count_correct,
                                 "unique_class_characters": total,
                                 "unique_class_characters_with_correct_key":
                                 sum(counts[c] for c in group if c in correct),
                                 "no_unique_evidence_classes": sum(counts[c] == 0 for c in group)})
    return metrics, group_statistics


def compare_metrics(saved: dict, expected: dict, label: str) -> None:
    for name, value in expected.items():
        require(name in saved, f"Missing metric {label}/{name}")
        if isinstance(value, float):
            same_number(saved[name], value, f"Metric mismatch {label}/{name}", 1e-12)
        else:
            require(saved[name] == value, f"Metric mismatch {label}/{name}")


def read_tables(path: Path) -> list[tuple[str, str, str, str]]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            role, table, letter = row["code"].split("_")
            require(role in {"unigram", "prefix", "suffix"} and bool(row["glyphs"]),
                    "Invalid source table entry")
            rows.append((role, table, letter, row["glyphs"]))
    expected = {(role, table, letter) for role in ("unigram", "prefix", "suffix")
                for table in TABLES for letter in ALPHABET}
    require(len(rows) == len(expected) and {row[:3] for row in rows} == expected,
            "Source table inventory mismatch")
    return rows


def table_parser(rows: list[tuple[str, str, str, str]], identities: dict | None = None):
    """Split glyph strings independently of the builder's Cartesian products."""
    roles = {name: defaultdict(set) for name in ("unigram", "prefix", "suffix")}
    for role, table, letter, glyphs in rows:
        roles[role][glyphs].add(identities[(table, letter)] if identities is not None else letter)

    def parse(token: str) -> set[tuple[str, ...]]:
        if token in roles["unigram"]:
            return {(value,) for value in roles["unigram"][token]}
        return {(first, second)
                for boundary in range(1, len(token))
                for first in roles["prefix"].get(token[:boundary], ())
                for second in roles["suffix"].get(token[boundary:], ())}

    return parse


def canonical_commit(value: str) -> str:
    revision = subprocess.check_output(["git", "rev-parse", value], cwd=ROOT, text=True).strip()
    require(revision == value and len(revision) == 40, "Use a full frozen commit identifier")
    return revision


def audit() -> dict:
    started = time.monotonic()
    checked: dict[str, str] = {}
    report_path = ROOT / "results/NAIBBE-002/evaluation.json"
    require(report_path.is_file(), "Evaluation must complete before answer access")
    report = json.loads(report_path.read_text())
    require(report["experiment"] == "NAIBBE-002", "Wrong evaluation experiment")
    require(set(report["arms"]) == set(ARMS) and set(report["oracle"]) == set(RANGES)
            and all(set(report["arms"][arm]) == set(RANGES) for arm in ARMS),
            "Incomplete or unexpected evaluation result set")
    revision = canonical_commit(report["key_freeze_commit"])
    subprocess.run(["git", "merge-base", "--is-ancestor", revision, "HEAD"], cwd=ROOT, check=True)
    manifest_path = ROOT / "data/manifests/naibbe002_data.json"
    manifest = json.loads(committed(manifest_path, revision))
    require(manifest["experiment"] == "NAIBBE-002", "Wrong data experiment")
    require((manifest["classes"], manifest["groups"], manifest["classes_per_group"]) == (138, 6, 23),
            "Unexpected experiment dimensions")
    for filename in ("scripts/evaluate_naibbe002.py", "scripts/evaluate_naibbe001.py",
                     "scripts/audit_naibbe002.py",
                     "scripts/audit_naibbe001.py"):
        committed(ROOT / filename, revision)
    fits, freezes = {}, {}
    source_commits = set()
    frozen_fields = ("arm", "fit_score", "seconds", "budget_stop", "key", "input_sha256",
                     "lm_sha256", "source_commit", "seed", "trace", "source_hashes", "environment")
    for arm in ARMS:
        freeze = json.loads(committed(ROOT / f"results/NAIBBE-002/{arm}_freeze.json", revision))
        path = ROOT / f"outputs/NAIBBE-002/{arm}.json"
        pinned(path, freeze["result_sha256"], checked)
        fitted = json.loads(path.read_text())
        for field in frozen_fields:
            require(fitted[field] == freeze[field], f"Learned result/freeze mismatch: {arm}/{field}")
        require(fitted["arm"] == arm and fitted["coordinated"] == (arm != "latin_local"),
                "Wrong search arm")
        require((fitted["seed"], fitted["restarts"], fitted["kicks"]) == (920201, 8, 6),
                "Search configuration differs from registration")
        require(fitted["input_sha256"] == manifest["split_inputs"]["fit"]["sha256"],
                "Fit hash not linked to manifest")
        require(fitted["lm_sha256"] == manifest["lms"][arm.split("_")[0]]["sha256"],
                "LM hash not linked to manifest")
        source_commit = canonical_commit(fitted["source_commit"])
        source_commits.add(source_commit)
        subprocess.run(["git", "merge-base", "--is-ancestor", source_commit, revision],
                       cwd=ROOT, check=True)
        require(set(fitted["source_hashes"]) == SEARCH_SOURCES, "Unexpected search source hash set")
        for filename, digest in fitted["source_hashes"].items():
            source_path = ROOT / filename
            pinned(source_path, digest, checked)
            committed(source_path, source_commit)
        for filename in ("data/manifests/naibbe002_data.json", "scripts/build_naibbe002_data.py",
                         "docs/experiments/NAIBBE-002.md"):
            committed(ROOT / filename, source_commit)
        require(fitted["trace"], "Missing optimization trace")
        best_seen = -math.inf
        for row in fitted["trace"]:
            require(math.isfinite(row["score"]), "Nonfinite trace objective")
            best_seen = max(best_seen, row["score"])
            same_number(row["best_score"], best_seen, "Trace incumbent mismatch")
        same_number(fitted["fit_score"], best_seen, "Frozen fit score differs from best trace")
        fits[arm], freezes[arm] = fitted, freeze
    for source in (*manifest["sources"].values(), *manifest["lms"].values(),
                   manifest["parent_manifest"], manifest["shared_parser_source"]):
        pinned(ROOT / source["path"], source["sha256"], checked)
    for filename in (manifest["parent_manifest"]["path"], manifest["shared_parser_source"]["path"]):
        committed(ROOT / filename, revision)
    parent = json.loads((ROOT / manifest["parent_manifest"]["path"]).read_text())
    require(parent["sources"] == manifest["sources"] and parent["lms"] == manifest["lms"],
            "Inherited inputs differ from frozen parent")
    require(parent["archive_revision"] == manifest["archive_revision"], "Archive revision mismatch")

    inputs = {}
    require(set(manifest["split_inputs"]) == set(RANGES), "Unexpected prepared split set")
    for split_name, source in manifest["split_inputs"].items():
        path = ROOT / source["path"]
        pinned(path, source["sha256"], checked)
        data = json.loads(path.read_text())
        validate_metadata(data)
        require(data["experiment"] == "NAIBBE-002" and data["split_name"] == split_name,
                "Mislabeled split input")
        require("".join(data["alphabet"]) == ALPHABET and len(data["class_ids"]) == 138
                and len(data["groups"]) == 6, "Prepared alphabet/group inventory mismatch")
        require(data["split"]["token_range"] == RANGES[split_name], "Unregistered split range")
        require(data["lms"] == manifest["lms"], "Prepared LM metadata mismatch")
        inputs[split_name] = data
    for field in ("alphabet", "class_ids", "groups", "group_ids", "lms"):
        require(inputs["fit"][field] == inputs["transfer"][field], "Fit/transfer key metadata mismatch")
    prediction_path = ROOT / "outputs/NAIBBE-002/evaluation_predictions.json"
    pinned(prediction_path, report["predictions_sha256"], checked)
    predictions = json.loads(prediction_path.read_text())
    require(set(predictions) == {f"{arm}_{split}" for arm in ("oracle", *ARMS) for split in RANGES},
            "Unexpected prediction key set")

    # Answer access follows completed evaluation, immutable keys, and provenance.
    answer_path = ROOT / manifest["answer_key"]["path"]
    require(manifest["answer_key"]["sha256"] == manifest["answer_key_sha256"]
            == report["answer_sha256"], "Answer hash links disagree")
    pinned(answer_path, manifest["answer_key_sha256"], checked)
    answers = json.loads(answer_path.read_text())
    require(answers["experiment"] == "NAIBBE-002", "Wrong answer experiment")
    class_to_letter = answers["class_to_letter"]
    require(set(class_to_letter) == set(inputs["fit"]["class_ids"]), "Gold class identity mismatch")
    require(Counter(class_to_letter.values()) == Counter({letter: 6 for letter in ALPHABET}),
            "Gold key is not a sixfold many-to-one mapping")
    gold_key = [ALPHABET.index(class_to_letter[c]) for c in inputs["fit"]["class_ids"]]
    key_mapping(inputs["fit"], gold_key)
    rows = read_tables(ROOT / manifest["sources"]["greshko_naibbe_tables.csv"]["path"])
    plaintext_parser = table_parser(rows)
    local_parser = None
    if "class_to_table" in answers:
        labels = answers["class_to_table"]
        require(set(labels) == set(class_to_letter) and set(labels.values()) == TABLES,
                "Invalid answer-only table labels")
        identities = {(labels[c], class_to_letter[c]): c for c in class_to_letter}
        require(len(identities) == 138, "Duplicated table/letter identity")
        require(all(len({labels[c] for c in group}) == 1 for group in inputs["fit"]["groups"]),
                "Supplied groups mix original source tables")
        local_parser = table_parser(rows, identities)
    ciphertext = (ROOT / manifest["sources"]["greshko_nathist_output_ciphertext.txt"]["path"]).read_text().split()
    plaintext = (ROOT / manifest["sources"]["greshko_nathist_pre_encryption_respaced_plaintext.txt"]["path"]).read_text().split()
    require(len(ciphertext) == len(plaintext) == 34764, "Published source alignment changed")
    lattice_tokens = 0
    for split_name, data in inputs.items():
        candidates = decoded_candidates(data, gold_key)
        reference = answers["splits"][split_name]
        start, stop = RANGES[split_name]
        require(reference["token_range"] == [start, stop], "Gold split range mismatch")
        require(data["split"]["tokens"] == ciphertext[start:stop], "Ciphertext slice mismatch")
        require(reference["plaintext_chunks"] == plaintext[start:stop], "Reference slice mismatch")
        require(len(candidates) == len(reference["plaintext_chunks"]) == stop - start,
                "Prepared split length mismatch")
        for token, local_paths, alternatives, gold in zip(data["split"]["tokens"],
                                                          data["split"]["candidates"], candidates,
                                                          reference["plaintext_chunks"], strict=True):
            require(set(alternatives) == {"".join(path) for path in plaintext_parser(token)},
                    "Plaintext lattice support differs from original tables")
            if local_parser is not None:
                require({tuple(path) for path in local_paths} == local_parser(token),
                        "Anonymous local-class lattice differs from original tables")
            require(gold in alternatives, "Reference outside legal lattice support")
        saved = manifest["splits"][split_name]
        require(saved["tokens"] == len(candidates)
                and saved["gold_characters"] == sum(map(len, reference["plaintext_chunks"]))
                and saved["token_range"] == [start, stop], "Manifest split count mismatch")
        histogram = {str(count): frequency for count, frequency in
                     Counter(map(len, data["split"]["candidates"])).items()}
        require(saved["candidate_count_histogram"] == histogram, "Candidate histogram mismatch")
        require(saved["unique_candidate_tokens"] == sum(len(paths) == 1 for paths in candidates),
                "Unique-candidate token count mismatch")
        lattice_tokens += len(candidates)
    lms = {}
    for language, source in manifest["lms"].items():
        text = (ROOT / source["path"]).read_text().strip()
        require(len(text) == source["characters"], "LM length mismatch")
        lms[language] = IndependentLM(text, ALPHABET)

    metrics, summaries, deltas = {}, {}, []
    emitted_chunks = 0
    for split_name, data in inputs.items():
        gold = answers["splits"][split_name]["plaintext_chunks"]
        gold_candidates = decoded_candidates(data, gold_key)
        gold_scores = {language: lm.best_lattice_score(gold_candidates) for language, lm in lms.items()}
        for arm in ("oracle", *ARMS):
            label = f"{arm}_{split_name}"
            key = gold_key if arm == "oracle" else freezes[arm]["key"]
            candidates = decoded_candidates(data, key)
            prediction = predictions[label]
            text = verify_prediction(prediction, candidates, label)
            language = "english" if arm == "english_joint" else "latin"
            lm = lms[language]
            score = lm.score(text)
            same_number(score, lm.best_lattice_score(candidates), f"Decode is not optimal: {label}")
            saved = report["oracle"][split_name] if arm == "oracle" else report["arms"][arm][split_name]
            row, group_statistics = independent_metrics(prediction["chunks"], gold, data, key, gold_key)
            compare_metrics(saved, row, label)
            same_number(saved["score"], score, f"Prediction score mismatch: {label}")
            same_number(saved["gold_key_score"], gold_scores[language], f"Gold score mismatch: {label}")
            same_number(saved["gold_minus_learned_objective"], gold_scores[language] - score,
                        f"Gold objective gap mismatch: {label}")
            deltas.append(abs(saved["score"] - score))
            if arm != "oracle" and split_name == "fit":
                fit = fits[arm]
                for field in ("class_ids", "alphabet", "groups"):
                    require(fit[field] == data[field], "Frozen fit key indexing changed")
                require(len(fit["choices"]) == len(candidates), "Frozen fit parse length mismatch")
                for choice, paths in zip(fit["choices"], candidates, strict=True):
                    require(type(choice) is int and 0 <= choice < len(paths), "Invalid frozen fit path")
                fitted_chunks = [paths[choice] for choice, paths in zip(fit["choices"], candidates, strict=True)]
                same_number(lm.score("".join(fitted_chunks)), fit["fit_score"], "Frozen path score mismatch")
                same_number(fit["fit_score"], score, "Frozen fit best decode mismatch")
            metrics[label] = row
            summaries[label] = {name: value for name, value in row.items() if name != "key_errors"}
            summaries[label]["unresolved_class_characters"] = (row["characters"]
                                                               - row["unique_class_characters"])
            summaries[label]["per_group_audit"] = group_statistics
            emitted_chunks += len(prediction["chunks"])
    primary, oracle = metrics["latin_joint_transfer"], metrics["oracle_transfer"]
    criteria = {"cer_at_most_02": primary["cer"] <= .02,
                "within_005_of_oracle": primary["cer"] <= oracle["cer"] + .005,
                "macro_key_at_least_95": primary["key_macro_accuracy"] >= .95,
                "unique_weighted_key_at_least_99": primary["unique_class_weighted_key_accuracy"] >= .99}
    require(report["criteria"] == criteria, "Primary criterion mismatch")
    decision = "PASS" if all(criteria.values()) else "FAIL"
    require(report["decision"] == decision, "Primary decision mismatch")
    return {"experiment": "NAIBBE-002", "audit_pass": True, "key_freeze_commit": revision,
            "source_commits": sorted(source_commits), "evaluation_sha256": sha(report_path),
            "manifest_sha256": sha(manifest_path), "auditor_sha256": sha(Path(__file__).resolve()),
            "independent_primitives_sha256": sha(ROOT / "scripts/audit_naibbe001.py"),
            "checked_hashes": checked, "source_table_lattice_tokens_verified": lattice_tokens,
            "exact_local_class_support_verified": local_parser is not None,
            "emitted_chunks_verified": emitted_chunks, "max_score_delta": max(deltas),
            "criteria": criteria, "decision": decision, "metrics": summaries,
            "seconds": time.monotonic() - started,
            "scope": ["Independent source-table glyph splits and published cipher/reference slices",
                      "Exact local-class supports if answer-only table labels are present; plaintext supports otherwise",
                      "All six within-group permutations, saved chunks and legal candidate choices",
                      "Independent dictionary LM and exact forward max DP for learned and gold objectives",
                      "Independent Wagner-Fischer CER, chunk/ambiguous recovery and macro key accuracy",
                      "Unique-class-position weighting only; per-group correct/error/evidence counts",
                      "All primary criteria, raw/derived/frozen hashes and local Git source/key consistency"],
            "not_established": ["No independent rerun of key search",
                                "No gold table-draw trace; weighting excludes ambiguous local-class positions",
                                "Local Git consistency does not prove remote timing or process isolation",
                                "Known role grammar, within-table links and source alphabet remain supplied",
                                "No Voynich plaintext or historical generator is established"]}


def self_test() -> dict:
    base = primitive_self_test()
    data = {"alphabet": ["x", "y"], "class_ids": ["a", "b", "c", "d"],
            "groups": [["a", "b"], ["c", "d"]], "group_ids": ["first", "second"],
            "split": {"candidates": [[["a"], ["c"]], [["a", "d"]]]}}
    gold_key, wrong_key = [0, 1, 0, 1], [0, 1, 1, 0]
    candidates = decoded_candidates(data, wrong_key)
    require(candidates == [["x", "y"], ["xx"]], "Toy local-class candidate failure")
    require(verify_prediction({"choices": [0, 0], "chunks": ["x", "xx"]}, candidates, "toy") == "xxx",
            "Toy prediction failure")
    row, groups = independent_metrics(["x", "xx"], ["x", "xy"], data, wrong_key, gold_key)
    expected = {"characters": 3, "predicted_characters": 3, "edit_distance": 1, "cer": 1 / 3,
                "chunks": 2, "exact_chunks": 1, "ambiguous_chunks": 1, "ambiguous_correct": 1,
                "key_types": 4, "key_types_correct": 2, "key_macro_accuracy": .5,
                "unique_class_characters": 2, "unique_class_weighted_key_accuracy": .5,
                "no_unique_evidence_classes": 2, "per_group_key_correct": [2, 0],
                "key_errors": [{"class_id": "c", "true": "x", "predicted": "y", "unique_class_count": 0},
                               {"class_id": "d", "true": "y", "predicted": "x", "unique_class_count": 1}]}
    require(row == expected and [group["errors"] for group in groups] == [0, 2],
            "Toy unique-class metric failure")
    rejected = 0
    for invalid in ([0, 0, 1, 1], [0, 1, 2, 3], [False, 1, 0, 1], [0, 1]):
        try:
            key_mapping(data, invalid)
        except AssertionError:
            rejected += 1
    require(rejected == 4, "Invalid per-table keys were not rejected")
    rows = [("unigram", "one", "x", "dar"), ("unigram", "two", "x", "dar"),
            ("prefix", "one", "x", "d"), ("suffix", "two", "y", "ar"),
            ("prefix", "one", "y", "z"), ("suffix", "two", "x", "q")]
    identities = {("one", "x"): "a", ("one", "y"): "b", ("two", "x"): "c", ("two", "y"): "d"}
    parse = table_parser(rows, identities)
    require(parse("dar") == {("a",), ("c",)} and parse("zq") == {("b", "c")},
            "Table collision or unigram precedence failure")
    require(parse("missing") == set(), "Unsupported source string accepted")
    # Exhaustively check DP under every pair of table permutations. Duplicate
    # plaintext alternatives from different classes must not add probability.
    lm = IndependentLM("xxyxyyyyxxxyxyx" * 5, "xy")
    paths_checked = 0
    for first, second in itertools.product(itertools.permutations(range(2)), repeat=2):
        alternatives = decoded_candidates(data, [*first, *second])
        brute = max(lm.score("".join(path)) for path in itertools.product(*alternatives))
        same_number(lm.best_lattice_score(alternatives), brute, "Homophone lattice score failure", 1e-12)
        paths_checked += math.prod(map(len, alternatives))
    return {**base, "homophone_metric_pass": True, "invalid_keys_rejected": rejected,
            "independent_local_table_parser_pass": True, "homophone_paths_checked": paths_checked,
            "experiment_files_read": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--after-evaluation", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    output = ROOT / "results/NAIBBE-002/audit.json"
    require(not output.exists(), "Preserve previous audit; do not overwrite")
    result = audit()
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: result[name] for name in
                      ("audit_pass", "decision", "emitted_chunks_verified",
                       "source_table_lattice_tokens_verified", "exact_local_class_support_verified",
                       "max_score_delta", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
