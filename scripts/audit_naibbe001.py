"""Independent post-evaluation audit for restricted NAIBBE-001 key recovery.

No search/evaluation implementation is imported. Source probabilities use
dictionary counts, lattice scores use a separate forward max DP, and edit
distance uses Wagner--Fischer rows rather than the evaluator's bit vectors.
``--self-test`` reads no experiment files. ``--after-evaluation`` deliberately
requires completed evaluation and verifies frozen artifacts before answers open.
Neither mode prints plaintext or keys.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
ARMS = ("latin_joint", "latin_fixed", "english_joint")
WEIGHTS = (.02, .08, .20, .70)


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_number(actual, expected, label: str, tolerance: float = 1e-6) -> None:
    require(isinstance(actual, (int, float)) and math.isfinite(actual), label)
    require(math.isclose(actual, expected, rel_tol=0., abs_tol=tolerance), label)


def pinned(path: Path, digest: str, checked: dict) -> None:
    require(sha(path) == digest, f"Hash mismatch: {path.name}")
    checked[str(path.relative_to(ROOT))] = digest


def committed(path: Path, revision: str) -> bytes:
    relative = str(path.relative_to(ROOT))
    body = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=ROOT)
    require(body == path.read_bytes(), f"Working file differs from frozen commit: {relative}")
    return body


def edit_distance(left: str, right: str) -> int:
    """Exact Levenshtein by a different recurrence from Myers bit vectors.

    After trimming equal edges, each row considers deletion/substitution into
    every column and all possible following insertions. A cumulative minimum
    computes those insertion chains. This is ordinary Wagner--Fischer DP in
    O(n*m) time and O(m) memory, with the inner loop evaluated by NumPy.
    """
    start = 0
    while start < min(len(left), len(right)) and left[start] == right[start]:
        start += 1
    left, right = left[start:], right[start:]
    end = 0
    while end < min(len(left), len(right)) and left[-1 - end] == right[-1 - end]:
        end += 1
    if end:
        left, right = left[:-end], right[:-end]
    if not left or not right:
        return max(len(left), len(right))
    if len(right) > len(left):
        left, right = right, left
    columns = np.arange(len(right) + 1, dtype=np.int64)
    previous = columns.copy()
    symbols = np.array(list(right))
    incoming = np.empty_like(previous)
    for row, char in enumerate(left, start=1):
        incoming[0] = row
        incoming[1:] = np.minimum(previous[1:] + 1,
                                  previous[:-1] + (symbols != char))
        previous = np.minimum.accumulate(incoming - columns) + columns
    return int(previous[-1])


class IndependentLM:
    """Separate dictionary implementation of the registered normalized LM."""

    def __init__(self, text: str, alphabet: str):
        require(len(text) >= 4 and set(text) <= set(alphabet), "Invalid LM text")
        require(len(alphabet) == len(set(alphabet)), "Duplicate LM alphabet symbols")
        self.alphabet = alphabet
        self.counts = []
        self.contexts = []
        self.cache: dict[str, float] = {}
        for width in range(1, 5):
            counts = Counter(text[i:i + width] for i in range(len(text) - width + 1))
            histories = Counter()
            for gram, count in counts.items():
                histories[gram[:-1]] += count
            self.counts.append(counts)
            self.contexts.append(histories)

    def log_next(self, gram: str) -> float:
        require(1 <= len(gram) <= 4, "Invalid source context length")
        if gram not in self.cache:
            terms = []
            for width in range(1, len(gram) + 1):
                suffix = gram[-width:]
                numerator = self.counts[width - 1][suffix] + .1
                denominator = self.contexts[width - 1][suffix[:-1]] + .1 * len(self.alphabet)
                terms.append(WEIGHTS[width - 1] * numerator / denominator)
            self.cache[gram] = math.log(math.fsum(terms) / math.fsum(WEIGHTS[:len(gram)]))
        return self.cache[gram]

    def append(self, history: str, emitted: str) -> tuple[str, float]:
        increment = 0.
        for char in emitted:
            gram = history + char
            increment += self.log_next(gram)
            history = gram[-3:]
        return history, increment

    def score(self, text: str) -> float:
        return math.fsum(self.log_next(text[max(0, i - 3):i + 1]) for i in range(len(text)))

    def best_lattice_score(self, candidates: list[list[str]]) -> float:
        scores = {"": 0.}
        for alternatives in candidates:
            require(bool(alternatives), "Empty candidate set")
            updated: dict[str, float] = {}
            for history, total in scores.items():
                for emitted in alternatives:
                    tail, increment = self.append(history, emitted)
                    updated[tail] = max(updated.get(tail, -math.inf), total + increment)
            scores = updated
        return max(scores.values())


def decoded_candidates(data: dict, key: list[int]) -> list[list[str]]:
    alphabet = data["alphabet"]
    identifiers = data["class_ids"]
    require(len(set(identifiers)) == len(identifiers) == len(alphabet), "Class alphabet mismatch")
    require(all(type(value) is int for value in key), "Noninteger key entry")
    require(sorted(key) == list(range(len(alphabet))), "Key is not a permutation")
    mapping = dict(zip(identifiers, (alphabet[value] for value in key), strict=True))
    return [["".join(mapping[identifier] for identifier in path) for path in alternatives]
            for alternatives in data["split"]["candidates"]]


def verify_prediction(prediction: dict, candidates: list[list[str]], label: str) -> str:
    choices, chunks = prediction["choices"], prediction["chunks"]
    require(len(choices) == len(chunks) == len(candidates), f"Prediction length mismatch: {label}")
    for choice, chunk, alternatives in zip(choices, chunks, candidates, strict=True):
        require(type(choice) is int and 0 <= choice < len(alternatives),
                f"Illegal candidate index: {label}")
        require(chunk == alternatives[choice], f"Chunk/key/candidate inconsistency: {label}")
    return "".join(chunks)


def independent_metrics(chunks: list[str], gold: list[str], data: dict,
                        key: list[int], class_to_letter: dict) -> dict:
    require(len(chunks) == len(gold) > 0, "Invalid metric chunk count")
    answer, prediction = "".join(gold), "".join(chunks)
    require(bool(answer), "Empty metric reference")
    alphabet = data["alphabet"]
    ids = data["class_ids"]
    frequencies = Counter(answer)
    correct_types = []
    errors = []
    for identifier, code in zip(ids, key, strict=True):
        true, predicted = class_to_letter[identifier], alphabet[code]
        if true == predicted:
            correct_types.append(identifier)
        else:
            errors.append({"true": true, "predicted": predicted,
                           "source_occurrences": frequencies[true]})
    correct_weight = sum(frequencies[class_to_letter[identifier]] for identifier in correct_types)
    ambiguous = [i for i, paths in enumerate(data["split"]["candidates"]) if len(paths) > 1]
    exact = sum(pred == ref for pred, ref in zip(chunks, gold, strict=True))
    distance = edit_distance(answer, prediction)
    return {
        "characters": len(answer), "predicted_characters": len(prediction),
        "edit_distance": distance, "cer": distance / len(answer),
        "exact_chunks": exact, "chunks": len(gold), "chunk_accuracy": exact / len(gold),
        "ambiguous_chunks": len(ambiguous),
        "ambiguous_correct": sum(chunks[i] == gold[i] for i in ambiguous),
        "key_types_correct": len(correct_types), "key_types": len(key),
        "key_frequency_weighted_accuracy": correct_weight / len(answer),
        "key_errors": errors,
        "unobserved_key_types": sum(frequencies[class_to_letter[identifier]] == 0 for identifier in ids),
    }


def compare_metrics(actual: dict, expected: dict, label: str) -> None:
    for name, value in expected.items():
        require(name in actual, f"Missing metric {label}/{name}")
        if isinstance(value, float):
            same_number(actual[name], value, f"Metric mismatch {label}/{name}", 1e-12)
        else:
            require(actual[name] == value, f"Metric mismatch {label}/{name}")


def table_parser(path: Path):
    roles = {role: defaultdict(set) for role in ("unigram", "prefix", "suffix")}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            role, _, letter = row["code"].split("_")
            roles[role][row["glyphs"]].add(letter)

    def parse(token: str) -> set[str]:
        if token in roles["unigram"]:
            return set(roles["unigram"][token])
        return {first + second
                for boundary in range(1, len(token))
                for first in roles["prefix"].get(token[:boundary], ())
                for second in roles["suffix"].get(token[boundary:], ())}

    return parse


def audit() -> dict:
    started = time.monotonic()
    checked: dict[str, str] = {}
    report_path = ROOT / "results/NAIBBE-001/evaluation.json"
    require(report_path.is_file(), "Evaluation must complete before answer access")
    report = json.loads(report_path.read_text())
    require(report["experiment"] == "NAIBBE-001", "Wrong evaluation experiment")
    revision = subprocess.check_output(["git", "rev-parse", report["key_freeze_commit"]],
                                       cwd=ROOT, text=True).strip()
    require(revision == report["key_freeze_commit"], "Use a full frozen commit identifier")
    manifest_path = ROOT / "data/manifests/naibbe001_data.json"
    manifest = json.loads(committed(manifest_path, revision))
    require(manifest["experiment"] == "NAIBBE-001", "Wrong data experiment")
    fits, freezes = {}, {}
    source_commits = set()
    for arm in ARMS:
        freeze_path = ROOT / f"results/NAIBBE-001/{arm}_freeze.json"
        freeze = json.loads(committed(freeze_path, revision))
        result_path = ROOT / f"outputs/NAIBBE-001/{arm}.json"
        pinned(result_path, freeze["result_sha256"], checked)
        result = json.loads(result_path.read_text())
        for name in ("arm", "fit_score", "seconds", "budget_stop", "key", "input_sha256",
                     "lm_sha256", "source_commit", "seed", "trace"):
            require(result[name] == freeze[name], f"Learned result/freeze mismatch: {arm}/{name}")
        require(result["arm"] == arm and result["joint"] == (arm != "latin_fixed"),
                "Wrong learned arm")
        require(result["input_sha256"] == manifest["split_inputs"]["fit"]["sha256"],
                "Fit hash not linked to manifest")
        require(result["lm_sha256"] == manifest["lms"][arm.split("_")[0]]["sha256"],
                "LM hash not linked to manifest")
        source_commits.add(result["source_commit"])
        fits[arm], freezes[arm] = result, freeze
    for source_commit in source_commits:
        subprocess.run(["git", "merge-base", "--is-ancestor", source_commit, revision],
                       cwd=ROOT, check=True)
        for filename in ("src/voynich/naibbe_key_search.py", "scripts/run_naibbe001.py",
                         "data/manifests/naibbe001_data.json"):
            committed(ROOT / filename, source_commit)
    for source in manifest["sources"].values():
        pinned(ROOT / source["path"], source["sha256"], checked)
    for source in manifest["lms"].values():
        pinned(ROOT / source["path"], source["sha256"], checked)
    inputs = {}
    for split_name, source in manifest["split_inputs"].items():
        path = ROOT / source["path"]
        pinned(path, source["sha256"], checked)
        inputs[split_name] = json.loads(path.read_text())
        require(inputs[split_name]["split_name"] == split_name, "Mislabeled split input")
    bundled_path = ROOT / "data/processed/naibbe001/solver_input.json"
    pinned(bundled_path, manifest["solver_input_sha256"], checked)
    bundled = json.loads(bundled_path.read_text())
    for split_name, data in inputs.items():
        for field in ("alphabet", "class_ids", "lms"):
            require(data[field] == bundled[field], "Split metadata differs from bundle")
        require(data["split"] == bundled["splits"][split_name], "Split differs from bundle")
    predictions_path = ROOT / "outputs/NAIBBE-001/evaluation_predictions.json"
    pinned(predictions_path, report["predictions_sha256"], checked)
    predictions = json.loads(predictions_path.read_text())
    expected_names = {f"{arm}_{split}" for arm in (*ARMS, "oracle") for split in ("fit", "transfer")}
    require(set(predictions) == expected_names, "Unexpected prediction key set")

    # Answer access occurs only after completed evaluation, frozen keys, and hashes.
    answer_path = ROOT / "data/processed/naibbe001/answer_key.json"
    pinned(answer_path, manifest["answer_key_sha256"], checked)
    require(report["answer_sha256"] == manifest["answer_key_sha256"], "Evaluation answer hash mismatch")
    answers = json.loads(answer_path.read_text())
    alphabet = "".join(bundled["alphabet"])
    class_to_letter = answers["class_to_letter"]
    require(set(class_to_letter) == set(bundled["class_ids"]), "Gold class identity mismatch")
    require(sorted(class_to_letter.values()) == sorted(alphabet), "Gold key not bijective")
    gold_key = [alphabet.index(class_to_letter[identifier]) for identifier in bundled["class_ids"]]
    parse = table_parser(ROOT / manifest["sources"]["greshko_naibbe_tables.csv"]["path"])
    lattice_tokens = 0
    for split_name, data in inputs.items():
        candidates = decoded_candidates(data, gold_key)
        gold = answers["splits"][split_name]
        require(data["split"]["token_range"] == gold["token_range"], "Gold split range mismatch")
        require(len(candidates) == len(gold["plaintext_chunks"]) == len(data["split"]["tokens"]),
                "Gold/cipher split length mismatch")
        for token, alternatives, reference in zip(data["split"]["tokens"], candidates,
                                                  gold["plaintext_chunks"], strict=True):
            require(set(alternatives) == parse(token), "Candidate lattice differs from source tables")
            require(reference in alternatives, "Reference outside legal candidate set")
        lattice_tokens += len(candidates)
    lms = {}
    for language, source in manifest["lms"].items():
        text = (ROOT / source["path"]).read_text().strip()
        require(len(text) == source["characters"], "LM character count mismatch")
        lms[language] = IndependentLM(text, alphabet)

    checked_predictions = 0
    summaries = {}
    recomputed = {}
    score_deltas = []
    for split_name in ("fit", "transfer"):
        data = inputs[split_name]
        gold = answers["splits"][split_name]["plaintext_chunks"]
        oracle_candidates = decoded_candidates(data, gold_key)
        for arm in ("oracle", *ARMS):
            label = f"{arm}_{split_name}"
            key = gold_key if arm == "oracle" else freezes[arm]["key"]
            candidates = decoded_candidates(data, key)
            prediction = predictions[label]
            text = verify_prediction(prediction, candidates, label)
            language = "english" if arm == "english_joint" else "latin"
            lm = lms[language]
            score = lm.score(text)
            expected_score = (lm.best_lattice_score(candidates) if arm != "latin_fixed"
                              else lm.score("".join(paths[0] for paths in candidates)))
            same_number(score, expected_score, f"Prediction is not the prescribed decode: {label}")
            if arm == "latin_fixed":
                require(not any(prediction["choices"]), "Fixed arm changed parse choices")
            saved = report["oracle"][split_name] if arm == "oracle" else report["arms"][arm][split_name]
            metrics = independent_metrics(prediction["chunks"], gold, data, key, class_to_letter)
            compare_metrics(saved, metrics, label)
            same_number(saved["score"], score, f"Source score mismatch: {label}")
            score_deltas.append(abs(saved["score"] - score))
            if arm != "oracle":
                gold_score = (lm.best_lattice_score(oracle_candidates) if arm != "latin_fixed"
                              else lm.score("".join(paths[0] for paths in oracle_candidates)))
                same_number(saved["gold_key_score"], gold_score, f"Gold objective mismatch: {label}")
                same_number(saved["gold_minus_learned_objective"], gold_score - score,
                            f"Objective gap mismatch: {label}")
                if split_name == "fit":
                    fitted = fits[arm]
                    require(fitted["alphabet"] == data["alphabet"]
                            and fitted["class_ids"] == data["class_ids"], "Fit key index mismatch")
                    fit_chunks = [paths[choice] for paths, choice in
                                  zip(candidates, fitted["choices"], strict=True)]
                    fitted_text = verify_prediction({"choices": fitted["choices"], "chunks": fit_chunks},
                                                     candidates, f"frozen_{arm}")
                    same_number(lm.score(fitted_text), fitted["fit_score"], "Frozen path score mismatch")
                    same_number(fitted["fit_score"], score, "Frozen fit objective failed replay")
            checked_predictions += len(prediction["chunks"])
            recomputed[label] = metrics
            summaries[label] = {name: metrics[name] for name in
                                ("chunks", "characters", "edit_distance", "cer", "exact_chunks",
                                 "ambiguous_chunks", "ambiguous_correct", "key_frequency_weighted_accuracy")}
            denominator = metrics["ambiguous_chunks"]
            summaries[label]["ambiguous_accuracy"] = (metrics["ambiguous_correct"] / denominator
                                                       if denominator else None)
    primary, oracle = recomputed["latin_joint_transfer"], recomputed["oracle_transfer"]
    criteria = {"cer_at_most_02": primary["cer"] <= .02,
                "cer_within_005_of_oracle": primary["cer"] <= oracle["cer"] + .005,
                "weighted_key_at_least_99": primary["key_frequency_weighted_accuracy"] >= .99}
    require(report["criteria"] == criteria, "Primary criterion mismatch")
    decision = "PASS" if all(criteria.values()) else "FAIL"
    require(report["decision"] == decision, "Primary decision mismatch")
    return {"experiment": "NAIBBE-001", "audit_pass": True,
            "key_freeze_commit": revision, "source_commits": sorted(source_commits),
            "evaluation_sha256": sha(report_path), "manifest_sha256": sha(manifest_path),
            "auditor_sha256": sha(Path(__file__).resolve()), "checked_hashes": checked,
            "source_table_lattice_tokens_verified": lattice_tokens,
            "emitted_chunks_verified": checked_predictions,
            "max_score_delta": max(score_deltas), "criteria": criteria, "decision": decision,
            "metrics": summaries, "seconds": time.monotonic() - started,
            "scope": ["Independent source-table parse support for all prepared split tokens",
                      "Every emitted chunk, legal parse choice, and frozen key",
                      "Independent dictionary LM counts, interpolation, source scoring and exact lattice DP",
                      "Independent Wagner-Fischer CER, exact and ambiguous chunks, weighted key accuracy",
                      "Every fit/transfer report score and metric, and all primary decision criteria",
                      "Pinned raw/derived inputs, local Git freeze consistency, learned result/key agreement"],
            "not_established": ["No independent rerun of key optimization",
                                "Local Git checks do not prove remote publication timing or historical process isolation",
                                "This is known-codebook calibration; it does not recover the Voynich plaintext"]}


def scalar_distance(left: str, right: str) -> int:
    row = list(range(len(right) + 1))
    for i, first in enumerate(left, 1):
        following = [i]
        for j, second in enumerate(right, 1):
            following.append(min(row[j] + 1, following[-1] + 1, row[j - 1] + (first != second)))
        row = following
    return row[-1]


def self_test() -> dict:
    strings = ["".join(chars) for length in range(6) for chars in itertools.product("ab", repeat=length)]
    for left, right in itertools.product(strings, repeat=2):
        require(edit_distance(left, right) == scalar_distance(left, right), "Toy edit-distance failure")
    for left, right in (("prefix_abc_suffix", "prefix_bca_suffix"), ("a" * 100, "b" * 100),
                        ("kitten", "sitting"), ("ab" * 100, "ba" * 100)):
        require(edit_distance(left, right) == scalar_distance(left, right), "Extended edit-distance failure")
    lm = IndependentLM("abacabadabacaba" * 5, "abcd")
    for width in range(4):
        for context in itertools.product("abcd", repeat=width):
            total = sum(math.exp(lm.log_next("".join(context) + letter)) for letter in "abcd")
            same_number(total, 1., "Toy probability normalization failure", 1e-12)
    candidates = [["a", "bc"], ["dd", "b"], ["cb", "a"], ["a", "cd"]]
    brute = max(lm.score("".join(path)) for path in itertools.product(*candidates))
    same_number(lm.best_lattice_score(candidates), brute, "Toy exact lattice failure", 1e-12)
    require(lm.best_lattice_score([]) == 0., "Empty lattice failure")
    return {"self_test_pass": True, "edit_distance_pairs": len(strings) ** 2 + 4,
            "lm_contexts": sum(4 ** length for length in range(4)),
            "exhaustive_lattice_paths": math.prod(map(len, candidates)),
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
    output = ROOT / "results/NAIBBE-001/audit.json"
    require(not output.exists(), "Preserve prior audit; do not overwrite")
    result = audit()
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: result[name] for name in ("audit_pass", "decision", "emitted_chunks_verified",
                                                      "source_table_lattice_tokens_verified",
                                                      "max_score_delta", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
