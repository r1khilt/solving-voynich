"""Independent post-evaluation replay; no production decoder/model is imported.

Reuse audit001's max DP/distance, audit002's local-table/metric primitives and
audit003_source's scalar counts/probabilities. Self-test never reads target data.
The answer-reading mode first requires a complete evaluation and frozen keys.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path

from scripts.audit_naibbe001 import IndependentLM, committed, pinned, require, same_number, sha
from scripts.audit_naibbe002 import (
    ALPHABET, TABLES, canonical_commit, compare_metrics, decoded_candidates,
    independent_metrics, key_mapping, read_tables, self_test as inherited_self_test,
    table_parser, validate_metadata, verify_prediction,
)
from scripts.audit_naibbe003_source import CONFIGS, ScalarSourceLM


ROOT = Path(__file__).resolve().parent.parent
ARMS = ("latin_joint", "english_joint", "legacy_latin")
RANGES = {"fit": [0, 8192], "transfer": [26624, 34764]}
OLD_KEY_COMMIT = "8b27c21f397d7c0e3eb5fa8a03f6fae5aa2372ba"
SEARCH_SOURCES = {"scripts/run_naibbe003.py", "src/voynich/recursive_character_lm.py",
                  "src/voynich/naibbe_homophone_search.py", "src/voynich/naibbe_key_search.py"}
OLD_SOURCES = {"scripts/run_naibbe002.py", "src/voynich/naibbe_homophone_search.py",
               "src/voynich/naibbe_key_search.py"}
PRIMITIVES = ("scripts/audit_naibbe001.py", "scripts/audit_naibbe002.py",
              "scripts/audit_naibbe003_source.py")


class LatticeLM(ScalarSourceLM):
    """Adapt independent scalar probabilities to the previously audited max DP."""

    def log_next(self, gram: str) -> float:
        return math.log(self.probability(gram[:-1], gram[-1]))

    append = IndependentLM.append
    best_lattice_score = IndependentLM.best_lattice_score


def model(text: str, alphabet: str, config: dict) -> LatticeLM:
    require(config in CONFIGS, "Unregistered source configuration")
    family = "baseline" if config["family"] == "legacy" else config["family"]
    return LatticeLM(text, alphabet, family, config["parameter"])


def decision(primary: dict, oracle: dict) -> tuple[dict, str]:
    criteria = {"cer_at_most_02": primary["cer"] <= .02,
                "within_005_of_oracle": primary["cer"] <= oracle["cer"] + .005,
                "macro_key_at_least_95": primary["key_macro_accuracy"] >= .95,
                "unique_weighted_key_at_least_99": primary["unique_class_weighted_key_accuracy"] >= .99}
    return criteria, "PASS" if all(criteria.values()) else "FAIL"


def compare_fit_freeze(fit: dict, freeze: dict) -> None:
    for field, value in freeze.items():
        if field != "result_sha256":
            require(field in fit and fit[field] == value, f"Fit/freeze disagreement: {field}")
    required = {"arm", "fit_score", "seconds", "budget_stop", "key", "input_sha256",
                "lm_sha256", "source_commit", "seed", "source_hashes", "environment", "trace"}
    require(required <= set(freeze), "Incomplete key freeze")
    require(fit["coordinated"] is True and (fit["seed"], fit["restarts"], fit["kicks"]) == (920201, 8, 6),
            "Wrong registered search configuration")
    require(bool(fit["trace"]), "Missing search trace")
    best = -math.inf
    for row in fit["trace"]:
        require(math.isfinite(row["score"]), "Nonfinite trace score")
        best = max(best, row["score"])
        same_number(row["best_score"], best, "Trace incumbent mismatch")
    same_number(fit["fit_score"], best, "Frozen score differs from best trace")


def audit_prediction(prediction: dict, data: dict, key: list[int], gold_key: list[int],
                     gold: list[str], lm: LatticeLM, saved: dict, label: str,
                     gold_score: float | None = None) -> tuple[dict, dict, float]:
    candidates = decoded_candidates(data, key)
    text = verify_prediction(prediction, candidates, label)
    score = lm.score(text)
    same_number(score, lm.best_lattice_score(candidates), f"Nonoptimal saved path: {label}")
    metrics, groups = independent_metrics(prediction["chunks"], gold, data, key, gold_key)
    compare_metrics(saved, metrics, label)
    same_number(saved["score"], score, f"Saved score mismatch: {label}")
    if gold_score is not None:
        same_number(saved["gold_key_score"], gold_score, f"Gold score mismatch: {label}")
        same_number(saved["gold_minus_learned_objective"], gold_score - score,
                    f"Objective gap mismatch: {label}")
    summary = {name: value for name, value in metrics.items() if name != "key_errors"}
    summary["unresolved_class_characters"] = metrics["characters"] - metrics["unique_class_characters"]
    summary["per_group_audit"] = groups
    return metrics, summary, abs(saved["score"] - score)


def ancestor(first: str, second: str) -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", first, second], cwd=ROOT, check=True)


def frozen_json(relative: str, revision: str) -> dict:
    return json.loads(committed(ROOT / relative, revision))


def audit() -> dict:
    started = time.monotonic()
    report_path = ROOT / "results/NAIBBE-003/evaluation.json"
    require(report_path.is_file(), "Complete evaluation before answer access")
    report = json.loads(report_path.read_text())
    require(report["experiment"] == "NAIBBE-003", "Wrong evaluation experiment")
    for kind in ("arms", "oracle"):
        require(set(report[kind]) == set(ARMS)
                and all(set(report[kind][arm]) == set(RANGES) for arm in ARMS),
                "Incomplete evaluation arms/oracles")
    revision = canonical_commit(report["key_freeze_commit"])
    ancestor(revision, "HEAD")
    require(report["old_key_commit"] == OLD_KEY_COMMIT, "Frozen baseline revision changed")
    ancestor(canonical_commit(OLD_KEY_COMMIT), revision)
    checked: dict[str, str] = {}
    for filename in (*PRIMITIVES, "scripts/audit_naibbe003.py", "tests/test_naibbe003_audit.py",
                     "scripts/evaluate_naibbe001.py", "scripts/evaluate_naibbe002.py",
                     "scripts/evaluate_naibbe003.py", "docs/experiments/NAIBBE-003.md"):
        committed(ROOT / filename, revision)
        checked[filename] = sha(ROOT / filename)
    manifest_path = ROOT / "data/manifests/naibbe003_data.json"
    manifest = frozen_json(str(manifest_path.relative_to(ROOT)), revision)
    require(manifest["experiment"] == "NAIBBE-003"
            and (manifest["classes"], manifest["groups"], manifest["classes_per_group"]) == (138, 6, 23),
            "Wrong data dimensions/experiment")
    selection_name = "results/NAIBBE-003/source_selection.json"
    source_audit_name = "results/NAIBBE-003/source_audit.json"
    selection = frozen_json(selection_name, revision)
    source_audit = frozen_json(source_audit_name, revision)
    selection_sha = sha(ROOT / selection_name)
    require(selection_sha == report["selection_sha256"] == source_audit["source_selection_sha256"],
            "Source-selection audit/hash mismatch")
    require(selection["cipher_phase_allowed"] is True and source_audit["audit_pass"] is True
            and source_audit["cipher_phase_allowed"] is True and source_audit["cipher_files_read"] is False,
            "Cipher phase lacked a passing independent source check")
    source_revision = canonical_commit(selection["source_commit"])
    require(source_audit["source_commit"] == source_revision, "Source-audit revision mismatch")
    ancestor(source_revision, revision)
    require(source_audit["auditor_sha256"] == sha(ROOT / PRIMITIVES[-1]), "Source auditor changed")
    require(selection["manifest_sha256"] == sha(manifest_path), "Selected source manifest changed")
    for filename, digest in source_audit["checked_hashes"].items():
        pinned(ROOT / filename, digest, checked)
    for filename, digest in selection["source_hashes"].items():
        pinned(ROOT / filename, digest, checked)
        committed(ROOT / filename, source_revision)

    # Learned outputs and all three keys are verified before opening any answers.
    fits, configs, commits = {}, {}, set()
    for arm in ARMS:
        old = arm == "legacy_latin"
        experiment, saved_arm = ("NAIBBE-002", "latin_joint") if old else ("NAIBBE-003", arm)
        freeze = frozen_json(f"results/{experiment}/{saved_arm}_freeze.json", OLD_KEY_COMMIT if old else revision)
        path = ROOT / f"outputs/{experiment}/{saved_arm}.json"
        pinned(path, freeze["result_sha256"], checked)
        fit = json.loads(path.read_text())
        compare_fit_freeze(fit, freeze)
        require(fit["arm"] == saved_arm, "Wrong learned arm")
        language = "english" if arm == "english_joint" else "latin"
        config = {"family": "legacy", "parameter": None} if old else selection["languages"][language]["selected_config"]
        require(config in CONFIGS, "Unregistered fitted prior")
        if not old:
            require(source_audit["languages"][language]["selected_config"] == config
                    and fit["prior_config"] == config and fit["selection_sha256"] == selection_sha,
                    "Fitted prior does not match audited selection")
        expected_input = manifest["parent_fit_input"] if old else manifest["split_inputs"]["fit"]
        require(fit["input_sha256"] == expected_input["sha256"], "Wrong fit source hash")
        require(fit["lm_sha256"] == manifest["lms"][language]["sha256"], "Wrong fitted source corpus")
        commit = canonical_commit(fit["source_commit"])
        ancestor(commit, OLD_KEY_COMMIT if old else revision)
        require(set(fit["source_hashes"]) == (OLD_SOURCES if old else SEARCH_SOURCES),
                "Unexpected fit source-code inventory")
        for filename, digest in fit["source_hashes"].items():
            pinned(ROOT / filename, digest, checked)
            committed(ROOT / filename, commit)
        if not old:
            ancestor(source_revision, commit)
            for filename in (selection_name, source_audit_name, str(manifest_path.relative_to(ROOT)),
                             "docs/experiments/NAIBBE-003.md", "scripts/build_naibbe003_data.py"):
                committed(ROOT / filename, commit)
        fits[arm], configs[arm] = fit, config
        commits.add(commit)

    for source in (*manifest["sources"].values(), *manifest["lms"].values(),
                   manifest["parent_manifest"], manifest["parent_fit_input"],
                   *manifest["builder_sources"]):
        pinned(ROOT / source["path"], source["sha256"], checked)
    for source in (manifest["parent_manifest"], *manifest["builder_sources"]):
        committed(ROOT / source["path"], revision)
    parent = json.loads((ROOT / manifest["parent_manifest"]["path"]).read_text())
    require(parent["sources"] == manifest["sources"] and parent["lms"] == manifest["lms"]
            and parent["archive_revision"] == manifest["archive_revision"], "Inherited raw inputs changed")
    require(manifest["parent_fit_input"] == parent["split_inputs"]["fit"]
            and manifest["parent_answer_key"] == parent["answer_key"], "Parent input/answer links changed")
    inputs = {}
    require(set(manifest["split_inputs"]) == set(RANGES), "Wrong prepared splits")
    for split_name, source in manifest["split_inputs"].items():
        path = ROOT / source["path"]
        pinned(path, source["sha256"], checked)
        data = json.loads(path.read_text())
        validate_metadata(data)
        require(data["experiment"] == "NAIBBE-003" and data["split_name"] == split_name
                and data["split"]["token_range"] == RANGES[split_name], "Wrong prepared range/experiment")
        require("".join(data["alphabet"]) == ALPHABET and len(data["class_ids"]) == 138
                and len(data["groups"]) == 6 and data["lms"] == manifest["lms"], "Wrong input metadata")
        inputs[split_name] = data
    for field in ("class_ids", "groups", "group_ids", "alphabet", "lms"):
        require(inputs["fit"][field] == inputs["transfer"][field], "Split identity/order mismatch")
    old_fit = json.loads((ROOT / manifest["parent_fit_input"]["path"]).read_text())
    require({**old_fit, "experiment": "NAIBBE-003"} == inputs["fit"], "Fit changed beyond experiment label")
    prediction_path = ROOT / "outputs/NAIBBE-003/evaluation_predictions.json"
    pinned(prediction_path, report["predictions_sha256"], checked)
    predictions = json.loads(prediction_path.read_text())
    expected = {f"{prefix}{arm}_{split}" for prefix in ("", "oracle_") for arm in ARMS for split in RANGES}
    require(set(predictions) == expected, "Incomplete prediction inventory")

    # Explicitly post-evaluation answer access, after immutable-key checks above.
    answer_path = ROOT / manifest["answer_key"]["path"]
    require(manifest["answer_key"]["sha256"] == manifest["answer_key_sha256"] == report["answer_sha256"],
            "Answer hash links disagree")
    pinned(answer_path, manifest["answer_key_sha256"], checked)
    pinned(ROOT / manifest["parent_answer_key"]["path"], manifest["parent_answer_key"]["sha256"], checked)
    answers = json.loads(answer_path.read_text())
    old_answers = json.loads((ROOT / manifest["parent_answer_key"]["path"]).read_text())
    require(answers["experiment"] == "NAIBBE-003" and set(answers["splits"]) == set(RANGES), "Wrong answers")
    for field in ("class_to_letter", "class_to_table"):
        require(answers[field] == old_answers[field], "Inherited answer identities changed")
    letters, tables = answers["class_to_letter"], answers["class_to_table"]
    require(set(letters) == set(tables) == set(inputs["fit"]["class_ids"]), "Gold class inventory mismatch")
    require(Counter(letters.values()) == Counter({c: 6 for c in ALPHABET}) and set(tables.values()) == TABLES,
            "Wrong gold table/alphabet inventory")
    identities = {(tables[c], letters[c]): c for c in letters}
    require(len(identities) == 138
            and all(len({tables[c] for c in group}) == 1 for group in inputs["fit"]["groups"]),
            "Gold local-table identity mismatch")
    gold_key = [ALPHABET.index(letters[c]) for c in inputs["fit"]["class_ids"]]
    key_mapping(inputs["fit"], gold_key)
    rows = read_tables(ROOT / manifest["sources"]["greshko_naibbe_tables.csv"]["path"])
    parse, plain_parse = table_parser(rows, identities), table_parser(rows)
    ciphertext = (ROOT / manifest["sources"]["greshko_nathist_output_ciphertext.txt"]["path"]).read_text().split()
    plaintext = (ROOT / manifest["sources"]["greshko_nathist_pre_encryption_respaced_plaintext.txt"]["path"]).read_text().split()
    require(len(ciphertext) == len(plaintext) == 34764, "Published alignment changed")
    lattice_tokens = 0
    for split, data in inputs.items():
        start, stop = RANGES[split]
        reference = answers["splits"][split]
        gold = reference["plaintext_chunks"]
        require(reference["token_range"] == [start, stop]
                and data["split"]["tokens"] == ciphertext[start:stop]
                and gold == plaintext[start:stop], "Raw source split mismatch")
        alternatives = decoded_candidates(data, gold_key)
        require(len(alternatives) == len(gold) == stop - start, "Wrong prepared split size")
        for token, paths, decoded, chunk in zip(data["split"]["tokens"], data["split"]["candidates"],
                                               alternatives, gold, strict=True):
            require({tuple(path) for path in paths} == parse(token), "Exact local-class support mismatch")
            require(set(decoded) == {"".join(path) for path in plain_parse(token)} and chunk in decoded,
                    "Reference/plaintext support mismatch")
        counts = {"tokens": len(gold), "gold_characters": sum(map(len, gold)), "token_range": [start, stop],
                  "candidate_count_histogram": dict(Counter(str(len(paths)) for paths in alternatives)),
                  "unique_candidate_tokens": sum(len(paths) == 1 for paths in alternatives)}
        require(manifest["splits"][split] == counts, "Manifest split statistics mismatch")
        lattice_tokens += len(gold)

    metrics, summaries, normalizations, deltas = {}, {}, {}, []
    emitted = 0
    for arm in ARMS:
        language = "english" if arm == "english_joint" else "latin"
        source = manifest["lms"][language]
        corpus = (ROOT / source["path"]).read_text().strip()
        require(len(corpus) == source["characters"] == 317326, "LM corpus length mismatch")
        lm = model(corpus, ALPHABET, configs[arm])
        normalizations[arm] = lm.normalization_audit()
        for split, data in inputs.items():
            gold = answers["splits"][split]["plaintext_chunks"]
            oracle_score = lm.best_lattice_score(decoded_candidates(data, gold_key))
            for oracle in (False, True):
                label = f"{'oracle_' if oracle else ''}{arm}_{split}"
                key = gold_key if oracle else fits[arm]["key"]
                saved = report["oracle" if oracle else "arms"][arm][split]
                row, summary, delta = audit_prediction(predictions[label], data, key, gold_key, gold, lm,
                                                        saved, label, None if oracle else oracle_score)
                if oracle:
                    same_number(saved["score"], oracle_score, "Oracle score mismatch")
                elif split == "fit":
                    fit = fits[arm]
                    for field in ("class_ids", "alphabet", "groups"):
                        require(fit[field] == data[field], "Frozen fit key indexing mismatch")
                    candidates = decoded_candidates(data, key)
                    choices = fit["choices"]
                    require(len(choices) == len(candidates), "Frozen fit path length mismatch")
                    require(all(type(choice) is int and 0 <= choice < len(paths)
                                for choice, paths in zip(choices, candidates, strict=True)), "Invalid fit path")
                    chunks = [paths[choice] for choice, paths in zip(choices, candidates, strict=True)]
                    same_number(lm.score("".join(chunks)), fit["fit_score"], "Frozen path score mismatch")
                    same_number(saved["score"], fit["fit_score"], "Frozen fit objective mismatch")
                metrics[label], summaries[label] = row, summary
                deltas.append(delta)
                emitted += len(predictions[label]["chunks"])
    primary, oracle = metrics["latin_joint_transfer"], metrics["oracle_latin_joint_transfer"]
    criteria, verdict = decision(primary, oracle)
    require(report["criteria"] == criteria and report["decision"] == verdict, "Primary gate mismatch")
    baseline = metrics["legacy_latin_transfer"]
    comparison = {"edit_distance_change": primary["edit_distance"] - baseline["edit_distance"],
                  "key_correct_change": primary["key_types_correct"] - baseline["key_types_correct"]}
    require(report["versus_frozen_baseline"] == comparison, "Frozen-baseline comparison mismatch")
    return {"experiment": "NAIBBE-003", "audit_pass": True, "key_freeze_commit": revision,
            "old_key_commit": OLD_KEY_COMMIT, "source_commits": sorted(commits),
            "evaluation_sha256": sha(report_path), "manifest_sha256": sha(manifest_path),
            "source_selection_sha256": selection_sha, "source_audit_sha256": sha(ROOT / source_audit_name),
            "auditor_sha256": sha(Path(__file__).resolve()),
            "independent_primitives_sha256": {name: sha(ROOT / name) for name in PRIMITIVES},
            "checked_hashes": checked, "normalization": normalizations,
            "source_table_lattice_tokens_verified": lattice_tokens, "exact_local_class_support_verified": True,
            "emitted_chunks_verified": emitted, "max_score_delta": max(deltas),
            "criteria": criteria, "decision": verdict, "versus_frozen_baseline": comparison,
            "metrics": summaries, "seconds": time.monotonic() - started,
            "scope": ["Independent raw counts, normalized selected/legacy priors and exact max lattice DP",
                      "All three learned/frozen keys and three gold-key oracles on fit and final transfer",
                      "Original-table exact local-class supports and raw source/answer slice integrity",
                      "Every chunk/choice, Wagner-Fischer CER, unique-position key weights and gates",
                      "Source selection/audit/fit/key local Git chains and pinned raw/derived files"],
            "not_established": ["Key optimization not independently rerun; no global-optimum claim",
                                "Source-grid calculations inherited from the separately frozen independent source audit",
                                "Local Git checks cannot prove remote timing or historical process isolation",
                                "Supplied table/role grammar remains; no Voynich decipherment"]}


def self_test() -> dict:
    result = inherited_self_test()
    paths = [["a", "bc"], ["dd", "b"], ["cb", "a"], ["a", "cd"]]
    verified = 0
    for config in CONFIGS:
        lm = model("abacabadabacaba" * 5, "abcd", config)
        lm.normalization_audit()
        brute = max(lm.score("".join(path)) for path in itertools.product(*paths))
        same_number(lm.best_lattice_score(paths), brute, "Source-family max-DP failure", 1e-12)
        same_number(lm.best_lattice_score([row + row for row in paths]), brute, "Duplicate paths add mass", 1e-12)
        require(lm.best_lattice_score([]) == 0., "Empty lattice failure")
        verified += math.prod(map(len, paths))
    return {**result, "source_configurations_checked": len(CONFIGS),
            "source_family_exhaustive_paths": verified, "experiment_files_read": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--after-evaluation", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    output = ROOT / "results/NAIBBE-003/audit.json"
    require(not output.exists(), "Preserve previous audit")
    result = audit()
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: result[name] for name in ("audit_pass", "decision", "emitted_chunks_verified",
                     "source_table_lattice_tokens_verified", "max_score_delta", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
