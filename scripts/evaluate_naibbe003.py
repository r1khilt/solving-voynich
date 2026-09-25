"""One post-freeze final-block evaluation; preserve all earlier FAIL decisions."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from scripts.evaluate_naibbe001 import decode
from scripts.evaluate_naibbe002 import metrics
from voynich.recursive_character_lm import train

ROOT = Path(__file__).resolve().parent.parent
NEW_ARMS = ("latin_joint", "english_joint")
ARMS = (*NEW_ARMS, "legacy_latin")
OLD_KEY_COMMIT = "8b27c21f397d7c0e3eb5fa8a03f6fae5aa2372ba"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def committed(relative, revision):
    raw = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=ROOT)
    if raw != (ROOT / relative).read_bytes():
        raise AssertionError(f"Artifact differs from frozen version: {relative}")
    return raw


def decision(primary, oracle):
    criteria = {"cer_at_most_02": primary["cer"] <= .02,
                "within_005_of_oracle": primary["cer"] <= oracle["cer"] + .005,
                "macro_key_at_least_95": primary["key_macro_accuracy"] >= .95,
                "unique_weighted_key_at_least_99": primary["unique_class_weighted_key_accuracy"] >= .99}
    return criteria, "PASS" if all(criteria.values()) else "FAIL"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-freeze-commit", required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", args.key_freeze_commit], cwd=ROOT, text=True).strip()
    output = ROOT / "outputs/NAIBBE-003/evaluation_predictions.json"
    report_path = ROOT / "results/NAIBBE-003/evaluation.json"
    if output.exists() or report_path.exists():
        raise FileExistsError("Preserve previous evaluation")
    for relative in ("scripts/evaluate_naibbe003.py", "scripts/evaluate_naibbe001.py",
                     "scripts/evaluate_naibbe002.py", "src/voynich/recursive_character_lm.py",
                     "src/voynich/naibbe_key_search.py"):
        committed(relative, revision)
    selection = json.loads(committed("results/NAIBBE-003/source_selection.json", revision))
    manifest = json.loads(committed("data/manifests/naibbe003_data.json", revision))
    freezes = {arm: json.loads(committed(f"results/NAIBBE-003/{arm}_freeze.json", revision)) for arm in NEW_ARMS}
    for arm, freeze in freezes.items():
        if sha(ROOT / f"outputs/NAIBBE-003/{arm}.json") != freeze["result_sha256"]:
            raise AssertionError("Fit artifact changed")
        if freeze["selection_sha256"] != sha(ROOT / "results/NAIBBE-003/source_selection.json"):
            raise AssertionError("Selection changed after fitting")
    freezes["legacy_latin"] = json.loads(committed("results/NAIBBE-002/latin_joint_freeze.json", OLD_KEY_COMMIT))
    answer_path = ROOT / manifest["answer_key"]["path"]
    if sha(answer_path) != manifest["answer_key_sha256"]:
        raise AssertionError("Answer changed")
    answers = json.loads(answer_path.read_text())
    lms = {}
    for arm in ARMS:
        language = "english" if arm == "english_joint" else "latin"
        source = manifest["lms"][language]
        path = ROOT / source["path"]
        if sha(path) != source["sha256"]:
            raise AssertionError("Source corpus changed")
        config = ({"family": "legacy", "parameter": None} if arm == "legacy_latin"
                  else selection["languages"][language]["selected_config"])
        if arm in NEW_ARMS and config != freezes[arm]["prior_config"]:
            raise AssertionError("Fitted prior differs from selected prior")
        lms[arm] = train(path.read_text().strip(), "abcdefghilmnopqrstuvxyz", config)
    report = {"experiment": "NAIBBE-003", "key_freeze_commit": revision,
              "old_key_commit": OLD_KEY_COMMIT, "answer_sha256": sha(answer_path),
              "selection_sha256": sha(ROOT / "results/NAIBBE-003/source_selection.json"),
              "arms": {}, "oracle": {}}
    predictions = {}
    for split_name in ("fit", "transfer"):
        source = manifest["split_inputs"][split_name]
        path = ROOT / source["path"]
        if sha(path) != source["sha256"]:
            raise AssertionError("Split changed")
        data = json.loads(path.read_text())
        gold_key = np.array([data["alphabet"].index(answers["class_to_letter"][c]) for c in data["class_ids"]])
        gold = answers["splits"][split_name]["plaintext_chunks"]
        for arm in ARMS:
            key = np.array(freezes[arm]["key"])
            ids = {c: i for i, c in enumerate(data["class_ids"])}
            if any(sorted(key[[ids[c] for c in group]]) != list(range(23)) for group in data["groups"]):
                raise AssertionError("Invalid within-table key")
            score, choices, chunks, _ = decode(data, lms[arm], key, True)
            gold_score, gold_choices, oracle_chunks, _ = decode(data, lms[arm], gold_key, True)
            row = metrics(chunks, gold, data, key, gold_key)
            row.update({"score": score, "gold_key_score": gold_score,
                        "gold_minus_learned_objective": gold_score - score})
            if split_name == "fit" and abs(score - freezes[arm]["fit_score"]) > 1e-6:
                raise AssertionError("Fit objective mismatch")
            report["arms"].setdefault(arm, {})[split_name] = row
            oracle = metrics(oracle_chunks, gold, data, gold_key, gold_key)
            oracle["score"] = gold_score
            report["oracle"].setdefault(arm, {})[split_name] = oracle
            predictions[f"{arm}_{split_name}"] = {"choices": choices, "chunks": chunks}
            predictions[f"oracle_{arm}_{split_name}"] = {"choices": gold_choices, "chunks": oracle_chunks}
    primary = report["arms"]["latin_joint"]["transfer"]
    baseline = report["arms"]["legacy_latin"]["transfer"]
    report["criteria"], report["decision"] = decision(primary, report["oracle"]["latin_joint"]["transfer"])
    report["versus_frozen_baseline"] = {"edit_distance_change": primary["edit_distance"] - baseline["edit_distance"],
                                        "key_correct_change": primary["key_types_correct"] - baseline["key_types_correct"]}
    output.write_text(json.dumps(predictions) + "\n")
    report["predictions_sha256"] = sha(output)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"decision": report["decision"], "primary": {key: primary[key]
                      for key in ("characters", "edit_distance", "key_types_correct", "cer")},
                      "versus_frozen_baseline": report["versus_frozen_baseline"]}, indent=2))


if __name__ == "__main__":
    main()
