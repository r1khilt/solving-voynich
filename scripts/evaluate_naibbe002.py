"""Post-freeze recovery metrics; table-draw identities are not observed gold."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from scripts.evaluate_naibbe001 import decode, edit_distance
from voynich.naibbe_key_search import CharacterLM

ROOT = Path(__file__).resolve().parent.parent
ARMS = ("latin_joint", "latin_local", "english_joint")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(chunks, gold, data, key, gold_key):
    text, answer = "".join(chunks), "".join(gold)
    ids = {c: i for i, c in enumerate(data["class_ids"])}
    counts = np.zeros(len(ids), dtype=int)
    ambiguous = []
    for i, paths in enumerate(data["split"]["candidates"]):
        if len(paths) > 1:
            ambiguous.append(i)
        if len({len(path) for path in paths}) != 1:
            raise AssertionError("Unique-position metric requires equal candidate lengths")
        for position in range(len(paths[0])):
            options = {path[position] for path in paths}
            if len(options) == 1:
                counts[ids[next(iter(options))]] += 1
    correct = np.asarray(key) == np.asarray(gold_key)
    distance = edit_distance(answer, text)
    group_indices = [[ids[c] for c in group] for group in data["groups"]]
    return {"characters": len(answer), "predicted_characters": len(text), "edit_distance": distance,
            "cer": distance / len(answer), "chunks": len(gold),
            "exact_chunks": sum(a == b for a, b in zip(chunks, gold, strict=True)),
            "ambiguous_chunks": len(ambiguous),
            "ambiguous_correct": sum(chunks[i] == gold[i] for i in ambiguous),
            "key_types": len(key), "key_types_correct": int(correct.sum()),
            "key_macro_accuracy": float(correct.mean()),
            "unique_class_characters": int(counts.sum()),
            "unique_class_weighted_key_accuracy": float(counts[correct].sum() / counts.sum()),
            "no_unique_evidence_classes": int((counts == 0).sum()),
            "per_group_key_correct": [int(correct[g].sum()) for g in group_indices],
            "key_errors": [{"class_id": data["class_ids"][i], "true": data["alphabet"][int(gold_key[i])],
                            "predicted": data["alphabet"][int(key[i])], "unique_class_count": int(counts[i])}
                           for i in range(len(key)) if not correct[i]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-freeze-commit", required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", args.key_freeze_commit], cwd=ROOT, text=True).strip()
    output = ROOT / "outputs/NAIBBE-002/evaluation_predictions.json"
    if output.exists():
        raise FileExistsError("Preserve previous evaluation")
    freezes = {}
    for arm in ARMS:
        relative = f"results/NAIBBE-002/{arm}_freeze.json"
        raw = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=ROOT)
        if raw != (ROOT / relative).read_bytes():
            raise AssertionError("Unfrozen key")
        freezes[arm] = json.loads(raw)
        if sha(ROOT / f"outputs/NAIBBE-002/{arm}.json") != freezes[arm]["result_sha256"]:
            raise AssertionError("Fit artifact changed")
    manifest = json.loads((ROOT / "data/manifests/naibbe002_data.json").read_text())
    answer_path = ROOT / "data/processed/naibbe002/answer_key.json"
    if sha(answer_path) != manifest["answer_key_sha256"]:
        raise AssertionError("Answer changed")
    answers = json.loads(answer_path.read_text())
    lms = {}
    for language, source in manifest["lms"].items():
        path = ROOT / source["path"]
        if sha(path) != source["sha256"]:
            raise AssertionError("Source prior changed")
        lms[language] = CharacterLM.train(path.read_text().strip(), "abcdefghilmnopqrstuvxyz")
    report = {"experiment": "NAIBBE-002", "key_freeze_commit": revision,
              "answer_sha256": sha(answer_path), "arms": {}, "oracle": {}}
    predictions = {}
    for split_name in ("fit", "transfer"):
        source = manifest["split_inputs"][split_name]
        path = ROOT / source["path"]
        if sha(path) != source["sha256"]:
            raise AssertionError("Split changed")
        data = json.loads(path.read_text())
        gold_key = np.array([data["alphabet"].index(answers["class_to_letter"][c]) for c in data["class_ids"]])
        gold = answers["splits"][split_name]["plaintext_chunks"]
        for arm in ("oracle", *ARMS):
            key = gold_key if arm == "oracle" else np.array(freezes[arm]["key"])
            ids = {c: i for i, c in enumerate(data["class_ids"])}
            if any(sorted(key[[ids[c] for c in g]]) != list(range(len(data["alphabet"]))) for g in data["groups"]):
                raise AssertionError("Invalid within-table key")
            language = "english" if arm == "english_joint" else "latin"
            score, choices, chunks, _ = decode(data, lms[language], key, True)
            gold_score, _, _, _ = decode(data, lms[language], gold_key, True)
            row = metrics(chunks, gold, data, key, gold_key)
            row.update({"score": score, "gold_key_score": gold_score,
                        "gold_minus_learned_objective": gold_score - score})
            if arm != "oracle" and split_name == "fit" and abs(score - freezes[arm]["fit_score"]) > 1e-6:
                raise AssertionError("Fit objective mismatch")
            if arm == "oracle":
                report["oracle"][split_name] = row
            else:
                report["arms"].setdefault(arm, {})[split_name] = row
            predictions[f"{arm}_{split_name}"] = {"choices": choices, "chunks": chunks}
    primary = report["arms"]["latin_joint"]["transfer"]
    report["criteria"] = {"cer_at_most_02": primary["cer"] <= .02,
                          "within_005_of_oracle": primary["cer"] <= report["oracle"]["transfer"]["cer"] + .005,
                          "macro_key_at_least_95": primary["key_macro_accuracy"] >= .95,
                          "unique_weighted_key_at_least_99": primary["unique_class_weighted_key_accuracy"] >= .99}
    report["decision"] = "PASS" if all(report["criteria"].values()) else "FAIL"
    output.write_text(json.dumps(predictions) + "\n")
    report["predictions_sha256"] = sha(output)
    (ROOT / "results/NAIBBE-002/evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"decision": report["decision"], "primary": primary}, indent=2))


if __name__ == "__main__":
    main()
