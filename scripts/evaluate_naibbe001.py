"""Answer access is permitted only after all three learned keys are frozen."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from voynich.naibbe_key_search import CharacterLM, encode_lattice, flatten, viterbi

ROOT = Path(__file__).resolve().parent.parent
ARMS = ("latin_joint", "latin_fixed", "english_joint")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edit_distance(left: str, right: str) -> int:
    """Myers bit-vector Levenshtein, unrestricted insertion/deletion/substitution."""
    if not left:
        return len(right)
    masks = {}
    for i, char in enumerate(left):
        masks[char] = masks.get(char, 0) | (1 << i)
    positive, negative, distance = ~0, 0, len(left)
    highest = 1 << (len(left) - 1)
    for char in right:
        equal = masks.get(char, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        positive_horizontal = negative | ~(horizontal | positive)
        negative_horizontal = positive & horizontal
        distance += bool(positive_horizontal & highest) - bool(negative_horizontal & highest)
        positive_horizontal = (positive_horizontal << 1) | 1
        negative_horizontal <<= 1
        positive = negative_horizontal | ~(vertical | positive_horizontal)
        negative = positive_horizontal & vertical
    return int(distance)


def metrics(chunks: list[str], gold: list[str], lattice, key, gold_key, alphabet) -> dict:
    text, answer = "".join(chunks), "".join(gold)
    ambiguous = [i for i, alternatives in enumerate(lattice) if len(alternatives) > 1]
    frequencies = np.bincount([alphabet.index(c) for c in answer], minlength=len(alphabet))
    correct = np.asarray(key) == np.asarray(gold_key)
    key_counts = frequencies[np.asarray(gold_key)]
    return {"characters": len(answer), "predicted_characters": len(text),
            "edit_distance": edit_distance(answer, text), "cer": edit_distance(answer, text) / len(answer),
            "exact_chunks": sum(a == b for a, b in zip(chunks, gold, strict=True)),
            "chunks": len(gold), "chunk_accuracy": sum(a == b for a, b in zip(chunks, gold, strict=True)) / len(gold),
            "ambiguous_chunks": len(ambiguous),
            "ambiguous_correct": sum(chunks[i] == gold[i] for i in ambiguous),
            "key_types_correct": int(correct.sum()), "key_types": len(key),
            "key_frequency_weighted_accuracy": float(key_counts[correct].sum() / key_counts.sum()),
            "key_errors": [{"true": alphabet[int(gold_key[i])], "predicted": alphabet[int(key[i])],
                            "source_occurrences": int(key_counts[i])}
                           for i in range(len(key)) if not correct[i]],
            "unobserved_key_types": int((key_counts == 0).sum())}


def decode(data, lm, key, joint):
    lattice = encode_lattice(data["split"], data["class_ids"])
    if joint:
        score, choices = viterbi(lattice, key, lm)
    else:
        choices = [0] * len(lattice)
        score = lm.score(key[flatten(lattice, choices)])
    alphabet = "".join(data["alphabet"])
    chunks = ["".join(alphabet[int(key[c])] for c in paths[choice])
              for paths, choice in zip(lattice, choices, strict=True)]
    return score, choices, chunks, lattice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-freeze-commit", required=True)
    args = parser.parse_args()
    freeze_commit = subprocess.check_output(["git", "rev-parse", args.key_freeze_commit],
                                            cwd=ROOT, text=True).strip()
    manifest = json.loads((ROOT / "data/manifests/naibbe001_data.json").read_text())
    freezes = {}
    for arm in ARMS:
        name = f"results/NAIBBE-001/{arm}_freeze.json"
        committed = subprocess.check_output(["git", "show", f"{freeze_commit}:{name}"], cwd=ROOT)
        if committed != (ROOT / name).read_bytes():
            raise AssertionError("Selected key not frozen at requested commit")
        freezes[arm] = json.loads(committed)
        if sha(ROOT / f"outputs/NAIBBE-001/{arm}.json") != freezes[arm]["result_sha256"]:
            raise AssertionError("Fit output mutated")
    answer_path = ROOT / "data/processed/naibbe001/answer_key.json"
    if sha(answer_path) != manifest["answer_key_sha256"]:
        raise AssertionError("Answer file changed")
    answers = json.loads(answer_path.read_text())
    lms = {}
    for language in ("latin", "english"):
        path = ROOT / manifest["lms"][language]["path"]
        if sha(path) != manifest["lms"][language]["sha256"]:
            raise AssertionError("Source model data changed")
        lms[language] = CharacterLM.train(path.read_text().strip(), "abcdefghilmnopqrstuvxyz")
    report = {"experiment": "NAIBBE-001", "key_freeze_commit": freeze_commit,
              "answer_sha256": sha(answer_path), "arms": {}, "oracle": {}}
    predictions = {}
    for split_name in ("fit", "transfer"):
        path = ROOT / manifest["split_inputs"][split_name]["path"]
        if sha(path) != manifest["split_inputs"][split_name]["sha256"]:
            raise AssertionError("Split hash mismatch")
        data = json.loads(path.read_text())
        alphabet = "".join(data["alphabet"])
        gold_key = np.array([alphabet.index(answers["class_to_letter"][c]) for c in data["class_ids"]])
        gold = answers["splits"][split_name]["plaintext_chunks"]
        oracle_score, oracle_choices, oracle_chunks, lattice = decode(data, lms["latin"], gold_key, True)
        report["oracle"][split_name] = metrics(oracle_chunks, gold, lattice, gold_key, gold_key, alphabet)
        report["oracle"][split_name]["score"] = oracle_score
        predictions[f"oracle_{split_name}"] = {"choices": oracle_choices, "chunks": oracle_chunks}
        for arm in ARMS:
            key = np.array(freezes[arm]["key"])
            if sorted(key) != list(range(len(alphabet))):
                raise AssertionError("Key must be bijective")
            lm = lms[arm.split("_")[0]]
            joint = arm != "latin_fixed"
            score, choices, chunks, lattice = decode(data, lm, key, joint)
            gold_score, _, _, _ = decode(data, lm, gold_key, joint)
            values = metrics(chunks, gold, lattice, key, gold_key, alphabet)
            values.update({"score": score, "gold_key_score": gold_score,
                           "gold_minus_learned_objective": gold_score - score})
            if split_name == "fit" and abs(score - freezes[arm]["fit_score"]) > 1e-6:
                raise AssertionError("Fit objective does not replay")
            report["arms"].setdefault(arm, {})[split_name] = values
            predictions[f"{arm}_{split_name}"] = {"choices": choices, "chunks": chunks}
    primary = report["arms"]["latin_joint"]["transfer"]
    oracle = report["oracle"]["transfer"]
    report["criteria"] = {"cer_at_most_02": primary["cer"] <= .02,
                          "cer_within_005_of_oracle": primary["cer"] <= oracle["cer"] + .005,
                          "weighted_key_at_least_99": primary["key_frequency_weighted_accuracy"] >= .99}
    report["decision"] = "PASS" if all(report["criteria"].values()) else "FAIL"
    output = ROOT / "outputs/NAIBBE-001/evaluation_predictions.json"
    if output.exists():
        raise FileExistsError("Preserve prior evaluation")
    output.write_text(json.dumps(predictions) + "\n")
    report["predictions_sha256"] = sha(output)
    (ROOT / "results/NAIBBE-001/evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
