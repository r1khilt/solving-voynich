"""Independent source-only replay; opens only pinned Caesar and Virgil text.

No production source estimator/scorer is imported. Ciphertext, Cicero, transfer
and answer artifacts are never opened. All text bytes are hash checked before
JSON decoding. Scoring uses sufficient statistics rather than a character walk.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "BLIND-CHANNEL-DEV-001"
GRID = [(0, 1.0), *((1, value) for value in (.25, 1., 4., 16., 64., 256.))]
TOLERANCE = 1e-10


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verified_json(root, artifact):
    raw = (root / artifact["path"]).read_bytes()
    if sha(raw) != artifact["sha256"]:
        raise ValueError("Source artifact hash mismatch: " + artifact["path"])
    return json.loads(raw)


def sufficient_statistics(payload, alphabet):
    text, boundaries = payload["text"], payload["body_boundaries"]
    if (not boundaries or boundaries[-1] != len(text) or boundaries != sorted(set(boundaries))
            or boundaries[0] <= 0 or set(text) - set(alphabet)):
        raise ValueError("Invalid source alphabet/body boundaries")
    unigram, starts, pairs = Counter(), Counter(), Counter()
    previous_end = 0
    for end in boundaries:
        record = text[previous_end:end]
        unigram.update(record)
        starts[record[0]] += 1
        pairs.update(zip(record[:-1], record[1:], strict=True))
        previous_end = end
    return unigram, starts, pairs


def probabilities(statistics, alphabet, order, tau):
    unigram, _, pairs = statistics
    denominator = sum(unigram.values()) + len(alphabet) / 2
    base = {letter: (unigram[letter] + .5) / denominator for letter in alphabet}
    model = {"": base}
    if order:
        for left in alphabet:
            total = sum(pairs[left, right] for right in alphabet)
            model[left] = {right: (pairs[left, right] + tau * base[right]) / (total + tau)
                           for right in alphabet}
    return model


def validation_bits(model, statistics, order):
    unigram, starts, pairs = statistics
    if not order:
        return -math.fsum(count * math.log2(model[""][letter]) for letter, count in unigram.items())
    return -math.fsum([*(count * math.log2(model[""][letter]) for letter, count in starts.items()),
                       *(count * math.log2(model[left][right]) for (left, right), count in pairs.items())])


def main():
    task_manifest_path = ROOT / "data/manifests/blind_channel_dev001.json"
    task_manifest_raw = task_manifest_path.read_bytes()
    # Only its source artifact identity is used; no generated case is opened.
    source_artifact = json.loads(task_manifest_raw)["source"]
    selected = verified_json(ROOT, source_artifact)
    corpus_path = ROOT / "data/manifests/blind_channel_development_corpora.json"
    corpus_raw = corpus_path.read_bytes()
    if sha(corpus_raw) != selected["corpus_manifest_sha256"]:
        raise ValueError("Corpus manifest differs from frozen source selection")
    corpus = json.loads(corpus_raw)
    alphabet = tuple(corpus["alphabet"])
    statistics, inputs = {}, {}
    for name, role in (("caesar", "P1"), ("virgil", "P2")):
        spec = corpus["sources"][name]
        if spec["role"] != role:
            raise ValueError("Source role differs")
        artifact = {"path": spec["derived_path"], "sha256": spec["derived_sha256"]}
        payload = verified_json(ROOT, artifact)
        if sha(payload["text"].encode()) != spec["selected_text_sha256"] or len(payload["text"]) != 50_000:
            raise ValueError("Selected source text differs")
        statistics[name] = sufficient_statistics(payload, alphabet)
        inputs[name] = artifact | {"selected_text_sha256": spec["selected_text_sha256"],
                                  "characters": len(payload["text"]), "bodies": len(payload["body_boundaries"]),
                                  "pairs": sum(statistics[name][2].values())}
    candidates = []
    for order, tau in GRID:
        model = probabilities(statistics["caesar"], alphabet, order, tau)
        bits = validation_bits(model, statistics["virgil"], order) / 50_000
        candidates.append({"order": order, "tau": tau, "validation_bits_per_character": bits})
    if len(selected["candidates"]) != len(candidates):
        raise ValueError("Source candidate inventory differs")
    errors, score_deltas = [], []
    for expected, reported in zip(candidates, selected["candidates"], strict=True):
        if expected["order"] != reported["order"] or expected["tau"] != reported["tau"]:
            errors.append("candidate_configuration")
        score_deltas.append(abs(expected["validation_bits_per_character"] - reported["validation_bits_per_character"]))
    chosen = min(candidates, key=lambda item: item["validation_bits_per_character"])
    if any(chosen[field] != selected["selected"][field] for field in ("order", "tau")):
        errors.append("selected_configuration")
    score_deltas.append(abs(chosen["validation_bits_per_character"] - selected["selected"]["validation_bits_per_character"]))
    combined = tuple(statistics["caesar"][index] + statistics["virgil"][index] for index in range(3))
    expected_final = probabilities(combined, alphabet, chosen["order"], chosen["tau"])
    final = selected["source_model"]
    if final["alphabet"] != list(alphabet) or final["order"] != chosen["order"] or final["schema_version"] != 1:
        errors.append("final_model_metadata")
    if set(final["probabilities"]) != set(expected_final):
        raise ValueError("Final source context inventory differs")
    probability_deltas, normalization_errors = [], []
    for context, row in expected_final.items():
        if set(final["probabilities"][context]) != set(row):
            raise ValueError("Final source probability support differs")
        probability_deltas.extend(abs(value - final["probabilities"][context][letter]) for letter, value in row.items())
        normalization_errors.append(abs(math.fsum(final["probabilities"][context].values()) - 1))
    if max(score_deltas) > TOLERANCE:
        errors.append("source_score_tolerance")
    if max(probability_deltas) > TOLERANCE or max(normalization_errors) > 1e-12:
        errors.append("source_probability_tolerance")
    result = {"experiment": EXPERIMENT, "status": "pass" if not errors else "fail", "errors": errors,
              "scope": "source_only_caesar_virgil", "source_input_artifacts": inputs,
              "source_selection_sha256": source_artifact["sha256"], "corpus_manifest_sha256": sha(corpus_raw),
              "task_manifest_sha256": sha(task_manifest_raw), "auditor_source_sha256": sha(Path(__file__).read_bytes()),
              "candidates_checked": len(candidates), "candidate_scores": candidates, "selected": chosen,
              "final_context_rows_checked": len(expected_final), "final_probabilities_checked": len(probability_deltas),
              "final_training_characters": sum(combined[0].values()),
              "final_training_pairs": sum(combined[2].values()),
              "maximum_score_delta_bits_per_character": max(score_deltas),
              "maximum_probability_delta": max(probability_deltas),
              "maximum_normalization_error": max(normalization_errors), "absolute_score_tolerance": TOLERANCE,
              "cicero_or_ciphertext_or_answers_opened": False, "paid_cost_usd": 0}
    destination = ROOT / f"results/{EXPERIMENT}/source_audit.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("source_input_artifacts", "candidate_scores")}, indent=2))
    if errors:
        raise SystemExit("Independent source-only audit failed")


if __name__ == "__main__":
    main()
