"""Post-result, fit-only diagnosis; no new transfer scoring or gate changes."""
from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

from scripts.audit_naibbe001 import IndependentLM, sha

ROOT = Path(__file__).resolve().parent.parent


def main():
    audit_path = ROOT / "results/NAIBBE-002/audit.json"
    audit = json.loads(audit_path.read_text())
    if not audit["audit_pass"]:
        raise AssertionError("Requires completed audit")
    data_path = ROOT / "data/processed/naibbe002/fit_input.json"
    data = json.loads(data_path.read_text())
    answers = json.loads((ROOT / "data/processed/naibbe002/answer_key.json").read_text())
    # Only fit candidates/answers are used; no transfer input or score is calculated.
    ids, alphabet = data["class_ids"], data["alphabet"]
    correct = answers["class_to_letter"]
    frozen = json.loads((ROOT / "results/NAIBBE-002/latin_joint_freeze.json").read_text())
    learned = {c: alphabet[code] for c, code in zip(ids, frozen["key"], strict=True)}
    if any(correct[c] != learned[c] for c in ids if correct[c] not in "yz"):
        raise AssertionError("The diagnostic assumes only the observed y/z errors")
    lm_source = ROOT / data["lms"]["latin"]["path"]
    lm = IndependentLM(lm_source.read_text().strip(), "".join(alphabet))
    candidates = data["split"]["candidates"]
    opportunity, unique = Counter(), Counter()
    for paths in candidates:
        for alternatives in zip(*paths, strict=True):
            options = set(alternatives)
            opportunity.update(options)
            if len(options) == 1:
                unique.update(options)
    pairs, groups = [], []
    for group in data["groups"]:
        y = next(c for c in group if correct[c] == "y")
        z = next(c for c in group if correct[c] == "z")
        pairs.append((y, z))
        groups.append({"y_possible_positions": opportunity[y], "z_possible_positions": opportunity[z],
                       "y_unique_positions": unique[y], "z_unique_positions": unique[z],
                       "learned_swapped": learned[y] == "z",
                       "both_absent": opportunity[y] == opportunity[z] == 0})
    rows = []
    for switches in itertools.product((0, 1), repeat=len(pairs)):
        mapping = correct.copy()
        for swap, (y, z) in zip(switches, pairs, strict=True):
            if swap:
                mapping[y], mapping[z] = "z", "y"
        strings = [["".join(mapping[c] for c in path) for path in paths] for paths in candidates]
        rows.append({"swaps": list(switches), "fit_score": lm.best_lattice_score(strings)})
    best = max(row["fit_score"] for row in rows)
    gold = rows[0]["fit_score"]
    learned_switches = [int(g["learned_swapped"]) for g in groups]
    selected = next(row["fit_score"] for row in rows if row["swaps"] == learned_switches)
    if abs(selected - frozen["fit_score"]) > 1e-6:
        raise AssertionError("Independent selected score mismatch")
    # Explain the actual score preference from the saved, audited fit parses.
    predictions = json.loads((ROOT / "outputs/NAIBBE-002/evaluation_predictions.json").read_text())
    gold_text = "".join(predictions["oracle_fit"]["chunks"])
    selected_text = "".join(predictions["latin_joint_fit"]["chunks"])
    if len(gold_text) != len(selected_text):
        raise AssertionError("Alignment assumption changed")
    contributions = []
    for i in range(len(gold_text)):
        gold_gram = gold_text[max(0, i - 3):i + 1]
        selected_gram = selected_text[max(0, i - 3):i + 1]
        if gold_gram != selected_gram:
            gain = lm.log_next(selected_gram) - lm.log_next(gold_gram)
            contributions.append({"position": i, "gold_key_gram": gold_gram,
                                  "selected_key_gram": selected_gram, "selected_minus_gold_log_score": gain,
                                  "gold_context_count": lm.contexts[len(gold_gram) - 1][gold_gram[:-1]],
                                  "selected_context_count": lm.contexts[len(selected_gram) - 1][selected_gram[:-1]]})
    if abs(sum(row["selected_minus_gold_log_score"] for row in contributions) - (selected - gold)) > 1e-6:
        raise AssertionError("Score decomposition mismatch")
    report = {"experiment": "NAIBBE-002", "stage": "posthoc_fit_only_diagnostic_after_registered_FAIL",
              "audit_sha256": sha(audit_path), "fit_input_sha256": sha(data_path),
              "lm_sha256": sha(lm_source), "groups": groups, "enumerated_keys": len(rows),
              "best_score": best, "gold_score": gold, "selected_score": selected,
              "selected_is_best_in_restricted_family": abs(best - selected) < 1e-8,
              "best_tie_count": sum(abs(row["fit_score"] - best) < 1e-8 for row in rows),
              "keys_scoring_above_gold": sum(row["fit_score"] > gold + 1e-8 for row in rows),
              "learned_swaps": learned_switches, "orientations": rows,
              "score_contributions": contributions,
              "claim_limit": "Conditional enumeration of six y/z swaps after observing errors; not global optimization, calibrated posterior, new holdout result, or changed gate. Published FAIL retained."}
    output = ROOT / "results/NAIBBE-002/posthoc_rare_key_diagnostic.json"
    if output.exists():
        raise FileExistsError("Preserve previous diagnosis")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("groups", "enumerated_keys", "gold_score", "selected_score",
                                                  "selected_is_best_in_restricted_family", "best_tie_count",
                                                  "keys_scoring_above_gold", "score_contributions")}, indent=2))


if __name__ == "__main__":
    main()
