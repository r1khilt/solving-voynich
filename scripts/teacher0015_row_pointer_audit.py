"""Independent visible-data audit of the row-pointer counterexample."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.teacher0015_clean_audit import (
    EXPECTED_MANIFESTS, SPLITS, _sha, read_manifests,
)
from scripts.teacher0015_finite_audit import audit_control_permutations


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _episode(group: dict, f: int, g: int, d: int, m: bool, o: int) -> dict:
    found = [cell["episode"] for cell in group["cells"] if (
        cell["f"], cell["g"], cell["distractor"], cell["marked"],
        cell["order"], cell["task"]) == (f, g, d, m, o, "composed")]
    if len(found) != 1:
        raise ValueError("Pointer audit cell missing/duplicate")
    return found[0]


def _position(episode: dict, key: int, value: int) -> int:
    edge = [key, value]
    found = [index for index, row in enumerate(episode["serialized_rows"])
             if row == edge]
    if len(found) != 1:
        raise ValueError("Pointer audit edge missing/duplicate")
    return found[0]


def _value(episode: dict, position: int) -> int:
    if not 0 <= position < len(episode["serialized_rows"]):
        raise ValueError("Pointer audit physical position out of range")
    return episode["serialized_rows"][position][1]


def expected_split(groups: list[dict], split: str) -> dict:
    wrong, deranged = audit_control_permutations(groups)
    names = ("transfer", "triples", "changed_g_non_injection",
             "reverse_eligible", "reverse_full", "marked", "marker_free",
             "wrong_key", "deranged", "random", "same_key", "final_donor",
             "base_clean", "target_clean")
    counters = {name: [0, 0] for name in names}
    signature = []
    same_slots = 0
    for group_index, group in enumerate(groups):
        key0, key1 = group["key0"], group["key1"]
        for d in (0, 1):
            for marked in (True, False):
                for order in (0, 1):
                    surface_index = d * 4 + (0 if marked else 2) + order
                    donor = _episode(group, 1, 0, d, marked, order)
                    reverse_donor = _episode(group, 0, 0, d, marked, order)
                    nuisance = _episode(group, 1, 0, d, not marked, order)
                    wrong_group = groups[wrong[group_index]]
                    deranged_group = groups[deranged[group_index]]
                    wrong_episode = _episode(
                        wrong_group, 1, 0, d, marked, order)
                    deranged_episode = _episode(
                        deranged_group, 1, 0, d, marked, order)
                    pointer = _position(donor, key1,
                                        group["recipient_outputs"][0][1])
                    reverse_pointer = _position(
                        reverse_donor, key0,
                        group["recipient_outputs"][0][0])
                    nuisance_pointer = _position(
                        nuisance, key1, group["recipient_outputs"][0][1])
                    wrong_pointer = _position(
                        wrong_episode, wrong_group["key1"],
                        wrong_group["recipient_outputs"][0][1])
                    deranged_pointer = _position(
                        deranged_episode, deranged_group["key1"],
                        deranged_group["recipient_outputs"][0][1])
                    slots = [pointer]
                    triple = True
                    for g in (0, 1, 2):
                        base = _episode(group, 0, g, d, marked, order)
                        target = _episode(group, 1, g, d, marked, order)
                        target_answer = target["answer"]
                        slots.append(_position(
                            base, key1, group["recipient_outputs"][g][1]))
                        null_token = ["TEACH-0015-pointer-null", split,
                                      group["group_id"], surface_index, g]
                        null_pointer = (int(_digest(null_token)[:16], 16) %
                                        len(base["serialized_rows"]))
                        guesses = {
                            "transfer": _value(base, pointer),
                            "reverse_eligible": _value(target, reverse_pointer),
                            "reverse_full": _value(target, reverse_pointer),
                            "wrong_key": _value(base, wrong_pointer),
                            "deranged": _value(base, deranged_pointer),
                            "random": _value(base, null_pointer),
                            "same_key": _value(base, nuisance_pointer),
                            "final_donor": donor["answer"],
                            "base_clean": base["answer"],
                            "target_clean": target_answer,
                        }
                        triple &= guesses["transfer"] == target_answer

                        def add(name: str, hit: bool) -> None:
                            counters[name][0] += int(hit)
                            counters[name][1] += 1

                        for name in ("transfer", "wrong_key", "deranged",
                                     "random", "same_key", "final_donor",
                                     "target_clean"):
                            add(name, guesses[name] == target_answer)
                        add("base_clean", True)
                        add("reverse_eligible", guesses["reverse_eligible"] ==
                            base["answer"])
                        add("reverse_full", guesses["reverse_full"] ==
                            base["answer"])
                        add("marked" if marked else "marker_free",
                            guesses["transfer"] == target_answer)
                        if g:
                            add("changed_g_non_injection",
                                guesses["transfer"] != donor["answer"])
                        signature.append([
                            group["group_id"], g, guesses["transfer"],
                            guesses["wrong_key"], guesses["deranged"],
                            guesses["random"]])
                    counters["triples"][0] += int(triple)
                    counters["triples"][1] += 1
                    same_slots += int(len(set(slots)) == 1)
    scores = {name: {"correct": pair[0], "total": pair[1],
                     "accuracy": pair[0] / pair[1] if pair[1] else 0.0}
              for name, pair in counters.items()}
    transfer = scores["transfer"]["accuracy"]
    margins = {name: transfer - scores[name]["accuracy"]
               for name in ("wrong_key", "deranged", "random")}
    passes = (transfer >= .75 and scores["triples"]["accuracy"] >= .60 and
              scores["changed_g_non_injection"]["accuracy"] >= .90 and
              scores["reverse_eligible"]["accuracy"] >= .70 and
              scores["marked"]["accuracy"] >= .70 and
              scores["marker_free"]["accuracy"] >= .70 and
              all(margin >= .35 for margin in margins.values()))
    return {"same_slot_surfaces": same_slots, "total_surfaces": 1024,
            "prediction_sha256": _digest(signature),
            "scores": scores, "control_margins": margins,
            "passes_behavioral_thresholds_only": passes}


def audit(suite_dir: Path, report_path: Path) -> dict:
    manifests = read_manifests(suite_dir)
    report = json.loads(report_path.read_text())
    if (report.get("experiment") != "TEACH-0015-row-pointer-counterexample" or
            report.get("scope") != "constructed_visible_row_pointer_no_neural_model" or
            report.get("manifest_sha256") != EXPECTED_MANIFESTS or
            report.get("suite_file_sha256") != {
                split: _sha(suite_dir / f"{split}.json") for split in SPLITS} or
            report.get("source_sha256") != {
                "script": _sha(Path(
                    "scripts/teacher0015_row_pointer_counterexample.py")),
                "registration": _sha(Path(
                    "docs/experiments/TEACH-0015-row-pointer-counterexample.md"))}):
        raise ValueError("Pointer counterexample provenance differs")
    expected = {split: expected_split(manifests[split]["groups"], split)
                for split in SPLITS}
    if (report.get("splits") != expected or
            report.get("all_behavioral_thresholds_pass") != all(
                row["passes_behavioral_thresholds_only"]
                for row in expected.values())):
        raise ValueError("Pointer counterexample metrics differ")
    return {"audit": "pass", "scope": "independent_visible_pointer_reconstruction",
            "report_sha256": _sha(report_path),
            "auditor_sha256": _sha(Path(__file__)),
            "manifest_sha256": EXPECTED_MANIFESTS,
            "same_slot_surfaces": {
                split: row["same_slot_surfaces"] for split, row in expected.items()},
            "all_behavioral_thresholds_pass": report[
                "all_behavioral_thresholds_pass"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path,
                        default=Path("outputs/TEACH-0015"))
    parser.add_argument("--report", type=Path, default=Path(
        "results/TEACH-0015-row-pointer/report.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/TEACH-0015-row-pointer/audit.json"))
    args = parser.parse_args()
    result = audit(args.suite_dir, args.report)
    args.output.write_text(json.dumps(result, sort_keys=True,
                                      indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
