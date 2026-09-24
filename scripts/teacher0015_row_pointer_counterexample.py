"""Constructive no-neural counterexample to unique key-symbol inference."""

import argparse
import json
from pathlib import Path

from scripts.teacher0015_clean_audit import (
    EXPECTED_MANIFESTS, SPLITS, _sha, read_manifests,
)
from scripts.teacher0015_finite_result_audit import SURFACES, score_rows
from voynich.workspace.teacher14_tasks import digest
from voynich.workspace.teacher15_pairs import control_permutations
from voynich.workspace.teacher15_tasks import generate_split


def _cell(group, f: int, g: int, surface: tuple[int, bool, int]):
    distractor, marked, order = surface
    return next(cell.episode for cell in group.cells if (
        cell.f, cell.g, cell.distractor, cell.marked,
        cell.order, cell.task) == (
            f, g, distractor, marked, order, "composed"))


def _index(episode, key: int, output: int) -> int:
    found = [index for index, row in enumerate(episode.serialized_rows)
             if row == (key, output)]
    if len(found) != 1:
        raise ValueError("Pointer oracle requires unique visible G edge")
    return found[0]


def _read(episode, index: int) -> int:
    if not 0 <= index < len(episode.serialized_rows):
        raise ValueError("Pointer oracle row index outside episode")
    return episode.serialized_rows[index][1]


def construct_split(groups: list, split: str) -> tuple[list[dict], dict]:
    if len(groups) != 128 or any(group.split != split for group in groups):
        raise ValueError("Pointer oracle requires frozen128-group split")
    wrong, deranged = control_permutations(groups)
    rows = []
    same_slot = 0
    for group_index, group in enumerate(groups):
        for surface_index, surface in enumerate(SURFACES):
            donor = _cell(group, 1, 0, surface)
            reverse_donor = _cell(group, 0, 0, surface)
            same_key = _cell(group, 1, 0,
                             (surface[0], not surface[1], surface[2]))
            wrong_source = _cell(groups[wrong[group_index]], 1, 0, surface)
            deranged_source = _cell(groups[deranged[group_index]], 1, 0, surface)
            pointer = _index(donor, group.key1, group.recipient_outputs[0][1])
            reverse_pointer = _index(
                reverse_donor, group.key0, group.recipient_outputs[0][0])
            nuisance_pointer = _index(
                same_key, group.key1, group.recipient_outputs[0][1])
            wrong_group = groups[wrong[group_index]]
            deranged_group = groups[deranged[group_index]]
            wrong_pointer = _index(
                wrong_source, wrong_group.key1,
                wrong_group.recipient_outputs[0][1])
            deranged_pointer = _index(
                deranged_source, deranged_group.key1,
                deranged_group.recipient_outputs[0][1])
            recipient_slots = []
            for g in (0, 1, 2):
                base = _cell(group, 0, g, surface)
                target = _cell(group, 1, g, surface)
                recipient_slots.append(_index(
                    base, group.key1, group.recipient_outputs[g][1]))
                random_pointer = int(digest([
                    "TEACH-0015-pointer-null", split, group.group_id,
                    surface_index, g])[:16], 16) % len(base.serialized_rows)
                rows.append({
                    "group_id": group.group_id,
                    "g": g, "marked": surface[1],
                    "base_answer": base.answer,
                    "target_answer": target.answer,
                    "fixed_donor_answer": donor.answer,
                    "base_prediction": base.answer,
                    "target_prediction": target.answer,
                    "donor_prediction": donor.answer,
                    "transfer_prediction": _read(base, pointer),
                    "same_key_prediction": _read(base, nuisance_pointer),
                    "reverse_prediction": _read(target, reverse_pointer),
                    "final_donor_prediction": donor.answer,
                    "wrong_key_prediction": _read(base, wrong_pointer),
                    "deranged_prediction": _read(base, deranged_pointer),
                    "random_prediction": _read(base, random_pointer),
                })
            same_slot += int(len(set(recipient_slots + [pointer])) == 1)
    if len(rows) != 3072:
        raise ValueError("Pointer oracle recipient grid incomplete")
    return rows, {"same_slot_surfaces": same_slot,
                  "total_surfaces": 1024,
                  "prediction_sha256": digest([
                      [row["group_id"], row["g"], row["transfer_prediction"],
                       row["wrong_key_prediction"],
                       row["deranged_prediction"], row["random_prediction"]]
                      for row in rows])}


def run(suite_dir: Path, result_path: Path) -> dict:
    manifests = read_manifests(suite_dir)
    result = {"experiment": "TEACH-0015-row-pointer-counterexample",
              "scope": "constructed_visible_row_pointer_no_neural_model",
              "manifest_sha256": EXPECTED_MANIFESTS,
              "suite_file_sha256": {split: _sha(suite_dir / f"{split}.json")
                                    for split in SPLITS},
              "splits": {}}
    for split in SPLITS:
        groups = generate_split(split)
        if [group.group_id for group in groups] != [
                group["group_id"] for group in manifests[split]["groups"]]:
            raise ValueError("Pointer oracle generated/frozen group order differs")
        rows, structure = construct_split(groups, split)
        scored = score_rows(rows, ["pointer", split])
        result["splits"][split] = {
            **structure,
            "scores": {name: {"correct": value["correct"],
                              "total": value["total"],
                              "accuracy": value["accuracy"]}
                       for name, value in scored["scores"].items()},
            "control_margins": scored["control_margins"],
            "passes_behavioral_thresholds_only": scored[
                "candidate_portable_state_pending_replay"],
        }
    result["all_behavioral_thresholds_pass"] = all(
        row["passes_behavioral_thresholds_only"]
        for row in result["splits"].values())
    result["source_sha256"] = {
        "script": _sha(Path(__file__)),
        "registration": _sha(Path(
            "docs/experiments/TEACH-0015-row-pointer-counterexample.md"))}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, sort_keys=True, indent=2,
                                      allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path,
                        default=Path("outputs/TEACH-0015"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/TEACH-0015-row-pointer/report.json"))
    args = parser.parse_args()
    report = run(args.suite_dir, args.output)
    print(json.dumps({"all_behavioral_thresholds_pass": report[
        "all_behavioral_thresholds_pass"],
        "split_scores": {split: row["scores"]["transfer"]
                         for split, row in report["splits"].items()}},
        sort_keys=True))


if __name__ == "__main__":
    main()
