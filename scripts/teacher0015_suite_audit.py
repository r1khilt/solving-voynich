"""Independent visible-stream and counterfactual audit for TEACH-0015.

Imports the already independent TEACH-0014 visible episode auditor, but no
generator, model or trainer. It checks the full 66-cell factorial per group.
"""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.teacher0014_suite_audit import (
    EDGE, SYMBOL_START, _audit_episode,
    audit_manifest as audit_teacher14_manifest,
)


NAMESPACE = "TEACH-0015-key-transfer-v1"
SEEDS = {"discovery": 85111, "confirmation": 85121}


def _digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _cell_key(cell: dict) -> tuple:
    return (cell["f"], cell["g"], cell["distractor"],
            cell["marked"], cell["order"], cell["task"])


def _roles(episode: dict) -> tuple[str, ...]:
    role_by_row = {}
    for index, path in enumerate(episode["signal_paths"]):
        for stage in (0, 1):
            role_by_row[tuple(path[stage:stage + 2])] = f"s{index}.{stage}"
    for index, path in enumerate(episode["distractor_paths"]):
        for stage in (0, 1):
            role_by_row[tuple(path[stage:stage + 2])] = f"d{index}.{stage}"
    if len(role_by_row) != len(episode["serialized_rows"]):
        raise ValueError("Duplicate logical row in counterfactual cell")
    return tuple(role_by_row[tuple(row)] for row in episode["serialized_rows"])


def _skeleton(episode: dict) -> tuple[int, ...]:
    return tuple(value if value < SYMBOL_START else -1
                 for value in episode["tokens"])


def audit_group(group: dict, split: str) -> dict:
    if not isinstance(group, dict) or set(group) != {
            "group_id", "split", "key0", "key1", "recipient_outputs",
            "donor_answer", "cells"} or group["split"] != split:
        raise ValueError("Malformed TEACH-0015 group")
    cells = group["cells"]
    if not isinstance(cells, list) or len(cells) != 66:
        raise ValueError("Incomplete 66-cell key-transfer group")
    by_key = {}
    render_ids = set()
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != {
                "f", "g", "distractor", "marked", "order", "task", "episode"}:
            raise ValueError("Malformed TEACH-0015 cell")
        key = _cell_key(cell)
        if key in by_key:
            raise ValueError("Duplicate TEACH-0015 cell")
        f, g, d, marked, order, task = key
        if (type(f) is not int or f not in (0, 1) or type(g) is not int or
                g not in (0, 1, 2) or type(d) is not int or d not in (0, 1) or
                type(marked) is not bool or type(order) is not int or
                order not in (0, 1) or task not in (
                    "composed", "first_hop", "direct", "copy")):
            raise ValueError("TEACH-0015 cell coordinates invalid")
        if task != "composed" and (d, marked, order) != (0, True, 0):
            raise ValueError("Auxiliary task outside fixed surface")
        episode = cell["episode"]
        _audit_episode(episode)
        if episode["stage_partitions"] != ["confirm", "confirm"]:
            raise ValueError("TEACH-0015 cell overlaps training stage family")
        has_marker = EDGE in episode["tokens"][1:-3]
        if has_marker != marked or (
                episode["marker_dropout"] != (0.0 if marked else 1.0)):
            raise ValueError("Marker-rich/free condition drift")
        if episode["render_id"] in render_ids:
            raise ValueError("Duplicate rendered episode in group")
        render_ids.add(episode["render_id"])
        by_key[key] = episode
    expected = {
        (f, g, d, marked, order, "composed")
        for f in (0, 1) for g in (0, 1, 2) for d in (0, 1)
        for marked in (True, False) for order in (0, 1)
    } | {
        (f, g, 0, True, 0, task)
        for f in (0, 1) for g in (0, 1, 2)
        for task in ("first_hop", "direct", "copy")
    }
    if set(by_key) != expected:
        raise ValueError("Counterfactual factorial coverage incomplete")
    base = by_key[(0, 0, 0, True, 0, "composed")]
    swap = by_key[(1, 0, 0, True, 0, "composed")]
    d1_base = by_key[(0, 0, 1, True, 0, "composed")]
    f0, f1 = base["signal_paths"], swap["signal_paths"]
    g_tables = [tuple(by_key[(0, g, 0, True, 0, "composed")][
        "signal_paths"][index][2] for index in range(4))
        for g in (0, 1, 2)]
    specification = {
        "f0": [path[:2] for path in f0],
        "f1": [path[:2] for path in f1],
        "g": g_tables,
        "d": [base["distractor_paths"], d1_base["distractor_paths"]],
    }
    group_id = _digest(specification)
    if group["group_id"] != group_id:
        raise ValueError("Counterfactual group ID mismatch")
    expected_split = ("discovery" if int(_digest([NAMESPACE, group_id])[:16], 16)
                      % 2 == 0 else "confirmation")
    if split != expected_split:
        raise ValueError("Counterfactual group split mismatch")
    query = f0[0][0]
    if (f1[0][0] != f0[1][0] or f1[1][0] != query or
            f1[0][1] != f0[0][1] or f1[1][1] != f0[1][1] or
            f1[2:] != f0[2:]):
        raise ValueError("F intervention changed more than queried binding")
    key0, key1 = f0[0][1], f1[1][1]
    outputs = [[table[0], table[1]] for table in g_tables]
    if (key0 == key1 or group["key0"] != key0 or
            group["key1"] != key1 or
            group["recipient_outputs"] != outputs or
            len({value for pair in outputs for value in pair}) != 6 or
            group["donor_answer"] != outputs[0][1]):
        raise ValueError("Recipient key/answer distinctions invalid")
    for d in (0, 1):
        for marked in (True, False):
            for order in (0, 1):
                reference = by_key[(0, 0, d, marked, order, "composed")]
                roles = _roles(reference)
                skeleton = _skeleton(reference)
                for f in (0, 1):
                    for g in (0, 1, 2):
                        episode = by_key[(f, g, d, marked, order, "composed")]
                        expected_paths = [
                            [f0[index][0] if f == 0 else f1[index][0],
                             f0[index][1], g_tables[g][index]]
                            for index in range(4)]
                        if (episode["signal_paths"] != expected_paths or
                                episode["distractor_paths"] != (
                                    base["distractor_paths"] if d == 0 else
                                    d1_base["distractor_paths"]) or
                                episode["query"] != query or
                                episode["answer"] != outputs[g][f] or
                                _roles(episode) != roles or
                                _skeleton(episode) != skeleton):
                            raise ValueError("Counterfactual content/surface drift")
            first_order = _roles(by_key[(0, 0, d, marked, 0, "composed")])
            second_order = _roles(by_key[(0, 0, d, marked, 1, "composed")])
            if first_order == second_order:
                raise ValueError("Two row-order conditions are identical")
    for f in (0, 1):
        for g in (0, 1, 2):
            for task in ("first_hop", "direct", "copy"):
                episode = by_key[(f, g, 0, True, 0, task)]
                expected_query = key0 if f == 0 else key1
                if (episode["signal_paths"] !=
                        by_key[(f, g, 0, True, 0, "composed")]["signal_paths"] or
                        episode["distractor_paths"] != base["distractor_paths"] or
                        episode["query"] != (expected_query if task == "direct"
                                             else query) or
                        episode["answer"] != (
                            expected_query if task == "first_hop" else
                            query if task == "copy" else outputs[g][f])):
                    raise ValueError("Auxiliary task oracle drift")
    return {"group_id": group_id, "render_ids": render_ids,
            "graph_ids": {cell["episode"]["graph_id"] for cell in cells},
            "logical_ids": {cell["episode"]["logical_id"] for cell in cells}}


def audit_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict) or set(manifest) != {
            "experiment", "namespace", "split", "seed", "group_count",
            "groups"} or manifest["experiment"] != "TEACH-0015" or (
            manifest["namespace"] != NAMESPACE):
        raise ValueError("TEACH-0015 manifest identity mismatch")
    split = manifest["split"]
    if split not in SEEDS or manifest["seed"] != SEEDS[split] or (
            type(manifest["group_count"]) is not int or
            manifest["group_count"] <= 0 or
            not isinstance(manifest["groups"], list) or
            len(manifest["groups"]) != manifest["group_count"]):
        raise ValueError("TEACH-0015 manifest seed/count mismatch")
    seen_groups = set()
    seen_graphs = set()
    seen_logical = set()
    seen_render = set()
    for group in manifest["groups"]:
        checked = audit_group(group, split)
        if (checked["group_id"] in seen_groups or
                checked["graph_ids"] & seen_graphs or
                checked["logical_ids"] & seen_logical or
                checked["render_ids"] & seen_render):
            raise ValueError("Repeated TEACH-0015 group/graph/logical/render")
        seen_groups.add(checked["group_id"])
        seen_graphs.update(checked["graph_ids"])
        seen_logical.update(checked["logical_ids"])
        seen_render.update(checked["render_ids"])
    return {"audit": "pass", "scope": "visible_three_recipient_counterfactuals",
            "split": split, "groups": len(seen_groups),
            "episodes": len(seen_render),
            "graph_ids": sorted(seen_graphs),
            "logical_ids": sorted(seen_logical),
            "manifest_sha256": _digest(manifest)}


def audit_splits(discovery: dict, confirmation: dict,
                 exposed_teacher14: list[dict] | None = None) -> dict:
    a = audit_manifest(discovery)
    b = audit_manifest(confirmation)
    if a["split"] != "discovery" or b["split"] != "confirmation":
        raise ValueError("TEACH-0015 split order mismatch")
    if set(a["graph_ids"]) & set(b["graph_ids"]) or (
            set(a["logical_ids"]) & set(b["logical_ids"])):
        raise ValueError("TEACH-0015 discovery/confirmation identity overlap")
    exposure_hashes = []
    for manifest in exposed_teacher14 or []:
        audited = audit_teacher14_manifest(manifest)
        exposure_hashes.append(audited["manifest_sha256"])
        graphs = {episode["graph_id"]
                  for panel in manifest["panels"].values() for episode in panel}
        logical = {episode["logical_id"]
                   for panel in manifest["panels"].values() for episode in panel}
        if ((set(a["graph_ids"]) | set(b["graph_ids"])) & graphs or
                (set(a["logical_ids"]) | set(b["logical_ids"])) & logical):
            raise ValueError("TEACH-0015 overlaps exposed TEACH-0014 graph")
    return {"audit": "pass", "discovery_sha256": a["manifest_sha256"],
            "confirmation_sha256": b["manifest_sha256"],
            "discovery_groups": a["groups"],
            "confirmation_groups": b["groups"],
            "teacher14_exposure_sha256": exposure_hashes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_manifest(json.loads(args.manifest.read_text())),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
