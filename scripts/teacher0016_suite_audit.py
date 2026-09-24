"""No-generator visible-data audit of fresh TEACH-0016 crossed-order groups."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.teacher0015_clean_audit import (
    EXPECTED_EXPOSURES, EXPECTED_MANIFESTS,
)
from scripts.teacher0014_suite_audit import audit_manifest as audit_teacher14
from scripts.teacher0015_suite_audit import (
    _stage_signatures, audit_group, audit_manifest as audit_teacher15,
)


SEEDS = {"discovery": 86111, "confirmation": 86121}
SURFACES = tuple((d, m, source_order)
                 for d in (0, 1) for m in (True, False)
                 for source_order in (0, 1))


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _episode(group: dict, f: int, g: int, d: int, marked: bool,
             order: int) -> dict:
    found = [cell["episode"] for cell in group["cells"] if (
        cell["f"], cell["g"], cell["distractor"], cell["marked"],
        cell["order"], cell["task"]) == (f, g, d, marked, order, "composed")]
    if len(found) != 1:
        raise ValueError("TEACH-0016 crossed-order cell missing/duplicate")
    return found[0]


def _index(episode: dict, key: int, value: int) -> int:
    positions = [index for index, row in enumerate(episode["serialized_rows"])
                 if row == [key, value]]
    if len(positions) != 1:
        raise ValueError("TEACH-0016 G edge missing/duplicate")
    return positions[0]


def audit_manifest(manifest: dict) -> dict:
    if (not isinstance(manifest, dict) or set(manifest) != {
            "experiment", "namespace", "generator", "split", "seed",
            "group_count", "groups"} or
            manifest["experiment"] != "TEACH-0016" or
            manifest["namespace"] != "TEACH-0016-cross-order-v1" or
            manifest["generator"] != "TEACH-0015-key-transfer-v1"):
        raise ValueError("TEACH-0016 suite identity differs")
    split = manifest["split"]
    if (split not in SEEDS or manifest["seed"] != SEEDS[split] or
            manifest["group_count"] != 128 or
            len(manifest["groups"]) != 128):
        raise ValueError("TEACH-0016 suite split/seed/count differs")
    group_ids = set()
    graphs = set()
    logical = set()
    renders = set()
    stages = set()
    shifted = 0
    all_pairs = 0
    null_hits = 0
    offdiag = 0
    for group in manifest["groups"]:
        checked = audit_group(group, split)
        if (checked["group_id"] in group_ids or
                checked["graph_ids"] & graphs or
                checked["logical_ids"] & logical or
                checked["render_ids"] & renders or
                checked["stage_families"] & stages):
            raise ValueError("TEACH-0016 duplicate group/graph/episode/family")
        group_ids.add(checked["group_id"])
        graphs.update(checked["graph_ids"])
        logical.update(checked["logical_ids"])
        renders.update(checked["render_ids"])
        stages.update(checked["stage_families"])
        for d, marked, source_order in SURFACES:
            recipient_order = 1 - source_order
            source_positions = [
                _index(_episode(group, 1, a, d, marked, source_order),
                       group["key1"], group["recipient_outputs"][a][1])
                for a in (0, 1, 2)]
            recipient_positions = [
                _index(_episode(group, 0, b, d, marked, recipient_order),
                       group["key1"], group["recipient_outputs"][b][1])
                for b in (0, 1, 2)]
            if (len(set(source_positions)) != 1 or
                    len(set(recipient_positions)) != 1):
                raise ValueError("TEACH-0016 within-order row-slot drift")
            all_pairs += 1
            shifted += int(source_positions[0] != recipient_positions[0])
            for a in (0, 1, 2):
                for b in (0, 1, 2):
                    if a == b:
                        continue
                    offdiag += 1
                    recipient = _episode(group, 0, b, d, marked,
                                         recipient_order)
                    predicted = recipient["serialized_rows"][
                        source_positions[a]][1]
                    null_hits += int(predicted == group[
                        "recipient_outputs"][b][1])
    if len(renders) != 128 * 66 or all_pairs != 1024 or offdiag != 6144:
        raise ValueError("TEACH-0016 visible episode/pair coverage differs")
    if shifted / all_pairs < .85:
        raise ValueError("TEACH-0016 frozen shifted-slot fraction below85%")
    return {"audit": "pass", "scope": "fresh_cross_order_visible_suite",
            "split": split, "seed": SEEDS[split],
            "groups": len(group_ids), "episodes": len(renders),
            "crossed_order_pairs": all_pairs,
            "shifted_slot_pairs": shifted,
            "offdiag_attempts": offdiag,
            "visible_row_pointer_offdiag_hits": null_hits,
            "visible_row_pointer_offdiag_total": offdiag,
            "manifest_sha256": _digest(manifest),
            "group_ids_sha256": _digest(sorted(group_ids)),
            "graph_ids_sha256": _digest(sorted(graphs)),
            "logical_ids_sha256": _digest(sorted(logical)),
            "render_ids_sha256": _digest(sorted(renders)),
            "stage_families_sha256": _digest(sorted(stages)),
            "_groups": group_ids, "_graphs": graphs,
            "_logical": logical, "_stages": stages, "_renders": renders}


def audit_splits(discovery: dict, confirmation: dict,
                 teach15: tuple[dict, dict],
                 teach14: tuple[dict, ...]) -> dict:
    a = audit_manifest(discovery)
    b = audit_manifest(confirmation)
    sources = [a, b]
    for left, right in ((a, b),):
        if (left["_groups"] & right["_groups"] or
                left["_graphs"] & right["_graphs"] or
                left["_logical"] & right["_logical"] or
                left["_stages"] & right["_stages"]):
            raise ValueError("TEACH-0016 split overlap")
    teach15_hashes = []
    for manifest in teach15:
        checked = audit_teacher15(manifest)
        teach15_hashes.append(checked["manifest_sha256"])
        prior_groups = {group["group_id"] for group in manifest["groups"]}
        prior_graphs = set(checked["graph_ids"])
        prior_logical = set(checked["logical_ids"])
        prior_stages = set(checked["stage_families"])
        prior_renders = {cell["episode"]["render_id"]
                         for group in manifest["groups"]
                         for cell in group["cells"]}
        for current in sources:
            if (current["_groups"] & prior_groups or
                    current["_graphs"] & prior_graphs or
                    current["_logical"] & prior_logical or
                    current["_stages"] & prior_stages or
                    current["_renders"] & prior_renders):
                raise ValueError("TEACH-0016 overlaps TEACH-0015 exposed suite")
    teach14_hashes = []
    for manifest in teach14:
        checked = audit_teacher14(manifest)
        teach14_hashes.append(checked["manifest_sha256"])
        graphs = {episode["graph_id"] for panel in manifest["panels"].values()
                  for episode in panel}
        logical = {episode["logical_id"] for panel in manifest["panels"].values()
                   for episode in panel}
        renders = {episode["render_id"] for panel in manifest["panels"].values()
                   for episode in panel}
        stages = set().union(*(_stage_signatures(episode)
                               for panel in manifest["panels"].values()
                               for episode in panel))
        for current in sources:
            if (current["_graphs"] & graphs or
                    current["_logical"] & logical or
                    current["_renders"] & renders or
                    current["_stages"] & stages):
                raise ValueError("TEACH-0016 overlaps TEACH-0014 exposed suite")
    cleaned = [{key: value for key, value in item.items()
                if not key.startswith("_")} for item in sources]
    if (teach15_hashes != [EXPECTED_MANIFESTS[split]
                          for split in ("discovery", "confirmation")] or
            tuple(teach14_hashes) != EXPECTED_EXPOSURES):
        raise ValueError("TEACH-0016 prior exposure manifest hashes differ")
    return {"audit": "pass", "scope": "fresh_cross_order_split_and_exposure",
            "discovery": cleaned[0], "confirmation": cleaned[1],
            "teacher15_exposure_sha256": teach15_hashes,
            "teacher14_exposure_sha256": teach14_hashes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path,
                        default=Path("outputs/TEACH-0016"))
    parser.add_argument("--teach15-dir", type=Path,
                        default=Path("outputs/TEACH-0015"))
    parser.add_argument("--teach14-manifests", nargs="*", type=Path,
                        default=[Path("outputs/TEACH-0016/teach14-74111.json"),
                                 Path("outputs/TEACH-0016/teach14-74117.json"),
                                 Path("results/TEACH-0014-v3/suite.json")])
    parser.add_argument("--output", type=Path, default=Path(
        "results/TEACH-0016/suite-audit.json"))
    args = parser.parse_args()
    own = {split: json.loads((args.suite_dir / f"{split}.json").read_text())
           for split in SEEDS}
    earlier = tuple(json.loads((args.teach15_dir / f"{split}.json").read_text())
                    for split in SEEDS)
    primary = tuple(json.loads(path.read_text())
                    for path in args.teach14_manifests)
    result = audit_splits(own["discovery"], own["confirmation"],
                          earlier, primary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({split: {key: value for key, value in result[split].items()
                              if key in ("manifest_sha256", "shifted_slot_pairs",
                                         "visible_row_pointer_offdiag_hits")}
                      for split in SEEDS}, sort_keys=True))


if __name__ == "__main__":
    main()
