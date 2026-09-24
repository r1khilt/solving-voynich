"""No-generator audit of fresh cross-distractor rank-shift qualification."""

import hashlib
import json
from pathlib import Path
import subprocess

from scripts.teacher0014_suite_audit import audit_manifest as audit_teacher14
from scripts.teacher0015_clean_audit import (
    EXPECTED_EXPOSURES as TEACH14_EXPOSURES,
    EXPECTED_MANIFESTS as TEACH15_MANIFESTS,
)
from scripts.teacher0015_suite_audit import (
    _stage_signatures, audit_group,
    audit_manifest as audit_teacher15,
)
from scripts.teacher0016_suite_audit import (
    _episode, audit_manifest as audit_teacher16,
)
from scripts.teacher0016_clean_audit import (
    EXPECTED_MANIFESTS as TEACH16_MANIFESTS,
)


SEEDS = {"discovery": 87111, "confirmation": 87121}
SURFACES = tuple((d, marked, order) for d in (0, 1)
                 for marked in (True, False) for order in (0, 1))
SOURCE_PATHS = (
    "docs/experiments/TEACH-0017-cross-context-design.md",
    "src/voynich/workspace/teacher17_tasks.py",
    "scripts/teacher0017_prepare.py",
    "scripts/teacher0017_suite_audit.py",
    "tests/test_teacher0017_suite.py",
)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(episode: dict) -> tuple[int, ...]:
    return tuple(sorted(left for left, _ in episode["serialized_rows"]))


def _rank(episode: dict, key: int) -> int:
    lefts = _identity(episode)
    if len(lefts) != len(set(lefts)) or key not in lefts:
        raise ValueError("TEACH-0017 intermediate key/row set invalid")
    return lefts.index(key)


def _attempt_sets(manifest: dict) -> dict:
    if (not isinstance(manifest, dict) or set(manifest) != {
            "experiment", "namespace", "generator", "split", "seed",
            "group_count", "groups"} or
            manifest["experiment"] != "TEACH-0017" or
            manifest["namespace"] != "TEACH-0017-cross-distractor-v1" or
            manifest["generator"] != "TEACH-0015-key-transfer-v1"):
        raise ValueError("TEACH-0017 suite identity differs")
    split = manifest["split"]
    if (split not in SEEDS or manifest["seed"] != SEEDS[split] or
            manifest["group_count"] != 128 or
            len(manifest["groups"]) != 128):
        raise ValueError("TEACH-0017 split/seed/count differs")
    sets = {name: set() for name in (
        "groups", "graphs", "logical", "renders", "stages")}
    shifted = 0
    total = 0
    shifted_offdiag = 0
    rank_null_hits = 0
    shifted_rank_null_hits = 0
    for group in manifest["groups"]:
        checked = audit_group(group, split)
        own = {
            "groups": {checked["group_id"]},
            "graphs": checked["graph_ids"],
            "logical": checked["logical_ids"],
            "renders": checked["render_ids"],
            "stages": checked["stage_families"],
        }
        if any(sets[name] & values for name, values in own.items()):
            raise ValueError("TEACH-0017 within-split duplicate/family")
        for name, values in own.items():
            sets[name].update(values)
        for d, marked, order in SURFACES:
            source = [_episode(group, 1, a, d, marked, order)
                      for a in range(3)]
            recipient = [_episode(group, 0, b, 1 - d, marked, 1 - order)
                         for b in range(3)]
            source_ranks = {_rank(item, group["key1"]) for item in source}
            recipient_ranks = {_rank(item, group["key1"])
                               for item in recipient}
            if len(source_ranks) != 1 or len(recipient_ranks) != 1:
                raise ValueError("TEACH-0017 within-G rank drift")
            source_rank = next(iter(source_ranks))
            recipient_rank = next(iter(recipient_ranks))
            is_shifted = source_rank != recipient_rank
            shifted += int(is_shifted)
            total += 1
            for a in range(3):
                for b in range(3):
                    if a == b:
                        continue
                    mapping = {left: right for left, right in
                               recipient[b]["serialized_rows"]}
                    if len(mapping) != len(recipient[b]["serialized_rows"]):
                        raise ValueError("TEACH-0017 recipient rows not functional")
                    rank_output = mapping[sorted(mapping)[source_rank]]
                    hit = rank_output == group["recipient_outputs"][b][1]
                    rank_null_hits += int(hit)
                    if is_shifted:
                        shifted_offdiag += 1
                        shifted_rank_null_hits += int(hit)
    if (len(sets["renders"]) != 128 * 66 or total != 1024 or
            shifted / total < .60):
        raise ValueError("TEACH-0017 visible coverage/rank-shift validity failed")
    return {"sets": sets, "summary": {
        "audit": "pass", "split": split, "seed": SEEDS[split],
        "groups": len(sets["groups"]), "episodes": len(sets["renders"]),
        "cross_distractor_pairs": total,
        "shifted_rank_pairs": shifted,
        "shifted_rank_fraction": shifted / total,
        "offdiag_attempts": total * 6,
        "shifted_rank_offdiag_attempts": shifted_offdiag,
        "relative_rank_null_offdiag_hits": rank_null_hits,
        "relative_rank_null_shifted_hits": shifted_rank_null_hits,
        "manifest_sha256": _digest(manifest),
        **{f"{name}_sha256": _digest(sorted(values))
           for name, values in sets.items()},
    }}


def _prior_sets(teach15: tuple[dict, ...], teach16: tuple[dict, ...],
                teach14: tuple[dict, ...]) -> list[dict[str, set[str]]]:
    result = []
    for manifest in teach15:
        checked = audit_teacher15(manifest)
        result.append({
            "groups": {group["group_id"] for group in manifest["groups"]},
            "graphs": set(checked["graph_ids"]),
            "logical": set(checked["logical_ids"]),
            "renders": {cell["episode"]["render_id"]
                        for group in manifest["groups"]
                        for cell in group["cells"]},
            "stages": set(checked["stage_families"]),
        })
    for manifest in teach16:
        checked = audit_teacher16(manifest)
        result.append({
            "groups": checked["_groups"], "graphs": checked["_graphs"],
            "logical": checked["_logical"],
            "renders": checked["_renders"], "stages": checked["_stages"],
        })
    for manifest in teach14:
        audit_teacher14(manifest)
        episodes = [episode for panel in manifest["panels"].values()
                    for episode in panel]
        result.append({
            "groups": set(),
            "graphs": {episode["graph_id"] for episode in episodes},
            "logical": {episode["logical_id"] for episode in episodes},
            "renders": {episode["render_id"] for episode in episodes},
            "stages": set().union(*(
                _stage_signatures(episode) for episode in episodes)),
        })
    return result


def audit_splits(discovery: dict, confirmation: dict,
                 teach15: tuple[dict, ...], teach16: tuple[dict, ...],
                 teach14: tuple[dict, ...]) -> dict:
    prior_hashes = {
        "TEACH-0015": [_digest(item) for item in teach15],
        "TEACH-0016": [_digest(item) for item in teach16],
        "TEACH-0014": [_digest(item) for item in teach14],
    }
    if (prior_hashes["TEACH-0015"] != [
            TEACH15_MANIFESTS[split] for split in SEEDS] or
            prior_hashes["TEACH-0016"] != [
                TEACH16_MANIFESTS[split] for split in SEEDS] or
            tuple(prior_hashes["TEACH-0014"]) != TEACH14_EXPOSURES):
        raise ValueError("TEACH-0017 prior exposure manifest hash differs")
    current = [_attempt_sets(discovery), _attempt_sets(confirmation)]
    old = _prior_sets(teach15, teach16, teach14)
    for index, item in enumerate(current):
        peers = [current[1 - index]["sets"], *old]
        if any(item["sets"][name] & peer[name] for peer in peers
               for name in item["sets"]):
            raise ValueError("TEACH-0017 exposed split/family overlap")
    return {"audit": "pass", "scope": "fresh_cross_distractor_visible_suite",
            "discovery": current[0]["summary"],
            "confirmation": current[1]["summary"],
            "prior_manifest_sha256": prior_hashes}


def main() -> None:
    root = Path.cwd()
    suite_dir = root / "outputs/TEACH-0017"
    output = root / "results/TEACH-0017/suite-audit.json"
    if output.exists():
        raise FileExistsError("No automatic TEACH-0017 audit overwrite")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_PATHS],
                           cwd=root, check=True, capture_output=True,
                           text=True).stdout.strip()
    if dirty:
        raise RuntimeError("Commit TEACH-0017 suite sources before audit")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          check=True, capture_output=True,
                          text=True).stdout.strip()
    current = {split: json.loads((suite_dir / f"{split}.json").read_text())
               for split in SEEDS}
    prior15 = tuple(json.loads((root / "outputs/TEACH-0015" /
                                f"{split}.json").read_text()) for split in SEEDS)
    prior16 = tuple(json.loads((root / "outputs/TEACH-0016" /
                                f"{split}.json").read_text()) for split in SEEDS)
    prior14 = tuple(json.loads(path.read_text()) for path in (
        root / "outputs/TEACH-0016/teach14-74111.json",
        root / "outputs/TEACH-0016/teach14-74117.json",
        root / "results/TEACH-0014-v3/suite.json"))
    result = audit_splits(current["discovery"], current["confirmation"],
                          prior15, prior16, prior14)
    result.update({"source_git_head": head,
                   "source_sha256": {name: _sha(root / name)
                                     for name in SOURCE_PATHS},
                   "suite_file_sha256": {split: _sha(suite_dir / f"{split}.json")
                                         for split in SEEDS}})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
