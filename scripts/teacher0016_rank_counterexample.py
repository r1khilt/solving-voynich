"""Visible-only relative-rank rival for TEACH-0016 behavioral claims.

This is a constructed symbolic program, not a trained neural mechanism. Its
intermediate state is only the ordinal rank of the first-hop key among visible
row left operands; it never stores that key's token ID in the carried state.
"""

import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.teacher0015_clean_audit import SPLITS, _sha
from scripts.teacher0015_finite_audit import audit_control_permutations
from scripts.teacher0016_clean_audit import read_manifests
from scripts.teacher0016_cross_score import score_split
from scripts.teacher0016_suite_audit import SURFACES, _episode, _index
from voynich.workspace.teacher14_tasks import strip_pair_rows


SOURCE_PATHS = (
    "docs/experiments/TEACH-0016-relative-rank-counterexample.md",
    "scripts/teacher0016_rank_counterexample.py",
    "scripts/teacher0016_rank_audit.py",
    "tests/test_teacher0016_rank_counterexample.py",
)


def _provenance(root: Path) -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_PATHS],
                           cwd=root, check=True, capture_output=True,
                           text=True).stdout.strip()
    if dirty:
        raise RuntimeError("Commit TEACH-0016 rank-rival sources before scoring")
    return {"source_git_head": head,
            "source_sha256": {name: _sha(root / name) for name in SOURCE_PATHS}}


def _mapping(episode: dict) -> dict[int, int]:
    rows = strip_pair_rows(tuple(episode["tokens"]))
    mapping = dict(rows)
    if len(mapping) != len(rows):
        raise ValueError("Rank-rival visible rows are not a function")
    return mapping


def _rank(episode: dict) -> tuple[int, tuple[int, ...]]:
    mapping = _mapping(episode)
    query = episode["tokens"][-2]
    key = mapping[query]
    lefts = tuple(sorted(mapping))
    if key not in mapping:
        raise ValueError("Rank-rival first-hop key lacks continuation")
    return lefts.index(key), lefts


def _decode(episode: dict, rank: int) -> int:
    mapping = _mapping(episode)
    lefts = tuple(sorted(mapping))
    if not 0 <= rank < len(lefts):
        raise ValueError("Rank-rival ordinal outside recipient table")
    return mapping[lefts[rank]]


def _random_rank(group_id: str, a: int, b: int, d: int,
                 marked: bool, order: int, size: int) -> int:
    message = f"TEACH-0016-rank-null|{group_id}|{a}|{b}|{d}|{int(marked)}|{order}"
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8],
                          "big") % size


def evaluate_split(manifest: dict) -> list[list[dict]]:
    groups = manifest["groups"]
    distinct, _ = audit_control_permutations(groups)
    surfaces = []
    for group_index, group in enumerate(groups):
        other = groups[distinct[group_index]]
        for d, marked, source_order in SURFACES:
            source = [_episode(group, 1, a, d, marked, source_order)
                      for a in range(3)]
            wrong = [_episode(group, 0, a, d, marked, source_order)
                     for a in range(3)]
            other_source = [_episode(other, 1, a, d, marked, source_order)
                            for a in range(3)]
            base = [_episode(group, 0, b, d, marked, 1 - source_order)
                    for b in range(3)]
            target = [_episode(group, 1, b, d, marked, 1 - source_order)
                      for b in range(3)]
            source_ranks = [_rank(item)[0] for item in source]
            wrong_ranks = [_rank(item)[0] for item in wrong]
            distinct_ranks = [_rank(item)[0] for item in other_source]
            rows = []
            for a in range(3):
                for b in range(3):
                    source_slot = _index(
                        source[a], group["key1"],
                        group["recipient_outputs"][a][1])
                    recipient_slot = _index(
                        base[b], group["key1"],
                        group["recipient_outputs"][b][1])
                    recipient_lefts = tuple(sorted(_mapping(base[b])))
                    if tuple(sorted(_mapping(source[a]))) != recipient_lefts:
                        raise ValueError("Rank-rival left-set changes across surfaces")
                    random_rank = _random_rank(
                        group["group_id"], a, b, d, marked, source_order,
                        len(recipient_lefts))
                    rows.append({
                        "group_id": group["group_id"],
                        "distractor": d, "marked": marked,
                        "source_order": source_order,
                        "source_g": a, "recipient_g": b,
                        "source_render_id": source[a]["render_id"],
                        "base_render_id": base[b]["render_id"],
                        "target_render_id": target[b]["render_id"],
                        "distinct_render_id": other_source[a]["render_id"],
                        "source_slot": source_slot,
                        "recipient_slot": recipient_slot,
                        "shifted_slot": source_slot != recipient_slot,
                        "source_rank_state": source_ranks[a],
                        "wrong_rank_state": wrong_ranks[a],
                        "distinct_rank_state": distinct_ranks[a],
                        "same_key_rank_state": source_ranks[b],
                        "random_rank_state": random_rank,
                        "base_answer": base[b]["answer"],
                        "target_answer": target[b]["answer"],
                        "source_answer": source[a]["answer"],
                        "base_prediction": base[b]["answer"],
                        "target_prediction": target[b]["answer"],
                        "source_prediction": source[a]["answer"],
                        "identity_prediction": _decode(base[b],
                                                       _rank(base[b])[0]),
                        "transfer_prediction": _decode(base[b],
                                                       source_ranks[a]),
                        "same_key_prediction": _decode(target[b],
                                                       source_ranks[b]),
                        "wrong_key_prediction": _decode(base[b],
                                                        wrong_ranks[a]),
                        "distinct_prediction": _decode(base[b],
                                                       distinct_ranks[a]),
                        "random_prediction": _decode(base[b], random_rank),
                        "reverse_prediction": _decode(target[b],
                                                      wrong_ranks[a]),
                        "final_donor_prediction": source[a]["answer"],
                    })
            surfaces.append(rows)
    return surfaces


def run(root: Path, suite_dir: Path, result_dir: Path) -> dict:
    if (result_dir / "report.json").exists():
        raise FileExistsError("No automatic rank-rival rerun")
    provenance = _provenance(root)
    manifests = read_manifests(suite_dir, root)
    result_dir.mkdir(parents=True, exist_ok=True)
    scores = {}
    archives = {}
    for split in SPLITS:
        surfaces = evaluate_split(manifests[split])
        scores[split] = score_split(
            manifests[split], surfaces, ["TEACH-0016-rank-rival", split])
        path = result_dir / f"rows-{split}.json.gz"
        path.write_bytes(gzip.compress(json.dumps(
            {"split": split, "rows": surfaces}, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode(),
            compresslevel=9, mtime=0))
        archives[split] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    result = {"experiment": "TEACH-0016-relative-rank-counterexample",
              "status": "symbolic_visible_only",
              "state_definition": "ordinal_of_first_hop_key_among_sorted_visible_left_operands",
              "neural_model_loaded": False,
              "manifest_sha256": {split: _sha(suite_dir / f"{split}.json")
                                  for split in SPLITS},
              "archives": archives, "scores": scores, **provenance}
    path = result_dir / "report.json"
    path.write_text(json.dumps(result, sort_keys=True, indent=2,
                               allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    root = Path.cwd()
    run(root, root / "outputs/TEACH-0016",
        root / "results/TEACH-0016-rank-rival")
