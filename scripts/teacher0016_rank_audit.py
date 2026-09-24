"""No-generator, no-neural audit of the relative-rank counterexample."""

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


SOURCE_PATHS = (
    "docs/experiments/TEACH-0016-relative-rank-counterexample.md",
    "scripts/teacher0016_rank_counterexample.py",
    "scripts/teacher0016_rank_audit.py",
    "tests/test_teacher0016_rank_counterexample.py",
)


def _source_check(root: Path, report: dict) -> None:
    hashes = report.get("source_sha256")
    head = report.get("source_git_head")
    if not isinstance(hashes, dict) or set(hashes) != set(SOURCE_PATHS) or (
            not isinstance(head, str) or len(head) != 40):
        raise ValueError("Rank-rival source provenance incomplete")
    for name in SOURCE_PATHS:
        committed = subprocess.run(["git", "show", f"{head}:{name}"],
                                   cwd=root, check=True,
                                   capture_output=True).stdout
        if (hashlib.sha256(committed).hexdigest() != hashes[name] or
                _sha(root / name) != hashes[name]):
            raise ValueError(f"Rank-rival source changed: {name}")


def _edge_map(episode: dict) -> dict[int, int]:
    rows = episode["serialized_rows"]
    mapping = {left: right for left, right in rows}
    if len(mapping) != len(rows):
        raise ValueError("Rank-rival metadata edges are not a function")
    return mapping


def _rank(episode: dict) -> int:
    mapping = _edge_map(episode)
    key = mapping[episode["tokens"][-2]]
    return sorted(mapping).index(key)


def _next(episode: dict, rank: int) -> int:
    mapping = _edge_map(episode)
    return mapping[sorted(mapping)[rank]]


def _random(group_id: str, a: int, b: int, d: int,
            marked: bool, order: int, size: int) -> int:
    value = f"TEACH-0016-rank-null|{group_id}|{a}|{b}|{d}|{int(marked)}|{order}"
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16) % size


def audit(root: Path, suite_dir: Path, result_dir: Path) -> dict:
    report_path = result_dir / "report.json"
    report = json.loads(report_path.read_text())
    _source_check(root, report)
    if (report.get("experiment") != "TEACH-0016-relative-rank-counterexample" or
            report.get("status") != "symbolic_visible_only" or
            report.get("neural_model_loaded") is not False or
            report.get("state_definition") !=
            "ordinal_of_first_hop_key_among_sorted_visible_left_operands"):
        raise ValueError("Rank-rival report identity differs")
    manifests = read_manifests(suite_dir, root)
    if (report.get("manifest_sha256") != {
            split: _sha(suite_dir / f"{split}.json") for split in SPLITS} or
            set(report.get("scores", {})) != set(SPLITS) or
            set(report.get("archives", {})) != set(SPLITS)):
        raise ValueError("Rank-rival suite/report coverage differs")
    checked = {}
    for split in SPLITS:
        path = result_dir / f"rows-{split}.json.gz"
        if (report["archives"][split].get("sha256") != _sha(path) or
                report["archives"][split].get("bytes") != path.stat().st_size):
            raise ValueError("Rank-rival archive hash/size differs")
        archive = json.loads(gzip.decompress(path.read_bytes()))
        surfaces = archive.get("rows")
        if (archive.get("split") != split or not isinstance(surfaces, list) or
                len(surfaces) != 1024):
            raise ValueError("Rank-rival surface coverage differs")
        groups = manifests[split]["groups"]
        distinct, _ = audit_control_permutations(groups)
        for group_index, group in enumerate(groups):
            other = groups[distinct[group_index]]
            for surface_index, (d, marked, source_order) in enumerate(SURFACES):
                rows = surfaces[group_index * 8 + surface_index]
                if not isinstance(rows, list) or len(rows) != 9:
                    raise ValueError("Rank-rival nine-pair coverage differs")
                for index, row in enumerate(rows):
                    a, b = divmod(index, 3)
                    source = _episode(group, 1, a, d, marked, source_order)
                    wrong = _episode(group, 0, a, d, marked, source_order)
                    distinct_source = _episode(
                        other, 1, a, d, marked, source_order)
                    same = _episode(group, 1, b, d, marked, source_order)
                    base = _episode(group, 0, b, d, marked, 1 - source_order)
                    target = _episode(group, 1, b, d, marked,
                                      1 - source_order)
                    rank = _rank(source)
                    wrong_rank = _rank(wrong)
                    distinct_rank = _rank(distinct_source)
                    same_rank = _rank(same)
                    source_slot = _index(source, group["key1"],
                                         group["recipient_outputs"][a][1])
                    recipient_slot = _index(base, group["key1"],
                                            group["recipient_outputs"][b][1])
                    if set(_edge_map(source)) != set(_edge_map(base)):
                        raise ValueError("Rank-rival left-set invariance fails")
                    random_rank = _random(
                        group["group_id"], a, b, d, marked, source_order,
                        len(_edge_map(base)))
                    expected = {
                        "group_id": group["group_id"],
                        "distractor": d, "marked": marked,
                        "source_order": source_order,
                        "source_g": a, "recipient_g": b,
                        "source_render_id": source["render_id"],
                        "base_render_id": base["render_id"],
                        "target_render_id": target["render_id"],
                        "distinct_render_id": distinct_source["render_id"],
                        "source_slot": source_slot,
                        "recipient_slot": recipient_slot,
                        "shifted_slot": source_slot != recipient_slot,
                        "source_rank_state": rank,
                        "wrong_rank_state": wrong_rank,
                        "distinct_rank_state": distinct_rank,
                        "same_key_rank_state": same_rank,
                        "random_rank_state": random_rank,
                        "base_answer": base["answer"],
                        "target_answer": target["answer"],
                        "source_answer": source["answer"],
                        "base_prediction": base["answer"],
                        "target_prediction": target["answer"],
                        "source_prediction": source["answer"],
                        "identity_prediction": _next(base, _rank(base)),
                        "transfer_prediction": _next(base, rank),
                        "same_key_prediction": _next(target, same_rank),
                        "wrong_key_prediction": _next(base, wrong_rank),
                        "distinct_prediction": _next(base, distinct_rank),
                        "random_prediction": _next(base, random_rank),
                        "reverse_prediction": _next(target, wrong_rank),
                        "final_donor_prediction": source["answer"],
                    }
                    if row != expected:
                        raise ValueError("Rank-rival archive row differs")
        score = score_split(manifests[split], surfaces,
                            ["TEACH-0016-rank-rival", split])
        if score != report["scores"][split]:
            raise ValueError("Rank-rival score differs")
        checked[split] = {
            "shifted_offdiag_attempts": score["shifted_offdiag_attempts"],
            "transfer_accuracy": score["scores"]["transfer"]["accuracy"],
            "distinct_accuracy": score["scores"]["distinct"]["accuracy"],
            "random_accuracy": score["scores"]["random"]["accuracy"],
            "candidate_behavioral_conjunction": score[
                "candidate_order_robust_state_pending_replay"],
        }
    return {"audit": "pass", "scope": "visible_relative_rank_counterexample",
            "report_sha256": _sha(report_path), "splits": checked,
            "numerical_neural_replay": "not_applicable_symbolic_oracle"}


if __name__ == "__main__":
    root = Path.cwd()
    result = audit(root, root / "outputs/TEACH-0016",
                   root / "results/TEACH-0016-rank-rival")
    path = root / "results/TEACH-0016-rank-rival/audit.json"
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
