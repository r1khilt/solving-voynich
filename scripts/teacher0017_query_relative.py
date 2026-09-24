"""Visible-token-only query-relative coding counterexample for TEACH-0017."""

import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.teacher0015_clean_audit import SPLITS, _sha
from scripts.teacher0015_finite_audit import audit_control_permutations
from scripts.teacher0016_suite_audit import SURFACES, _episode
from scripts.teacher0017_suite_audit import _digest
from voynich.workspace.teacher14_tasks import (
    SYMBOL_COUNT, SYMBOL_START, strip_pair_rows,
)


SOURCE_PATHS = (
    "docs/experiments/TEACH-0017-cross-context-design.md",
    "docs/experiments/TEACH-0017-query-relative-counterexample.md",
    "scripts/teacher0017_query_relative.py",
    "scripts/teacher0017_query_relative_audit.py",
    "tests/test_teacher0017_query_relative.py",
)


def _provenance(root: Path) -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_PATHS],
                           cwd=root, check=True, capture_output=True,
                           text=True).stdout.strip()
    if dirty:
        raise RuntimeError("Commit query-relative sources before scoring")
    return {"source_git_head": head,
            "source_sha256": {name: _sha(root / name) for name in SOURCE_PATHS}}


def _suite(root: Path, suite_dir: Path) -> dict:
    audit = json.loads((root / "results/TEACH-0017/suite-audit.json").read_text())
    if audit.get("audit") != "pass":
        raise ValueError("TEACH-0017 suite audit is missing")
    manifests = {}
    for split in SPLITS:
        path = suite_dir / f"{split}.json"
        item = json.loads(path.read_text())
        if (_sha(path) != audit["suite_file_sha256"][split] or
                _digest(item) != audit[split]["manifest_sha256"] or
                item.get("experiment") != "TEACH-0017" or
                item.get("group_count") != 128):
            raise ValueError("TEACH-0017 source-frozen suite differs")
        manifests[split] = item
    return manifests


def _edges(episode: dict) -> dict[int, int]:
    rows = strip_pair_rows(tuple(episode["tokens"]))
    edges = dict(rows)
    if len(edges) != len(rows):
        raise ValueError("Visible rows do not form a function")
    return edges


def _query(episode: dict) -> int:
    return int(episode["tokens"][-2])


def _encode(episode: dict) -> int:
    edges = _edges(episode)
    query = _query(episode)
    key = edges[query]
    if key not in edges:
        raise ValueError("First-hop key has no visible continuation")
    return (key - query) % SYMBOL_COUNT


def _decode(episode: dict, code: int) -> int:
    key = SYMBOL_START + ((code + _query(episode) - SYMBOL_START) % SYMBOL_COUNT)
    return _edges(episode).get(key, -1)


def _rank_prediction(source: dict, recipient: dict) -> tuple[int, bool]:
    source_edges = _edges(source)
    recipient_edges = _edges(recipient)
    rank = sorted(source_edges).index(source_edges[_query(source)])
    recipient_lefts = sorted(recipient_edges)
    return recipient_edges[recipient_lefts[rank]], rank != recipient_lefts.index(
        source_edges[_query(source)])


def _random(group_id: str, d: int, marked: bool, order: int,
            a: int, b: int) -> int:
    label = (f"TEACH-0017-query-offset-null|{group_id}|{d}|"
             f"{int(marked)}|{order}|{a}|{b}")
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],
                          "big") % SYMBOL_COUNT


def evaluate_split(manifest: dict) -> list[list[dict]]:
    groups = manifest["groups"]
    distinct, _ = audit_control_permutations(groups)
    surfaces = []
    for group_index, group in enumerate(groups):
        other = groups[distinct[group_index]]
        for d, marked, order in SURFACES:
            donors = [_episode(group, 1, a, d, marked, order)
                      for a in range(3)]
            wrong = [_episode(group, 0, a, d, marked, order)
                     for a in range(3)]
            foreign = [_episode(other, 1, a, d, marked, order)
                       for a in range(3)]
            base = [_episode(group, 0, b, 1 - d, marked, 1 - order)
                    for b in range(3)]
            target = [_episode(group, 1, b, 1 - d, marked, 1 - order)
                      for b in range(3)]
            donor_codes = [_encode(item) for item in donors]
            wrong_codes = [_encode(item) for item in wrong]
            foreign_codes = [_encode(item) for item in foreign]
            rows = []
            for a in range(3):
                for b in range(3):
                    rank_prediction, shifted = _rank_prediction(
                        donors[a], base[b])
                    rows.append({
                        "group_id": group["group_id"], "distractor": d,
                        "marked": marked, "source_order": order,
                        "source_g": a, "recipient_g": b,
                        "source_render_id": donors[a]["render_id"],
                        "base_render_id": base[b]["render_id"],
                        "target_render_id": target[b]["render_id"],
                        "distinct_render_id": foreign[a]["render_id"],
                        "source_query": _query(donors[a]),
                        "recipient_query": _query(base[b]),
                        "source_code": donor_codes[a],
                        "wrong_code": wrong_codes[a],
                        "distinct_code": foreign_codes[a],
                        "shifted_rank": shifted,
                        "base_answer": base[b]["answer"],
                        "target_answer": target[b]["answer"],
                        "source_answer": donors[a]["answer"],
                        "identity_prediction": _decode(base[b],
                                                       _encode(base[b])),
                        "transfer_prediction": _decode(base[b],
                                                       donor_codes[a]),
                        "same_key_prediction": _decode(target[b],
                                                       donor_codes[b]),
                        "wrong_key_prediction": _decode(base[b],
                                                        wrong_codes[a]),
                        "distinct_prediction": _decode(base[b],
                                                       foreign_codes[a]),
                        "random_prediction": _decode(base[b], _random(
                            group["group_id"], d, marked, order, a, b)),
                        "reverse_prediction": _decode(target[b],
                                                      wrong_codes[a]),
                        "rank_prediction": rank_prediction,
                        "final_donor_prediction": donors[a]["answer"],
                    })
            surfaces.append(rows)
    return surfaces


def _counts(surfaces: list[list[dict]]) -> dict:
    all_rows = [row for surface in surfaces for row in surface]
    selected = [row for row in all_rows if row["shifted_rank"] and
                row["source_g"] != row["recipient_g"]]
    names = ("identity", "transfer", "same_key", "wrong_key", "distinct",
             "random", "reverse", "rank", "final_donor")
    return {"surfaces": len(surfaces), "attempts": len(all_rows),
            "same_query_attempts": sum(row["source_query"] == row[
                "recipient_query"] for row in all_rows),
            "shifted_rank_offdiag_attempts": len(selected),
            "shifted_rank_offdiag_hits": {
                name: sum(row[f"{name}_prediction"] == row[
                    "base_answer" if name in ("identity", "reverse")
                    else "target_answer"]
                    for row in selected) for name in names}}


def run(root: Path, suite_dir: Path, result_dir: Path) -> dict:
    if (result_dir / "report.json").exists():
        raise FileExistsError("No automatic query-relative rival rerun")
    provenance = _provenance(root)
    manifests = _suite(root, suite_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    archives = {}
    for split in SPLITS:
        surfaces = evaluate_split(manifests[split])
        summaries[split] = _counts(surfaces)
        path = result_dir / f"rows-{split}.json.gz"
        path.write_bytes(gzip.compress(json.dumps(
            {"split": split, "rows": surfaces}, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode(),
            compresslevel=9, mtime=0))
        archives[split] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    report = {"experiment": "TEACH-0017-query-relative-counterexample",
              "status": "symbolic_visible_only", "neural_model_loaded": False,
              "state_definition": "first_hop_key_minus_query_mod_2048",
              "suite_file_sha256": {split: _sha(suite_dir / f"{split}.json")
                                    for split in SPLITS},
              "suite_audit_sha256": _sha(
                  root / "results/TEACH-0017/suite-audit.json"),
              "summaries": summaries, "archives": archives, **provenance}
    (result_dir / "report.json").write_text(json.dumps(
        report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    root = Path.cwd()
    run(root, root / "outputs/TEACH-0017",
        root / "results/TEACH-0017-query-relative-rival")
