"""Independent metadata-based audit of the visible query-relative rival."""

import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.teacher0015_clean_audit import SPLITS, _sha
from scripts.teacher0015_finite_audit import audit_control_permutations
from scripts.teacher0016_suite_audit import SURFACES, _episode
from scripts.teacher0017_suite_audit import _digest
from voynich.workspace.teacher14_tasks import SYMBOL_COUNT, SYMBOL_START


SOURCE_PATHS = (
    "docs/experiments/TEACH-0017-cross-context-design.md",
    "docs/experiments/TEACH-0017-query-relative-counterexample.md",
    "scripts/teacher0017_query_relative.py",
    "scripts/teacher0017_query_relative_audit.py",
    "tests/test_teacher0017_query_relative.py",
)


def _source_check(root: Path, report: dict) -> None:
    head = report.get("source_git_head")
    hashes = report.get("source_sha256")
    if (not isinstance(head, str) or len(head) != 40 or
            not isinstance(hashes, dict) or set(hashes) != set(SOURCE_PATHS)):
        raise ValueError("Query-relative source provenance incomplete")
    for name in SOURCE_PATHS:
        committed = subprocess.run(["git", "show", f"{head}:{name}"],
                                   cwd=root, check=True,
                                   capture_output=True).stdout
        if (hashlib.sha256(committed).hexdigest() != hashes[name] or
                _sha(root / name) != hashes[name]):
            raise ValueError(f"Query-relative source changed: {name}")


def _map(episode: dict) -> dict[int, int]:
    pairs = episode["serialized_rows"]
    mapping = dict(pairs)
    if len(mapping) != len(pairs):
        raise ValueError("Metadata rows not a function")
    return mapping


def _query(episode: dict) -> int:
    return int(episode["query"])


def _code(episode: dict) -> int:
    mapping = _map(episode)
    query = _query(episode)
    key = mapping[query]
    if key not in mapping:
        raise ValueError("Metadata first-hop key lacks continuation")
    return (key - query) % SYMBOL_COUNT


def _output(episode: dict, code: int) -> int:
    key = SYMBOL_START + ((code + _query(episode) - SYMBOL_START) % SYMBOL_COUNT)
    return _map(episode).get(key, -1)


def _random(group_id: str, d: int, marked: bool, order: int,
            a: int, b: int) -> int:
    message = (f"TEACH-0017-query-offset-null|{group_id}|{d}|"
               f"{int(marked)}|{order}|{a}|{b}")
    return int(hashlib.sha256(message.encode()).hexdigest()[:16], 16) % SYMBOL_COUNT


def _expected(group: dict, foreign_group: dict, d: int, marked: bool,
              order: int, a: int, b: int) -> dict:
    source = _episode(group, 1, a, d, marked, order)
    wrong = _episode(group, 0, a, d, marked, order)
    foreign = _episode(foreign_group, 1, a, d, marked, order)
    same = _episode(group, 1, b, d, marked, order)
    base = _episode(group, 0, b, 1 - d, marked, 1 - order)
    target = _episode(group, 1, b, 1 - d, marked, 1 - order)
    source_key = _map(source)[_query(source)]
    recipient_key = _map(base)[_query(base)]
    if (source_key != group["key1"] or recipient_key != group["key0"] or
            _query(source) != _query(base)):
        raise ValueError("TEACH-0017 query/key structure differs")
    source_lefts = sorted(_map(source))
    recipient_lefts = sorted(_map(base))
    donor_rank = source_lefts.index(source_key)
    recipient_rank = recipient_lefts.index(source_key)
    source_code = _code(source)
    wrong_code = _code(wrong)
    foreign_code = _code(foreign)
    return {
        "group_id": group["group_id"], "distractor": d,
        "marked": marked, "source_order": order,
        "source_g": a, "recipient_g": b,
        "source_render_id": source["render_id"],
        "base_render_id": base["render_id"],
        "target_render_id": target["render_id"],
        "distinct_render_id": foreign["render_id"],
        "source_query": _query(source), "recipient_query": _query(base),
        "source_code": source_code, "wrong_code": wrong_code,
        "distinct_code": foreign_code,
        "shifted_rank": donor_rank != recipient_rank,
        "base_answer": base["answer"],
        "target_answer": target["answer"],
        "source_answer": source["answer"],
        "identity_prediction": _output(base, _code(base)),
        "transfer_prediction": _output(base, source_code),
        "same_key_prediction": _output(target, _code(same)),
        "wrong_key_prediction": _output(base, wrong_code),
        "distinct_prediction": _output(base, foreign_code),
        "random_prediction": _output(base, _random(
            group["group_id"], d, marked, order, a, b)),
        "reverse_prediction": _output(target, wrong_code),
        "rank_prediction": _map(base)[recipient_lefts[donor_rank]],
        "final_donor_prediction": source["answer"],
    }


def _count(rows: list[list[dict]]) -> dict:
    flat = [row for surface in rows for row in surface]
    selected = [row for row in flat if row["shifted_rank"] and
                row["source_g"] != row["recipient_g"]]
    names = ("identity", "transfer", "same_key", "wrong_key", "distinct",
             "random", "reverse", "rank", "final_donor")
    return {"surfaces": len(rows), "attempts": len(flat),
            "same_query_attempts": sum(row["source_query"] == row[
                "recipient_query"] for row in flat),
            "shifted_rank_offdiag_attempts": len(selected),
            "shifted_rank_offdiag_hits": {
                name: sum(row[f"{name}_prediction"] == row[
                    "base_answer" if name in ("identity", "reverse")
                    else "target_answer"] for row in selected)
                for name in names}}


def audit(root: Path, suite_dir: Path, result_dir: Path) -> dict:
    report_path = result_dir / "report.json"
    report = json.loads(report_path.read_text())
    _source_check(root, report)
    if (report.get("experiment") != "TEACH-0017-query-relative-counterexample" or
            report.get("status") != "symbolic_visible_only" or
            report.get("neural_model_loaded") is not False or
            report.get("state_definition") != "first_hop_key_minus_query_mod_2048" or
            report.get("suite_audit_sha256") != _sha(
                root / "results/TEACH-0017/suite-audit.json")):
        raise ValueError("Query-relative report identity differs")
    suite_audit = json.loads((root / "results/TEACH-0017/suite-audit.json").read_text())
    if suite_audit.get("audit") != "pass":
        raise ValueError("TEACH-0017 suite audit missing")
    if (set(report.get("summaries", {})) != set(SPLITS) or
            set(report.get("archives", {})) != set(SPLITS)):
        raise ValueError("Query-relative split coverage differs")
    checked = {}
    for split in SPLITS:
        manifest_path = suite_dir / f"{split}.json"
        manifest = json.loads(manifest_path.read_text())
        if (_sha(manifest_path) != report["suite_file_sha256"][split] or
                _sha(manifest_path) != suite_audit["suite_file_sha256"][split] or
                _digest(manifest) != suite_audit[split]["manifest_sha256"]):
            raise ValueError("Query-relative suite hash differs")
        archive_path = result_dir / f"rows-{split}.json.gz"
        if (report["archives"][split]["sha256"] != _sha(archive_path) or
                report["archives"][split]["bytes"] != archive_path.stat().st_size):
            raise ValueError("Query-relative archive identity differs")
        archive = json.loads(gzip.decompress(archive_path.read_bytes()))
        surfaces = archive.get("rows")
        if (archive.get("split") != split or not isinstance(surfaces, list) or
                len(surfaces) != 1024):
            raise ValueError("Query-relative surface coverage differs")
        groups = manifest["groups"]
        deranged, _ = audit_control_permutations(groups)
        for i, group in enumerate(groups):
            foreign_group = groups[deranged[i]]
            for j, (d, marked, order) in enumerate(SURFACES):
                rows = surfaces[i * 8 + j]
                if not isinstance(rows, list) or len(rows) != 9:
                    raise ValueError("Query-relative pair grid differs")
                for k, row in enumerate(rows):
                    a, b = divmod(k, 3)
                    if row != _expected(group, foreign_group, d,
                                        marked, order, a, b):
                        raise ValueError("Query-relative output differs")
        counted = _count(surfaces)
        if (counted != report["summaries"][split] or
                counted["shifted_rank_offdiag_attempts"] !=
                suite_audit[split]["shifted_rank_offdiag_attempts"]):
            raise ValueError("Query-relative count differs")
        checked[split] = counted
    return {"audit": "pass", "scope": "visible_query_relative_counterexample",
            "report_sha256": _sha(report_path), "summaries": checked,
            "numerical_neural_replay": "not_applicable_symbolic_oracle"}


if __name__ == "__main__":
    root = Path.cwd()
    result = audit(root, root / "outputs/TEACH-0017",
                   root / "results/TEACH-0017-query-relative-rival")
    path = root / "results/TEACH-0017-query-relative-rival/audit.json"
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
