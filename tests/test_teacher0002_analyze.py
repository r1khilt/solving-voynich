"""Synthetic CPU-only checks for the independent TEACH-0002 audit."""

import gzip
import json
from pathlib import Path
import subprocess
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import teacher0002_analyze as audit  # noqa: E402


def perfect_rows(suite: dict) -> dict:
    result = {}
    for name, examples in suite.items():
        rows = []
        for index, (tokens, answer) in enumerate(examples):
            task, _, f_family, g_family, _, _ = audit.parse_tokens(tokens)
            candidates = (tokens[3], tokens[5]) if task == "first_hop" else (tokens[8], tokens[10])
            rows.append({"index": index, "tokens": tokens, "answer": answer,
                         "prediction": answer, "task": task,
                         "f_family": [list(x) for x in f_family],
                         "g_family": [list(x) for x in g_family],
                         "candidate_member": answer in candidates})
        result[name] = rows
    return result


def test_suite_is_reconstructed_from_frozen_seed() -> None:
    from voynich.workspace.teacher2_tasks import evaluation_suite

    expected = audit.expected_suite()
    audit.check_suite_structure(expected)
    actual = evaluation_suite(62311, 256)
    assert {name: [(list(ep.tokens), ep.answer) for ep in episodes]
            for name, episodes in actual.items()} == expected


def test_independent_oracle_and_family_partition() -> None:
    tokens, answer = audit.episode((9, 10), (21, 22), (22, 21), (33, 34),
                                   "composed", 9)
    task, computed, f_family, g_family, f_part, g_part = audit.parse_tokens(tokens)
    assert (task, answer, computed) == ("composed", 34, 34)
    assert f_family == ((9, 10), (21, 22))
    assert g_family == ((21, 22), (33, 34))
    assert (f_part, g_part) == (audit.partition("F", (9, 10), (21, 22)),
                               audit.partition("G", (22, 21), (33, 34)))
    tokens[5] = 21
    with pytest.raises(audit.AuditError):
        audit.parse_tokens(tokens)


def test_archive_recomputes_every_score_and_rejects_changed_label(tmp_path: Path) -> None:
    suite = audit.expected_suite()
    rows = perfect_rows(suite)
    path = tmp_path / "predictions.json.gz"
    path.write_bytes(gzip.compress(json.dumps(rows).encode(), mtime=0))
    scores, _ = audit.check_archive(path, suite)
    assert scores["first_hop_pairs_holdout"]["exact_groups"] == 128
    assert scores["direct_pairs_holdout"]["exact_groups"] == 128
    assert scores["factorial"]["exact_groups"] == 128
    assert scores["factorial"]["f_swap_prediction_changes"] == 256
    assert scores["factorial"]["g_remap_prediction_changes"] == 256
    old_answer = rows["composed_holdout_holdout"][0]["answer"]
    rows["composed_holdout_holdout"][0]["answer"] = 33 if old_answer != 33 else 34
    path.write_bytes(gzip.compress(json.dumps(rows).encode(), mtime=0))
    with pytest.raises(audit.AuditError, match="changed evaluation example"):
        audit.check_archive(path, suite)


def test_query_reversal_exact_requires_both_answers() -> None:
    suite = audit.expected_suite()
    rows = perfect_rows(suite)["first_hop_pairs_holdout"]
    first_key = rows[0]["prediction"]
    rows[1]["prediction"] = first_key
    rows[1]["candidate_member"] = True
    score = audit.score_cell(rows, "first_hop_pairs_holdout")
    assert score["correct"] == 255
    assert score["exact_groups"] == 127
    assert score["prediction_reversals"] == 127
    behavior = audit.paired_shortcut_diagnostics(rows, "first_hop_pairs_holdout")
    assert behavior["prediction_changes_on_query"] == 127
    assert behavior["pair_exact"] == 127
    assert behavior["candidate_positions"]["noncandidate"] == 0
    assert behavior["unique_families"] > 0


def test_verdict_requires_both_seeds_and_matched_baseline_gain() -> None:
    def cells(compose: float, factorial: float) -> dict:
        return {name: {"accuracy": compose, "group_accuracy": factorial}
                for name in audit.CELL_SPECS}

    scores = {rep: {"curriculum": cells(.95, .90), "baseline": cells(.50, .20),
                    "null": cells(.20, .00)} for rep in audit.REPLICATES}
    assert audit.decision(scores)["verdict"] == "qualified"
    scores["1"]["baseline"]["composed_holdout_holdout"]["accuracy"] = .80
    result = audit.decision(scores)
    assert result["verdict"] == "not_qualified"
    assert not result["clauses_by_replicate"]["1"]["composed_gain_vs_baseline_at_least_0.20"]
    assert result["clauses_by_replicate"]["0"]["composed_gain_vs_baseline_at_least_0.20"]


def test_source_and_config_hashes_bind_frozen_commit() -> None:
    root = Path(__file__).resolve().parents[1]
    hashes = {}
    for path in audit.SOURCE_PATHS:
        source = subprocess.run(["git", "show", f"{audit.REVISION}:{path}"],
                                cwd=root, capture_output=True, check=True).stdout
        hashes[path] = audit.digest(source)
    encoded = json.dumps(audit.CONFIG, sort_keys=True, separators=(",", ":")).encode()
    report = {"source_git_head": audit.REVISION, "source_worktree_status": [],
              "source_sha256": hashes, "config": audit.CONFIG,
              "config_sha256": audit.digest(encoded)}
    audit.check_source(root, report)
    report["source_sha256"][audit.SOURCE_PATHS[0]] = "0" * 64
    with pytest.raises(audit.AuditError, match="Source hash mismatch"):
        audit.check_source(root, report)
