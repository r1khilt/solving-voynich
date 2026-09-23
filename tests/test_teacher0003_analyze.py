"""Synthetic CPU-only checks for the independent TEACH-0003 auditor."""

import gzip
import json
from pathlib import Path
import subprocess
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import teacher0003_analyze as audit  # noqa: E402


def perfect_rows(suite):
    archived = {}
    for name, examples in suite.items():
        rows = []
        for index, (tokens, answer) in enumerate(examples):
            task, _, f_family, g_family, _, _, f_row, g_row = audit.parse_tokens(tokens)
            candidates = ((tokens[3], tokens[5]) if task == "first_hop"
                          else (tokens[8], tokens[10]))
            rows.append({"index": index, "tokens": tokens, "task": task,
                         "answer": answer, "prediction": answer,
                         "f_family": [list(x) for x in f_family],
                         "g_family": [list(x) for x in g_family],
                         "candidate_member": answer in candidates,
                         "f_row_answer": f_row, "g_row_answer": g_row,
                         "f_row_prediction": f_row if f_row is not None else 0,
                         "g_row_prediction": g_row if g_row is not None else 0})
        archived[name] = rows
    return archived


def test_fresh_suite_matches_generator_and_independent_oracle():
    from voynich.workspace.teacher3_tasks import evaluation_suite

    expected = audit.expected_suite()
    audit.check_suite_structure(expected)
    generated = evaluation_suite(63311, 256)
    assert {name: [(list(ep.tokens), ep.answer) for ep in episodes]
            for name, episodes in generated.items()} == expected
    assert sum(len(items) for items in expected.values()) == 3328
    tokens, answer = audit.episode((9, 10), (21, 22), (22, 21), (33, 34),
                                   "composed", 9)
    assert audit.parse_tokens(tokens)[1] == answer == 34
    tokens[5] = 21
    with pytest.raises(audit.AuditError):
        audit.parse_tokens(tokens)


def test_archive_scores_every_item_and_rejects_changed_label(tmp_path):
    suite = audit.expected_suite()
    archived = perfect_rows(suite)
    path = tmp_path / "predictions.json.gz"
    path.write_bytes(gzip.compress(json.dumps(archived).encode(), mtime=0))
    scores = audit.check_archive(path, suite)
    assert scores["factorial"]["exact_groups"] == 128
    assert scores["first_hop_pairs_holdout"]["f_row_exact_pairs"] == 128
    assert scores["direct_pairs_holdout"]["g_row_exact_pairs"] == 128
    archived["factorial"][0]["answer"] = 33 if archived["factorial"][0]["answer"] != 33 else 34
    path.write_bytes(gzip.compress(json.dumps(archived).encode(), mtime=0))
    with pytest.raises(audit.AuditError, match="changed evaluation item"):
        audit.check_archive(path, suite)


def test_loss_hash_progress_means_and_tamper(tmp_path):
    losses = [float(step % 100) / 100 for step in range(5000)]
    path = tmp_path / "losses.json.gz"
    path.write_bytes(gzip.compress(json.dumps(losses).encode(), mtime=0))
    record = {"losses_sha256": audit.file_digest(path),
              "history": [{"step": step,
                           "last_100_loss": sum(losses[step-100:step])/100}
                          for step in range(250, 5001, 250)]}
    assert audit.check_losses(path, record) == record["losses_sha256"]
    record["history"][0]["last_100_loss"] += .01
    with pytest.raises(audit.AuditError, match="last-100 loss"):
        audit.check_losses(path, record)


def test_source_hashes_bind_frozen_commit_not_worktree():
    root = Path(__file__).resolve().parents[1]
    hashes = {}
    for path in audit.SOURCE_PATHS:
        blob = subprocess.run(["git", "show", f"{audit.REVISION}:{path}"], cwd=root,
                              capture_output=True, check=True).stdout
        hashes[path] = audit.digest(blob)
    config = json.dumps(audit.CONFIG, sort_keys=True, separators=(",", ":")).encode()
    report = {"source_git_head": audit.REVISION, "source_worktree_status": [],
              "source_sha256": hashes, "config": audit.CONFIG,
              "config_sha256": audit.digest(config)}
    audit.check_source(root, report)
    report["source_sha256"][audit.SOURCE_PATHS[0]] = "0" * 64
    with pytest.raises(audit.AuditError, match="Source hash mismatch"):
        audit.check_source(root, report)


def test_independent_decision_requires_both_seeds_and_alignment_controls():
    def cell(value):
        return {name: {"accuracy": value, "group_accuracy": value,
                       "f_row_accuracy": value, "g_row_accuracy": value,
                       "f_row_pair_accuracy": value, "g_row_pair_accuracy": value}
                for name in audit.CELL_SPECS}

    scores = {rep: {"dense_small": cell(.4), "dense_medium": cell(.5),
                    "dense_large": cell(.95), "align_true": cell(1.0),
                    "align_random": cell(.5)} for rep in audit.REPLICATES}
    decision = audit.decision(scores)
    assert decision["dense_scale_qualified"] is True
    assert decision["alignment_qualified"] is False  # .05 gain over dense_large
    scores["1"]["dense_large"]["factorial"]["group_accuracy"] = .5
    decision = audit.decision(scores)
    assert decision["verdict"] == "not_qualified"
    assert decision["dense_clauses_by_seed"]["1"]["factorial_at_least_0.65"] is False


def test_active_campaign_is_reported_incomplete_without_scoring(tmp_path):
    result_dir = tmp_path / "results/TEACH-0003"
    result_dir.mkdir(parents=True)
    (result_dir / "status.json").write_text(json.dumps({
        "experiment": "TEACH-0003", "status": "running",
        "completed_arms": [[0, "dense_small"]],
    }))
    result = audit.audit(tmp_path)
    assert result["audit"] == "incomplete"
    assert result["completed_arm_count_reported"] == 1
    assert "decision" not in result
