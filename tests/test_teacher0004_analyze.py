"""Independent TEACH-0004 auditor checks without a trained model or GPU."""

import gzip
import json

import pytest

from voynich.workspace import teacher4_tasks

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import teacher0004_analyze as audit4  # noqa: E402


def test_frozen_suite_is_reconstructed_and_oracle_checked():
    independent = audit4.expected_suite()
    native = teacher4_tasks.evaluation_suite(64311, 256)
    audit4.check_suite_structure(independent)
    assert sum(len(rows) for rows in independent.values()) == 3328
    assert all(independent[name] == [(list(item.tokens), item.answer) for item in items]
               for name, items in native.items())


def test_memory_and_dense_archives_have_distinct_attention_contracts(tmp_path):
    suite = audit4.expected_suite()
    archived = {}
    for name, examples in suite.items():
        rows = []
        for index, (tokens, answer) in enumerate(examples):
            _, _, f_family, g_family, _, _, f_read, g_read = audit4.parse_tokens(tokens)
            rows.append({"index": index, "tokens": tokens,
                         "task": audit4.TASK_MARKERS[tokens[11]], "answer": answer,
                         "prediction": answer,
                         "f_family": [list(x) for x in f_family],
                         "g_family": [list(x) for x in g_family],
                         "candidate_member": (answer in ((tokens[3], tokens[5])
                                                       if tokens[11] == 8
                                                       else (tokens[8], tokens[10]))),
                         "f_read_answer": f_read, "g_read_answer": g_read,
                         "f_read_prediction": f_read if f_read is not None else 0,
                         "g_read_prediction": g_read if g_read is not None else 0})
        archived[name] = rows
    memory = tmp_path/"predictions-rep0-two_read.json.gz"
    memory.write_bytes(gzip.compress(json.dumps(archived).encode()))
    scores = audit4.check_archive(memory, suite)
    assert scores["factorial"]["exact_groups"] == 128
    assert scores["first_hop_pairs_holdout"]["exact_groups"] == 128
    dense = tmp_path/"predictions-rep0-dense_row.json.gz"
    dense_rows = json.loads(json.dumps(archived))
    for rows in dense_rows.values():
        for row in rows:
            row["f_read_prediction"] = row["g_read_prediction"] = None
    dense.write_bytes(gzip.compress(json.dumps(dense_rows).encode()))
    assert audit4.check_archive(dense, suite)["factorial"]["exact_groups"] == 128
    dense_rows["direct_holdout"][0]["answer"] = 33  # corrupted oracle label
    dense.write_bytes(gzip.compress(json.dumps(dense_rows).encode()))
    with pytest.raises(audit4.AuditError, match="changed evaluation item"):
        audit4.check_archive(dense, suite)


def test_both_seed_decision_requires_gain_over_each_control():
    def cell(value):
        return {name: {"accuracy": value, "group_accuracy": value}
                for name in audit4.CELL_SPECS}
    scores = {rep: {"two_read": cell(.95), "one_read": cell(.50),
                    "dense_row": cell(.50)} for rep in audit4.REPLICATES}
    assert audit4.decision(scores)["two_read_qualified"] is True
    scores["1"]["dense_row"]["factorial"]["group_accuracy"] = .90
    assert audit4.decision(scores)["two_read_qualified"] is False
