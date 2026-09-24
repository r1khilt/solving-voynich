"""Fresh native-read replication gates and prior-evidence invariants."""

import json
from pathlib import Path

from scripts.teacher0025_fresh_hard_read_audit import _decision, SEEDS, CONDITIONS
from scripts.teacher0024_soft_read_audit import _score


ROOT = Path(__file__).resolve().parents[1]


def test_each_cell_must_gain_thirteen_items():
    scores = {str(seed): {} for seed in SEEDS}
    for seed in SEEDS:
        for rep in ("0", "1"):
            scores[str(seed)][rep] = {"conditions": {
                "clean": {"correct": 90},
                "identity": {"correct": 90},
                "hard_first": {"correct": 103},
                "gold_last": {"correct": 125},
                "wrong_first": {"correct": 4},
            }}
    assert _decision(scores)["fresh_replicated"]
    scores[str(SEEDS[1])]["1"]["conditions"]["hard_first"]["correct"] = 102
    assert not _decision(scores)["fresh_replicated"]


def test_prior_checkpoint_and_assay_audit_exist():
    prior = ROOT / "results/TEACH-0024-soft-read"
    status = json.loads((prior / "status.json").read_text())
    report = json.loads((prior / "audit.json").read_text())
    replay = json.loads((prior / "replay-audit.json").read_text())
    assert status["status"] == "complete"
    assert report["audit"] == replay["audit"] == "pass"
    assert set(status["checkpoint_sha256"]) == {"0", "1"}


def test_paired_scorer_counts_damage_and_rescue():
    rows = []
    for clean, hard in ((18, 17), (19, 20)):
        rows.append({"answer": 17 if clean == 18 else 19,
                     "target_rows": [0, 1], "native_top_rows": [0, 1],
                     "attention": [[1, 0], [0, 1]],
                     "predictions": {name: hard if name == "hard_first"
                                     else clean for name in CONDITIONS}})
    score = _score(rows)
    assert score["conditions"]["hard_first"]["rescued_clean_errors"] == 1
    assert score["conditions"]["hard_first"]["damaged_clean_correct"] == 1
