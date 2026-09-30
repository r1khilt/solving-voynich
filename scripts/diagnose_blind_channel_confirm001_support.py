"""Fixed answer-informed interventions; no search and no qualification repair."""
from __future__ import annotations

import argparse
import json
import math
import resource
import signal
import time

from scripts import audit_blind_channel_dev004 as independent
from scripts.run_blind_channel_confirm001 import (
    ROOT, SOURCE_PATHS, inference_rows, sources, write_new,
)
from scripts.run_blind_channel_dev001 import checked_artifact, digest, require_frozen
from scripts.run_blind_channel_dev004 import metrics, resource_report


def main(commit):
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    def timeout(*_):
        raise TimeoutError("Diagnostic wall limit")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(180)
    started, cpu = time.monotonic(), time.process_time()
    directory = ROOT / "results/BLIND-CHANNEL-CONFIRM-001-DIAG-A"
    require_frozen(commit, ["scripts/diagnose_blind_channel_confirm001_support.py",
        "docs/experiments/BLIND-CHANNEL-CONFIRM-001-DIAG-A.md",
        "results/BLIND-CHANNEL-CONFIRM-001/evaluation.json",
        "results/BLIND-CHANNEL-CONFIRM-001/B-key6_freeze.json"])
    write_new(directory / "started.json", {"freeze": commit, "start_unix": time.time()})
    _, source, models = sources()
    manifest = json.loads((ROOT / "data/manifests/blind_channel_confirm001.json").read_text())
    require_frozen(manifest["pipeline_freeze"], SOURCE_PATHS)
    case = manifest["cases"]["B-key6"]
    fit = checked_artifact(case["artifacts"]["fit"])
    transfer = checked_artifact(case["artifacts"]["transfer"])
    answer = checked_artifact(case["artifacts"]["answer"])
    chosen = json.loads((ROOT / "results/BLIND-CHANNEL-CONFIRM-001/B-key6_freeze.json").read_text())
    evaluation = json.loads((ROOT / "results/BLIND-CHANNEL-CONFIRM-001/evaluation.json").read_text())
    base = tuple(chosen["units"])
    assert "F" not in base
    assert next(r for r in answer["gold_channel"]["rows"] if r["letter"] == "y")["emissions"][0]["glyphs"] == "F"
    assert all(letter not in "".join(answer["plaintext"]["fit"]) for letter in "kyz")
    full, summaries, maximum = {}, {}, 0.
    for arm, letter in (("learned", None), ("repair_y", "y"), ("control_k", "k"), ("control_z", "z")):
        units = list(base)
        if letter is not None:
            index = source.alphabet.index(letter)
            assert len(units[index]) == 1
            units[index] = "F"
        cost = independent.literal_model_bits(tuple(units), fit["context"])
        assert cost == independent.literal_model_bits(base, fit["context"])
        full[arm], summaries[arm] = {}, {"model_bits": cost, "changed_letter": letter}
        for split, data in (("fit", fit), ("transfer", transfer)):
            rows, delta = inference_rows(source, models["order3"], tuple(units), data["records"], fit["context"]["stop_probability"])
            maximum = max(maximum, delta)
            value = metrics(rows, answer["plaintext"][split])
            assert value["record_edits"] == [independent.edit_distance(r["plaintext"] or "", gold)
                for r, gold in zip(rows, answer["plaintext"][split], strict=True)]
            if arm == "learned":
                assert all(value[k] == evaluation["cases"]["B-key6"][arm][split][k] for k in value)
            value["record_log_likelihoods"] = [r["log_likelihood"] for r in rows]
            value["log_likelihood"] = None if any(r["log_likelihood"] is None for r in rows) else math.fsum(r["log_likelihood"] for r in rows)
            value["total_bits"] = None if value["log_likelihood"] is None else cost - value["log_likelihood"] / math.log(2)
            summaries[arm][split], full[arm][split] = value, rows
        summaries[arm]["fit_bits_minus_learned"] = summaries[arm]["fit"]["total_bits"] - chosen["score"]["total_bits"]
    archive = write_new(ROOT / "outputs/BLIND-CHANNEL-CONFIRM-001-DIAG-A/predictions.json.gz", full, compressed=True)
    output = {"experiment": "BLIND-CHANNEL-CONFIRM-001-DIAG-A", "freeze": commit,
        "status": "exploratory_answer_informed_counterfactual_not_qualification",
        "evaluation_sha256": digest((ROOT / "results/BLIND-CHANNEL-CONFIRM-001/evaluation.json").read_bytes()),
        "arms": summaries, "record_predictions_replayed": 24, "maximum_score_delta": maximum,
        "predictions": archive, "resources": resource_report(started, cpu)}
    write_new(directory / "evaluation.json", output)
    print(json.dumps(summaries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    main(parser.parse_args().freeze)
