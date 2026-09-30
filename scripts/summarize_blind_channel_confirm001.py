"""Post-evaluation diagnostics; never selects a model or reruns inference.

This root-authored descriptive supplement is outside the frozen qualification
pipeline. It may read answers only after that pipeline has saved its evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "BLIND-CHANNEL-CONFIRM-001"
NAMES = tuple(f"B-key{k}{suffix}" for k in range(1, 9) for suffix in ("", "-shuffle"))
ALPHABET = "abcdefghiklmnopqrstuxyz"


def artifact(root, spec):
    raw = (root / spec["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise ValueError("Artifact hash differs")
    if "bytes" in spec and len(raw) != spec["bytes"]:
        raise ValueError("Artifact size differs")
    return json.loads(gzip.decompress(raw) if spec["path"].endswith(".gz") else raw)


def posterior_surprisal(rows):
    """Conditional uncertainty under one fixed source/key, not correctness."""
    result = []
    for row in rows:
        marginal, joint = row["log_likelihood"], row["joint_log_probability"]
        if marginal is None or joint is None:
            result.append(None)
        else:
            bits = (marginal - joint) / math.log(2)
            if bits < -1e-7 or not math.isfinite(bits):
                raise ValueError("Invalid marginal versus joint score")
            result.append(max(0., bits))
    return result


def key_diagnostic(learned, gold, truth):
    if len(gold) != len(ALPHABET) or (learned is not None and len(learned) != len(gold)):
        raise ValueError("Unexpected key width")
    exposure = {split: Counter("".join(texts)) for split, texts in truth.items()}
    matches = None if learned is None else [a == b for a, b in zip(learned, gold, strict=True)]
    return {
        "literal_rows_correct": None if matches is None else sum(matches),
        "row_count": len(gold),
        "rows": [{"letter": letter, "correct": None if matches is None else matches[i],
                  "fit_occurrences": exposure["fit"][letter],
                  "transfer_occurrences": exposure["transfer"][letter]}
                 for i, letter in enumerate(ALPHABET)],
        "incorrect_row_occurrences": {
            split: None if matches is None else sum(counts[a] for a, ok in zip(ALPHABET, matches, strict=True) if not ok)
            for split, counts in exposure.items()},
    }


def summarize(root=ROOT):
    directory = root / "results" / EXPERIMENT
    # These reads must succeed before any answer or prediction is opened.
    evaluation = json.loads((directory / "evaluation.json").read_text())
    attempt = json.loads((directory / "evaluation_started.json").read_text())
    campaign = json.loads((directory / "campaign.json").read_text())
    if evaluation["key_freeze"] != attempt["key_freeze"]:
        raise ValueError("Evaluation freeze differs")
    if set(evaluation["cases"]) != set(NAMES):
        raise ValueError("Evaluation panel differs")
    jobs = {row["case"]: row for row in campaign["results"]}
    if len(campaign["results"]) != len(NAMES) or set(jobs) != set(NAMES):
        raise ValueError("Campaign panel differs")
    manifest = json.loads((root / "data/manifests/blind_channel_confirm001.json").read_text())
    full = artifact(root, evaluation["predictions"])
    if set(full) != set(NAMES):
        raise ValueError("Prediction panel differs")
    positives, resource_rows = {}, []
    for name in NAMES:
        chosen = None
        if jobs[name]["returncode"] == 0:
            chosen = json.loads((directory / f"{name}_freeze.json").read_text())
            resource_rows.append(chosen["resources"])
        if name.endswith("-shuffle"):
            continue
        answer = artifact(root, manifest["cases"][name]["artifacts"]["answer"])
        rows = answer["gold_channel"]["rows"]
        if len(rows) != len(ALPHABET) or len({r["letter"] for r in rows}) != len(ALPHABET):
            raise ValueError("Generating rows differ")
        mapping = {row["letter"]: row["emissions"][0]["glyphs"] for row in rows}
        gold = tuple(mapping[a] for a in ALPHABET)
        report = key_diagnostic(None if chosen is None else chosen["units"], gold, answer["plaintext"])
        values = evaluation["cases"][name]
        learned_bits = values["learned"]["fit"]["total_bits"]
        report["learned_minus_oracle_fit_bits"] = None if learned_bits is None else learned_bits - values["oracle"]["fit"]["total_bits"]
        report["conditional_map_surprisal_bits"] = {
            arm: {split: posterior_surprisal(rs) for split, rs in splits.items()}
            for arm, splits in full[name].items()}
        report["transfer_edits"] = {arm: values[arm]["transfer"]["edits"] for arm in full[name]}
        report["transfer_learned_minus_stage1_edits"] = report["transfer_edits"]["learned"] - report["transfer_edits"]["stage1"]
        positives[name] = report
    complete = all(row["returncode"] == 0 for row in jobs.values())
    rates = {arm: [Fraction(evaluation["cases"][name][arm]["transfer"]["edits"],
                           evaluation["cases"][name][arm]["transfer"]["gold_characters"])
                   for name in positives] for arm in ("learned", "oracle", "frequency")}
    gates = {
        "all_jobs_complete": complete,
        "each_transfer_cer_at_most_005": all(x <= Fraction(1, 20) for x in rates["learned"]),
        "macro_transfer_cer_at_most_002": sum(rates["learned"]) / 8 <= Fraction(1, 50),
        "each_oracle_excess_at_most_002": all(x-y <= Fraction(1, 50) for x, y in zip(rates["learned"], rates["oracle"], strict=True)),
        "each_beats_frequency": all(x < y for x, y in zip(rates["learned"], rates["frequency"], strict=True)),
    }
    decision = evaluation["decision"]
    if gates != decision["recovery_gates"] or all(gates.values()) != decision["recovery_pass"]:
        raise ValueError("Exact rational recovery gate disagrees")
    screen = complete and all(evaluation["cases"][n]["iid_diagnostic"]["flag"] == (not n.endswith("-shuffle")) for n in NAMES)
    if screen != decision["iid_screen_pass"] or decision["semantic_rejection_qualified"]:
        raise ValueError("Screening or semantic claim differs")
    return {"experiment": EXPERIMENT, "status": "post_evaluation_descriptive_supplement",
            "evaluation_sha256": hashlib.sha256((directory / "evaluation.json").read_bytes()).hexdigest(),
            "key_freeze": evaluation["key_freeze"], "positives": positives,
            "exact_rational_gate_replay": gates,
            "resources": {"successful_fit_cpu_seconds": sum(r["cpu_seconds"] for r in resource_rows),
                          "evaluation_cpu_seconds": evaluation["resources"]["cpu_seconds"],
                          "campaign_wall_seconds": campaign["wall_seconds"],
                          "largest_successful_worker_rss_bytes": max((r["peak_rss_bytes"] for r in resource_rows), default=None),
                          "failed_job_resources_included": False},
            "limitations": ["No new numerical inference or independent author review",
                            "Positive objective gap proves only one better known key exists",
                            "Posterior uncertainty is conditional on one fixed source and key",
                            "Literal row errors are not plaintext edit counts"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize()
    with args.output.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
