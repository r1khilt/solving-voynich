"""Open exposed development answers only after selected models are frozen."""
from __future__ import annotations

import argparse
import json
import math

from scripts.evaluate_naibbe001 import edit_distance
from scripts.run_blind_channel_dev001 import (
    EXPERIMENT, ROOT, baseline_log_likelihood, checked_artifact, digest, require_frozen,
)
from voynich.finite_state_channel import Channel, SourceModel, forward_log_probability, viterbi_decode


def evaluate_channel(source: SourceModel, channel: Channel | None, records: list[str],
                     gold: list[str] | None = None) -> dict:
    if gold is not None and len(gold) != len(records):
        raise ValueError("Prediction/gold record mismatch")
    rows = []
    for i, record in enumerate(records):
        score = -math.inf if channel is None else forward_log_probability(source, channel, record)
        decoded = None if channel is None else viterbi_decode(source, channel, record)
        plaintext = None if decoded is None else decoded.plaintext
        joint = -math.inf if decoded is None else decoded.log_probability
        rows.append({"log_likelihood": score if math.isfinite(score) else None,
                     "joint_log_probability": joint if math.isfinite(joint) else None,
                     "plaintext": plaintext, "prediction_supported": plaintext is not None,
                     "edits": None if gold is None else edit_distance(plaintext or "", gold[i]),
                     "gold_characters": None if gold is None else len(gold[i])})
    all_supported = all(row["prediction_supported"] for row in rows)
    return {"log_likelihood": math.fsum(row["log_likelihood"] for row in rows) if all_supported else None,
            "edits": None if gold is None else sum(row["edits"] for row in rows),
            "gold_characters": None if gold is None else sum(map(len, gold)),
            "decoded_characters": sum(len(row["plaintext"] or "") for row in rows),
            "exact_records": None if gold is None else sum(row["plaintext"] == truth for row, truth in zip(rows, gold, strict=True)),
            "records": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-freeze", required=True)
    args = parser.parse_args()
    manifest_path = ROOT / "data/manifests/blind_channel_dev001.json"
    manifest = json.loads(manifest_path.read_text())
    freeze_paths = [f"results/{EXPERIMENT}/{name}_freeze.json" for name in manifest["cases"]]
    require_frozen(args.key_freeze, [*freeze_paths, "scripts/evaluate_blind_channel_dev001.py",
                                    "scripts/audit_blind_channel_dev001.py"])
    source = SourceModel.from_dict(checked_artifact(manifest["source"])["source_model"])
    result = {"experiment": EXPERIMENT, "status": "exposed_development", "key_freeze": args.key_freeze,
              "manifest_sha256": digest(manifest_path.read_bytes()), "source_sha256": manifest["source"]["sha256"],
              "cases": {}}
    # Full prediction strings remain in ignored output; compact report strips them.
    for name, case in manifest["cases"].items():
        path = ROOT / f"results/{EXPERIMENT}/{name}_freeze.json"
        freeze = json.loads(path.read_text())
        if freeze["source_sha256"] != manifest["source"]["sha256"] or freeze["input_sha256"] != case["artifacts"]["fit"]["sha256"]:
            raise ValueError("Frozen fit/source identity changed")
        channel = None if freeze["channel"] is None else Channel.from_dict(freeze["channel"])
        answer = checked_artifact(case["artifacts"]["answer"])
        oracle = None if answer["gold_channel"] is None else Channel.from_dict(answer["gold_channel"])
        report = {"freeze_sha256": digest(path.read_bytes()), "positive": case["positive"], "family": case["family"],
                  "splits": {}, "model_bits_with_selector": None if freeze["score"] is None else freeze["score"]["model_bits"] + 1,
                  "baseline_model_bits": freeze["baseline"]["model_bits"]}
        for split in ("fit", "transfer"):
            data = checked_artifact(case["artifacts"][split])
            gold = None if answer["plaintext"] is None else answer["plaintext"][split]
            learned = evaluate_channel(source, channel, data["records"], gold)
            baseline = baseline_log_likelihood(freeze["baseline"], data["records"])
            item = {"learned": learned, "baseline_log_likelihood": baseline,
                    "glyph_characters": sum(map(len, data["records"]))}
            if oracle is not None:
                item["oracle"] = evaluate_channel(source, oracle, data["records"], gold)
            report["splits"][split] = item
        fit_ll = report["splits"]["fit"]["learned"]["log_likelihood"]
        transfer_ll = report["splits"]["transfer"]["learned"]["log_likelihood"]
        if fit_ll is None or transfer_ll is None:
            margin = transfer_gain = None
        else:
            margin = (report["baseline_model_bits"] - report["splits"]["fit"]["baseline_log_likelihood"] / math.log(2)
                      - report["model_bits_with_selector"] + fit_ll / math.log(2))
            transfer_gain = ((transfer_ll - report["splits"]["transfer"]["baseline_log_likelihood"])
                             / math.log(2) / report["splits"]["transfer"]["glyph_characters"])
        report.update(fit_saving_vs_iid_bits=margin, transfer_gain_vs_iid_bits_per_glyph=transfer_gain,
                      diagnostic_flag=margin is not None and margin >= 32 and transfer_gain >= .05)
        result["cases"][name] = report
    raw_path = ROOT / f"outputs/{EXPERIMENT}/evaluation_predictions.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    raw_path.write_bytes(raw)
    # The independent audit reads the full predictions, verifies hash, then also
    # checks this tracked compact report against that exact artifact.
    compact = json.loads(raw)
    for case in compact["cases"].values():
        for split in case["splits"].values():
            for arm in ("learned", "oracle"):
                if arm in split:
                    split[arm].pop("records")
    compact["full_predictions"] = {"path": str(raw_path.relative_to(ROOT)), "sha256": digest(raw), "bytes": len(raw)}
    positives = [row for row in compact["cases"].values() if row["positive"]]
    nulls = [row for row in compact["cases"].values() if not row["positive"]]
    compact["diagnostic_summary"] = {
        "positives_cer_at_most_10pct": sum(row["splits"]["transfer"]["learned"]["edits"] <= .1 * row["splits"]["transfer"]["learned"]["gold_characters"] for row in positives),
        "positives_flagged": sum(row["diagnostic_flag"] for row in positives),
        "nulls_flagged": sum(row["diagnostic_flag"] for row in nulls),
        "scope": "Development diagnostics; no sealed benchmark or source-language decision"}
    (ROOT / f"results/{EXPERIMENT}/evaluation.json").write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(compact["diagnostic_summary"]))


if __name__ == "__main__":
    main()
