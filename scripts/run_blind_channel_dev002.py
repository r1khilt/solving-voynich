"""Separately frozen, exposed-development exact-neighborhood A diagnostic."""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

from scripts.audit_blind_channel_dev001 import compare_known, evaluate_channel as independent_evaluate
from scripts.evaluate_blind_channel_dev001 import evaluate_channel
from scripts.run_blind_channel_dev001 import checked_artifact, digest, require_frozen
from voynich.bijective_channel_search import BijectiveSearchConfig, search_bijective_channel
from voynich.finite_state_channel import Channel, SourceModel
from voynich.finite_state_channel_fit import CodingContext


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "BLIND-CHANNEL-DEV-002"


def save(path: Path, data: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(raw), "bytes": len(raw)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("fit", "evaluate"))
    parser.add_argument("--freeze", required=True)
    args = parser.parse_args()
    manifest_path = "data/manifests/blind_channel_dev001.json"
    manifest = json.loads((ROOT / manifest_path).read_text())
    names = [name for name in manifest["cases"] if name.startswith("A-")]
    source_raw = checked_artifact(manifest["source"])["source_model"]
    source = SourceModel.from_dict(source_raw)
    paths = ["scripts/run_blind_channel_dev002.py", "src/voynich/bijective_channel_search.py",
             "src/voynich/finite_state_channel_search.py", "src/voynich/finite_state_channel.py",
             "src/voynich/finite_state_channel_fit.py", "scripts/evaluate_blind_channel_dev001.py",
             "scripts/audit_blind_channel_dev001.py", "scripts/run_blind_channel_dev001.py",
             "docs/experiments/BLIND-CHANNEL-DEV-002.md", manifest_path, manifest["source"]["path"]]
    if args.stage == "evaluate":
        paths += [f"results/{EXPERIMENT}/{name}_freeze.json" for name in names]
    require_frozen(args.freeze, paths)
    cases = {}
    for index, name in enumerate(names):
        case = manifest["cases"][name]
        if args.stage == "fit":
            cpu_start = time.process_time()
            fit = checked_artifact(case["artifacts"]["fit"])
            result = search_bijective_channel(source, fit["records"], CodingContext(**fit["context"]),
                config=BijectiveSearchConfig(seed=51101 + index * 101, restarts=16, max_sweeps=80, max_seconds=30.))
            payload = result.to_dict()
            encoded = gzip.compress(json.dumps(payload, sort_keys=True, allow_nan=False).encode(), mtime=0)
            full_path = ROOT / f"outputs/{EXPERIMENT}/{name}.json.gz"
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_bytes(encoded)
            compact = {key: value for key, value in payload.items() if key not in ("trace", "statistics")}
            compact.update(case_id=name, source_freeze=args.freeze, source_sha256=manifest["source"]["sha256"],
                input_sha256=case["artifacts"]["fit"]["sha256"], cpu_seconds=time.process_time() - cpu_start,
                full_output={"path": str(full_path.relative_to(ROOT)), "sha256": digest(encoded), "bytes": len(encoded)})
            save(ROOT / f"results/{EXPERIMENT}/{name}_freeze.json", compact)
            print(json.dumps({"case_id": name, "seconds": result.seconds,
                              "evaluated_neighbors": result.evaluated_neighbors, "score": result.score}), flush=True)
        else:
            frozen_path = ROOT / f"results/{EXPERIMENT}/{name}_freeze.json"
            frozen = json.loads(frozen_path.read_text())
            if frozen["source_sha256"] != manifest["source"]["sha256"] or frozen["input_sha256"] != case["artifacts"]["fit"]["sha256"]:
                raise ValueError("Source or fit identity mismatch")
            full = (ROOT / frozen["full_output"]["path"]).read_bytes()
            if digest(full) != frozen["full_output"]["sha256"]:
                raise ValueError("Full fit archive hash mismatch")
            payload = json.loads(gzip.decompress(full))
            if payload["channel"] != frozen["channel"] or payload["score"] != frozen["score"]:
                raise ValueError("Full/compact model mismatch")
            answer = checked_artifact(case["artifacts"]["answer"])
            model = None if frozen["channel"] is None else Channel.from_dict(frozen["channel"])
            report = {"positive": case["positive"], "freeze_sha256": digest(frozen_path.read_bytes()),
                      "splits": {}, "maximum_independent_delta": 0.}
            for split in ("fit", "transfer"):
                observed = checked_artifact(case["artifacts"][split])
                gold = None if answer["plaintext"] is None else answer["plaintext"][split]
                measured = evaluate_channel(source, model, observed["records"], gold)
                independent = independent_evaluate(source_raw, frozen["channel"], observed["records"], gold)
                difference = compare_known(independent, measured, name + "." + split)
                report["maximum_independent_delta"] = max(report["maximum_independent_delta"], difference)
                report["splits"][split] = measured
            if model is not None:
                compare_known(frozen["score"]["log_likelihood"], report["splits"]["fit"]["log_likelihood"], "fit_score")
            cases[name] = report
    if args.stage == "evaluate":
        output = {"experiment": EXPERIMENT, "status": "exposed_diagnostic_independent_replay_pass",
                  "key_freeze": args.freeze, "source_sha256": manifest["source"]["sha256"],
                  "manifest_sha256": digest((ROOT / manifest_path).read_bytes()), "cases": cases}
        artifact = save(ROOT / f"outputs/{EXPERIMENT}/predictions.json", output)
        for case in cases.values():
            for split in case["splits"].values():
                split.pop("records")
        output["full_predictions"] = artifact
        save(ROOT / f"results/{EXPERIMENT}/evaluation.json", output)
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
