"""Frozen exposed-development unit search, with separate fitting and evaluation."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import resource
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

from scripts.audit_blind_channel_dev001 import (
    compare_known,
    evaluate_channel as independent_evaluate,
    model_bits as independent_model_bits,
)
from scripts.evaluate_blind_channel_dev001 import evaluate_channel
from scripts.run_blind_channel_dev001 import checked_artifact, digest, require_frozen
from voynich.finite_state_channel import Channel, SourceModel
from voynich.finite_state_channel_fit import CodingContext
from voynich.unit_channel_search import UnitSearchConfig, search_unit_channel


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "BLIND-CHANNEL-DEV-003"
MANIFEST = "data/manifests/blind_channel_dev001.json"
SOURCE_PATHS = [
    "scripts/run_blind_channel_dev003.py",
    "src/voynich/unit_channel_search.py",
    "src/voynich/batched_unit_channel.py",
    "src/voynich/finite_state_channel.py",
    "src/voynich/finite_state_channel_fit.py",
    "src/voynich/finite_state_channel_search.py",
    "scripts/run_blind_channel_dev001.py",
    "scripts/evaluate_blind_channel_dev001.py",
    "scripts/evaluate_naibbe001.py",
    "src/voynich/naibbe_key_search.py",
    "scripts/audit_blind_channel_dev001.py",
    "docs/experiments/BLIND-CHANNEL-DEV-003.md",
    "tests/test_batched_unit_channel.py",
    "tests/test_batched_unit_channel_independent.py",
    "tests/test_unit_channel_search.py",
    "tests/test_unit_channel_search_independent.py",
    "tests/test_run_blind_channel_dev003.py",
    MANIFEST,
]


def save(path: Path, data: dict) -> dict:
    raw = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(raw), "bytes": len(raw)}


def case_names(manifest: dict) -> list[str]:
    names = sorted(name for name in manifest["cases"] if name.startswith("B-"))
    if names != ["B-key1", "B-key1-shuffle", "B-key2", "B-key2-shuffle"]:
        raise ValueError("The registered four-case panel changed")
    return names


def freeze_path(name: str) -> Path:
    return ROOT / f"results/{EXPERIMENT}/{name}_freeze.json"


def paths_for(manifest: dict) -> list[str]:
    return [*SOURCE_PATHS, manifest["source"]["path"]]


def fit_case(name: str, commit: str, manifest: dict) -> None:
    cpu_start = time.process_time()
    # This process is disposable; cap total process CPU, including serialization.
    resource.setrlimit(resource.RLIMIT_CPU, (360, 360))
    names = case_names(manifest)
    if name not in names:
        raise ValueError("Unregistered case")
    if freeze_path(name).exists():
        raise FileExistsError("Refusing to overwrite a selected-model freeze")
    paths = paths_for(manifest)
    require_frozen(commit, paths)
    source = SourceModel.from_dict(checked_artifact(manifest["source"])["source_model"])
    case = manifest["cases"][name]
    fit = checked_artifact(case["artifacts"]["fit"])
    context = CodingContext(**fit["context"])
    config = UnitSearchConfig(seed=52101 + 101 * names.index(name), restarts=16,
                              max_sweeps=80, max_seconds=300., batch_size=256,
                              max_units=256, improvement_tolerance_bits=1e-8)
    result = search_unit_channel(source, fit["records"], context, config=config)
    payload = result.to_dict()
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    archive = gzip.compress(raw, mtime=0)
    if len(archive) > 200 * 1024 * 1024:
        raise RuntimeError("Search trace exceeds the 200MiB per-case cap")
    full_path = ROOT / f"outputs/{EXPERIMENT}/{name}.json.gz"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_bytes(archive)
    compact = {key: value for key, value in payload.items() if key != "trace"}
    compact.update(
        experiment=EXPERIMENT, case_id=name, source_freeze=commit,
        source_sha256=manifest["source"]["sha256"], input_sha256=case["artifacts"]["fit"]["sha256"],
        full_output={"path": str(full_path.relative_to(ROOT)), "bytes": len(archive), "sha256": digest(archive)},
        cpu_seconds_before_final_write=time.process_time() - cpu_start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        environment={"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                     "numerical_thread_limits": {key: os.environ.get(key) for key in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}},
        source_hashes={path: digest((ROOT / path).read_bytes()) for path in paths},
    )
    save(freeze_path(name), compact)
    print(json.dumps({"case_id": name, "seconds": result.seconds, "score": result.score,
                      "stop_reason": result.stop_reason, "evaluated_neighbors": result.evaluated_neighbors}), flush=True)


def campaign(commit: str, manifest: dict) -> None:
    names = case_names(manifest)
    if any(freeze_path(name).exists() for name in names):
        raise FileExistsError("Refusing to rerun any part of the registered panel")
    require_frozen(commit, paths_for(manifest))
    env = dict(os.environ, PYTHONPATH=".:src", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    started = time.monotonic()
    status = {"experiment": EXPERIMENT, "source_freeze": commit, "status": "running",
              "workers": 2, "cases": {}, "wall_seconds": 0.}
    status_path = ROOT / f"outputs/{EXPERIMENT}/campaign_status.json"
    save(status_path, status)

    def launch(name: str) -> dict:
        log = ROOT / f"outputs/{EXPERIMENT}/{name}.log"
        begin = time.monotonic()
        with log.open("w") as stream:
            try:
                completed = subprocess.run(
                    [sys.executable, "scripts/run_blind_channel_dev003.py", "fit", "--case", name, "--freeze", commit],
                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=420, check=False,
                )
                return {"returncode": completed.returncode, "wall_seconds": time.monotonic() - begin,
                        "log": str(log.relative_to(ROOT))}
            except subprocess.TimeoutExpired:
                return {"returncode": None, "failure": "420_second_wall_timeout",
                        "wall_seconds": time.monotonic() - begin, "log": str(log.relative_to(ROOT))}

    # Queue only the next pair while previous workers are healthy. A failing
    # process never causes a replacement run; the already-running peer finishes.
    failed = False
    remaining = iter(names)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = {executor.submit(launch, name): name for name in (next(remaining), next(remaining))}
        while pending:
            done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                name = pending.pop(future)
                try:
                    row = future.result()
                except Exception as error:
                    row = {"returncode": None, "failure": f"{type(error).__name__}: {error}"}
                status["cases"][name] = row
                failed |= row["returncode"] != 0
                print(json.dumps({"completed_case": name, **row}), flush=True)
            if not failed:
                while len(pending) < 2:
                    name = next(remaining, None)
                    if name is None:
                        break
                    pending[executor.submit(launch, name)] = name
            status["wall_seconds"] = time.monotonic() - started
            save(status_path, status)
    status["status"] = "failed" if failed else "complete"
    status["wall_seconds"] = time.monotonic() - started
    save(status_path, status)
    save(ROOT / f"results/{EXPERIMENT}/campaign.json", status)
    if failed:
        raise RuntimeError("At least one registered worker failed; inspect preserved logs")


def evaluate(commit: str, manifest: dict) -> dict:
    names = case_names(manifest)
    destination = ROOT / f"results/{EXPERIMENT}/evaluation.json"
    if destination.exists():
        raise FileExistsError("Refusing to overwrite the registered evaluation")
    require_frozen(commit, [*paths_for(manifest), *(str(freeze_path(n).relative_to(ROOT)) for n in names)])
    source_raw = checked_artifact(manifest["source"])["source_model"]
    source = SourceModel.from_dict(source_raw)
    output = {"experiment": EXPERIMENT, "status": "exposed_diagnostic_independent_replay_pass",
              "key_freeze": commit, "manifest_sha256": digest((ROOT / MANIFEST).read_bytes()),
              "source_sha256": manifest["source"]["sha256"], "cases": {}}
    for name in names:
        case = manifest["cases"][name]
        frozen = json.loads(freeze_path(name).read_text())
        if (frozen["source_sha256"] != manifest["source"]["sha256"]
                or frozen["input_sha256"] != case["artifacts"]["fit"]["sha256"]):
            raise ValueError("Frozen source or fit identity changed")
        full = (ROOT / frozen["full_output"]["path"]).read_bytes()
        if digest(full) != frozen["full_output"]["sha256"] or len(full) != frozen["full_output"]["bytes"]:
            raise ValueError("Full trace archive identity changed")
        payload = json.loads(gzip.decompress(full))
        for key, value in payload.items():
            if key != "trace" and value != frozen[key]:
                raise ValueError("Frozen payload differs from trace archive: " + key)
        fit = checked_artifact(case["artifacts"]["fit"])
        answer = checked_artifact(case["artifacts"]["answer"])
        channel = None if frozen["channel"] is None else Channel.from_dict(frozen["channel"])
        row = {"positive": case["positive"], "freeze_sha256": digest(freeze_path(name).read_bytes()),
               "maximum_independent_delta": 0., "splits": {}}
        if channel is not None:
            bits = independent_model_bits(frozen["channel"], fit["context"])
            compare_known(bits, frozen["score"]["model_bits"], name + ".model_bits")
        for split in ("fit", "transfer"):
            observed = fit if split == "fit" else checked_artifact(case["artifacts"][split])
            gold = None if answer["plaintext"] is None else answer["plaintext"][split]
            measured = evaluate_channel(source, channel, observed["records"], gold)
            independent = independent_evaluate(source_raw, frozen["channel"], observed["records"], gold)
            delta = compare_known(independent, measured, name + "." + split)
            row["maximum_independent_delta"] = max(row["maximum_independent_delta"], delta)
            row["splits"][split] = measured
        if channel is not None:
            compare_known(frozen["score"]["log_likelihood"], row["splits"]["fit"]["log_likelihood"], name + ".fit_score")
        output["cases"][name] = row
    artifact = save(ROOT / f"outputs/{EXPERIMENT}/predictions.json", output)
    for row in output["cases"].values():
        for split in row["splits"].values():
            split.pop("records")
    output["full_predictions"] = artifact
    save(destination, output)
    print(json.dumps(output, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("campaign", "fit", "evaluate"))
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--case")
    args = parser.parse_args()
    manifest = json.loads((ROOT / MANIFEST).read_text())
    if args.stage == "campaign":
        campaign(args.freeze, manifest)
    elif args.stage == "fit":
        if args.case is None:
            parser.error("--case is required for a single fit")
        fit_case(args.case, args.freeze, manifest)
    else:
        evaluate(args.freeze, manifest)


if __name__ == "__main__":
    main()
