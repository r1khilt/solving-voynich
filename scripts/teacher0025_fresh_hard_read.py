"""One-shot fresh-suite native hard-read assay; no model training."""

import json
from pathlib import Path
import time

import torch

from scripts.teacher0020_dev_run import episode_from_json
from scripts.teacher0025_fresh_hard_read_audit import (
    CONDITIONS, PANEL, SEEDS, provenance, sha, suite_audit,
)
from scripts.teacher0024_soft_read_run import evaluate_batch
from voynich.workspace.teacher14_tasks import evaluation_suite, suite_manifest
from voynich.workspace.teacher14_train import Config, new_model


MAX_SECONDS = 1200
MAX_BYTES = 100 * 1024**2


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False) + "\n")


def prepare(root: Path) -> dict:
    source = provenance(root)
    output = root / "outputs/TEACH-0025"
    if output.exists():
        raise FileExistsError("No automatic TEACH-0025 suite redraw")
    output.mkdir(parents=True)
    for seed in SEEDS:
        manifest = suite_manifest(evaluation_suite(seed, 128),
                                  seed=seed, group_count=128)
        (output / f"suite-{seed}.json").write_text(json.dumps(
            manifest, sort_keys=True, separators=(",", ":")) + "\n")
    checked = suite_audit(root)
    report = root / "results/TEACH-0025/suite-audit.json"
    if report.exists():
        raise FileExistsError("No automatic TEACH-0025 suite audit overwrite")
    write_json(report, {**checked, **source})
    return {"audit": checked["audit"], "source_head": source["source_head"]}


def run(root: Path) -> dict:
    source = provenance(root)
    result_dir = root / "results/TEACH-0025"
    suite_path = result_dir / "suite-audit.json"
    stored = json.loads(suite_path.read_text())
    checked = suite_audit(root)
    if any(stored[key] != value for key, value in checked.items()) or (
            stored["source_sha256"] != source["source_sha256"]):
        raise ValueError("Fresh suite/source audit differs")
    status_path = result_dir / "status.json"
    if status_path.exists():
        raise FileExistsError("No automatic TEACH-0025 run rerun")
    older = root / "results/TEACH-0024-soft-read"
    prior = json.loads((older / "status.json").read_text())
    if (prior["status"] != "complete" or
            json.loads((older / "audit.json").read_text())["audit"] != "pass" or
            json.loads((older / "replay-audit.json").read_text())["audit"] != "pass"):
        raise ValueError("Prior causal assay not audited")
    start = time.monotonic()
    status = {"experiment": "TEACH-0025", "status": "running",
              "source_head": source["source_head"],
              "source_sha256": source["source_sha256"],
              "suite_audit_sha256": sha(suite_path),
              "prior_audit_sha256": sha(older / "audit.json"),
              "prior_replay_sha256": sha(older / "replay-audit.json"),
              "checkpoint_sha256": prior["checkpoint_sha256"],
              "suite_file_sha256": {str(seed): sha(
                  root / f"outputs/TEACH-0025/suite-{seed}.json")
                  for seed in SEEDS}}
    write_json(status_path, status)
    try:
        cells = {}
        with torch.no_grad():
            for seed in SEEDS:
                suite = json.loads((root / f"outputs/TEACH-0025/suite-{seed}.json").read_text())
                episodes = [episode_from_json(item) for item in suite[
                    "panels"][PANEL]]
                cells[str(seed)] = {}
                for rep in (0, 1):
                    checkpoint = root / f"outputs/TEACH-0014-v3/rep{rep}-oracle_rows_workspace-step6000.pt"
                    if sha(checkpoint) != status["checkpoint_sha256"][str(rep)]:
                        raise ValueError("Checkpoint differs")
                    saved = torch.load(checkpoint, map_location="cpu",
                                       weights_only=True)
                    model, optimizer = new_model(
                        Config(), "oracle_rows_workspace", rep, "cpu")
                    model.load_state_dict(saved["model"], strict=True)
                    model.eval()
                    rows = []
                    samples = {name: [] for name in CONDITIONS}
                    for offset in range(0, 128, 32):
                        if time.monotonic() - start > MAX_SECONDS:
                            raise TimeoutError("TEACH-0025 CPU wall cap")
                        chunk_rows, logits = evaluate_batch(
                            model, episodes[offset:offset + 32])
                        rows.extend(chunk_rows)
                        for local in range(len(chunk_rows)):
                            index = offset + local
                            if index in (0, 64, 127):
                                for name in CONDITIONS:
                                    samples[name].append({"index": index,
                                                          "logits": logits[name][local]})
                    cells[str(seed)][str(rep)] = {"rows": rows,
                                                   "samples": samples}
                    del model, optimizer, saved
        archive_path = result_dir / "rows.json"
        archive_path.write_text(json.dumps({**status, "status": "complete",
                                            "cells": cells},
                                           separators=(",", ":"),
                                           allow_nan=False) + "\n")
        total_bytes = archive_path.stat().st_size + sum(
            path.stat().st_size for path in (result_dir / "suite-audit.json",
                                            status_path)) + sum(
            (root / f"outputs/TEACH-0025/suite-{seed}.json").stat().st_size
            for seed in SEEDS)
        if total_bytes > MAX_BYTES:
            raise RuntimeError("TEACH-0025 artifact cap")
        status.update({"status": "complete", "rows_sha256": sha(archive_path),
                       "rows_bytes": archive_path.stat().st_size,
                       "total_new_bytes": total_bytes,
                       "elapsed_seconds": time.monotonic() - start})
        write_json(status_path, status)
        return {"status": status["status"],
                "rows_sha256": status["rows_sha256"],
                "elapsed_seconds": status["elapsed_seconds"]}
    except Exception as exc:
        status.update({"status": "stopped",
                       "reason": f"{type(exc).__name__}: {exc}"})
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    args = parser.parse_args()
    print(json.dumps(prepare(Path.cwd()) if args.mode == "prepare"
                     else run(Path.cwd()), indent=2, sort_keys=True))
