"""ONE bounded, exclusively created train-only paragraph-state transport study."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
import os
import platform
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from voynich import paragraph_carry as pc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / pc.EXPERIMENT
BULK = ROOT / "outputs" / pc.EXPERIMENT
PATHS = (
    "src/voynich/paragraph_carry.py", "src/voynich/boundary_cross_transcription.py",
    "src/voynich/data.py",
    "scripts/run_paragraph_carry001.py", "scripts/audit_paragraph_carry001.py",
    "tests/test_paragraph_carry.py", "docs/experiments/PARAGRAPH-CARRY-001.md",
    "docs/research/paragraph-carry-mechanism-2026-10-10.md", "docs/research/PROTOCOL.md",
    "data/raw/ZL3b-n.txt", "data/raw/v101/GC2a-n.txt", "data/processed/zl3b/train.jsonl",
    "data/manifests/zl3b_split.json", "data/manifests/zl3b_preparation.json",
)
EXPECTED = {"data/raw/ZL3b-n.txt": "bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc",
            "data/raw/v101/GC2a-n.txt": "b09570cb6c993bc2d87134d115e60a978650a8a6495483ddbb1f6005a586096f",
            "data/processed/zl3b/train.jsonl": "49618c7be69cef573fe9ad8ae3627ad9f7b495601af1897aafb6082837fceca4",
            "data/manifests/zl3b_split.json": "9fc80cb4b000fdd5d952b7c2416b37b6b92f07bc56486a63309142953ed6634e"}


def write_exclusive(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def fingerprint():
    return {path: pc.sha(ROOT / path) for path in PATHS}


def require_frozen(freeze, hashes):
    for path, expected in hashes.items():
        if path.startswith("data/raw/") or path.startswith("data/processed/"):
            continue  # Raw bytes are pinned separately, never added to Git.
        committed = subprocess.check_output(["git", "show", f"{freeze}:{path}"], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != expected:
            raise ValueError(f"Working code/manifest differs from frozen commit: {path}")


def hard_limits():
    def timeout(_signum, _frame):
        raise TimeoutError("Registered process wall/CPU cap")
    signal.signal(signal.SIGALRM, timeout)
    signal.signal(signal.SIGPROF, timeout)
    signal.alarm(600)
    signal.setitimer(signal.ITIMER_PROF, 550)


def clear_limits():
    signal.alarm(0)
    signal.setitimer(signal.ITIMER_PROF, 0)


def guard(start_wall, start_cpu):
    if time.monotonic() - start_wall >= 600 or time.process_time() - start_cpu >= 550:
        raise RuntimeError("Registered time cap")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 1024 ** 3:
        raise RuntimeError("Registered 1GiB process RSS cap")


def input_views():
    for path, expected in EXPECTED.items():
        if pc.sha(ROOT / path) != expected:
            raise ValueError(f"Changed source: {path}")
    assignment = json.loads((ROOT / "data/manifests/zl3b_split.json").read_text())["leaf_assignments"]
    zl, zl_counts = pc.load_zl_train(ROOT / "data/processed/zl3b/train.jsonl")
    gc, gc_counts = pc.load_gc_train(ROOT / "data/raw/v101/GC2a-n.txt", assignment)
    views, matched = pc.matched_views(zl, gc)
    if any(assignment.get(row.leaf) != "train" for rows in views.values() for row in rows):
        raise ValueError("Nontraining physical leaf")
    return views, {"zl": zl_counts, "gc": gc_counts, "matching": matched}


def evaluate(name, rows, bootstrap_seed, permutation_seed):
    ledger, manifest = pc.cross_fit(rows)
    summary = pc.summarize(ledger, bootstrap_seed, n_boot=pc.N_BOOT)
    null = pc.permutation_null(rows, ledger, permutation_seed, n_perm=pc.N_PERM)
    return {"name": name, "input_rows_sha": pc.json_sha([asdict(row) for row in rows]),
            "manifest": manifest, "summary": summary, "null": null,
            "bootstrap_seed": bootstrap_seed, "permutation_seed": permutation_seed,
            "ledger_sha": pc.json_sha(ledger)}, ledger


def decisions(cells):
    expected_names = {f"control:{family}:{seed}:{mode}" for family, seed in pc.CONTROL_ALLOCATION for mode in pc.MODES}
    expected_names |= {"zl", "gc"}
    if len(cells) != len(expected_names) or {cell["name"] for cell in cells} != expected_names:
        raise ValueError("Missing, duplicate or unexpected cell")
    controls = [cell for cell in cells if cell["name"].startswith("control:")]
    expected = {"paragraph_carry": "paragraph_carry", "line_reset": "line_reset_compatible",
                "always_carry": "unbroken_carry"}
    planted = all(cell["summary"]["classification"] == expected[cell["name"].split(":")[-1]]
                  for cell in controls if cell["name"].split(":")[-1] in expected)
    negatives = all(cell["summary"]["classification"] != "paragraph_carry"
                    for cell in controls if cell["name"].split(":")[-1] in ("copy_mutate", "iid"))
    manuscript = [cell for cell in cells if cell["name"] in ("zl", "gc")]
    signal = all(cell["summary"]["classification"] == "paragraph_carry"
                 and cell["null"]["movable_events"] * 4 >= cell["null"]["events"]
                 and cell["null"]["one_sided_p"] is not None and cell["null"]["one_sided_p"] <= .05
                 for cell in manuscript)
    return {"all_planted_mechanisms_classified": planted,
            "structured_and_iid_controls_reject_paragraph_carry": negatives,
            "common_cause_classifications": {cell["name"]: cell["summary"]["classification"]
                for cell in controls if cell["name"].endswith(":paragraph_common_cause")},
            "both_manuscript_views_support_transport": signal,
            "qualified_paragraph_carry_signal": planted and negatives and signal,
            "causal_carry_identified": False,
            "decipherment": False}


def run(freeze):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != freeze:
        raise ValueError("Wrong frozen commit")
    hashes = fingerprint()
    require_frozen(freeze, hashes)
    OUT.mkdir(parents=True, exist_ok=False)
    BULK.mkdir(parents=True, exist_ok=False)
    start_wall, start_cpu = time.monotonic(), time.process_time()
    hard_limits()
    write_exclusive(OUT / "started.json", {"experiment": pc.EXPERIMENT, "freeze": freeze,
                    "started_utc": datetime.now(timezone.utc).isoformat(), "inputs": hashes,
                    "limits": {"wall_s": 600, "cpu_s": 550, "process_rss_bytes": 1024 ** 3,
                               "private_bytes": 16 * 1024 ** 2, "workers": 1, "paid_usd": 0}})
    try:
        views, extraction = input_views()
        cells, ledgers = [], {}
        # Synthetic qualification outcomes cannot tune the manuscript model.
        # Its fixed law and all choices are already in the published code.
        allocation = [(f"control:{family}:{seed}:{mode}", pc.controls(family, seed, mode))
                      for family, seed in pc.CONTROL_ALLOCATION for mode in pc.MODES]
        allocation.extend(views.items())
        for index, (name, rows) in enumerate(allocation):
            guard(start_wall, start_cpu)
            cell, ledger = evaluate(name, rows, 97701 + index, 97801 + index)
            cells.append(cell)
            ledgers[name] = ledger
            print(f"{index + 1}/{len(allocation)} completed", flush=True)
        archive = BULK / "ledger.json.gz"
        with archive.open("xb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
                stream.write(json.dumps(ledgers, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        if archive.stat().st_size > 16 * 1024 ** 2:
            raise RuntimeError("Private archive cap")
        guard(start_wall, start_cpu)
        if fingerprint() != hashes:
            raise ValueError("Frozen inputs changed")
        require_frozen(freeze, hashes)
        result = {"experiment": pc.EXPERIMENT, "freeze": freeze, "inputs": hashes,
                  "software": {"python": sys.version, "platform": platform.platform(),
                               "random_generator": "Python random.Random MT19937",
                               "model": "fixed terminal density-ratio transport with concentrations 50/20"},
                  "extraction": extraction, "cells": cells, "decisions": decisions(cells),
                  "private_archive": {"path": str(archive.relative_to(ROOT)), "sha256": pc.sha(archive),
                                      "bytes": archive.stat().st_size},
                  "resource": {"wall_s": time.monotonic() - start_wall,
                               "cpu_s": time.process_time() - start_cpu,
                               "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                               "paid_usd": 0}, "complete": True}
        write_exclusive(OUT / "result.json", result)
        return result
    except BaseException as exc:
        write_exclusive(OUT / "failure.json", {"error": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        clear_limits()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    args = parser.parse_args()
    run(args.freeze)
