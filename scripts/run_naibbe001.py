"""Fit only: the training entry point cannot open answer/dev/transfer files."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from voynich.naibbe_key_search import CharacterLM, encode_lattice, search

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["latin_joint", "latin_fixed", "english_joint"])
    args = parser.parse_args()
    output = ROOT / "outputs/NAIBBE-001"
    result_path = output / f"{args.arm}.json"
    if result_path.exists():
        raise FileExistsError("Preserve prior run; do not overwrite")
    source_hashes = {}
    for source in ("scripts/run_naibbe001.py", "src/voynich/naibbe_key_search.py"):
        frozen = subprocess.check_output(["git", "show", f"HEAD:{source}"], cwd=ROOT)
        if frozen != (ROOT / source).read_bytes():
            raise AssertionError("Search source differs from committed freeze")
        source_hashes[source] = sha(ROOT / source)
    path = ROOT / "data/processed/naibbe001/fit_input.json"
    manifest = json.loads((ROOT / "data/manifests/naibbe001_data.json").read_text())
    if sha(path) != manifest["split_inputs"]["fit"]["sha256"]:
        raise AssertionError("Fit data hash mismatch")
    data = json.loads(path.read_text())
    if data["split_name"] != "fit":
        raise AssertionError("Fit split only")
    language = args.arm.split("_")[0]
    lm_path = ROOT / data["lms"][language]["path"]
    if sha(lm_path) != data["lms"][language]["sha256"]:
        raise AssertionError("LM hash mismatch")
    lm = CharacterLM.train(lm_path.read_text().strip(), "".join(data["alphabet"]))
    lattice = encode_lattice(data["split"], data["class_ids"])
    result = search(lattice, lm, seed=910101, joint=args.arm != "latin_fixed",
                    restarts=8, kicks=6, max_seconds=3600)
    result.update({"arm": args.arm, "input_sha256": sha(path), "lm_sha256": sha(lm_path),
                   "source_hashes": source_hashes,
                   "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                   "platform": platform.platform()},
                   "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                            cwd=ROOT, text=True).strip(),
                   "class_ids": data["class_ids"], "alphabet": data["alphabet"]})
    output.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    compact = {key: result[key] for key in ("arm", "fit_score", "seconds", "budget_stop", "key",
                                           "input_sha256", "lm_sha256", "source_commit", "seed",
                                           "source_hashes", "environment")}
    compact["result_sha256"] = sha(result_path)
    compact["trace"] = result["trace"]
    tracked = ROOT / "results/NAIBBE-001"
    tracked.mkdir(parents=True, exist_ok=True)
    (tracked / f"{args.arm}_freeze.json").write_text(json.dumps(compact, indent=2) + "\n")
    print(json.dumps({key: compact[key] for key in ("arm", "fit_score", "seconds", "budget_stop")}), flush=True)


if __name__ == "__main__":
    main()
