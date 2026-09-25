"""Fit six unknown table permutations using only the exposed fit block."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from voynich.naibbe_homophone_search import search
from voynich.naibbe_key_search import CharacterLM, encode_lattice

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["latin_joint", "latin_local", "english_joint"], required=True)
    args = parser.parse_args()
    output = ROOT / "outputs/NAIBBE-002"
    result_path = output / f"{args.arm}.json"
    if result_path.exists():
        raise FileExistsError("Preserve prior run")
    source_hashes = {}
    for source in ("scripts/run_naibbe002.py", "src/voynich/naibbe_homophone_search.py",
                   "src/voynich/naibbe_key_search.py"):
        if subprocess.check_output(["git", "show", f"HEAD:{source}"], cwd=ROOT) != (ROOT / source).read_bytes():
            raise AssertionError("Search source differs from committed freeze")
        source_hashes[source] = sha(ROOT / source)
    manifest = json.loads((ROOT / "data/manifests/naibbe002_data.json").read_text())
    path = ROOT / "data/processed/naibbe002/fit_input.json"
    if sha(path) != manifest["split_inputs"]["fit"]["sha256"]:
        raise AssertionError("Fit input changed")
    data = json.loads(path.read_text())
    if data["split_name"] != "fit":
        raise AssertionError("Wrong split")
    ids = {c: i for i, c in enumerate(data["class_ids"])}
    groups = [[ids[c] for c in group] for group in data["groups"]]
    language = args.arm.split("_")[0]
    lm_path = ROOT / data["lms"][language]["path"]
    if sha(lm_path) != data["lms"][language]["sha256"]:
        raise AssertionError("Source model data changed")
    lm = CharacterLM.train(lm_path.read_text().strip(), "".join(data["alphabet"]))
    result = search(encode_lattice(data["split"], data["class_ids"]), lm, groups,
                    seed=920201, coordinated=args.arm != "latin_local",
                    restarts=8, kicks=6, max_seconds=1200)
    result.update({"arm": args.arm, "input_sha256": sha(path), "lm_sha256": sha(lm_path),
                   "source_hashes": source_hashes,
                   "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                            cwd=ROOT, text=True).strip(),
                   "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                   "platform": platform.platform()},
                   "class_ids": data["class_ids"], "alphabet": data["alphabet"], "groups": data["groups"]})
    output.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    compact = {key: result[key] for key in ("arm", "fit_score", "seconds", "budget_stop", "key",
                                           "input_sha256", "lm_sha256", "source_commit", "seed",
                                           "source_hashes", "environment", "trace")}
    compact["result_sha256"] = sha(result_path)
    tracked = ROOT / "results/NAIBBE-002"
    tracked.mkdir(parents=True, exist_ok=True)
    (tracked / f"{args.arm}_freeze.json").write_text(json.dumps(compact, indent=2) + "\n")
    print(json.dumps({key: compact[key] for key in ("arm", "fit_score", "seconds", "budget_stop")}), flush=True)


if __name__ == "__main__":
    main()
