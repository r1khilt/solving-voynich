"""Fit only after the source-only selection and its independent audit freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from voynich.naibbe_homophone_search import search
from voynich.naibbe_key_search import encode_lattice
from voynich.recursive_character_lm import train

ROOT = Path(__file__).resolve().parent.parent
SEARCH_SOURCES = ("scripts/run_naibbe003.py", "src/voynich/recursive_character_lm.py",
                  "src/voynich/naibbe_homophone_search.py", "src/voynich/naibbe_key_search.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_json(relative):
    path = ROOT / relative
    if subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT) != path.read_bytes():
        raise AssertionError(f"Unfrozen artifact: {relative}")
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["latin_joint", "english_joint"], required=True)
    args = parser.parse_args()
    result_path = ROOT / f"outputs/NAIBBE-003/{args.arm}.json"
    compact_path = ROOT / f"results/NAIBBE-003/{args.arm}_freeze.json"
    if result_path.exists() or compact_path.exists():
        raise FileExistsError("Preserve prior fit")
    source_hashes = {}
    for source in SEARCH_SOURCES:
        if subprocess.check_output(["git", "show", f"HEAD:{source}"], cwd=ROOT) != (ROOT / source).read_bytes():
            raise AssertionError("Search source differs from committed freeze")
        source_hashes[source] = sha(ROOT / source)
    selection = frozen_json("results/NAIBBE-003/source_selection.json")
    audit = frozen_json("results/NAIBBE-003/source_audit.json")
    if not selection["cipher_phase_allowed"] or not audit["audit_pass"]:
        raise AssertionError("Source calibration or independent audit failed")
    selection_sha = sha(ROOT / "results/NAIBBE-003/source_selection.json")
    if audit["source_selection_sha256"] != selection_sha:
        raise AssertionError("Audit does not cover selected configuration")
    manifest = frozen_json("data/manifests/naibbe003_data.json")
    path = ROOT / manifest["split_inputs"]["fit"]["path"]
    if sha(path) != manifest["split_inputs"]["fit"]["sha256"]:
        raise AssertionError("Fit input changed")
    data = json.loads(path.read_text())
    if data["split_name"] != "fit":
        raise AssertionError("Wrong split")
    ids = {c: i for i, c in enumerate(data["class_ids"])}
    groups = [[ids[c] for c in group] for group in data["groups"]]
    language = args.arm.split("_")[0]
    config = selection["languages"][language]["selected_config"]
    lm_path = ROOT / data["lms"][language]["path"]
    if sha(lm_path) != data["lms"][language]["sha256"]:
        raise AssertionError("Source data changed")
    lm = train(lm_path.read_text().strip(), "".join(data["alphabet"]), config)
    result = search(encode_lattice(data["split"], data["class_ids"]), lm, groups,
                    seed=920201, coordinated=True, restarts=8, kicks=6, max_seconds=1200)
    result.update({"arm": args.arm, "prior_config": config, "selection_sha256": selection_sha,
                   "input_sha256": sha(path), "lm_sha256": sha(lm_path),
                   "source_hashes": source_hashes,
                   "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                            cwd=ROOT, text=True).strip(),
                   "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                   "platform": platform.platform()},
                   "class_ids": data["class_ids"], "alphabet": data["alphabet"], "groups": data["groups"]})
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    compact = {key: value for key, value in result.items()
               if key not in {"choices", "class_ids", "alphabet", "groups"}}
    compact["result_sha256"] = sha(result_path)
    compact_path.write_text(json.dumps(compact, indent=2) + "\n")
    print(json.dumps({key: compact[key] for key in ("arm", "prior_config", "fit_score", "seconds", "budget_stop")}), flush=True)


if __name__ == "__main__":
    main()
