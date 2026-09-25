"""Frozen source-only selection; this program cannot open Pliny inputs/answers."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from voynich.recursive_character_lm import CONFIGS, log_terms, train

ROOT = Path(__file__).resolve().parent.parent
ALPHABET = "abcdefghilmnopqrstuvxyz"
SOURCES = ("scripts/select_naibbe003_source.py", "src/voynich/recursive_character_lm.py",
           "src/voynich/naibbe_key_search.py", "docs/experiments/NAIBBE-003.md")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_gain(gains):
    blocks = [{"sum": float(gains[start:start + 1000].sum()),
               "count": len(gains[start:start + 1000])} for start in range(0, len(gains), 1000)]
    rng = np.random.default_rng(920301)
    indices = rng.integers(0, len(blocks), size=(2000, len(blocks)))
    sums = np.array([block["sum"] for block in blocks])
    counts = np.array([block["count"] for block in blocks])
    draws = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return blocks, np.quantile(draws, [.025, .975]).tolist()


def main():
    started = time.monotonic()
    output = ROOT / "results/NAIBBE-003/source_selection.json"
    if output.exists():
        raise FileExistsError("Preserve the source-selection result")
    hashes = {}
    for name in SOURCES:
        path = ROOT / name
        if subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT) != path.read_bytes():
            raise AssertionError("Unfrozen source-selection code or protocol")
        hashes[name] = sha(path)
    manifest_path = ROOT / "data/manifests/naibbe003_data.json"
    if subprocess.check_output(["git", "show", "HEAD:data/manifests/naibbe003_data.json"], cwd=ROOT) != manifest_path.read_bytes():
        raise AssertionError("Unfrozen corpus manifest")
    manifest = json.loads(manifest_path.read_text())
    report = {"experiment": "NAIBBE-003", "stage": "source_only_selection",
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_hashes": hashes, "manifest_sha256": sha(manifest_path), "alphabet": ALPHABET,
              "ranges": {"train": [0, 220000], "dev": [220000, 268000], "check": [268000, 317326]},
              "environment": {"python": platform.python_version(), "numpy": np.__version__},
              "languages": {}}
    for language in ("latin", "english"):
        source = manifest["lms"][language]
        path = ROOT / source["path"]
        if sha(path) != source["sha256"]:
            raise AssertionError("Corpus changed")
        text = path.read_text().strip()
        if len(text) != 317326:
            raise AssertionError("Source length differs from registered range")
        train_text, dev, check = text[:220000], text[220000:268000], text[268000:]
        candidates = []
        for config in CONFIGS:
            lm = train(train_text, ALPHABET, config)
            candidates.append({"config": config, "dev_bits_per_character": float(-log_terms(lm, dev).mean() / np.log(2))})
        selected_index = min(range(len(candidates)), key=lambda i: (candidates[i]["dev_bits_per_character"], i))
        selected = candidates[selected_index]["config"]
        baseline = train(train_text, ALPHABET, CONFIGS[0])
        chosen = train(train_text, ALPHABET, selected)
        base_terms, chosen_terms = log_terms(baseline, check), log_terms(chosen, check)
        gains = (chosen_terms - base_terms) / np.log(2)
        blocks, interval = bootstrap_gain(gains)
        gain = float(gains.mean())
        criteria = {"new_family_selected": selected["family"] != "legacy",
                    "check_gain_at_least_005_bits": gain >= .005,
                    "block_bootstrap_lower_positive": interval[0] > 0}
        report["languages"][language] = {
            "source": source, "candidates": candidates, "selected_index": selected_index,
            "selected_config": selected, "check": {"characters": len(check),
            "legacy_bits_per_character": float(-base_terms.mean() / np.log(2)),
            "selected_bits_per_character": float(-chosen_terms.mean() / np.log(2)),
            "gain_bits_per_character": gain, "gain_ci95": interval, "gain_blocks": blocks},
            "criteria": criteria, "decision": "PASS" if all(criteria.values()) else "FAIL"}
    report["cipher_phase_allowed"] = report["languages"]["latin"]["decision"] == "PASS"
    report["seconds"] = time.monotonic() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({language: {k: row[k] for k in ("selected_config", "decision")}
                      | {"check_gain": row["check"]["gain_bits_per_character"]}
                      for language, row in report["languages"].items()}, indent=2))


if __name__ == "__main__":
    main()
