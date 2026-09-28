"""Fit one bounded case using its ciphertext fit file and frozen source only."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import resource
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from voynich.finite_state_channel import SourceModel
from voynich.finite_state_channel_fit import CodingContext, _composition_rank, _field, _round_counts
from voynich.finite_state_channel_search import SearchConfig, search_channel


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "BLIND-CHANNEL-DEV-001"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def checked_artifact(artifact: dict) -> dict:
    raw = (ROOT / artifact["path"]).read_bytes()
    if digest(raw) != artifact["sha256"]:
        raise ValueError("Artifact hash mismatch: " + artifact["path"])
    return json.loads(raw)


def require_frozen(commit: str, paths: list[str]) -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=True)
    for path in paths:
        frozen = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
        if frozen != (ROOT / path).read_bytes():
            raise ValueError("Unfrozen source/input: " + path)


def fit_glyph_baseline(records: list[str], glyphs: tuple[str, ...]) -> dict:
    """Proper iid-glyph null, with explicitly coded finite-grid weights/stop."""
    if not records or set("".join(records)) - set(glyphs):
        raise ValueError("Bad baseline observations")
    counts = Counter("".join(records))
    total = sum(counts.values()) + .5 * len(glyphs)
    weights = _round_counts([(counts[glyph] + .5) / total for glyph in glyphs], 256, True)
    # The geometric MLE is 1/(1+mean glyph length); nearest declared grid point.
    stop_count = min(4095, max(1, round(4096 * len(records) / (len(records) + sum(counts.values())))))
    code = "1" + _field(stop_count - 1, 4095) + _field(*_composition_rank(weights, 256, True))
    return {"family": "iid_glyph_geometric", "glyph_alphabet": list(glyphs),
            "denominator": 256, "counts": weights, "stop_denominator": 4096,
            "stop_count": stop_count, "model_code": code, "model_bits": len(code)}


def baseline_log_likelihood(model: dict, records: list[str]) -> float:
    rho = model["stop_count"] / model["stop_denominator"]
    probabilities = {g: count / model["denominator"]
                     for g, count in zip(model["glyph_alphabet"], model["counts"], strict=True)}
    return math.fsum(math.log(rho) + len(record) * math.log1p(-rho)
                     + math.fsum(math.log(probabilities[g]) for g in record) for record in records)


def main() -> None:
    cpu_start = time.process_time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--source-freeze", required=True)
    args = parser.parse_args()
    manifest_path = "data/manifests/blind_channel_dev001.json"
    manifest = json.loads((ROOT / manifest_path).read_text())
    case = manifest["cases"][args.case]
    paths = [manifest_path, manifest["source"]["path"], "scripts/run_blind_channel_dev001.py",
             "src/voynich/finite_state_channel.py", "src/voynich/finite_state_channel_fit.py",
             "src/voynich/finite_state_channel_search.py", "docs/experiments/BLIND-CHANNEL-DEV-001.md"]
    require_frozen(args.source_freeze, paths)
    source_data = checked_artifact(manifest["source"])
    fit = checked_artifact(case["artifacts"]["fit"])
    source = SourceModel.from_dict(source_data["source_model"])
    context = CodingContext(**fit["context"])
    config = SearchConfig(seed=case["search_seed"], state_counts=(1, 2), restarts_per_state=2,
                          proposals_per_restart=1000, em_iterations=1, max_seconds=120., max_units=128)
    result = search_channel(source, fit["records"], context, config=config)
    payload = result.to_dict()
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    compressed = gzip.compress(raw, mtime=0)
    if len(compressed) > 200 * 1024 * 1024:
        raise RuntimeError("Trace exceeds registered 200MiB per-case cap")
    full_path = ROOT / f"outputs/{EXPERIMENT}/{args.case}.json.gz"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_bytes(compressed)
    freeze = {"experiment": EXPERIMENT, "case_id": args.case, "source_freeze": args.source_freeze,
              "source_sha256": manifest["source"]["sha256"], "input_sha256": case["artifacts"]["fit"]["sha256"],
              "channel": payload["channel"], "score": result.score, "config": asdict(config),
              "seconds": result.seconds, "stop_reason": result.stop_reason,
              "proposals": result.proposals, "completed_candidates": result.completed_candidates,
              "full_output": {"path": str(full_path.relative_to(ROOT)), "sha256": digest(compressed),
                              "bytes": len(compressed)},
              "baseline": fit_glyph_baseline(fit["records"], context.glyph_alphabet),
              "cpu_seconds_before_final_write": time.process_time() - cpu_start,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
              "environment": {"python": platform.python_version(), "platform": platform.platform()},
              "source_hashes": {path: digest((ROOT / path).read_bytes()) for path in paths}}
    path = ROOT / f"results/{EXPERIMENT}/{args.case}_freeze.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(freeze, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"case_id": args.case, "score": result.score, "seconds": result.seconds,
                      "stop_reason": result.stop_reason, "proposals": result.proposals,
                      "completed_candidates": result.completed_candidates}))


if __name__ == "__main__":
    main()
