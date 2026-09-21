"""Archive WMD-0001 qualification with explicit negative results and provenance."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run, out = Path(args.run_dir), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    qualification = run / "qualification"
    manifest = read(qualification / "manifest.json")
    for name, expected in manifest["files"].items():
        assert digest(qualification / name) == expected, name
    for path in sorted(qualification.glob("*.json")):
        if path.name.endswith("-audit.json"):
            (out / (path.name + ".gz")).write_bytes(gzip.compress(path.read_bytes(), mtime=0))
        else:
            name = "qualification-manifest.json" if path.name == "manifest.json" else path.name
            shutil.copyfile(path, out / name)
    for folder in ("joint", "actions"):
        for name in ("manifest.json", "summary.json"):
            shutil.copyfile(run / folder / name, out / f"{folder}-{name}")
    for name in ("interventions.json", "demo-inference.json"):
        shutil.copyfile(run / name, out / name)
    shutil.copyfile(run / "manuscript-format-export-v2" / "manifest.json", out / "manuscript-format-manifest.json")

    summary = {
        "schema_version": 1,
        "decision": "software qualification completed; mixed development results, no recovery pass claim",
        "interval_policy": "Post-run descriptive paired percentile bootstrap; not preregistered inference or across-seed uncertainty",
        "bootstrap_seed": 41023, "bootstrap_replicates": 10000,
        "comparisons": {}, "candidate_audits": {}, "untrained_replay_exact": {},
    }
    rng = np.random.default_rng(41023)
    for anchors in (0, 2):
        trained = read(qualification / f"trained-a{anchors}.json")
        untrained = read(qualification / f"untrained-a{anchors}.json")
        assert trained["input_sha256"] == untrained["input_sha256"]
        assert [w["observation_sha256"] for w in trained["worlds"]] == [
            w["observation_sha256"] for w in untrained["worlds"]
        ]
        original = read(run / f"untrained-a{anchors}.json")
        exact = all(original[key] == untrained[key] for key in ("metrics", "worlds", "input_sha256"))
        assert exact, "Untrained replay changed"
        summary["untrained_replay_exact"][str(anchors)] = exact
        indices = rng.integers(0, 64, size=(10000, 64))
        comparisons = {}
        for metric in ("decoded_similarity", "selected_valid", "grammar_accuracy", "family_accuracy"):
            delta = np.array([
                float(left["methods"]["neural"][metric]) - float(right["methods"]["neural"][metric])
                for left, right in zip(trained["worlds"], untrained["worlds"], strict=True)
            ])
            interval = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975]).tolist()
            comparisons[metric] = {"trained_minus_untrained": float(delta.mean()),
                                   "descriptive_paired_95_percent_interval": interval}
        summary["comparisons"][str(anchors)] = comparisons
        for condition in ("trained", "untrained"):
            name = f"{condition}-a{anchors}"
            audit = read(qualification / f"{name}-audit.json")["inferences"]
            counts = Counter()
            selected = Counter()
            valid_semantics = Counter()
            for record in audit:
                compiler = record["compiler_search"]
                assert compiler["expansions"] <= 20000
                counts["compiler_expansions"] += compiler["expansions"]
                counts["budget_exhausted_worlds"] += compiler["budget_exhausted"]
                counts["worlds_with_pruning"] += any(c["beam_pruned"] for c in compiler["combination_reports"])
                for candidate in record["candidates"]:
                    counts[candidate["source"] + "_candidates"] += 1
                    counts[candidate["source"] + "_valid"] += bool(candidate["verification"]["valid"])
                    counts["candidates_with_projection_changes"] += bool(candidate["projection_changes"])
                    if candidate["verification"]["valid"]:
                        valid_semantics[candidate["verification"]["semantics"]["status"]] += 1
                if record["candidates"]:
                    selected[record["candidates"][0]["source"]] += 1
            summary["candidate_audits"][name] = {
                "counts": dict(counts), "selected_sources": dict(selected),
                "valid_semantic_statuses": dict(valid_semantics),
            }
    save(out / "comparison.json", summary)
    save(out / "artifact-manifest.json", {
        "schema_version": 1,
        "training_source_commit": read(run / "joint" / "manifest.json")["environment"]["git_commit"],
        "local_checkpoints": {str(p): {"sha256": digest(p), "bytes": p.stat().st_size}
                              for p in (run / "joint" / "best.pt", run / "joint" / "last.pt", run / "actions" / "model.pt")},
        "packaging_script_sha256": digest(Path(__file__)),
        "files": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir())},
        "retention": "Checkpoints and generated observations remain local and ignored. Audit gzip files decompress to bytes hashed in qualification-manifest.json.",
    })
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
