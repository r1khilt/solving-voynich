"""Post-hoc artifact audit only: consumes frozen results, never fits a model."""
from collections import defaultdict
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np

from voynich.blind_recovery import keyed_hmm

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-root", default="outputs/EXP-0008")
parser.add_argument("--data-root", default="data/processed/exp0008-blind")
parser.add_argument("--destination", required=True)
args = parser.parse_args()
root = Path(args.source_root)
draft = Path(args.destination)
data = Path(args.data_root)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (draft / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


report = json.loads((root / "report.json").read_text())
freeze = json.loads((root / "FROZEN_BEFORE_TRUTH.json").read_text())
manifest = json.loads((root / "dataset_manifest.json").read_text())
run_manifest = json.loads((root / "run_manifest.json").read_text())
assert report["status"] == "complete" and report["source_unchanged"]
assert report["manuscript_test_scored"] is False and report["paid_api_calls"] == 0
assert freeze["truth_opened"] is False
assert digest(root / "FROZEN_BEFORE_TRUTH.json") == report["freeze_sha256"]
assert digest(data / "manifest.json") == freeze["data_manifest_sha256"]
assert run_manifest["environment"]["git_dirty"] is False
for path, expected in run_manifest["environment"]["source_sha256"].items():
    assert digest(Path(path)) == expected, f"Registered source changed: {path}"
assert len(report["runs"]) == 4 and len(report["reports"]) == 80 and len(freeze["artifacts"]) == 80
assert {item["id"] for item in report["runs"]} == {
    "compact-seed42", "compact-seed43", "large-seed42", "large-seed43"}
assert freeze["elapsed_seconds"] < report["elapsed_seconds"]
assert len(manifest["datasets"]) == 20
for name, expected in manifest["files"].items():
    assert digest(data / name) == expected, name
artifact_count = 0
for record in freeze["artifacts"]:
    for name, expected in record["files"].items():
        assert digest(root / name) == expected, name
        artifact_count += 1
for item in report["runs"]:
    assert digest(root / item["id"] / "best.pt") == item["summary"]["best_sha256"]
    assert item["summary"]["history"][-1]["step"] == 4000
    assert [row["step"] for row in item["summary"]["history"]] == list(range(500, 4001, 500))
for item in report["reports"]:
    if item["family"] in {"copy_lag", "iid"}:
        assert item["transfer"] == "fresh_stream_control"
    elif item["family"] == "rrxor":
        assert item["transfer"] == "novel_family"
    else:
        assert item["transfer"] in {"seen_key", "unseen_key"}
    for method in ("residual", "forecast"):
        assert len(item["methods"][method]["per_context_three_horizon_bits"]) == 1024
        choice = item["decisions"][method]
        candidates = choice["candidates"]
        minimum = min(c["validation_bits_mean_three_horizons"] for c in candidates)
        expected = next(c["count"] for c in candidates if c["validation_bits_mean_three_horizons"] <= minimum + .01)
        assert choice["selected_count"] == expected

# Registered extraction is already finished and frozen above. This post-run
# check reads sealed generator keys solely to detect equivalent encodings.
keys = json.loads((data / "sealed_generator_keys.json").read_text())
key_equivalence = []
for family in ("cycle_null", "branch_null"):
    for test_key in (8, 9):
        test_edge, test_prior = keyed_hmm(family, keys[f"{family}-key{test_key}"]["permutation"])
        for train_key in range(8):
            train_edge, train_prior = keyed_hmm(family, keys[f"{family}-key{train_key}"]["permutation"])
            matching = []
            for permutation in itertools.permutations(range(len(train_prior))):
                order = list(permutation)
                if (np.allclose(test_edge, train_edge[:, order][:, :, order], atol=1e-12, rtol=0)
                        and np.allclose(test_prior, train_prior[order], atol=1e-12, rtol=0)):
                    matching.append(order)
            test_bigram = np.einsum("i,xij->xj", test_prior, test_edge) @ test_edge.sum(axis=2).T
            train_bigram = np.einsum("i,xij->xj", train_prior, train_edge) @ train_edge.sum(axis=2).T
            difference = np.abs(test_bigram - train_bigram)
            witness = np.unravel_index(difference.argmax(), difference.shape)
            key_equivalence.append({"family": family, "heldout_key": test_key, "training_key": train_key,
                                    "equivalent_under_state_permutation": bool(matching),
                                    "matching_permutations": matching,
                                    "max_abs_stationary_visible_bigram_difference": float(difference.max()),
                                    "observable_bigram_equal_within_1e_minus_12": bool(difference.max() <= 1e-12),
                                    "witness_symbols_zero_based": [int(v) for v in witness],
                                    "heldout_witness_probability": float(test_bigram[witness]),
                                    "training_witness_probability": float(train_bigram[witness])})

groups = defaultdict(list)
for item in report["reports"]:
    size = item["run"].split("-seed")[0]
    groups[(size, item["family"], item["transfer"])].append(item)
aggregates = []
for (size, family, transfer), rows in groups.items():
    record = {"size": size, "family": family, "transfer": transfer, "n_model_key_cases": len(rows),
              "baselines": {}, "methods": {}}
    for baseline in ("unigram", "last_symbol", "last_four_backoff"):
        record["baselines"][baseline] = float(np.mean([r["baselines"][baseline]["mean_three_horizon_bits"] for r in rows]))
    for field in ("bits", "oracle_kl_bits"):
        record["baselines"]["neural_next_" + field] = float(np.mean([r["baselines"]["neural_next"][field] for r in rows]))
    for method in ("residual", "forecast"):
        measurements = [r["methods"][method] for r in rows]
        value = {key: float(np.mean([r[key] for r in measurements])) for key in
                 ("mean_three_horizon_bits", "actual_label_ari", "posterior_map_ari", "unrecoverable_nuisance_ari",
                  "last_symbol_map_ari", "mean_bits_gain_over_last_symbol")}
        value["per_horizon_oracle_kl_bits"] = np.mean([r["per_horizon_oracle_kl_bits"] for r in measurements], axis=0).tolist()
        value["per_horizon_bits"] = np.mean([r["per_horizon_bits"] for r in measurements], axis=0).tolist()
        value["predictive_partition_passes"] = sum(r["predictive_partition_criterion"] for r in measurements)
        value["beyond_last_symbol_passes"] = sum(r["beyond_last_symbol_criterion"] for r in measurements)
        value["chosen_counts"] = [r["decisions"][method]["selected_count"] for r in rows]
        record["methods"][method] = value
    aggregates.append(record)

compact = copy.deepcopy(report)
for item in compact["reports"]:
    for value in item["methods"].values():
        del value["per_context_three_horizon_bits"]
compact["raw_report_sha256"] = digest(root / "report.json")
compact["omitted"] = "Per-context loss arrays remain in ignored raw report with the recorded digest."
draft.mkdir(parents=True, exist_ok=True)
for name in ("run_manifest.json", "dataset_manifest.json", "FROZEN_BEFORE_TRUTH.json"):
    shutil.copyfile(root / name, draft / name)
for item in report["runs"]:
    (draft / item["id"]).mkdir(exist_ok=True)
    for name in ("history.json", "summary.json"):
        shutil.copyfile(root / item["id"] / name, draft / item["id"] / name)
for item in freeze["artifacts"]:
    relative = Path("abstractions") / item["run"] / item["dataset"] / "decisions.json"
    (draft / relative).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / relative, draft / relative)
write("compact_report.json", compact)
write("aggregate.json", aggregates)
write("post_run_key_equivalence.json", {
    "scope": "Post-run diagnostic only; no fitting/selection changes. Exact edge/initial-prior equivalence under all state relabelings plus stationary visible bigram witnesses. A nonzero visible bigram difference proves these particular stationary processes have unequal observable distributions; it does not prove general latent-state identifiability.",
    "key_table_sha256": digest(data / "sealed_generator_keys.json"),
    "comparisons": key_equivalence,
    "alias_count": sum(row["equivalent_under_state_permutation"] for row in key_equivalence),
    "unequal_observable_bigram_witness_count": sum(not row["observable_bigram_equal_within_1e_minus_12"] for row in key_equivalence)})
write("audit.json", {"source_revision": run_manifest["environment"]["git_commit"],
      "complete": True, "model_runs": 4, "updates": 16000, "diagnostic_datasets": 20,
      "representation_fits": 160, "verified_frozen_files": artifact_count,
      "dataset_files_verified": len(manifest["files"]), "freeze_sha256": report["freeze_sha256"],
      "raw_report_sha256": digest(root / "report.json"), "elapsed_seconds": report["elapsed_seconds"],
      "peak_process_rss_bytes": report["peak_process_rss_bytes"],
      "mps_driver_allocated_bytes_at_completion": report.get("mps_driver_allocated_bytes"),
      "truth_diagnostics_after_artifact_freeze": True,
      "manuscript_test_scored": False, "paid_api_calls": 0})
print(json.dumps({"audit": "passed", "groups": aggregates}, indent=2))
