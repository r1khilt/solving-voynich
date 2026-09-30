"""Frozen fixed-channel source/decision diagnostics; no channel optimization."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import platform
import resource
import signal
import sys
import time
from pathlib import Path

from scripts.build_blind_channel_dev001 import segments
from scripts.evaluate_naibbe001 import edit_distance
from scripts.run_blind_channel_dev001 import checked_artifact, digest, require_frozen
from voynich.finite_state_channel import Channel, SourceModel, forward_log_probability, viterbi_decode
from voynich.finite_state_channel_fit import CodingContext, encode_channel
from voynich.unit_channel_edit_floor import minimum_unit_edit_distance


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "BLIND-CHANNEL-DEV-004"
MANIFEST = "data/manifests/blind_channel_dev001.json"
CORPUS = "data/manifests/blind_channel_development_corpora.json"
CASE_NAMES = ("B-key1", "B-key1-shuffle", "B-key2", "B-key2-shuffle")
TAUS = (.25, 1., 4., 16., 64., 256.)
SELECTION = f"results/{EXPERIMENT}/source_selection.json"
PREDICTIONS = f"results/{EXPERIMENT}/prediction_manifest.json"
SOURCE_PATHS = [
    "scripts/run_blind_channel_dev004.py", "scripts/build_blind_channel_dev001.py",
    "scripts/run_blind_channel_dev001.py", "scripts/evaluate_naibbe001.py",
    "src/voynich/finite_state_channel.py", "src/voynich/finite_state_channel_fit.py",
    "src/voynich/higher_order_unit_channel.py", "src/voynich/unit_channel_decision.py",
    "src/voynich/unit_channel_edit_floor.py", "tests/test_unit_channel_edit_floor.py",
    "scripts/audit_blind_channel_dev004.py",
    "tests/test_higher_order_unit_channel.py", "tests/test_run_blind_channel_dev004.py",
    "tests/test_higher_order_unit_channel_independent.py", "tests/test_unit_channel_decision.py",
    "tests/test_unit_channel_decision_independent.py", "tests/test_unit_channel_edit_floor_independent.py",
    "tests/test_blind_channel_dev004_audit.py",
    "docs/experiments/BLIND-CHANNEL-DEV-004.md", MANIFEST, CORPUS,
]


def save_new(path: Path, payload: dict, *, compressed: bool = False) -> dict:
    """Atomic exclusive creation: never replace a prior stage result."""
    raw = (json.dumps(payload, indent=None if compressed else 2,
                      sort_keys=True, allow_nan=False) + "\n").encode()
    if compressed:
        raw = gzip.compress(raw, mtime=0)
    if len(raw) > 100 * 1024 * 1024:
        raise ValueError("Artifact exceeds 100 MiB cap")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(raw), "bytes": len(raw)}


def load_archive(artifact: dict) -> dict:
    raw = (ROOT / artifact["path"]).read_bytes()
    if digest(raw) != artifact["sha256"] or len(raw) != artifact["bytes"]:
        raise ValueError("Archive identity mismatch")
    return json.loads(gzip.decompress(raw))


def fixed_inputs(commit: str) -> dict:
    manifest = json.loads((ROOT / MANIFEST).read_text())
    require_frozen(commit, [*SOURCE_PATHS, manifest["source"]["path"], *[
        f"results/BLIND-CHANNEL-DEV-003/{name}_freeze.json" for name in CASE_NAMES]])
    if digest((ROOT / MANIFEST).read_bytes()) != "8587ae6010de94996bdd57d74ea2e1eee3f71940cdb13c1cc3125adfdfc9acea":
        raise ValueError("Registered case manifest changed")
    return manifest


def get_units(channel: Channel, alphabet: tuple[str, ...]) -> tuple[str, ...]:
    if len(channel.states) != 1 or channel.initial[channel.states[0]] != 1.:
        raise ValueError("Diagnostic requires one deterministic state")
    state = channel.states[0]
    rows = [channel.rows[state, char] for char in alphabet]
    if any(len(row) != 1 or row[0].probability != 1. or row[0].next_state != state for row in rows):
        raise ValueError("Diagnostic requires deterministic unit rows")
    return tuple(row[0].glyphs for row in rows)


def resource_report(start_wall: float, start_cpu: float) -> dict:
    return {"wall_seconds": time.monotonic() - start_wall,
            "cpu_seconds": time.process_time() - start_cpu,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "python": platform.python_version(), "platform": platform.platform(),
            "paid_spend_usd": 0,
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}


def metrics(rows: list[dict], truth: list[str] | None, key: str = "plaintext") -> dict:
    if truth is not None and len(rows) != len(truth):
        raise ValueError("Prediction/truth record mismatch")
    texts = [row[key] for row in rows]
    return {"supported_records": sum(text is not None for text in texts),
            "decoded_characters": sum(len(text or "") for text in texts),
            "gold_characters": None if truth is None else sum(map(len, truth)),
            "edits": None if truth is None else sum(edit_distance(text or "", gold)
                                                     for text, gold in zip(texts, truth, strict=True)),
            "exact_records": None if truth is None else sum(text == gold
                                                             for text, gold in zip(texts, truth, strict=True)),
            "record_edits": None if truth is None else [edit_distance(text or "", gold)
                                                         for text, gold in zip(texts, truth, strict=True)]}


def select(commit: str) -> None:
    # Imports are local to keep basic runner invariants testable independently.
    from voynich.higher_order_unit_channel import estimate_source, source_bits

    start_wall, start_cpu = time.monotonic(), time.process_time()
    manifest = fixed_inputs(commit)
    if (ROOT / SELECTION).exists():
        raise FileExistsError("Source selection already exists")
    corpus = json.loads((ROOT / CORPUS).read_text())
    bodies = {}
    for name in ("caesar", "virgil"):
        spec = corpus["sources"][name]
        payload = checked_artifact({"path": spec["derived_path"], "sha256": spec["derived_sha256"]})
        bodies[name] = segments(payload)
        if sum(map(len, bodies[name])) != 50_000:
            raise ValueError("Registered author length changed")
    alphabet = tuple(corpus["alphabet"])
    candidates = []
    for order, tau in [(0, 1.), *((order, tau) for order in (1, 2, 3) for tau in TAUS)]:
        source = estimate_source(bodies["caesar"], alphabet, order, tau)
        candidates.append({"order": order, "tau": tau,
                           "validation_bits_per_character": source_bits(source, bodies["virgil"]) / 50_000})
    per_order = {str(order): min((row for row in candidates if row["order"] == order),
                                key=lambda row: row["validation_bits_per_character"])
                 for order in (0, 1, 2, 3)}
    selected = min(candidates, key=lambda row: row["validation_bits_per_character"])
    models = {f"order{order}": estimate_source(bodies["caesar"] + bodies["virgil"], alphabet,
                                               row["order"], row["tau"]).to_dict()
              for order, row in per_order.items()}
    old = SourceModel.from_dict(checked_artifact(manifest["source"])["source_model"])
    regression = estimate_source(bodies["caesar"] + bodies["virgil"], alphabet, 1, 64.)
    # Comparing actual rows makes source/record-boundary compatibility explicit.
    delta = max(abs(regression.probabilities[context][char] - old.probabilities[context][char])
                for context in old.probabilities for char in alphabet)
    if delta > 1e-12:
        raise ValueError("Higher-order fitter does not reproduce old source")
    model_artifact = save_new(ROOT / f"outputs/{EXPERIMENT}/selected_sources.json.gz",
                              {"models": models}, compressed=True)
    save_new(ROOT / SELECTION, {"experiment": EXPERIMENT, "source_freeze": commit,
                               "status": "source_only_development_selection",
                               "candidates": candidates, "selected_per_order": per_order,
                               "primary_model": f"order{selected['order']}", "selected": selected,
                               "old_source_regression_max_delta": delta,
                               "models": model_artifact, "old_source": manifest["source"],
                               "corpus_manifest_sha256": digest((ROOT / CORPUS).read_bytes()),
                               "source_characters_per_author": 50_000,
                               "source_authors_accessed": ["caesar", "virgil"],
                               "resources": resource_report(start_wall, start_cpu)})
    print(json.dumps({"selected": selected, "per_order": per_order, "regression_delta": delta}))


def predict(commit: str) -> None:
    from voynich.higher_order_unit_channel import MarkovSource, decode_units
    from voynich.unit_channel_decision import sample_mbr_decode

    start_wall, start_cpu = time.monotonic(), time.process_time()
    manifest = fixed_inputs(commit)
    require_frozen(commit, [SELECTION])
    if (ROOT / PREDICTIONS).exists():
        raise FileExistsError("Prediction stage already exists")
    selection = json.loads((ROOT / SELECTION).read_text())
    sources = {name: MarkovSource.from_dict(raw) for name, raw in load_archive(selection["models"])["models"].items()}
    old = SourceModel.from_dict(checked_artifact(manifest["source"])["source_model"])
    predictions = {"experiment": EXPERIMENT, "selection_freeze": commit,
                   "source_selection_sha256": digest((ROOT / SELECTION).read_bytes()), "cases": {}}
    baseline_delta = 0.
    for case_index, name in enumerate(CASE_NAMES):
        case = manifest["cases"][name]
        freeze_path = ROOT / f"results/BLIND-CHANNEL-DEV-003/{name}_freeze.json"
        frozen = json.loads(freeze_path.read_text())
        if frozen["source_sha256"] != manifest["source"]["sha256"] or frozen["input_sha256"] != case["artifacts"]["fit"]["sha256"]:
            raise ValueError("Frozen learned key identity mismatch")
        channels = {"learned": Channel.from_dict(frozen["channel"])}
        # Generating channel is a declared evaluator-only intervention; plaintext
        # fields in the answer container are never used in this stage.
        if case["positive"]:
            channels["oracle"] = Channel.from_dict(checked_artifact(case["artifacts"]["answer"])["gold_channel"])
        item = {"positive": case["positive"], "freeze_sha256": digest(freeze_path.read_bytes()), "arms": {}}
        for arm, channel in channels.items():
            units = get_units(channel, old.alphabet)
            fit = checked_artifact(case["artifacts"]["fit"])
            model_bits = len(encode_channel(channel, CodingContext(**fit["context"]))) + 1
            report = {"model_bits_with_selector": model_bits, "sources": {}, "decisions": {}}
            for split_index, split in enumerate(("fit", "transfer")):
                observed = checked_artifact(case["artifacts"][split])["records"]
                report["decisions"][split] = []
                for source_name in ("old", *sources):
                    report["sources"].setdefault(source_name, {})[split] = []
                for record_index, record in enumerate(observed):
                    old_score = forward_log_probability(old, channel, record)
                    old_map = viterbi_decode(old, channel, record)
                    old_row = {"log_likelihood": old_score, "plaintext": old_map.plaintext if old_map else None,
                               "joint_log_probability": old_map.log_probability if old_map else None}
                    report["sources"]["old"][split].append(old_row)
                    for source_name, source in sources.items():
                        decoded = decode_units(source, units, record, channel.stop_probability)
                        report["sources"][source_name][split].append(decoded.to_dict())
                    seed = 61101 + 1000 * case_index + 100 * split_index + record_index + (10000 if arm == "oracle" else 0)
                    decision = sample_mbr_decode(old, units, record, channel.stop_probability,
                                                 candidate_draws=32, risk_draws=256, seed=seed)
                    decision_row = decision.to_dict()
                    if decision_row["map_plaintext"] != old_row["plaintext"]:
                        raise ValueError("Decision decoder MAP differs from original engine")
                    baseline_delta = max(baseline_delta, abs(decision_row["log_likelihood"] - old_score))
                    if baseline_delta > 1e-8:
                        raise ValueError("Decision backward likelihood differs from original engine")
                    report["decisions"][split].append(decision_row)
            item["arms"][arm] = report
        predictions["cases"][name] = item
        print(json.dumps({"completed": name}), flush=True)
    artifact = save_new(ROOT / f"outputs/{EXPERIMENT}/predictions.json.gz", predictions, compressed=True)
    save_new(ROOT / PREDICTIONS, {"experiment": EXPERIMENT, "selection_freeze": commit,
                                  "source_selection_sha256": predictions["source_selection_sha256"],
                                  "predictions": artifact, "baseline_marginal_max_delta": baseline_delta,
                                  "cases": list(CASE_NAMES), "plaintext_metrics_computed": False,
                                  "resources": resource_report(start_wall, start_cpu)})


def evaluate(commit: str) -> None:
    start_wall, start_cpu = time.monotonic(), time.process_time()
    manifest = fixed_inputs(commit)
    require_frozen(commit, [SELECTION, PREDICTIONS])
    prediction_manifest = json.loads((ROOT / PREDICTIONS).read_text())
    predictions = load_archive(prediction_manifest["predictions"])
    if predictions["source_selection_sha256"] != digest((ROOT / SELECTION).read_bytes()):
        raise ValueError("Prediction source selection changed")
    result = {"experiment": EXPERIMENT, "prediction_freeze": commit,
              "status": "adaptive_fixed_dictionary_diagnostic", "cases": {}}
    for name, case in predictions["cases"].items():
        answer = checked_artifact(manifest["cases"][name]["artifacts"]["answer"])
        item = {"positive": case["positive"], "arms": {}}
        for arm, report in case["arms"].items():
            measured = {"model_bits_with_selector": report["model_bits_with_selector"], "sources": {}, "decisions": {}}
            if case["positive"]:
                if arm == "oracle":
                    channel = Channel.from_dict(answer["gold_channel"])
                else:
                    channel = Channel.from_dict(json.loads((ROOT / f"results/BLIND-CHANNEL-DEV-003/{name}_freeze.json").read_text())["channel"])
                alphabet = tuple(checked_artifact(manifest["source"])["source_model"]["alphabet"])
                units = get_units(channel, alphabet)
                measured["dictionary_edit_floor"] = {}
                for split in ("fit", "transfer"):
                    records = checked_artifact(manifest["cases"][name]["artifacts"][split])["records"]
                    floors = [minimum_unit_edit_distance(alphabet, units, record, gold)
                              for record, gold in zip(records, answer["plaintext"][split], strict=True)]
                    measured["dictionary_edit_floor"][split] = {"record_edits": floors,
                        "edits": None if any(value is None for value in floors) else sum(floors)}
            for source_name, splits in report["sources"].items():
                measured["sources"][source_name] = {}
                for split, rows in splits.items():
                    truth = answer["plaintext"][split] if case["positive"] else None
                    value = metrics(rows, truth)
                    value["log_likelihood"] = math.fsum(row["log_likelihood"] for row in rows)
                    value["data_bits"] = -value["log_likelihood"] / math.log(2)
                    value["total_bits"] = report["model_bits_with_selector"] + value["data_bits"]
                    value["map_surprisal_bits"] = [(row["log_likelihood"] - row["joint_log_probability"]) / math.log(2)
                                                   for row in rows]
                    measured["sources"][source_name][split] = value
            for split, rows in report["decisions"].items():
                truth = answer["plaintext"][split] if case["positive"] else None
                measured["decisions"][split] = metrics(rows, truth, "plaintext")
            item["arms"][arm] = measured
        result["cases"][name] = item
    result["resources"] = resource_report(start_wall, start_cpu)
    result["predictions"] = prediction_manifest["predictions"]
    save_new(ROOT / f"results/{EXPERIMENT}/evaluation.json", result)
    print(json.dumps({name: {arm: {s: r["transfer"]["edits"] for s, r in row["sources"].items()}
                             for arm, row in case["arms"].items()}
                      for name, case in result["cases"].items()}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("select", "predict", "evaluate"))
    parser.add_argument("--freeze", required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, ({"select": 180, "predict": 1100, "evaluate": 500}[args.stage],) * 2)

    def timed_out(_signum, _frame):
        raise TimeoutError("Registered 20-minute wall deadline exceeded")

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(1200)
    {"select": select, "predict": predict, "evaluate": evaluate}[args.stage](args.freeze)


if __name__ == "__main__":
    main()
