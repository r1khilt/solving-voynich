"""No-model audit of the frozen TEACH-0014 gate-rescue diagnosis.

This checks every visible example and control assignment. A numerical replay
of sampled intervention logits is a separate, required final gate.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess

import torch

from scripts.teacher0014_suite_audit import audit_manifest


CONDITIONS = {"native", "gold", "false_only", "reversed", "count_random"}
ARM = "latent_rows_answer"
SYMBOL_START = 16
VOCAB_SIZE = 2064
TOLERANCE = 2e-3
MAX_SECONDS = 2 * 3600
MAX_MPS_BYTES = 12 * 1024**3
MAX_ARTIFACT_BYTES = 2 * 1024**3
SOURCE_PATHS = (
    "docs/experiments/TEACH-0014-diagnostic-preregistration.md",
    "src/voynich/workspace/teacher14_diagnose.py",
    "scripts/teacher0014_diagnostic_run.py",
    "scripts/teacher0014_diagnostic_audit.py",
    "scripts/teacher0014_diagnostic_replay.py",
    "tests/test_teacher0014_diagnose.py",
    "tests/test_teacher0014_diagnostic_audit.py",
    "tests/test_teacher0014_diagnostic_replay.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _source_check(root: Path, status: dict) -> None:
    hashes = status.get("diagnostic_source_sha256")
    head = status.get("diagnostic_git_head")
    if not isinstance(hashes, dict) or set(hashes) != set(SOURCE_PATHS) or (
            not isinstance(head, str) or len(head) != 40):
        raise ValueError("Diagnostic source provenance incomplete")
    for name in SOURCE_PATHS:
        committed = subprocess.run(["git", "show", f"{head}:{name}"],
                                   cwd=root, check=True, capture_output=True).stdout
        if hashlib.sha256(committed).hexdigest() != hashes[name] or (
                _sha(root / name) != hashes[name]):
            raise ValueError(f"Diagnostic source hash differs: {name}")


def _visible_path(episode: dict) -> tuple[list[int], int]:
    ordinary = [token for token in episode["tokens"][1:-3]
                if token >= SYMBOL_START]
    if len(ordinary) < 2 or len(ordinary) % 2:
        raise ValueError("Malformed visible row operands")
    rows = list(zip(ordinary[::2], ordinary[1::2], strict=True))
    mapping = dict(rows)
    if len(mapping) != len(rows):
        raise ValueError("Visible row mapping not functional")
    value = episode["query"]
    path = []
    for _ in range(episode["hops"]):
        next_value = mapping[value]
        path.append(2 * rows.index((value, next_value)))
        value = next_value
    if value != episode["answer"]:
        raise ValueError("Visible answer differs from manifest")
    return path, 2 * len(rows) - 1


def _random_indices(render_id: str, count: int, rows: int,
                    replicate: int) -> list[int]:
    # Recompute the declared CPU Philox/MT draw without importing the
    # intervention implementation or reading the model.
    seed = int(_canonical(["TEACH-0014-gate-random", render_id,
                           replicate])[:16], 16) % 2**63
    if rows != (count + 1) // 2:
        raise ValueError("Invalid count-matched random control")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return sorted(torch.randperm(count, generator=generator)[:rows].tolist())


def _fraction(hits: list[bool]) -> dict:
    return {"correct": sum(hits), "total": len(hits),
            "accuracy": sum(hits) / len(hits) if hits else 0.0}


def _paired_bootstrap(left: list[bool], right: list[bool],
                      identity: object, draws: int = 4000) -> dict:
    if len(left) != len(right) or not left or draws < 100:
        raise ValueError("Paired diagnostic bootstrap inputs invalid")
    effects = [int(a) - int(b) for a, b in zip(left, right, strict=True)]
    seed = int(_canonical(["TEACH-0014-gate-bootstrap", identity])[:16], 16)
    rng = random.Random(seed)
    estimates = sorted(
        sum(effects[rng.randrange(len(effects))]
            for _ in range(len(effects))) / len(effects)
        for _ in range(draws))
    return {"difference": sum(effects) / len(effects),
            "lower_95": estimates[int(.025 * (draws - 1))],
            "upper_95": estimates[int(.975 * (draws - 1))],
            "groups": len(effects), "draws": draws, "seed": seed}


def _benchmark_check(path: Path, manifest: dict, source_hashes: dict,
                     primary_suite_file_sha: str) -> None:
    row = json.loads(path.read_text())
    names = {"factorial", "boundary_groups", "long_ood",
             "hop_4_long_ood", "copy_confirm", "alias_inner"}
    timings = row.get("timings_seconds")
    if (not isinstance(timings, dict) or set(timings) != names or any(
            not isinstance(values, list) or len(values) != 4 or any(
                type(value) not in (int, float) or
                not math.isfinite(value) or value <= 0 for value in values)
            for values in timings.values())):
        raise ValueError("Diagnostic benchmark timing grid invalid")
    medians = {name: statistics.median(values)
               for name, values in timings.items()}
    batches = sum(math.ceil(len(episodes) / 32)
                  for episodes in manifest["panels"].values())
    slowest = max(medians.values())
    projection = 1.75 * slowest * 2 * batches + 300
    if (row.get("experiment") != "TEACH-0014-gate-diagnostic" or
            row.get("diagnostic_source_sha256") != source_hashes or
            row.get("primary_suite_sha256") != primary_suite_file_sha or
            row.get("batch_size") != 32 or row.get("random_rep") != 0 or
            row.get("warmup_batches") != 6 or row.get("timed_batches") != 24 or
            row.get("batches_per_seed") != batches or
            row.get("panel_median_seconds") != medians or
            row.get("slowest_panel_median_seconds") != slowest or
            not math.isclose(row.get("conservative_projected_seconds", math.inf),
                             projection, rel_tol=1e-9) or
            row.get("admitted") is not True or projection >= MAX_SECONDS or
            row.get("max_seconds") != MAX_SECONDS or
            row.get("max_mps_bytes") != MAX_MPS_BYTES or
            row.get("max_artifact_bytes") != MAX_ARTIFACT_BYTES or
            row.get("sampled_mps_allocated_bytes", math.inf) > MAX_MPS_BYTES or
            row.get("peak_sampled_mps_allocated_bytes", math.inf)
            > MAX_MPS_BYTES):
        raise ValueError("Diagnostic benchmark resource gate invalid")


def _validate_row(source: dict, row: dict, *, index: int,
                  primary_prediction: int, primary_logits: list[float] | None,
                  sampled: bool, primary_gates: list[float] | None = None
                  ) -> dict[str, bool]:
    path, count = _visible_path(source)
    if (row.get("render_id") != source["render_id"] or
            row.get("answer") != source["answer"] or
            row.get("candidate_count") != count or
            row.get("row_count") != (count + 1) // 2 or
            row.get("path_candidate_indices") != path or
            row.get("random_rep") != 0 or
            type(row.get("identity_max_abs_logit_error")) not in (int, float) or
            not 0 <= row["identity_max_abs_logit_error"] <= TOLERANCE):
        raise ValueError(f"Diagnostic row identity/path/identity error at {index}")
    choices = row.get("random_selected_indices")
    if (not isinstance(choices, list) or
            len(choices) != row["row_count"] or
            len(set(choices)) != len(choices) or
            any(type(choice) is not int or not 0 <= choice < count
                for choice in choices) or
            choices != _random_indices(source["render_id"], count,
                                       row["row_count"], 0)):
        raise ValueError("Invalid count-matched random assignment")
    gates = row.get("native_gate_logits")
    if (not isinstance(gates, list) or len(gates) != count or
            any(type(value) not in (int, float) or not math.isfinite(value)
                for value in gates)):
        raise ValueError("Invalid native gate logits")
    if primary_gates is not None and (
            len(primary_gates) != count or any(
                abs(a - b) > TOLERANCE + TOLERANCE * abs(b)
                for a, b in zip(gates, primary_gates, strict=True))):
        raise ValueError("Native gates differ from archived primary run")
    assignments = row.get("gate_assignment_logits")
    if not isinstance(assignments, dict) or set(assignments) != CONDITIONS:
        raise ValueError("Diagnostic gate assignments incomplete")
    expected_assignments = {
        "native": gates,
        "gold": [20.0 if index % 2 == 0 else -20.0
                 for index in range(count)],
        "false_only": [gates[index] if index % 2 == 0 else -20.0
                       for index in range(count)],
        "reversed": [-20.0 if index % 2 == 0 else 20.0
                     for index in range(count)],
        "count_random": [20.0 if index in set(choices) else -20.0
                         for index in range(count)],
    }
    for name in CONDITIONS:
        values = assignments[name]
        if (not isinstance(values, list) or len(values) != count or any(
                type(value) not in (int, float) or not math.isfinite(value)
                or abs(value - expected) > 1e-6
                for value, expected in zip(values, expected_assignments[name],
                                           strict=True))):
            raise ValueError(f"Diagnostic {name} gate assignment differs")
    details = row.get("conditions")
    if not isinstance(details, dict) or set(details) != CONDITIONS:
        raise ValueError("Missing or extra gate condition")
    hits = {}
    for name, condition in details.items():
        prediction = condition.get("prediction")
        if type(prediction) is not int or not SYMBOL_START <= prediction < VOCAB_SIZE:
            raise ValueError("Invalid diagnostic answer")
        for key in ("target_attention", "false_attention", "target_rank"):
            if not isinstance(condition.get(key), list) or (
                    len(condition[key]) != len(path)):
                raise ValueError("Invalid readout length")
        if (any(type(value) not in (int, float) or not 0 <= value <= 1
                for key in ("target_attention", "false_attention")
                for value in condition[key]) or
                any(type(rank) is not int or not 1 <= rank <= count
                    for rank in condition["target_rank"])):
            raise ValueError("Invalid attention/rank readout")
        for key in ("first_value_norm", "first_state_norm"):
            value = condition.get(key)
            if ((not path and value is not None) or
                    (path and (type(value) not in (int, float) or
                               not math.isfinite(value) or value < 0))):
                raise ValueError("Invalid read/state norm")
        logits = condition.get("sampled_logits")
        if (logits is not None) != sampled:
            raise ValueError("Sampled intervention logit selection differs")
        if logits is not None:
            if (not isinstance(logits, list) or len(logits) != VOCAB_SIZE or
                    any(type(value) not in (int, float) or
                        not math.isfinite(value) for value in logits)):
                raise ValueError("Invalid sampled full logits")
            observed = max(range(SYMBOL_START, VOCAB_SIZE),
                           key=lambda token: logits[token])
            if observed != prediction:
                raise ValueError("Sampled logits and answer disagree")
            if name == "native" and primary_logits is not None and any(
                    abs(a - b) > TOLERANCE + TOLERANCE * abs(b)
                    for a, b in zip(logits, primary_logits, strict=True)):
                raise ValueError("Native logits differ from archived primary run")
        if name == "native" and prediction != primary_prediction:
            raise ValueError("Native answer differs from primary run")
        hits[name] = prediction == source["answer"]
    return hits


def audit_diagnostic(primary_dir: Path, result_dir: Path, root: Path,
                     *, runs: dict | None = None) -> dict:
    status = json.loads((result_dir / "status.json").read_text())
    _source_check(root, status)
    primary_report = json.loads((primary_dir / "report.json").read_text())
    if status.get("primary_report_sha256") != _sha(primary_dir / "report.json") or (
            status.get("primary_artifact_audit_sha256") != _sha(
                primary_dir / "artifact-audit.json") or
            status.get("primary_replay_audit_sha256") != _sha(
                primary_dir / "replay-audit.json") or
            status.get("benchmark_sha256") != _sha(result_dir / "benchmark.json") or
            status.get("primary_source_git_head") !=
            primary_report.get("source_git_head") or
            status.get("arm") != ARM or status.get("random_rep") != 0):
        raise ValueError("Diagnostic and primary provenance disagree")
    manifest = json.loads((primary_dir / "suite.json").read_text())
    suite = audit_manifest(manifest)
    _benchmark_check(result_dir / "benchmark.json", manifest,
                     status["diagnostic_source_sha256"],
                     _sha(primary_dir / "suite.json"))
    if status.get("manifest_sha256") != suite["manifest_sha256"]:
        raise ValueError("Diagnostic suite SHA differs")
    predictions = json.loads(gzip.decompress(
        (primary_dir / "predictions.json.gz").read_bytes()))
    replay = json.loads(gzip.decompress(
        (primary_dir / "replay-logits.json.gz").read_bytes()))
    primary_gates = json.loads(gzip.decompress(
        (primary_dir / "parser-gates.json.gz").read_bytes()))
    behavior = json.loads((primary_dir / "behavior-audit.json").read_text())
    parser = json.loads((primary_dir / "parser-audit.json").read_text())
    if runs is None:
        runs = json.loads((result_dir / "report.json").read_text())["runs"]
    if not isinstance(runs, dict) or set(runs) != {"0", "1"}:
        raise ValueError("Both diagnostic seeds required")
    scores = {}
    outcome_vectors = {}
    for rep in ("0", "1"):
        path = result_dir / f"diagnostic-rep{rep}.json.gz"
        metadata = runs[rep]
        if (metadata.get("sha256") != _sha(path) or
                metadata.get("bytes") != path.stat().st_size or
                metadata.get("checkpoint_sha256") != primary_report[
                    "training"][ARM][rep]["final_checkpoint"]["sha256"]):
            raise ValueError("Diagnostic file/checkpoint provenance invalid")
        archive = json.loads(gzip.decompress(path.read_bytes()))
        if archive.get("checkpoint_sha256") != metadata["checkpoint_sha256"] or (
                set(archive.get("panels", {})) != set(manifest["panels"])):
            raise ValueError("Diagnostic panel/checkpoint archive incomplete")
        scores[rep] = {}
        outcome_vectors[rep] = {}
        for name, episodes in manifest["panels"].items():
            rows = archive["panels"][name]
            if not isinstance(rows, list) or len(rows) != len(episodes):
                raise ValueError("Diagnostic item coverage incomplete")
            hits = {condition: [] for condition in CONDITIONS}
            samples = {row["index"]: row["logits"] for row in replay[
                "runs"][ARM][rep][name]}
            for index, (source, row) in enumerate(zip(episodes, rows, strict=True)):
                selected = index in {0, len(rows) // 2, len(rows) - 1}
                evidence = _validate_row(
                    source, row, index=index,
                    primary_prediction=predictions["runs"][ARM][rep][
                        "panels"][name][index]["prediction"],
                    primary_logits=samples.get(index), sampled=selected,
                    primary_gates=primary_gates["runs"][ARM][rep][name][index][
                        "logits"])
                for condition in CONDITIONS:
                    hits[condition].append(evidence[condition])
            scores[rep][name] = {}
            if name in ("factorial", "boundary_groups"):
                outcome_vectors[rep][name] = {}
            for condition in CONDITIONS:
                values = hits[condition]
                metrics = {"items": _fraction(values)}
                if name in ("factorial", "boundary_groups"):
                    if len(values) % 4:
                        raise ValueError("Incomplete diagnostic quartet")
                    metrics["groups"] = _fraction([
                        all(values[index:index + 4])
                        for index in range(0, len(values), 4)])
                    if name == "factorial":
                        outcome_vectors[rep][name][condition] = [
                            all(values[index:index + 4])
                            for index in range(0, len(values), 4)]
                if name == "boundary_groups":
                    metrics["marker_free"] = _fraction(values[3::4])
                    outcome_vectors[rep][name][condition] = values[3::4]
                scores[rep][name][condition] = metrics
    def rate(rep: str, panel: str, condition: str, metric: str) -> float:
        return scores[rep][panel][condition][metric]["accuracy"]

    rescue = all(
        rate(rep, panel, "gold", metric) - rate(rep, panel, "native", metric)
        >= .40 and all(
            rate(rep, panel, "gold", metric) - rate(rep, panel, control, metric)
            >= .35 for control in ("reversed", "count_random"))
        for rep in ("0", "1") for panel, metric in (
            ("factorial", "groups"), ("boundary_groups", "marker_free")))
    parser_failed_both = all(not parser["runs"][ARM][rep]["parser_qualified"]
                             for rep in ("0", "1"))
    raw_absolute_failed_both = all(
        not behavior["scores"][ARM][rep]["twohop_absolute"]
        for rep in ("0", "1"))
    eligible = (behavior["decisions"]["null_valid"] and
                behavior["decisions"]["oracle_twohop"] and
                raw_absolute_failed_both and
                parser_failed_both)
    intervals = {}
    for rep in ("0", "1"):
        intervals[rep] = {}
        for panel in ("factorial", "boundary_groups"):
            intervals[rep][panel] = {
                control: _paired_bootstrap(
                    outcome_vectors[rep][panel]["gold"],
                    outcome_vectors[rep][panel][control],
                    [rep, panel, control])
                for control in ("native", "reversed", "count_random")}
    return {"audit": "pass", "scope": "all_panel_gate_diagnosis_no_model",
            "arm": ARM, "seeds": 2,
            "manifest_sha256": suite["manifest_sha256"],
            "scores": scores, "paired_group_bootstrap": intervals,
            "entry_eligible": eligible,
            "rescue_threshold_met": rescue,
            "candidate_label_pending_numerical_replay": (
                "PARSER-GATING BOTTLENECK CANDIDATE" if eligible and rescue
                else "NO REGISTERED PARSER-GATE SUPPORT"),
            "sampled_intervention_checkpoint_replay": "pending"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", type=Path,
                        default=Path("results/TEACH-0014-v3"))
    parser.add_argument("--result-dir", type=Path,
                        default=Path("results/TEACH-0014-diagnostic"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_diagnostic(args.primary_dir, args.result_dir, args.root)
    raw = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(raw)
    else:
        print(raw, end="")


if __name__ == "__main__":
    main()
