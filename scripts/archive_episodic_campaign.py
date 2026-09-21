"""Archive a completed EXP-0012/0013 campaign without copying arrays or weights.

Usage: python scripts/archive_episodic_campaign.py --out outputs/EXP-0012
The registered rank-four gate is fixed here before confirmation. Missing causal
metrics never become zero or a pass. This command does not fit or select models.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import statistics
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ("shuffled_targets", "pca", "output_jacobian", "random_norm_0", "random_norm_1", "random_norm_2")
FAMILIES = ("gru_fresh", "signed_delta_fresh")
METRIC = "donor_conditional_after_first_bits"
SUBSET = "immediate_matched_pairs"
BOOTSTRAP_SEED = 130013
BOOTSTRAP_SAMPLES = 2000


def fail(message):
    raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    def nonfinite(value):
        fail(f"Nonfinite JSON constant {value} in {path}")
    with Path(path).open() as handle:
        result = json.load(handle, parse_constant=nonfinite)
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def mapping(value, name):
    if not isinstance(value, dict):
        fail(f"Expected object: {name}")
    return value


def safe_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        fail(f"Unsafe artifact path: {relative}")
    result = root / relative
    if not result.is_file() or not result.resolve().is_relative_to(root.resolve()):
        fail(f"Missing or external artifact: {result}")
    return result


def metric(report, method, name=METRIC):
    """Read independent-key statistics and validate counts without using row means."""
    value = report.get("confirmation", {}).get(method, {}).get(SUBSET, {}).get(name)
    if not isinstance(value, dict):
        return None, f"Missing {method}/{SUBSET}/{name}"
    means, counts = value.get("per_key_means"), value.get("per_key_counts")
    if not isinstance(means, dict) or not isinstance(counts, dict) or set(means) != set(counts):
        return None, f"Malformed per-key means/counts: {method}/{name}"
    if not means or any(not finite(x) for x in means.values()):
        return None, f"No finite matched-key values: {method}/{name}"
    if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in counts.values()):
        return None, f"Invalid key counts: {method}/{name}"
    if value.get("rows") != sum(counts.values()) or value.get("keys") != len(means):
        return None, f"Key counts disagree with aggregate: {method}/{name}"
    average = statistics.mean(means.values())
    if not finite(value.get("equal_key_mean")) or not math.isclose(
            average, value["equal_key_mean"], rel_tol=1e-7, abs_tol=1e-10):
        return None, f"Equal-key mean disagrees: {method}/{name}"
    return {"per_key_means": means, "per_key_counts": counts,
            "mean": average, "rows": sum(counts.values()), "keys": len(means)}, None


def check_numeric(report):
    numerical = report.get("numerical_controls", {})
    restoration = report.get("model_restoration", {})
    missing, failures = [], []
    for key in ("parameters_and_buffers_unchanged", "parameters_and_buffers_restored"):
        if key not in restoration:
            missing.append(f"Missing restoration field {key}")
        elif restoration[key] is not True:
            failures.append(f"Restoration field {key} is not true")
    for key, limit in (("identity_state_max_abs", 1e-6), ("identity_joint_max_abs", 1e-6),
                       ("state_encoding_replay_max_abs", 1e-6),
                       ("maximum_joint_normalization_error", 1e-5)):
        if not finite(numerical.get(key)):
            missing.append(f"Missing/nonfinite numerical field {key}")
        elif not 0 <= numerical[key] <= limit:
            failures.append(f"{key}={numerical[key]} exceeds {limit}")
    return missing, failures


def seed_gate(report):
    """All registered per-seed requirements, with explicit inconclusive controls."""
    missing, failures = check_numeric(report)
    if report.get("key_disjointness_verified") is not True:
        missing.append("Independent task-key split verification unavailable")
    teacher = report.get("teacher_oracle_joint_kl_bits")
    if not finite(teacher):
        missing.append("Missing/nonfinite teacher oracle H3 KL")
    elif teacher > .30:
        failures.append("Teacher oracle H3 KL exceeds 0.30 bits")
    diagnostics = {"teacher_oracle_joint_kl_bits": teacher}
    fields = {}
    for method, name in (("unchanged", METRIC), ("learned", METRIC),
                         ("learned", "recipient_immediate_tv"),
                         ("learned", "commuting_symmetric_kl_bits")):
        value, error = metric(report, method, name)
        if error:
            missing.append(error)
        else:
            fields[(method, name)] = value
    base, learned = fields.get(("unchanged", METRIC)), fields.get(("learned", METRIC))
    reduction = None
    if base:
        diagnostics.update(matched_rows=base["rows"], matched_keys=base["keys"], unchanged_kl_bits=base["mean"])
        if base["rows"] < 32 or base["keys"] < 4:
            missing.append("Fewer than 32 matched pairs or four independent keys")
        if base["mean"] < .01:
            missing.append("Unchanged beyond-first KL below qualifying 0.01-bit gap")
        if any(value["per_key_counts"] != base["per_key_counts"] for value in fields.values()):
            missing.append("Metrics do not describe the same matched pairs/keys")
    if base and learned:
        absolute = base["mean"] - learned["mean"]
        reduction = absolute / base["mean"] if base["mean"] > 0 else None
        diagnostics.update(learned_kl_bits=learned["mean"], absolute_reduction_bits=absolute,
                           relative_reduction=reduction)
        if absolute < .005 or reduction is None or reduction < .20:
            failures.append("Learned intervention misses 0.005-bit or 20-percent reduction")
    for name, limit in (("recipient_immediate_tv", .02), ("commuting_symmetric_kl_bits", .02)):
        value = fields.get(("learned", name))
        if value:
            diagnostics[name] = value["mean"]
            if value["mean"] > limit:
                failures.append(f"Learned {name} exceeds {limit}")
    untrained = report.get("untrained_control", {})
    specificity = {"status": "inconclusive"}
    if untrained.get("status") != "completed" or not isinstance(untrained.get("report"), dict):
        missing.append("Untrained control missing or incomplete")
    else:
        control = untrained["report"]
        problems, numeric_failures = check_numeric(control)
        missing.extend("Untrained: " + x for x in problems)
        failures.extend("Untrained: " + x for x in numeric_failures)
        ub, eb = metric(control, "unchanged")
        ul, el = metric(control, "learned")
        missing.extend("Untrained: " + x for x in (eb, el) if x)
        if ub and ul:
            ur = (ub["mean"] - ul["mean"]) / ub["mean"] if ub["mean"] > 0 else None
            specificity.update(unchanged_kl_bits=ub["mean"], learned_kl_bits=ul["mean"],
                               relative_reduction=ur, matched_rows=ub["rows"], matched_keys=ub["keys"])
            if ub["per_key_counts"] != ul["per_key_counts"]:
                missing.append("Untrained intervention counts do not match its baseline")
            elif ub["rows"] < 32 or ub["keys"] < 4 or ub["mean"] < .01:
                missing.append("Untrained control lacks a comparable qualifying gap; specificity inconclusive")
            elif reduction is None or ur is None:
                missing.append("Relative reduction unavailable for specificity comparison")
            elif ur >= reduction:
                specificity["status"] = "failed"
                failures.append("Qualifying untrained control matches/exceeds trained relative reduction")
            else:
                specificity["status"] = "passed"
    diagnostics["untrained_specificity"] = specificity
    return {"status": "passed" if not missing and not failures else "not_established",
            "missing_or_ineligible": missing, "failed_requirements": failures, **diagnostics}


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap(values, seed=BOOTSTRAP_SEED):
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP_SAMPLES)]
    return {"mean": statistics.mean(values), "lo95": quantile(means, .025),
            "hi95": quantile(means, .975), "independent_keys": len(values),
            "resamples": BOOTSTRAP_SAMPLES, "seed": seed}


def family_gate(reports, seeds):
    per_seed = {str(seed): seed_gate(reports[seed]) if seed in reports else {
        "status": "not_established", "missing_or_ineligible": ["Missing rank-four seed report"],
        "failed_requirements": []} for seed in seeds}
    comparisons, errors = {}, []
    if len(seeds) != 3 or len(set(seeds)) != 3:
        errors.append("Primary claim requires exactly three distinct trained seeds")
    for control in CONTROLS:
        pairs = []
        for seed in seeds:
            if seed not in reports:
                errors.append(f"Missing seed {seed} for control {control}")
                continue
            learned, le = metric(reports[seed], "learned")
            baseline, be = metric(reports[seed], control)
            if le or be:
                errors.extend(f"Seed {seed}: {x}" for x in (le, be) if x)
                continue
            if learned["per_key_counts"] != baseline["per_key_counts"]:
                errors.append(f"Seed {seed}: learned and {control} pair counts differ")
                continue
            pairs.append((learned["per_key_means"], baseline["per_key_means"]))
        if len(pairs) != len(seeds):
            comparisons[control] = {"status": "not_established", "reason": "Missing matched seed metrics"}
            continue
        common = sorted(set.intersection(*(set(left) for left, _ in pairs)))
        if len(common) < 4:
            comparisons[control] = {"status": "not_established", "reason": "Fewer than four keys in all three seeds",
                                    "common_keys": common}
            continue
        differences = {key: statistics.mean(left[key] - right[key] for left, right in pairs) for key in common}
        interval = bootstrap(list(differences.values()))
        passed = interval["hi95"] < 0 and interval["mean"] <= -.005
        comparisons[control] = {"status": "passed" if passed else "not_established", **interval,
                                "seed_averaged_per_key_differences": differences,
                                "excluded_key_counts_by_seed": [len(left) - len(common) for left, _ in pairs]}
    passed = not errors and all(x["status"] == "passed" for x in per_seed.values()) and all(
        x["status"] == "passed" for x in comparisons.values())
    return {"status": "established" if passed else "not_established", "rank": 4,
            "per_seed": per_seed, "control_comparisons_learned_minus_control": comparisons,
            "missing_or_ineligible": errors,
            "claim": "qualified predictive-state transplantation, not latent identification or semantics"}


def descriptive(report):
    result = {"run": report["run"], "rank": report["rank"],
              "teacher_oracle_joint_kl_bits": report.get("teacher_oracle_joint_kl_bits"), "methods": {}}
    for name in ("unchanged", "learned", "future_jacobian", "delayed_jacobian"):
        value, error = metric(report, name)
        result["methods"][name] = {"unavailable": error} if error else value
    result["future_sensitivity"] = report.get("future_sensitivity", {})
    result["held_out_affine_closure"] = report.get("held_out_affine_closure", {})
    return result


def resource_summary(path, summaries, histories):
    count, errors, peak, final_time = 0, 0, 0, 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            errors += row.get("memory_sampling_error") is not None
            value = row.get("sum_process_group_rss_bytes")
            if finite(value) and value >= 0:
                peak = max(peak, value)
            if finite(row.get("elapsed_seconds")):
                final_time = max(final_time, row["elapsed_seconds"])
    allocations = [row.get("resources", {}).get("mps_driver_bytes")
                   for summary, history in zip(summaries, histories, strict=True) for row in [summary, *history]]
    return {"supervisor_samples": count, "memory_sampling_errors": errors,
            "resource_log_sha256": digest(path),
            "peak_sampled_sum_process_group_rss_bytes": peak, "last_supervisor_sample_seconds": final_time,
            "max_reported_mps_driver_bytes": max((x for x in allocations if finite(x)), default=None),
            "summed_run_elapsed_seconds": sum(s["elapsed_seconds"] for s in summaries),
            "sampled_training_targets": sum(s["sampled_targets"] for s in summaries),
            "accounting": "Sampled summed process RSS is not physical/GPU peak. MPS and RSS overlap; never add them."}


def archive(out, results):
    out, results = Path(out).resolve(), Path(results).resolve()
    # Collect and validate every required input before creating either destination.
    complete = mapping(read_json(safe_path(out, "complete.json")), "complete")
    status = mapping(read_json(safe_path(out, "supervisor/status.json")), "supervisor status")
    analysis_status = mapping(read_json(safe_path(out, "analysis-supervisor/status.json")), "analysis status")
    supervisor = mapping(read_json(safe_path(out, "supervisor/manifest.json")), "supervisor manifest")
    launch_source = mapping(supervisor.get("source"), "launch source")
    if (launch_source.get("git_dirty") is not False or not isinstance(launch_source.get("git_commit"), str)
            or not isinstance(launch_source.get("source_sha256"), dict)):
        fail("Launch lacks clean, versioned scientific-source provenance")
    if launch_source["source_sha256"].get("scripts/archive_episodic_campaign.py") != digest(Path(__file__)):
        fail("Archival decision code differs from the frozen campaign source")
    if status.get("state") != "completed" or not status.get("jobs") or any(
            job.get("state") != "completed" or job.get("returncode") != 0 for job in status["jobs"].values()):
        fail("All supervisor workers must have completed successfully")
    if analysis_status.get("state") != "completed" or not analysis_status.get("jobs") or any(
            job.get("state") != "completed" or job.get("returncode") != 0 for job in analysis_status["jobs"].values()):
        fail("Analysis supervisor must have completed successfully")
    if complete.get("no_paid_api") is not True or complete.get("no_manuscript_inputs") is not True:
        fail("Completion marker lacks registered scope assertions")
    if complete.get("source") != supervisor.get("source"):
        fail("Completion and launch source provenance differ")
    manifest = mapping(read_json(safe_path(out, "data_manifest.json")), "data manifest")
    config = mapping(manifest.get("config"), "registered config")
    seeds = config.get("model_seeds")
    if (not isinstance(seeds, list) or len(seeds) != 3 or len(set(seeds)) != 3 or
            any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds)):
        fail("Expected the three registered model seeds")
    expected_runs = []
    for condition in config["conditions"]:
        name = condition["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_]+", name):
            fail("Unsafe or missing condition name")
        expected_runs.extend(f"{name}-s{seed}" for seed in seeds)
    if len(set(expected_runs)) != len(expected_runs):
        fail("Duplicate registered runs")
    confirmation = mapping(read_json(safe_path(out, "confirmation_report.json")), "confirmation report")
    if {x.get("run") for x in confirmation.get("results", [])} != set(expected_runs):
        fail("Confirmation report does not contain every registered run")
    explicit = mapping(read_json(safe_path(out, "explicit_report.json")), "explicit report")
    if len(explicit.get("tasks", [])) != 6 * config["explicit_tasks_per_family"]:
        fail("Explicit comparator did not complete every registered task")
    files12 = {name: name for name in ("complete.json", "data_manifest.json", "confirmation_report.json",
                                      "confirmation_baselines.json", "explicit_report.json")}
    files12.update({"supervisor/manifest.json": "supervisor_manifest.json",
                    "supervisor/status.json": "supervisor_status.json",
                    "analysis-supervisor/manifest.json": "analysis_supervisor_manifest.json",
                    "analysis-supervisor/status.json": "analysis_supervisor_status.json"})
    weights, summaries, histories = [], [], []
    for name in expected_runs:
        for kind in ("manifest", "history", "summary", "confirmation"):
            files12[f"runs/{name}/{kind}.json"] = f"{name}__{kind}.json"
        summary = mapping(read_json(safe_path(out, f"runs/{name}/summary.json")), name)
        if summary.get("data_manifest_sha256") != digest(out / "data_manifest.json"):
            fail(f"Run manifest identity mismatch: {name}")
        checkpoint = safe_path(out, f"runs/{name}/best.pt")
        actual = digest(checkpoint)
        if actual != summary.get("best_checkpoint_sha256"):
            fail(f"Checkpoint hash mismatch: {name}")
        weights.append({"path": str(checkpoint.relative_to(out)), "sha256": actual,
                        "bytes": checkpoint.stat().st_size, "archived": False})
        if not finite(summary.get("elapsed_seconds")) or not isinstance(summary.get("sampled_targets"), int):
            fail(f"Missing run resource accounting: {name}")
        summaries.append(summary)
        history = read_json(safe_path(out, f"runs/{name}/history.json"))
        if not isinstance(history, list) or not history:
            fail(f"Missing run history: {name}")
        histories.append(history)
    causal_summary = mapping(read_json(safe_path(out, "causal/summary.json")), "causal summary")
    expected_causal = {f"{family}-s{seed}-r{rank}" for family in FAMILIES for seed in seeds for rank in (4, 16)}
    if {x.get("run") for x in causal_summary.get("audits", [])} != expected_causal:
        fail("Causal phase did not complete every registered audit")
    files13 = {"causal/summary.json": "campaign_summary.json", "causal/task_manifest.json": "task_manifest.json"}
    reports, causal_artifacts = {}, []
    for name in sorted(expected_causal):
        relative = f"causal/{name}.json"
        report = mapping(read_json(safe_path(out, relative)), name)
        if (report.get("schema_version") != 1 or report.get("run") != name or
                report.get("rank") != int(name.rsplit("r", 1)[1]) or
                report.get("horizon") != 3 or report.get("alphabet") != 4):
            fail(f"Unsupported or inconsistent causal report schema: {name}")
        reports[name] = report
        artifact = safe_path(out, f"causal/{name}.pt")
        actual = digest(artifact)
        if report.get("artifact_sha256") != actual:
            fail(f"Causal artifact hash mismatch: {name}")
        causal_artifacts.append({"path": str(artifact.relative_to(out)), "sha256": actual,
                                 "bytes": artifact.stat().st_size, "archived": False})
        files13[relative] = f"{name}.json"
    for relative in set(files12) | set(files13):
        read_json(safe_path(out, relative))
    resources = resource_summary(safe_path(out, "supervisor/resources.jsonl"), summaries, histories)
    outcomes13 = {"schema_version": 1, "experiment": "EXP-0013", "primary_families": {
        family: family_gate({seed: reports[f"{family}-s{seed}-r4"] for seed in seeds}, seeds) for family in FAMILIES},
        "aggregation": "Matched subset; equal-key means; seed differences averaged within keys present in all three seeds.",
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "interval": "two-sided percentile 95%"},
        "specificity_rule": "Untrained gap >=.01 bits, >=32 pairs and >=4 keys is comparable. Smaller gap is inconclusive and blocks the strict overall gate.",
        "secondary_descriptive_only": [descriptive(reports[name]) for name in sorted(reports)],
        "limitations": ["Rank16 and Fisher candidates cannot promote a primary claim.",
                        "Immediate-output preservation does not establish a semantic nuisance variable.",
                        "A positive result would concern model behavior, not historical decipherment."]}
    outcomes12 = {"schema_version": 1, "experiment": "EXP-0012", "completed_runs": len(expected_runs),
                  "registered_contrasts": confirmation.get("contrasts", []), "resources": resources,
                  "scope": "Predictive synthetic comparisons; explicit fitting uses extra per-task observations; no decipherment."}
    for experiment in ("EXP-0012", "EXP-0013"):
        target = results / experiment
        if target.exists():
            fail(f"Refusing to overwrite existing archive: {target}")
    results.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".episodic-archive-", dir=results) as temporary:
        stage = Path(temporary)
        for experiment, files, outcome, artifacts in (("EXP-0012", files12, outcomes12, weights),
                                                      ("EXP-0013", files13, outcomes13, causal_artifacts)):
            target = stage / experiment
            target.mkdir()
            for source, destination in files.items():
                shutil.copy2(out / source, target / destination)
            write_json(target / "summary.json", outcome)
            write_json(target / "local_artifact_hashes.json", {"source_root": str(out), "artifacts": artifacts})
            (target / "README.md").write_text(
                f"# {experiment} compact archive\n\n"
                "Read `summary.json` for the registered outcomes and full reports for all metrics. "
                "Raw arrays and model/basis weights remain local; their digests are retained. "
                "Missing or ineligible causal evidence is not established, never a pass. "
                "Rank16 and Fisher results are descriptive. No artifact establishes a decipherment.\n")
            hashes = {path.name: digest(path) for path in sorted(target.iterdir()) if path.is_file()}
            write_json(target / "archive_manifest.json", {"schema_version": 1,
                       "created_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(out),
                       "archiver_sha256": digest(Path(__file__)), "sha256": hashes})
        (stage / "EXP-0012").rename(results / "EXP-0012")
        (stage / "EXP-0013").rename(results / "EXP-0013")
    return {"archives": [str(results / x) for x in ("EXP-0012", "EXP-0013")],
            "primary_causal_status": {family: value["status"] for family, value in outcomes13["primary_families"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/EXP-0012")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    print(json.dumps(archive(args.out, args.results), indent=2))


if __name__ == "__main__":
    main()
