"""Independent, no-model audit of the full TEACH-0015 finite assay."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess

import numpy as np

from scripts.teacher0015_clean_audit import (
    EXPECTED_MANIFESTS, SPLITS, _sha, read_manifests,
)
from scripts.teacher0015_finite_audit import (
    audit_control_permutations, audit_surface,
)


SOURCE_PATHS = (
    "docs/experiments/TEACH-0015-finite-registration.md",
    "src/voynich/workspace/teacher15_intervene.py",
    "src/voynich/workspace/teacher15_pairs.py",
    "scripts/teacher0015_finite_audit.py",
    "scripts/teacher0015_finite_result_audit.py",
    "scripts/teacher0015_finite_run.py",
    "scripts/teacher0015_finite_replay.py",
    "tests/test_teacher0015_finite_audit.py",
    "tests/test_teacher0015_intervene.py",
    "tests/test_teacher0015_finite_result_audit.py",
    "tests/test_teacher0015_finite_replay.py",
)
VECTOR_NAMES = (
    "base_native", "target_native", "donor_native", "same_key_native",
    "reverse_base", "final_donor", "random", "wrong_key",
    "wrong_source_native", "deranged", "deranged_source_native",
)
SURFACES = tuple((distractor, marked, order)
                 for distractor in (0, 1)
                 for marked in (True, False)
                 for order in (0, 1))
SAMPLE_SURFACES = (0, 512, 1023)
MAX_SECONDS = 7200
MAX_MPS_BYTES = 12 * 1024**3
MAX_ARTIFACT_BYTES = 2 * 1024**3
TOTAL_WORST_SURFACES = 3 * 2 * 2 * 128 * 8


def _canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _source_check(root: Path, status: dict) -> None:
    hashes = status.get("finite_source_sha256")
    head = status.get("finite_git_head")
    if not isinstance(hashes, dict) or set(hashes) != set(SOURCE_PATHS):
        raise ValueError("Finite assay source list incomplete")
    if not isinstance(head, str) or len(head) != 40:
        raise ValueError("Finite assay source commit missing")
    for name in SOURCE_PATHS:
        committed = subprocess.run(["git", "show", f"{head}:{name}"],
                                   cwd=root, check=True,
                                   capture_output=True).stdout
        if (hashlib.sha256(committed).hexdigest() != hashes[name] or
                _sha(root / name) != hashes[name]):
            raise ValueError(f"Finite assay source changed: {name}")


def _benchmark_check(benchmark: dict, hashes: dict, manifests: dict) -> None:
    timings = benchmark.get("timings_seconds")
    if (not isinstance(timings, dict) or set(timings) != {
            str(index) for index in range(8)} or
            any(not isinstance(values, list) or len(values) != 3 or
                any(type(value) not in (float, int) or
                    not math.isfinite(value) or value <= 0
                    for value in values) for values in timings.values())):
        raise ValueError("Finite benchmark surface timings malformed")
    slowest = max(statistics.median(values) for values in timings.values())
    projected = 1.75 * slowest * TOTAL_WORST_SURFACES + 300
    if (benchmark.get("finite_source_sha256") != hashes or
            benchmark.get("manifest_sha256") != EXPECTED_MANIFESTS or
            benchmark.get("suite_file_sha256") != {
                split: _sha(manifests[split]["path"]) for split in SPLITS} or
            benchmark.get("warmup_surfaces") != 8 or
            benchmark.get("timed_surfaces") != 24 or
            benchmark.get("slowest_type_median_seconds") != slowest or
            not math.isclose(benchmark.get("conservative_projected_seconds",
                                           math.inf), projected, rel_tol=1e-9) or
            benchmark.get("worst_case_surfaces") != TOTAL_WORST_SURFACES or
            benchmark.get("max_seconds") != MAX_SECONDS or
            benchmark.get("max_mps_bytes") != MAX_MPS_BYTES or
            benchmark.get("max_artifact_bytes") != MAX_ARTIFACT_BYTES or
            benchmark.get("peak_sampled_mps_allocated_bytes", math.inf)
            > MAX_MPS_BYTES or projected >= MAX_SECONDS or
            benchmark.get("admitted") is not True):
        raise ValueError("Finite benchmark resource gate failed")


def _bootstrap(group_counts: list[tuple[int, int]], identity: object) -> dict:
    if len(group_counts) != 128:
        raise ValueError("Finite bootstrap requires128 logical groups")
    seed = int(_canonical(["TEACH-0015-finite-bootstrap", identity])[:16], 16)
    rng = random.Random(seed)
    rates = []
    for _ in range(4000):
        sample = [group_counts[rng.randrange(128)] for _ in range(128)]
        denominator = sum(pair[1] for pair in sample)
        rates.append(sum(pair[0] for pair in sample) / denominator
                     if denominator else 0.0)
    rates.sort()
    return {"lower_95": rates[99], "upper_95": rates[3899],
            "groups": 128, "draws": 4000, "seed": seed}


def score_rows(rows: list[dict], identity: object) -> dict:
    """Score every attempted recipient and resample whole logical groups."""
    if len(rows) != 3072 or any(row["g"] != index % 3
                                for index, row in enumerate(rows)):
        raise ValueError("Finite recipient denominator/order differs")
    names = ("transfer", "triples", "changed_g_non_injection",
             "reverse_eligible", "reverse_full", "marked", "marker_free",
             "wrong_key", "deranged", "random", "same_key", "final_donor",
             "base_clean", "target_clean")
    counts = {name: [] for name in names}
    surface_counts = [{name: [0, 0] for name in names} for _ in SURFACES]
    per_group_counts = []
    for group_index in range(128):
        group = rows[group_index * 24:(group_index + 1) * 24]
        group_counts = {name: [0, 0] for name in names}

        def add(name: str, hit: bool, surface: int) -> None:
            group_counts[name][0] += int(hit)
            group_counts[name][1] += 1
            surface_counts[surface][name][0] += int(hit)
            surface_counts[surface][name][1] += 1

        for surface in range(8):
            triple = group[surface * 3:(surface + 1) * 3]
            add("triples", all(row["transfer_prediction"] ==
                               row["target_answer"] for row in triple), surface)
            for row in triple:
                target = row["target_answer"]
                add("transfer", row["transfer_prediction"] == target, surface)
                if row["g"] in (1, 2):
                    add("changed_g_non_injection",
                        row["transfer_prediction"] != row["fixed_donor_answer"],
                        surface)
                eligible = (row["base_prediction"] == row["base_answer"]
                            and row["target_prediction"] == target)
                if eligible:
                    add("reverse_eligible", row["reverse_prediction"] ==
                        row["base_answer"], surface)
                add("reverse_full", row["reverse_prediction"] ==
                    row["base_answer"], surface)
                add("marked" if row["marked"] else "marker_free",
                    row["transfer_prediction"] == target, surface)
                for name in ("wrong_key", "deranged", "random", "same_key",
                             "final_donor"):
                    add(name, row[f"{name}_prediction"] == target, surface)
                add("base_clean", row["base_prediction"] ==
                    row["base_answer"], surface)
                add("target_clean", row["target_prediction"] == target, surface)
        for name in names:
            counts[name].append(tuple(group_counts[name]))
        per_group_counts.append({"group_index": group_index,
                                 "group_id": group[0].get("group_id"),
                                 "counts": group_counts})
    expected = {"transfer": 3072, "triples": 1024,
                "changed_g_non_injection": 2048, "reverse_full": 3072,
                "marked": 1536, "marker_free": 1536,
                "wrong_key": 3072, "deranged": 3072, "random": 3072,
                "same_key": 3072, "final_donor": 3072,
                "base_clean": 3072, "target_clean": 3072}
    scores = {}
    for name, per_group in counts.items():
        hits = sum(pair[0] for pair in per_group)
        denominator = sum(pair[1] for pair in per_group)
        if name in expected and denominator != expected[name]:
            raise ValueError(f"Finite {name} denominator differs")
        scores[name] = {"correct": hits, "total": denominator,
                        "accuracy": hits / denominator if denominator else 0.0,
                        "group_bootstrap": (_bootstrap(per_group, [identity, name])
                                            if denominator else None)}
    transfer = scores["transfer"]["accuracy"]
    margins = {name: transfer - scores[name]["accuracy"]
               for name in ("wrong_key", "deranged", "random")}
    qualified = (
        transfer >= .75 and scores["triples"]["accuracy"] >= .60 and
        scores["changed_g_non_injection"]["accuracy"] >= .90 and
        scores["reverse_eligible"]["total"] > 0 and
        scores["reverse_eligible"]["accuracy"] >= .70 and
        scores["marked"]["accuracy"] >= .70 and
        scores["marker_free"]["accuracy"] >= .70 and
        all(margin >= .35 for margin in margins.values())
    )
    per_surface = []
    for index, coordinates in enumerate(SURFACES):
        per_surface.append({
            "surface_index": index,
            "distractor": coordinates[0], "marked": coordinates[1],
            "order": coordinates[2],
            "scores": {name: {"correct": pair[0], "total": pair[1],
                              "accuracy": pair[0] / pair[1] if pair[1] else 0.0}
                       for name, pair in surface_counts[index].items()},
        })
    return {"scores": scores, "per_surface": per_surface,
            "per_group_counts": per_group_counts,
            "control_margins": margins,
            "candidate_portable_state_pending_replay": qualified}


def _screen_predictions(screen_dir: Path, arm: str, rep: str,
                        split: str) -> dict[str, int]:
    source = screen_dir / f"predictions-{arm}-rep{rep}.json.gz"
    archive = json.loads(gzip.decompress(source.read_bytes()))
    return {row["render_id"]: row["prediction"]
            for row in archive["splits"][split]["rows"]}


def audit_finite(primary_dir: Path, suite_dir: Path, screen_dir: Path,
                 result_dir: Path, root: Path, *, runs: dict | None = None
                 ) -> dict:
    status = json.loads((result_dir / "status.json").read_text())
    _source_check(root, status)
    manifests = read_manifests(suite_dir)
    manifest_paths = {split: {"path": suite_dir / f"{split}.json"}
                      for split in SPLITS}
    suite_hashes = {split: _sha(suite_dir / f"{split}.json")
                    for split in SPLITS}
    benchmark = json.loads((result_dir / "benchmark.json").read_text())
    _benchmark_check(benchmark, status["finite_source_sha256"], manifest_paths)
    if (status.get("manifest_sha256") != EXPECTED_MANIFESTS or
            status.get("suite_file_sha256") != suite_hashes or
            status.get("benchmark_sha256") != _sha(result_dir / "benchmark.json") or
            status.get("screen_report_sha256") != _sha(screen_dir / "report.json") or
            status.get("screen_decision_audit_sha256") != _sha(
                screen_dir / "decision-audit.json") or
            status.get("screen_replay_audit_sha256") != _sha(
                screen_dir / "replay-audit.json") or
            status.get("primary_report_sha256") != _sha(primary_dir / "report.json")):
        raise ValueError("Finite entry provenance differs")
    screen_decision = json.loads((screen_dir / "decision-audit.json").read_text())
    screen_replay = json.loads((screen_dir / "replay-audit.json").read_text())
    arms = screen_decision["candidate_competent_arms_pending_replay"]
    if (screen_decision.get("audit") != "pass" or
            screen_replay.get("audit") != "pass" or
            screen_replay.get("clean_competent_arms") != arms or
            status.get("eligible_arms") != arms or not arms):
        raise ValueError("Finite fresh competence entry differs")
    if runs is None:
        runs = json.loads((result_dir / "report.json").read_text())["runs"]
    if set(runs) != set(arms):
        raise ValueError("Finite run arm coverage differs")
    primary_report = json.loads((primary_dir / "report.json").read_text())
    results = {}
    total_surfaces = 0
    max_identity = 0.0
    max_final = 0.0
    for arm in arms:
        if set(runs[arm]) != {"0", "1"}:
            raise ValueError("Finite run replicate coverage differs")
        results[arm] = {}
        for rep in ("0", "1"):
            run = runs[arm][rep]
            if run.get("checkpoint_sha256") != primary_report[
                    "training"][arm][rep]["final_checkpoint"]["sha256"]:
                raise ValueError("Finite checkpoint hash differs")
            results[arm][rep] = {}
            for split in SPLITS:
                files = run["splits"][split]
                row_path = result_dir / f"rows-{arm}-rep{rep}-{split}.json.gz"
                vec_path = result_dir / f"vectors-{arm}-rep{rep}-{split}.npy"
                if (files["rows_sha256"] != _sha(row_path) or
                        files["vectors_sha256"] != _sha(vec_path) or
                        files["rows_bytes"] != row_path.stat().st_size or
                        files["vectors_bytes"] != vec_path.stat().st_size):
                    raise ValueError("Finite archived file hash/size differs")
                archive = json.loads(gzip.decompress(row_path.read_bytes()))
                if (archive.get("checkpoint_sha256") != run[
                        "checkpoint_sha256"] or
                        archive.get("split") != split or
                        len(archive.get("rows", [])) != 1024):
                    raise ValueError("Finite surface archive incomplete")
                vectors = np.load(vec_path, mmap_mode="r", allow_pickle=False)
                if (vectors.dtype != np.float32 or vectors.shape != (
                        128, 8, 3, len(VECTOR_NAMES), 512) or
                        not np.isfinite(vectors).all()):
                    raise ValueError("Finite archived vector tensor invalid")
                manifest_groups = manifests[split]["groups"]
                wrong, deranged = audit_control_permutations(manifest_groups)
                screen = _screen_predictions(screen_dir, arm, rep, split)
                reconstructed = []
                for group_index, group in enumerate(manifest_groups):
                    for surface_index, (distractor, marked, order) in enumerate(
                            SURFACES):
                        index = group_index * 8 + surface_index
                        stored = archive["rows"][index]
                        if len(stored) != 3:
                            raise ValueError("Finite surface recipient count differs")
                        rows = []
                        for recipient, compact in enumerate(stored):
                            row = dict(compact)
                            row["replacement_vectors"] = {
                                name: vectors[group_index, surface_index,
                                              recipient, vector_index].tolist()
                                for vector_index, name in enumerate(VECTOR_NAMES)}
                            rows.append(row)
                        checked = audit_surface(
                            group, rows, distractor=distractor, marked=marked,
                            order=order, wrong_group=manifest_groups[wrong[group_index]],
                            deranged_group=manifest_groups[deranged[group_index]],
                            full_logits=index in SAMPLE_SURFACES)
                        max_identity = max(max_identity,
                                           checked["max_identity_error"])
                        max_final = max(max_final,
                                        checked["max_final_donor_error"])
                        for row in rows:
                            if (screen.get(row["base_render_id"]) != row[
                                    "base_prediction"] or
                                    screen.get(row["target_render_id"]) != row[
                                        "target_prediction"] or
                                    screen.get(row["donor_render_id"]) != row[
                                        "donor_prediction"]):
                                raise ValueError("Finite native/screen answer differs")
                        reconstructed.extend(rows)
                        total_surfaces += 1
                results[arm][rep][split] = score_rows(
                    reconstructed, [arm, rep, split])
    return {"audit": "pass", "scope": "full_finite_state_grid_no_model",
            "manifest_sha256": EXPECTED_MANIFESTS,
            "eligible_arms": arms, "surfaces": total_surfaces,
            "max_identity_error": max_identity,
            "max_final_donor_error": max_final,
            "scores": results,
            "candidate_arms_pending_checkpoint_replay": [
                arm for arm in arms if all(results[arm][rep]["confirmation"][
                    "candidate_portable_state_pending_replay"]
                    for rep in ("0", "1"))],
            "sampled_checkpoint_replay": "pending"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", type=Path,
                        default=Path("results/TEACH-0014-v3"))
    parser.add_argument("--suite-dir", type=Path,
                        default=Path("outputs/TEACH-0015"))
    parser.add_argument("--screen-dir", type=Path,
                        default=Path("results/TEACH-0015-screen"))
    parser.add_argument("--result-dir", type=Path,
                        default=Path("results/TEACH-0015-finite"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_finite(args.primary_dir, args.suite_dir,
                          args.screen_dir, args.result_dir, args.root)
    raw = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(raw)
    else:
        print(raw, end="")


if __name__ == "__main__":
    main()
