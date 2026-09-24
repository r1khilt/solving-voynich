"""No-model clean-competence audit for the conditional TEACH-0015 assay."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess

from scripts.teacher0015_suite_audit import audit_manifest, audit_splits


ARMS = ("latent_rows_answer", "latent_rows_causal", "latent_rows_edge_aux")
SPLITS = ("discovery", "confirmation")
EXPECTED_MANIFESTS = {
    "discovery": "ff044ec42ceaaea180a4cde43bbb1bd0dd73dc0c4f53c476876855831402261c",
    "confirmation": "941cce744278d677655d01ef71f4dfa4943352e7e477ce66efcbd5233c4db975",
}
EXPECTED_EXPOSURES = (
    "09930778461abea7618db8d31eb9aa82c610be0e05ca4d87493a7ff9834c98af",
    "f4c47f734810e6e7997c69d72af33cb433a16d5a9fa7240b4ffc8b1c6e8e50a6",
    "6af176d376921caccc0d40641002b94a462b42670fc4794f5b354457d17fa827",
)
SOURCE_PATHS = (
    "docs/experiments/TEACH-0015-clean-screen-registration.md",
    "src/voynich/workspace/teacher15_tasks.py",
    "scripts/teacher0015_suite_audit.py",
    "scripts/teacher0015_clean_run.py",
    "scripts/teacher0015_clean_audit.py",
    "scripts/teacher0015_clean_replay.py",
    "tests/test_teacher0015_clean_audit.py",
    "tests/test_teacher0015_clean_replay.py",
)
MAX_SECONDS = 3600
MAX_MPS_BYTES = 12 * 1024**3
MAX_ARTIFACT_BYTES = 1024**3
SAMPLE_INDICES = (0, 4224, 8447)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def eligible_arms(primary_dir: Path) -> tuple[str, ...]:
    behavior = json.loads((primary_dir / "behavior-audit.json").read_text())
    parser = json.loads((primary_dir / "parser-audit.json").read_text())
    if not (behavior["decisions"]["null_valid"] and
            behavior["decisions"]["oracle_twohop"]):
        return ()
    return tuple(arm for arm in ARMS if all(
        behavior["scores"][arm][str(rep)]["twohop_absolute"] and
        parser["runs"][arm][str(rep)]["parser_qualified"]
        for rep in (0, 1)))


def read_manifests(suite_dir: Path) -> dict[str, dict]:
    manifests = {split: json.loads((suite_dir / f"{split}.json").read_text())
                 for split in SPLITS}
    for split, manifest in manifests.items():
        audited = audit_manifest(manifest)
        if audited["manifest_sha256"] != EXPECTED_MANIFESTS[split] or (
                len(manifest["groups"]) != 128):
            raise ValueError("TEACH-0015 clean manifest hash/count differs")
    cross = audit_splits(manifests["discovery"],
                         manifests["confirmation"])
    saved = json.loads((suite_dir / "suite-audit.json").read_text())
    if (saved.get("audit") != "pass" or
            saved.get("discovery_sha256") != cross["discovery_sha256"] or
            saved.get("confirmation_sha256") != cross[
                "confirmation_sha256"] or
            tuple(saved.get("teacher14_exposure_sha256", ())) !=
            EXPECTED_EXPOSURES):
        raise ValueError("TEACH-0015 prior exposure audit differs")
    return manifests


def _source_check(root: Path, status: dict) -> None:
    hashes = status.get("screen_source_sha256")
    head = status.get("screen_git_head")
    if (not isinstance(hashes, dict) or set(hashes) != set(SOURCE_PATHS) or
            not isinstance(head, str) or len(head) != 40):
        raise ValueError("TEACH-0015 screen source provenance incomplete")
    for name in SOURCE_PATHS:
        committed = subprocess.run(["git", "show", f"{head}:{name}"],
                                   cwd=root, check=True,
                                   capture_output=True).stdout
        if (hashlib.sha256(committed).hexdigest() != hashes[name] or
                _sha(root / name) != hashes[name]):
            raise ValueError(f"TEACH-0015 screen source differs: {name}")


def _fraction(hits: list[bool]) -> dict:
    return {"correct": sum(hits), "total": len(hits),
            "accuracy": sum(hits) / len(hits) if hits else 0.0}


def _bootstrap(per_group: list[float], identity: object,
               draws: int = 4000) -> dict:
    if not per_group:
        raise ValueError("Clean-screen group-bootstrap denominator invalid")
    count = len(per_group)
    seed = int(_canonical(["TEACH-0015-clean-bootstrap", identity])[:16], 16)
    rng = random.Random(seed)
    estimates = sorted(
        sum(per_group[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(draws))
    return {"lower_95": estimates[int(.025 * (draws - 1))],
            "upper_95": estimates[int(.975 * (draws - 1))],
            "groups": count, "draws": draws, "seed": seed}


def score_split(manifest: dict, rows: list[dict], samples: list[dict],
                identity: object) -> dict:
    groups = manifest["groups"]
    cells = [cell for group in groups for cell in group["cells"]]
    if not groups or len(cells) != 66 * len(groups) or (
            not isinstance(rows, list) or len(rows) != len(cells)):
        raise ValueError("Clean-screen episode coverage incomplete")
    sample_indices = (0, len(cells) // 2, len(cells) - 1)
    if (not isinstance(samples, list) or
            [row.get("index") for row in samples] != list(sample_indices)):
        raise ValueError("Clean-screen sampled-logit positions differ")
    sample_by_index = {row["index"]: row for row in samples}
    hits_by_group = []
    for group_index, group in enumerate(groups):
        hit = {}
        for local, cell in enumerate(group["cells"]):
            index = group_index * 66 + local
            source = cell["episode"]
            row = rows[index]
            if (not isinstance(row, dict) or set(row) != {
                    "render_id", "prediction"} or
                    row["render_id"] != source["render_id"] or
                    type(row["prediction"]) is not int or
                    not 16 <= row["prediction"] < 2064):
                raise ValueError("Clean-screen render ID/prediction differs")
            key = (cell["f"], cell["g"], cell["distractor"],
                   cell["marked"], cell["order"], cell["task"])
            hit[key] = row["prediction"] == source["answer"]
            if index in sample_by_index:
                sample = sample_by_index[index]
                logits = sample.get("logits")
                if (sample.get("render_id") != row["render_id"] or
                        not isinstance(logits, list) or len(logits) != 2064 or
                        any(type(value) not in (int, float) or
                            not math.isfinite(value) for value in logits) or
                        max(range(16, 2064), key=lambda token: logits[token])
                        != row["prediction"]):
                    raise ValueError("Clean-screen sampled logits invalid")
        hits_by_group.append(hit)
    metrics = {name: [] for name in (
        "composed_items", "recipient_triples", "marked_free_pairs",
        "first_hop_items", "direct_items", "copy_items")}
    group_rates = {name: [] for name in metrics}
    for hit in hits_by_group:
        outcomes = {name: [] for name in metrics}
        for f in (0, 1):
            for g in (0, 1, 2):
                for d in (0, 1):
                    for marked in (True, False):
                        for order in (0, 1):
                            outcomes["composed_items"].append(
                                hit[(f, g, d, marked, order, "composed")])
                for task in ("first_hop", "direct", "copy"):
                    outcomes[f"{task}_items"].append(
                        hit[(f, g, 0, True, 0, task)])
            for d in (0, 1):
                for marked in (True, False):
                    for order in (0, 1):
                        outcomes["recipient_triples"].append(all(
                            hit[(f, g, d, marked, order, "composed")]
                            for g in (0, 1, 2)))
            for g in (0, 1, 2):
                for d in (0, 1):
                    for order in (0, 1):
                        outcomes["marked_free_pairs"].append(all(
                            hit[(f, g, d, marked, order, "composed")]
                            for marked in (True, False)))
        expected = {"composed_items": 48, "recipient_triples": 16,
                    "marked_free_pairs": 24, "first_hop_items": 6,
                    "direct_items": 6, "copy_items": 6}
        for name, values in outcomes.items():
            if len(values) != expected[name]:
                raise ValueError("Clean-screen metric denominator differs")
            metrics[name].extend(values)
            group_rates[name].append(sum(values) / len(values))
    return {name: {**_fraction(values),
                   "group_bootstrap": _bootstrap(group_rates[name],
                                                 [identity, name])}
            for name, values in metrics.items()}


def _competent(scores: dict) -> bool:
    thresholds = {"composed_items": .90, "recipient_triples": .80,
                  "marked_free_pairs": .80, "first_hop_items": .95,
                  "direct_items": .95, "copy_items": .98}
    return all(scores[name]["accuracy"] >= threshold
               for name, threshold in thresholds.items())


def _benchmark_check(benchmark: dict, manifests: dict, hashes: dict) -> None:
    timings = benchmark.get("timings_seconds")
    if (not isinstance(timings, list) or len(timings) != 24 or
            any(type(value) not in (int, float) or not math.isfinite(value)
                or value <= 0 for value in timings)):
        raise ValueError("Clean-screen benchmark timings invalid")
    median = statistics.median(timings)
    batches = sum(math.ceil(sum(len(group["cells"])
                                for group in manifests[split]["groups"]) / 32)
                  for split in SPLITS)
    projected = 1.75 * median * batches * len(ARMS) * 2 + 300
    if (benchmark.get("screen_source_sha256") != hashes or
            benchmark.get("batch_size") != 32 or
            benchmark.get("warmup_batches") != 6 or
            benchmark.get("timed_batches") != 24 or
            benchmark.get("batches_per_arm_seed") != batches or
            benchmark.get("median_batch_seconds") != median or
            not math.isclose(benchmark.get("conservative_projected_seconds",
                                           math.inf), projected, rel_tol=1e-9) or
            benchmark.get("admitted") is not True or
            projected >= MAX_SECONDS or
            benchmark.get("max_seconds") != MAX_SECONDS or
            benchmark.get("max_mps_bytes") != MAX_MPS_BYTES or
            benchmark.get("max_artifact_bytes") != MAX_ARTIFACT_BYTES or
            benchmark.get("peak_sampled_mps_allocated_bytes", math.inf)
            > MAX_MPS_BYTES or
            benchmark.get("manifest_sha256") != {
                split: EXPECTED_MANIFESTS[split] for split in SPLITS}):
        raise ValueError("Clean-screen benchmark resource gate invalid")


def audit_screen(primary_dir: Path, suite_dir: Path, result_dir: Path,
                 root: Path, *, runs: dict | None = None) -> dict:
    status = json.loads((result_dir / "status.json").read_text())
    _source_check(root, status)
    manifests = read_manifests(suite_dir)
    final_teacher14 = json.loads((primary_dir / "suite.json").read_text())
    final_overlap = audit_splits(
        manifests["discovery"], manifests["confirmation"],
        [final_teacher14])
    if tuple(final_overlap["teacher14_exposure_sha256"]) != (
            EXPECTED_EXPOSURES[-1],):
        raise ValueError("TEACH-0015 overlap with final TEACH-0014 suite")
    suite_files = {split: _sha(suite_dir / f"{split}.json")
                   for split in SPLITS}
    if status.get("manifest_sha256") != EXPECTED_MANIFESTS or (
            status.get("suite_file_sha256") != suite_files):
        raise ValueError("Clean-screen suite file hash differs")
    benchmark = json.loads((result_dir / "benchmark.json").read_text())
    _benchmark_check(benchmark, manifests, status["screen_source_sha256"])
    if benchmark.get("suite_file_sha256") != suite_files:
        raise ValueError("Clean-screen benchmark suite file hash differs")
    primary_report = json.loads((primary_dir / "report.json").read_text())
    if (status.get("primary_report_sha256") != _sha(
            primary_dir / "report.json") or
            status.get("primary_artifact_audit_sha256") != _sha(
                primary_dir / "artifact-audit.json") or
            status.get("primary_replay_audit_sha256") != _sha(
                primary_dir / "replay-audit.json") or
            status.get("primary_source_git_head") != primary_report[
                "source_git_head"] or
            primary_report.get("behavior_audit_sha256") != _sha(
                primary_dir / "behavior-audit.json") or
            primary_report.get("parser_audit_sha256") != _sha(
                primary_dir / "parser-audit.json") or
            status.get("benchmark_sha256") != _sha(
                result_dir / "benchmark.json")):
        raise ValueError("Clean-screen primary provenance differs")
    arms = eligible_arms(primary_dir)
    if not arms or status.get("eligible_arms") != list(arms):
        raise ValueError("Clean-screen arm eligibility differs")
    if runs is None:
        runs = json.loads((result_dir / "report.json").read_text())["runs"]
    if not isinstance(runs, dict) or set(runs) != set(arms):
        raise ValueError("Clean-screen run arm set differs")
    scores = {}
    for arm in arms:
        if set(runs[arm]) != {"0", "1"}:
            raise ValueError("Clean-screen seed set incomplete")
        scores[arm] = {}
        for rep in ("0", "1"):
            path = result_dir / f"predictions-{arm}-rep{rep}.json.gz"
            detail = runs[arm][rep]
            primary_checkpoint = primary_report["training"][arm][rep][
                "final_checkpoint"]["sha256"]
            if (detail.get("sha256") != _sha(path) or
                    detail.get("bytes") != path.stat().st_size or
                    detail.get("checkpoint_sha256") != primary_checkpoint):
                raise ValueError("Clean-screen archive/checkpoint hash differs")
            archive = json.loads(gzip.decompress(path.read_bytes()))
            if (archive.get("checkpoint_sha256") != primary_checkpoint or
                    set(archive.get("splits", {})) != set(SPLITS)):
                raise ValueError("Clean-screen split archive incomplete")
            scores[arm][rep] = {
                split: score_split(manifests[split],
                                   archive["splits"][split]["rows"],
                                   archive["splits"][split]["samples"],
                                   [arm, rep, split])
                for split in SPLITS}
    return {"audit": "pass", "scope": "all_fresh_clean_competence_no_model",
            "eligible_arms": list(arms),
            "manifest_sha256": EXPECTED_MANIFESTS,
            "scores": scores,
            "candidate_competent_arms_pending_replay": [
                arm for arm in arms if all(_competent(
                    scores[arm][rep]["confirmation"])
                    for rep in ("0", "1"))],
            "sampled_checkpoint_replay": "pending"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", type=Path,
                        default=Path("results/TEACH-0014-v3"))
    parser.add_argument("--suite-dir", type=Path,
                        default=Path("outputs/TEACH-0015"))
    parser.add_argument("--result-dir", type=Path,
                        default=Path("results/TEACH-0015-screen"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_screen(args.primary_dir, args.suite_dir,
                          args.result_dir, args.root)
    raw = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(raw)
    else:
        print(raw, end="")


if __name__ == "__main__":
    main()
