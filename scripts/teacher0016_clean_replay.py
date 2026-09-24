"""CPU checkpoint replay for sampled TEACH-0016 fresh clean predictions."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch

from scripts.teacher0015_clean_audit import SPLITS
from scripts.teacher0015_clean_replay import _compare
from scripts.teacher0015_clean_run import _episodes
from scripts.teacher0016_clean_audit import (
    MAX_ARTIFACT_BYTES, MAX_MPS_BYTES, MAX_SECONDS, SAMPLE_INDICES,
    audit_screen, read_manifests,
)
from voynich.workspace.teacher14_models import CandidateEdgeWorkspace
from voynich.workspace.teacher14_objectives import padded_tokens


def replay(primary_dir: Path, suite_dir: Path, finite_dir: Path,
           result_dir: Path, root: Path) -> dict:
    audited = audit_screen(primary_dir, suite_dir, finite_dir,
                           result_dir, root)
    decision_path = result_dir / "decision-audit.json"
    archived = json.loads(decision_path.read_text())
    report = json.loads((result_dir / "report.json").read_text())
    artifact_bytes = sum(path.stat().st_size for path in result_dir.rglob("*")
                         if path.is_file())
    if (audited != archived or report.get("status") != "complete" or
            report.get("decision_audit_sha256") != hashlib.sha256(
                decision_path.read_bytes()).hexdigest() or
            report.get("elapsed_seconds", math.inf) > MAX_SECONDS or
            report.get("peak_sampled_mps_allocated_bytes", math.inf)
            > MAX_MPS_BYTES or artifact_bytes > MAX_ARTIFACT_BYTES):
        raise ValueError("TEACH-0016 screen archive/resource audit differs")
    primary = json.loads((primary_dir / "report.json").read_text())
    for name, expected in primary["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"TEACH-0016 checkpoint source changed: {name}")
    manifests = read_manifests(suite_dir, root)
    max_error = 0.0
    vectors = 0
    per_run = {}
    with torch.no_grad():
        for arm in audited["eligible_arms"]:
            per_run[arm] = {}
            for rep in ("0", "1"):
                checkpoint = root / "outputs/TEACH-0014-v3" / (
                    f"rep{rep}-{arm}-step6000.pt")
                expected_sha = report["runs"][arm][rep]["checkpoint_sha256"]
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_sha:
                    raise ValueError("TEACH-0016 replay checkpoint hash differs")
                saved = torch.load(checkpoint, map_location="cpu",
                                   weights_only=True)
                model = CandidateEdgeWorkspace()
                model.load_state_dict(saved["model"], strict=True)
                model.eval()
                local_error = 0.0
                local_count = 0
                archive = json.loads(gzip.decompress((result_dir / (
                    f"predictions-{arm}-rep{rep}.json.gz")).read_bytes()))
                for split in SPLITS:
                    episodes = _episodes(manifests[split])
                    selected = [episodes[index] for index in SAMPLE_INDICES]
                    output = model(padded_tokens(selected, "cpu"))
                    samples = archive["splits"][split]["samples"]
                    for local, index in enumerate(SAMPLE_INDICES):
                        sample = samples[local]
                        if sample["render_id"] != selected[local].render_id:
                            raise ValueError("TEACH-0016 replay render ID differs")
                        prediction = archive["splits"][split]["rows"][index][
                            "prediction"]
                        error = _compare(output.logits[local].float().tolist(),
                                         sample["logits"], prediction)
                        local_error = max(local_error, error)
                        local_count += 1
                per_run[arm][rep] = {"vectors": local_count,
                                     "max_abs_logit_error": local_error}
                max_error = max(max_error, local_error)
                vectors += local_count
                del model, saved
    return {"audit": "pass", "scope": "sampled_cross_order_clean_checkpoint_replay",
            "manifest_sha256": audited["manifest_sha256"],
            "vectors": vectors, "max_abs_logit_error": max_error,
            "absolute_tolerance": 2e-3, "relative_tolerance": 2e-3,
            "per_run": per_run,
            "clean_competent_arms": audited[
                "candidate_competent_arms_pending_replay"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", type=Path,
                        default=Path("results/TEACH-0014-v3"))
    parser.add_argument("--suite-dir", type=Path,
                        default=Path("outputs/TEACH-0016"))
    parser.add_argument("--finite-dir", type=Path,
                        default=Path("results/TEACH-0015-finite"))
    parser.add_argument("--result-dir", type=Path,
                        default=Path("results/TEACH-0016-screen"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = replay(args.primary_dir, args.suite_dir, args.finite_dir,
                    args.result_dir, args.root)
    raw = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(raw)
    else:
        print(raw, end="")


if __name__ == "__main__":
    main()
