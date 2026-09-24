"""Numerically replay frozen TEACH-0014 logits from saved checkpoints.

Unlike the no-model artifact auditor this imports the registered model code.
It runs only after the independent file/source audit succeeds and requires the
current source files to match the archived campaign commit exactly.
"""

import argparse
from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path

import torch

from scripts.teacher0014_artifact_audit import (
    ARMS, SOURCE_PATHS, _canonical_sha, _sha_file, audit_artifacts,
)
from voynich.workspace.teacher14_tasks import evaluation_suite, suite_manifest
from voynich.workspace.teacher14_train import (
    Config, model_output, new_model,
)


ABS_TOLERANCE = 2e-3
REL_TOLERANCE = 2e-3


def _verify_vector(actual: torch.Tensor, expected: list[float],
                   prediction: int) -> float:
    if actual.shape != (2064,) or len(expected) != 2064:
        raise ValueError("Replay logit shape mismatch")
    reference = torch.tensor(expected, dtype=torch.float32)
    observed = actual.detach().float().cpu()
    if not bool(torch.isfinite(observed).all().item()):
        raise ValueError("Nonfinite replayed logits")
    error = (observed - reference).abs()
    allowance = ABS_TOLERANCE + REL_TOLERANCE * reference.abs()
    if bool((error > allowance).any().item()):
        raise ValueError(f"Checkpoint logit replay differs: max error {error.max().item()}")
    guess = int(observed[16:].argmax().item()) + 16
    if guess != prediction:
        raise ValueError("Checkpoint answer replay differs")
    return float(error.max().item())


def replay(result_dir: Path, output_dir: Path, root: Path) -> dict:
    artifact = audit_artifacts(result_dir, output_dir, root)
    report = json.loads((result_dir / "report.json").read_text())
    if json.loads(json.dumps(asdict(Config()))) != report["config"]:
        raise ValueError("Current replay configuration differs from archived run")
    for path in SOURCE_PATHS:
        if _sha_file(root / path) != report["source_sha256"][path]:
            raise ValueError(f"Current replay source differs: {path}")
    suite = evaluation_suite(Config().eval_seed, Config().eval_groups)
    if _canonical_sha(suite_manifest(
            suite, seed=Config().eval_seed,
            group_count=Config().eval_groups)) != artifact["manifest_sha256"]:
        raise ValueError("Regenerated replay suite differs from archived manifest")
    predictions = json.loads(gzip.decompress(
        (result_dir / "predictions.json.gz").read_bytes()))
    archive = json.loads(gzip.decompress(
        (result_dir / "replay-logits.json.gz").read_bytes()))
    max_error = 0.0
    count = 0
    per_run = {}
    with torch.no_grad():
        for arm in sorted(ARMS):
            per_run[arm] = {}
            for replicate in (0, 1):
                checkpoint = output_dir / (
                    f"rep{replicate}-{arm}-step{Config().steps_per_arm}.pt")
                saved = torch.load(checkpoint, map_location="cpu",
                                   weights_only=True)
                if (json.loads(json.dumps(saved.get("config"))) != report["config"] or
                        saved.get("replicate") != replicate or
                        saved.get("arm") != arm or
                        saved.get("step") != Config().steps_per_arm or
                        saved.get("parameters") != report["training"][arm][
                            str(replicate)]["parameters"]):
                    raise ValueError(f"{arm}/{replicate}: checkpoint metadata drift")
                model, optimizer = new_model(Config(), arm, replicate, "cpu")
                model.load_state_dict(saved["model"], strict=True)
                model.eval()
                run_error = 0.0
                run_count = 0
                for name, episodes in suite.items():
                    samples = archive["runs"][arm][str(replicate)][name]
                    selected = [episodes[row["index"]] for row in samples]
                    logits = model_output(model, arm, selected, "cpu").logits
                    for index, row in enumerate(samples):
                        prediction = predictions["runs"][arm][str(replicate)][
                            "panels"][name][row["index"]]["prediction"]
                        error = _verify_vector(logits[index], row["logits"], prediction)
                        run_error = max(run_error, error)
                        run_count += 1
                per_run[arm][str(replicate)] = {
                    "samples": run_count, "max_abs_logit_error": run_error}
                max_error = max(max_error, run_error)
                count += run_count
                del model, optimizer, saved
    if count != artifact["replay_logit_samples"] or not math.isfinite(max_error):
        raise ValueError("Numerical replay sample count or error invalid")
    return {"audit": "pass", "scope": "sampled_checkpoint_logit_replay",
            "source_git_head": report["source_git_head"],
            "manifest_sha256": artifact["manifest_sha256"],
            "samples": count, "max_abs_logit_error": max_error,
            "absolute_tolerance": ABS_TOLERANCE,
            "relative_tolerance": REL_TOLERANCE,
            "per_run": per_run}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path,
                        default=Path("results/TEACH-0014"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/TEACH-0014"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = replay(args.result_dir, args.output_dir, args.root)
    raw = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(raw)
    else:
        print(raw, end="")


if __name__ == "__main__":
    main()
