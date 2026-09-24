"""CPU sampled-logit checkpoint replay for TEACH-0020 development scores."""

import gzip
import json
import math
from pathlib import Path
import time

import torch

from scripts.teacher0020_dev_audit import audit, sha
from voynich.workspace.teacher14_tasks import Episode
from voynich.workspace.teacher14_train import Config, model_output, new_model


PANELS = ("composed_confirm_confirm", "factorial",
          "boundary_groups", "long_ood")
ABS_TOL = 2e-3
REL_TOL = 2e-3
MAX_SECONDS = 1800.0


def _compare(actual: list[float], archived: list[float], prediction: int) -> float:
    if len(actual) != 2064 or len(archived) != 2064:
        raise ValueError("Replay vector length differs")
    if any(not math.isfinite(value) for value in actual + archived):
        raise ValueError("Replay vector is nonfinite")
    errors = [abs(a - b) for a, b in zip(actual, archived, strict=True)]
    if any(error > ABS_TOL + REL_TOL * abs(saved)
           for error, saved in zip(errors, archived, strict=True)):
        raise ValueError("Replay logits exceed frozen tolerance")
    if 16 + max(range(2048), key=lambda i: actual[16 + i]) != prediction:
        raise ValueError("Replay argmax differs")
    return max(errors)


def replay(root: Path, primary_dir: Path, output_dir: Path,
           suite_path: Path, result_dir: Path) -> dict:
    if (result_dir / "replay-audit.json").exists():
        raise FileExistsError("No automatic TEACH-0020 replay rerun")
    checked = audit(root, primary_dir, output_dir, suite_path, result_dir)
    saved_audit = json.loads((result_dir / "audit.json").read_text())
    if checked != saved_audit or checked.get("audit") != "pass":
        raise ValueError("Independent development score audit differs")
    report = json.loads((result_dir / "report.json").read_text())
    manifest = json.loads(suite_path.read_text())
    start = time.monotonic()
    per_run = {}
    max_error = 0.0
    vectors = 0
    with torch.no_grad():
        for arm, seeds in report["completed_checkpoint_files"].items():
            per_run[arm] = {}
            for rep, details in seeds.items():
                if time.monotonic() - start > MAX_SECONDS:
                    raise TimeoutError("TEACH-0020 CPU replay time cap")
                checkpoint = output_dir / f"rep{rep}-{arm}-step6000.pt"
                if sha(checkpoint) != details["checkpoint_sha256"]:
                    raise ValueError("Replay checkpoint hash differs")
                saved = torch.load(checkpoint, map_location="cpu",
                                   weights_only=True)
                model, optimizer = new_model(Config(), arm, int(rep), "cpu")
                model.load_state_dict(saved["model"], strict=True)
                model.eval()
                archive_path = result_dir / f"rows-{arm}-rep{rep}.json.gz"
                archived = json.loads(gzip.decompress(archive_path.read_bytes()))
                local_error = 0.0
                local_count = 0
                for name in PANELS:
                    source = manifest["panels"][name]
                    indices = [0, len(source) // 2, len(source) - 1]
                    episodes = [Episode(**source[index]) for index in indices]
                    output = model_output(model, arm, episodes, "cpu")
                    samples = archived["samples"][name]
                    for local, (index, sample) in enumerate(zip(
                            indices, samples, strict=True)):
                        if (sample["index"] != index or sample["render_id"] !=
                                source[index]["render_id"]):
                            raise ValueError("Replay sample identity differs")
                        prediction = archived["panels"][name][index]["prediction"]
                        error = _compare(output.logits[local].float().tolist(),
                                         sample["logits"], prediction)
                        local_error = max(local_error, error)
                        local_count += 1
                per_run[arm][rep] = {"vectors": local_count,
                                     "max_abs_logit_error": local_error}
                vectors += local_count
                max_error = max(max_error, local_error)
                del saved, model, optimizer, archived
    return {"audit": "pass", "scope": "sampled_development_checkpoint_replay",
            "score_audit_sha256": sha(result_dir / "audit.json"),
            "report_sha256": sha(result_dir / "report.json"),
            "vectors": vectors, "max_abs_logit_error": max_error,
            "absolute_tolerance": ABS_TOL, "relative_tolerance": REL_TOL,
            "elapsed_seconds": time.monotonic() - start,
            "per_run": per_run}


if __name__ == "__main__":
    root = Path.cwd()
    result = replay(root, root / "results/TEACH-0014-v3",
                    root / "outputs/TEACH-0014-v3",
                    root / "outputs/TEACH-0016/teach14-74111.json",
                    root / "results/TEACH-0020-interrupted-dev")
    (root / "results/TEACH-0020-interrupted-dev/replay-audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
