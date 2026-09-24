"""Fixed-item CPU checkpoint replay for TEACH-0025."""

import json
import math
from pathlib import Path
import time

import torch

from scripts.teacher0020_dev_run import episode_from_json
from scripts.teacher0025_fresh_hard_read_audit import (
    CONDITIONS, PANEL, SEEDS, audit, sha,
)
from scripts.teacher0024_soft_read_run import evaluate_batch
from voynich.workspace.teacher14_train import Config, new_model


MAX_SECONDS = 600
ABS_TOL = .002
REL_TOL = .002


def replay(root: Path) -> dict:
    result_dir = root / "results/TEACH-0025"
    output = result_dir / "replay-audit.json"
    if output.exists():
        raise FileExistsError("No automatic replay overwrite")
    if audit(root) != json.loads((result_dir / "audit.json").read_text()):
        raise ValueError("Independent audit differs")
    archive = json.loads((result_dir / "rows.json").read_text())
    start = time.monotonic()
    count = 0
    maximum = 0.0
    with torch.no_grad():
        for seed in SEEDS:
            suite = json.loads((root / f"outputs/TEACH-0025/suite-{seed}.json").read_text())
            indices = (0, 64, 127)
            episodes = [episode_from_json(suite["panels"][PANEL][i])
                        for i in indices]
            for rep in (0, 1):
                if time.monotonic() - start > MAX_SECONDS:
                    raise TimeoutError("TEACH-0025 replay wall cap")
                checkpoint = root / f"outputs/TEACH-0014-v3/rep{rep}-oracle_rows_workspace-step6000.pt"
                if sha(checkpoint) != archive["checkpoint_sha256"][str(rep)]:
                    raise ValueError("Checkpoint differs")
                saved = torch.load(checkpoint, map_location="cpu",
                                   weights_only=True)
                model, optimizer = new_model(Config(),
                                             "oracle_rows_workspace", rep, "cpu")
                model.load_state_dict(saved["model"], strict=True)
                model.eval()
                rows, logits = evaluate_batch(model, episodes)
                original = archive["cells"][str(seed)][str(rep)]
                for local, index in enumerate(indices):
                    old = original["rows"][index]
                    if (rows[local]["predictions"] != old["predictions"] or
                            rows[local]["target_rows"] != old["target_rows"] or
                            rows[local]["native_top_rows"] != old[
                                "native_top_rows"]):
                        raise ValueError("Replay prediction/target differs")
                    for hop in (0, 1):
                        for current, expected in zip(
                                rows[local]["attention"][hop],
                                old["attention"][hop], strict=True):
                            if abs(current - expected) > 1e-5:
                                raise ValueError("Replay attention differs")
                    for name in CONDITIONS:
                        sample = original["samples"][name][local]
                        if sample["index"] != index:
                            raise ValueError("Replay sample index differs")
                        for current, expected in zip(
                                logits[name][local], sample["logits"], strict=True):
                            error = abs(current - expected)
                            if (not math.isfinite(current) or error >
                                    ABS_TOL + REL_TOL * abs(expected)):
                                raise ValueError("Replay logits differ")
                            maximum = max(maximum, error)
                        count += 1
                del model, optimizer, saved
    return {"audit": "pass", "vectors": count,
            "max_abs_logit_error": maximum,
            "absolute_tolerance": ABS_TOL,
            "relative_tolerance": REL_TOL,
            "elapsed_seconds": time.monotonic() - start,
            "score_audit_sha256": sha(result_dir / "audit.json")}


if __name__ == "__main__":
    root = Path.cwd()
    result = replay(root)
    (root / "results/TEACH-0025/replay-audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
