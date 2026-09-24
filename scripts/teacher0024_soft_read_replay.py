"""Sampled CPU checkpoint replay for native hard-read interventions."""

import json
import math
from pathlib import Path
import time

import torch

from scripts.teacher0020_dev_run import episode_from_json
from scripts.teacher0024_soft_read_audit import audit, sha
from scripts.teacher0024_soft_read_run import CONDITIONS, PANEL, evaluate_batch
from voynich.workspace.teacher14_train import Config, new_model


MAX_SECONDS = 600
ABS_TOL = .002
REL_TOL = .002


def replay(root: Path) -> dict:
    result_dir = root / "results/TEACH-0024-soft-read"
    if (result_dir / "replay-audit.json").exists():
        raise FileExistsError("No automatic TEACH-0024 replay rerun")
    checked = audit(root)
    if checked != json.loads((result_dir / "audit.json").read_text()):
        raise ValueError("Independent score audit differs")
    archive = json.loads((result_dir / "rows.json").read_text())
    suite = json.loads((root / "outputs/TEACH-0016/teach14-74111.json").read_text())
    start = time.monotonic()
    maximum = 0.0
    vectors = 0
    with torch.no_grad():
        for rep in ("0", "1"):
            if time.monotonic() - start > MAX_SECONDS:
                raise TimeoutError("TEACH-0024 replay wall cap")
            checkpoint = root / f"outputs/TEACH-0014-v3/rep{rep}-oracle_rows_workspace-step6000.pt"
            if sha(checkpoint) != archive["checkpoint_sha256"][rep]:
                raise ValueError("Replay checkpoint differs")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model, optimizer = new_model(
                Config(), "oracle_rows_workspace", int(rep), "cpu")
            model.load_state_dict(saved["model"], strict=True)
            model.eval()
            indices = (0, 64, 127)
            episodes = [episode_from_json(
                suite["panels"][PANEL][index]) for index in indices]
            rows, logits = evaluate_batch(model, episodes)
            original = archive["data"][rep]
            for local, index in enumerate(indices):
                row = original["rows"][index]
                if (rows[local]["predictions"] != row["predictions"] or
                        rows[local]["target_rows"] != row["target_rows"] or
                        rows[local]["native_top_rows"] != row[
                            "native_top_rows"]):
                    raise ValueError("Replay row prediction/attention differs")
                for hop in (0, 1):
                    for actual, old in zip(rows[local]["attention"][hop],
                                           row["attention"][hop], strict=True):
                        if abs(actual - old) > 1e-5:
                            raise ValueError("Replay attention differs")
                for name in CONDITIONS:
                    sample = original["samples"][name][local]
                    if sample["index"] != index:
                        raise ValueError("Replay sample index differs")
                    actual = logits[name][local]
                    for current, expected in zip(
                            actual, sample["logits"], strict=True):
                        error = abs(current - expected)
                        if (not math.isfinite(current) or
                                error > ABS_TOL + REL_TOL * abs(expected)):
                            raise ValueError("Replay logits differ")
                        maximum = max(maximum, error)
                    vectors += 1
            del model, optimizer, saved
    return {"audit": "pass", "vectors": vectors,
            "max_abs_logit_error": maximum,
            "absolute_tolerance": ABS_TOL, "relative_tolerance": REL_TOL,
            "elapsed_seconds": time.monotonic() - start,
            "score_audit_sha256": sha(result_dir / "audit.json")}


if __name__ == "__main__":
    root = Path.cwd()
    result = replay(root)
    (root / "results/TEACH-0024-soft-read/replay-audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
