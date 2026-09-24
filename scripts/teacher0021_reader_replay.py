"""Sampled full-logit CPU checkpoint replay of TEACH-0021 interventions."""

import json
import math
from pathlib import Path
import time

import torch

from scripts.teacher0020_dev_run import episode_from_json
from scripts.teacher0021_reader_audit import audit
from scripts.teacher0021_reader_run import CONDITIONS, PANELS, evaluate_batch, sha
from voynich.workspace.teacher14_train import Config, new_model


MAX_SECONDS = 600
ABS_TOL = 0.002
REL_TOL = 0.002


def replay(root: Path) -> dict:
    result_dir = root / "results/TEACH-0021-oracle-reader"
    if (result_dir / "replay-audit.json").exists():
        raise FileExistsError("No automatic TEACH-0021 replay rerun")
    checked = audit(root)
    if checked != json.loads((result_dir / "audit.json").read_text()):
        raise ValueError("Score audit differs")
    rows_path = result_dir / "rows.json"
    data = json.loads(rows_path.read_text())
    manifest = json.loads((root / "outputs/TEACH-0016/teach14-74111.json").read_text())
    start = time.monotonic()
    max_error = 0.0
    vectors = 0
    with torch.no_grad():
        for rep in ("0", "1"):
            checkpoint = root / f"outputs/TEACH-0014-v3/rep{rep}-oracle_rows_workspace-step6000.pt"
            if sha(checkpoint) != data["checkpoint_sha256"][rep]:
                raise ValueError("Replay checkpoint differs")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model, optimizer = new_model(Config(), "oracle_rows_workspace",
                                         int(rep), "cpu")
            model.load_state_dict(saved["model"], strict=True)
            model.eval()
            for panel in PANELS:
                if time.monotonic() - start > MAX_SECONDS:
                    raise TimeoutError("TEACH-0021 replay wall cap")
                indices = (0, 64, 127)
                episodes = [episode_from_json(
                    manifest["panels"][panel][index]) for index in indices]
                hops = 2 if panel == "composed_confirm_confirm" else 1
                replay_rows, replay_logits = evaluate_batch(model, episodes, hops)
                original = data["panels"][rep][panel]
                for local, index in enumerate(indices):
                    row = original["rows"][index]
                    if replay_rows[local]["predictions"] != row["predictions"]:
                        raise ValueError("Replay prediction differs")
                    if replay_rows[local]["target_rows"] != row["target_rows"]:
                        raise ValueError("Replay target row differs")
                    for name in CONDITIONS:
                        sample = original["samples"][name][local]
                        if sample["index"] != index:
                            raise ValueError("Replay sample index differs")
                        actual = replay_logits[name][local]
                        archived = sample["logits"]
                        if len(actual) != len(archived):
                            raise ValueError("Replay vector length differs")
                        for current, expected in zip(actual, archived, strict=True):
                            if not math.isfinite(current) or (
                                    abs(current - expected) >
                                    ABS_TOL + REL_TOL * abs(expected)):
                                raise ValueError("Replay logits differ")
                            max_error = max(max_error, abs(current - expected))
                        vectors += 1
            del model, optimizer, saved
    return {"audit": "pass", "vectors": vectors,
            "max_abs_logit_error": max_error,
            "absolute_tolerance": ABS_TOL, "relative_tolerance": REL_TOL,
            "elapsed_seconds": time.monotonic() - start,
            "rows_sha256": sha(rows_path),
            "score_audit_sha256": sha(result_dir / "audit.json")}


if __name__ == "__main__":
    root = Path.cwd()
    result = replay(root)
    (root / "results/TEACH-0021-oracle-reader/replay-audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
