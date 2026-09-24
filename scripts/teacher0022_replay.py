"""Sampled CPU checkpoint replay after the independent TEACH-0022 audit."""

import gzip
import json
import math
from pathlib import Path
import time

import torch

from scripts.teacher0022_audit import ARMS, SEEDS, audit, sha
from scripts.teacher0022_score import _panel
from scripts.teacher0022_train import new_arm
from voynich.workspace.teacher14_train import Config as OldConfig
from voynich.workspace.teacher14_train import new_model


PANELS = ("composed_confirm_confirm", "factorial",
          "boundary_groups", "long_ood")
MAX_SECONDS = 1800
ABS_TOL = .002
REL_TOL = .002


def replay(root: Path) -> dict:
    result_dir = root / "results/TEACH-0022"
    if (result_dir / "replay-audit.json").exists():
        raise FileExistsError("No automatic TEACH-0022 replay rerun")
    checked = audit(root)
    saved = json.loads((result_dir / "audit.json").read_text())
    if checked != saved or saved["audit"] != "pass":
        raise ValueError("Independent score audit differs")
    trained = json.loads((result_dir / "status.json").read_text())
    begin = time.monotonic()
    count = 0
    maximum = 0.0
    with torch.no_grad():
        for arm in ARMS:
            for rep in (0, 1):
                if arm == "oracle_rows_workspace":
                    model, optimizer = new_model(
                        OldConfig(), arm, rep, "cpu")
                    checkpoint = root / f"outputs/TEACH-0014-v3/rep{rep}-{arm}-step6000.pt"
                    expected = trained["baseline"]["files"][str(rep)][
                        "checkpoint_sha256"]
                else:
                    model, optimizer = new_arm(arm, rep, "cpu")
                    checkpoint = root / f"outputs/TEACH-0022/rep{rep}-{arm}-step6000.pt"
                    expected = trained["runs"][arm][str(rep)][
                        "checkpoint_sha256"]
                if sha(checkpoint) != expected:
                    raise ValueError("Replay checkpoint differs")
                state = torch.load(checkpoint, map_location="cpu",
                                   weights_only=True)
                model.load_state_dict(state["model"], strict=True)
                model.eval()
                for seed in SEEDS:
                    manifest = json.loads((root / f"outputs/TEACH-0022/suite-{seed}.json").read_text())
                    archive_path = result_dir / f"predictions-{arm}-rep{rep}-seed{seed}.json.gz"
                    archive = json.loads(gzip.decompress(
                        archive_path.read_bytes()))
                    for name in PANELS:
                        if time.monotonic() - begin > MAX_SECONDS:
                            raise TimeoutError("TEACH-0022 replay wall cap")
                        source = manifest["panels"][name]
                        indices = (0, len(source) // 2, len(source) - 1)
                        subset = [source[index] for index in indices]
                        reexecuted = _panel(model, subset, "cpu")
                        original = archive["panels"][name]
                        for local, index in enumerate(indices):
                            row = original["rows"][index]
                            if reexecuted["rows"][local] != row:
                                raise ValueError("Replay prediction/attention differs")
                            sample = original["samples"][local]
                            if sample["index"] != index or sample[
                                    "render_id"] != subset[local]["render_id"]:
                                raise ValueError("Replay sample identity differs")
                            actual = reexecuted["samples"][local]["logits"]
                            for current, expected_logit in zip(
                                    actual, sample["logits"], strict=True):
                                error = abs(current - expected_logit)
                                if (not math.isfinite(current) or error >
                                        ABS_TOL + REL_TOL * abs(expected_logit)):
                                    raise ValueError("Replay logits differ")
                                maximum = max(maximum, error)
                            count += 1
                del model, optimizer, state
    return {"audit": "pass", "vectors": count,
            "max_abs_logit_error": maximum, "absolute_tolerance": ABS_TOL,
            "relative_tolerance": REL_TOL,
            "elapsed_seconds": time.monotonic() - begin,
            "score_audit_sha256": sha(result_dir / "audit.json")}


if __name__ == "__main__":
    root = Path.cwd()
    result = replay(root)
    (root / "results/TEACH-0022/replay-audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
