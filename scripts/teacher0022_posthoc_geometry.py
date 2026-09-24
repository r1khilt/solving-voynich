"""Exploratory checkpoint read-mass and value/update norms on exposed dev."""

import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch

from scripts.teacher0020_dev_run import episode_from_json
from scripts.teacher0022_audit import ARMS, sha
from scripts.teacher0022_score import _path_indices
from scripts.teacher0022_train import new_arm
from voynich.workspace.teacher14_models import public_row_tensors
from voynich.workspace.teacher14_objectives import padded_tokens
from voynich.workspace.teacher14_tasks import SYMBOL_START
from voynich.workspace.teacher14_train import Config, new_model


SEED = 84411
PANEL = "composed_confirm_confirm"
MAX_SECONDS = 300


def run(root: Path) -> dict:
    output = root / "results/TEACH-0022/posthoc-geometry.json"
    if output.exists():
        raise FileExistsError("No automatic posthoc geometry overwrite")
    result_dir = root / "results/TEACH-0022"
    previous = json.loads((result_dir / "audit.json").read_text())
    replay = json.loads((result_dir / "replay-audit.json").read_text())
    if previous["audit"] != replay["audit"] or previous["audit"] != "pass":
        raise ValueError("Primary result is not audited")
    suite_path = root / f"outputs/TEACH-0022/suite-{SEED}.json"
    suite = json.loads(suite_path.read_text())
    episodes = [episode_from_json(item) for item in suite["panels"][PANEL]]
    start = time.monotonic()
    summary = {}
    with torch.no_grad():
        for arm in ARMS:
            summary[arm] = {}
            for rep in (0, 1):
                if arm == "oracle_rows_workspace":
                    model, optimizer = new_model(Config(), arm, rep, "cpu")
                    checkpoint = root / f"outputs/TEACH-0014-v3/rep{rep}-{arm}-step6000.pt"
                else:
                    model, optimizer = new_arm(arm, rep, "cpu")
                    checkpoint = root / f"outputs/TEACH-0022/rep{rep}-{arm}-step6000.pt"
                trained = json.loads((result_dir / "status.json").read_text())
                expected = (trained["baseline"]["files"][str(rep)][
                    "checkpoint_sha256"] if arm == "oracle_rows_workspace"
                    else trained["runs"][arm][str(rep)]["checkpoint_sha256"])
                if sha(checkpoint) != expected:
                    raise ValueError("Checkpoint differs")
                model.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                                 weights_only=True)["model"])
                model.eval()
                values = {name: [] for name in (
                    "first_target_mass", "second_target_mass",
                    "first_top_mass", "second_top_mass",
                    "first_value_norm", "first_update_norm",
                    "first_update_over_value")}
                correct = first_hits = second_hits = 0
                for offset in range(0, len(episodes), 32):
                    if time.monotonic() - start > MAX_SECONDS:
                        raise TimeoutError("TEACH-0022 posthoc CPU cap")
                    chunk = episodes[offset:offset + 32]
                    ids = padded_tokens(chunk, "cpu")
                    left, right, mask = public_row_tensors(ids)
                    result = model(ids, row_left=left, row_right=right,
                                   row_mask=mask, capture=True)
                    paths = _path_indices(left, right, mask,
                                          [item.query for item in chunk],
                                          [item.hops for item in chunk],
                                          [item.answer for item in chunk])
                    predictions = (result.logits[:, SYMBOL_START:].argmax(-1)
                                   + SYMBOL_START).tolist()
                    first_value = result.cache["read.0.value"]
                    first_update = model.update(torch.cat((
                        result.cache["query.0"], first_value), dim=-1))
                    for index, (item, path) in enumerate(zip(chunk, paths,
                                                              strict=True)):
                        count = int(mask[index].sum())
                        first = result.cache["read.0.attention"][index, :count]
                        second = result.cache["read.1.attention"][index, :count]
                        correct += predictions[index] == item.answer
                        first_hits += int(first.argmax()) == path[0]
                        second_hits += int(second.argmax()) == path[1]
                        for name, value in (
                            ("first_target_mass", first[path[0]]),
                            ("second_target_mass", second[path[1]]),
                            ("first_top_mass", first.max()),
                            ("second_top_mass", second.max()),
                            ("first_value_norm", first_value[index].norm()),
                            ("first_update_norm", first_update[index].norm()),
                            ("first_update_over_value", first_update[index].norm()
                             / (first_value[index].norm() + 1e-8))):
                            number = float(value)
                            if not math.isfinite(number):
                                raise ValueError("Nonfinite read geometry")
                            values[name].append(number)
                prior = previous["scores"][arm][str(rep)][str(SEED)][
                    "panels"][PANEL]
                if (correct != prior["items"]["correct"] or
                        first_hits != prior["first_target_row_hits"] or
                        second_hits != prior["second_target_row_hits"]):
                    raise ValueError("Posthoc behavior differs from independent audit")
                row = {"correct": correct, "first_hits": first_hits,
                       "second_hits": second_hits,
                       "means": {name: statistics.mean(series)
                                 for name, series in values.items()},
                       "checkpoint_sha256": expected}
                if hasattr(model.update, "log_scale"):
                    row["learned_update_scale"] = float(
                        model.update.log_scale.exp())
                summary[arm][str(rep)] = row
                del model, optimizer
    result = {"scope": "posthoc_exposed_development_only",
              "suite_seed": SEED, "panel": PANEL,
              "suite_file_sha256": sha(suite_path),
              "primary_audit_sha256": sha(result_dir / "audit.json"),
              "primary_replay_sha256": sha(result_dir / "replay-audit.json"),
              "source_sha256": hashlib.sha256((root /
                  "scripts/teacher0022_posthoc_geometry.py").read_bytes()).hexdigest(),
              "elapsed_seconds": time.monotonic() - start,
              "cells": summary}
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    result = run(Path.cwd())
    print(json.dumps({"scope": result["scope"],
                      "elapsed_seconds": result["elapsed_seconds"]},
                     indent=2, sort_keys=True))
