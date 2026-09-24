"""Finite soft-to-hard native read interventions on exposed oracle rows."""

import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from scripts.teacher0020_dev_run import episode_from_json
from scripts.teacher0021_reader_run import target_rows
from voynich.workspace.teacher14_models import public_row_tensors
from voynich.workspace.teacher14_objectives import padded_tokens
from voynich.workspace.teacher14_tasks import SYMBOL_START, VOCAB_SIZE
from voynich.workspace.teacher14_train import Config, new_model


SUITE_SHA = "09930778461abea7618db8d31eb9aa82c610be0e05ca4d87493a7ff9834c98af"
PANEL = "composed_confirm_confirm"
CONDITIONS = ("clean", "identity", "hard_first", "hard_second",
              "hard_both", "gold_first", "gold_last", "wrong_first")
SOURCES = ("docs/experiments/TEACH-0024-soft-read-localization.md",
           "scripts/teacher0024_soft_read_run.py",
           "scripts/teacher0024_soft_read_audit.py",
           "scripts/teacher0024_soft_read_replay.py",
           "tests/test_teacher0024_soft_read.py",
           "scripts/teacher0021_reader_run.py",
           "src/voynich/workspace/teacher14_models.py")
MAX_SECONDS = 1200
MAX_BYTES = 100 * 1024**2


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def provenance(root: Path) -> dict:
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *SOURCES],
                           cwd=root, check=True, capture_output=True,
                           text=True).stdout.strip()
    if dirty:
        raise RuntimeError("Commit TEACH-0024 sources before inference")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()
    return {"source_head": head,
            "source_sha256": {name: sha(root / name) for name in SOURCES}}


def _vectors(model, right: torch.Tensor,
             row_ids: list[int]) -> torch.Tensor:
    batch = torch.arange(right.shape[0], device=right.device)
    chosen = right[batch, torch.tensor(row_ids, device=right.device)]
    return model.value_proj(model.symbol(chosen))


def _predictions(logits: torch.Tensor) -> list[int]:
    return (logits[:, SYMBOL_START:VOCAB_SIZE].argmax(dim=1) +
            SYMBOL_START).tolist()


def evaluate_batch(model, episodes: list) -> tuple[list[dict], dict]:
    ids = padded_tokens(episodes, "cpu")
    left, right, mask = public_row_tensors(ids)
    target, symbols = target_rows(
        left, right, mask, [episode.query for episode in episodes],
        2, [episode.answer for episode in episodes])
    clean = model(ids, row_left=left, row_right=right, row_mask=mask,
                  capture=True)
    hard = [[int(clean.cache[f"read.{hop}.attention"][item].argmax())
             for hop in (0, 1)] for item in range(len(episodes))]
    wrong = []
    for item, path in enumerate(target):
        correct = int(right[item, path[0]])
        choice = next((index for index in range(right.shape[1])
                       if bool(mask[item, index]) and int(right[item, index]) !=
                       correct), None)
        if choice is None:
            raise ValueError("No distinct wrong visible row")
        wrong.append(choice)
    first_hard = _vectors(model, right, [path[0] for path in hard])
    second_hard = _vectors(model, right, [path[1] for path in hard])
    first_gold = _vectors(model, right, [path[0] for path in target])
    second_gold = _vectors(model, right, [path[1] for path in target])
    first_wrong = _vectors(model, right, wrong)
    interventions = {
        "identity": {"read.0.value": clean.cache["read.0.value"],
                     "read.1.value": clean.cache["read.1.value"]},
        "hard_first": {"read.0.value": first_hard},
        "hard_second": {"read.1.value": second_hard},
        "hard_both": {"read.0.value": first_hard,
                      "read.1.value": second_hard},
        "gold_first": {"read.0.value": first_gold},
        "gold_last": {"read.1.value": second_gold},
        "wrong_first": {"read.0.value": first_wrong},
    }
    outputs = {"clean": clean}
    for name, replacement in interventions.items():
        outputs[name] = model(ids, row_left=left, row_right=right,
                              row_mask=mask, interventions=replacement)
    if not torch.allclose(clean.logits, outputs["identity"].logits,
                          atol=1e-5, rtol=0):
        raise ValueError("Identity intervention changed logits")
    guesses = {name: _predictions(output.logits)
               for name, output in outputs.items()}
    rows = []
    for item, episode in enumerate(episodes):
        count = int(mask[item].sum())
        attentions = [clean.cache[f"read.{hop}.attention"][
            item, :count].float().tolist() for hop in (0, 1)]
        rows.append({
            "render_id": episode.render_id, "answer": episode.answer,
            "target_rows": target[item], "target_symbols": symbols[item],
            "native_top_rows": hard[item], "wrong_first_row": wrong[item],
            "attention": attentions,
            "predictions": {name: values[item]
                            for name, values in guesses.items()},
        })
    return rows, {name: output.logits.float().tolist()
                  for name, output in outputs.items()}


def run(root: Path) -> dict:
    output = root / "results/TEACH-0024-soft-read"
    if output.exists():
        raise FileExistsError("No automatic TEACH-0024 rerun")
    source = provenance(root)
    suite_path = root / "outputs/TEACH-0016/teach14-74111.json"
    suite = json.loads(suite_path.read_text())
    if canonical(suite) != SUITE_SHA or len(suite["panels"][PANEL]) != 128:
        raise ValueError("Exposed suite differs")
    prior_path = root / "results/TEACH-0021-oracle-reader/audit.json"
    prior = json.loads(prior_path.read_text())
    if prior["audit"] != "pass" or prior["rows_sha256"] != sha(
            root / "results/TEACH-0021-oracle-reader/rows.json"):
        raise ValueError("Prior finite-value audit differs")
    replay_path = root / "results/TEACH-0021-oracle-reader/replay-audit.json"
    if json.loads(replay_path.read_text())["audit"] != "pass":
        raise ValueError("Prior checkpoint replay differs")
    report = json.loads((root / "results/TEACH-0020-interrupted-dev-v2/report.json").read_text())
    output.mkdir(parents=True)
    start = time.monotonic()
    status = {"experiment": "TEACH-0024", "status": "running",
              "scope": "exposed_development_only",
              "suite_sha256": SUITE_SHA,
              "suite_file_sha256": sha(suite_path),
              "prior_audit_sha256": sha(prior_path),
              "prior_replay_sha256": sha(replay_path),
              "checkpoint_sha256": {rep: report["completed_checkpoint_files"][
                  "oracle_rows_workspace"][rep]["checkpoint_sha256"]
                                    for rep in ("0", "1")}, **source}
    (output / "status.json").write_text(json.dumps(status, indent=2,
                                                   sort_keys=True) + "\n")
    try:
        data = {}
        with torch.no_grad():
            for rep in ("0", "1"):
                checkpoint = root / f"outputs/TEACH-0014-v3/rep{rep}-oracle_rows_workspace-step6000.pt"
                if sha(checkpoint) != status["checkpoint_sha256"][rep]:
                    raise ValueError("Checkpoint hash differs")
                saved = torch.load(checkpoint, map_location="cpu",
                                   weights_only=True)
                model, optimizer = new_model(
                    Config(), "oracle_rows_workspace", int(rep), "cpu")
                model.load_state_dict(saved["model"], strict=True)
                model.eval()
                episodes = [episode_from_json(item)
                            for item in suite["panels"][PANEL]]
                rows = []
                samples = {name: [] for name in CONDITIONS}
                for offset in range(0, 128, 32):
                    if time.monotonic() - start > MAX_SECONDS:
                        raise TimeoutError("TEACH-0024 wall cap")
                    chunk_rows, logits = evaluate_batch(
                        model, episodes[offset:offset + 32])
                    rows.extend(chunk_rows)
                    for local in range(len(chunk_rows)):
                        index = offset + local
                        if index in (0, 64, 127):
                            for name in CONDITIONS:
                                samples[name].append({
                                    "index": index,
                                    "logits": logits[name][local]})
                data[rep] = {"rows": rows, "samples": samples}
                del model, optimizer, saved
        archive = output / "rows.json"
        archive.write_text(json.dumps({**status, "status": "complete",
                                       "data": data},
                                      separators=(",", ":"),
                                      allow_nan=False) + "\n")
        if archive.stat().st_size > MAX_BYTES:
            raise RuntimeError("TEACH-0024 artifact cap")
        status.update({"status": "complete", "rows_sha256": sha(archive),
                       "artifact_bytes": archive.stat().st_size,
                       "elapsed_seconds": time.monotonic() - start})
        (output / "status.json").write_text(json.dumps(status, indent=2,
                                                       sort_keys=True) + "\n")
        return status
    except Exception as exc:
        status.update({"status": "stopped",
                       "reason": f"{type(exc).__name__}: {exc}"})
        (output / "status.json").write_text(json.dumps(status, indent=2,
                                                       sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2, sort_keys=True))
