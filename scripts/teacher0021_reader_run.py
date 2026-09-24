"""Frozen exposed-development intervention on the TEACH-0014 oracle reader."""

import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from scripts.teacher0020_dev_run import episode_from_json
from voynich.workspace.teacher14_models import public_row_tensors
from voynich.workspace.teacher14_tasks import SYMBOL_START, VOCAB_SIZE
from voynich.workspace.teacher14_train import Config, new_model, padded_tokens


PANELS = ("first_hop_confirm", "direct_confirm", "composed_confirm_confirm")
CONDITIONS = ("clean", "identity", "gold_first", "gold_last", "gold_both",
              "wrong_both")
SUITE_SHA = "09930778461abea7618db8d31eb9aa82c610be0e05ca4d87493a7ff9834c98af"
SOURCES = ("docs/experiments/TEACH-0021-oracle-reader-localization.md",
           "scripts/teacher0021_reader_run.py",
           "scripts/teacher0021_reader_audit.py",
           "scripts/teacher0021_reader_replay.py",
           "tests/test_teacher0021_reader.py")
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
        raise RuntimeError("Commit TEACH-0021 source before inference")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()
    return {"source_head": head,
            "source_sha256": {name: sha(root / name) for name in SOURCES}}


def target_rows(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor,
                queries: list[int], hops: int,
                answers: list[int]) -> tuple[list[list[int]], list[list[int]]]:
    """Follow visible rows only; metadata paths never choose the targets."""
    all_indices: list[list[int]] = []
    all_symbols: list[list[int]] = []
    for item, (query, answer) in enumerate(zip(queries, answers, strict=True)):
        rows = {int(left[item, j]): (j, int(right[item, j]))
                for j in range(left.shape[1]) if bool(mask[item, j])}
        if len(rows) != int(mask[item].sum()):
            raise ValueError("Visible left symbols are not unique")
        indices, symbols = [], []
        value = query
        for _ in range(hops):
            if value not in rows:
                raise ValueError("Visible path missing")
            index, value = rows[value]
            indices.append(index)
            symbols.append(value)
        if value != answer:
            raise ValueError("Visible path disagrees with answer")
        all_indices.append(indices)
        all_symbols.append(symbols)
    return all_indices, all_symbols


def alternate_symbols(right: torch.Tensor, mask: torch.Tensor,
                      targets: list[list[int]]) -> list[list[int]]:
    alternates = []
    for item, path in enumerate(targets):
        valid = [int(right[item, j]) for j in range(right.shape[1])
                 if bool(mask[item, j])]
        chosen = []
        for target in path:
            correct = int(right[item, target])
            choice = next((candidate for candidate in valid
                           if candidate != correct), None)
            if choice is None:
                raise ValueError("No distinct wrong visible row value")
            chosen.append(choice)
        alternates.append(chosen)
    return alternates


def predict(logits: torch.Tensor) -> list[int]:
    return (logits[:, SYMBOL_START:VOCAB_SIZE].argmax(dim=-1) +
            SYMBOL_START).tolist()


def evaluate_batch(model: torch.nn.Module, episodes: list,
                   hops: int) -> tuple[list[dict], dict[str, list[list[float]]]]:
    ids = padded_tokens(episodes, "cpu")
    left, right, mask = public_row_tensors(ids)
    queries = [episode.query for episode in episodes]
    answers = [episode.answer for episode in episodes]
    targets, symbols = target_rows(left, right, mask, queries, hops, answers)
    wrong = alternate_symbols(right, mask, targets)
    base = model(ids, row_left=left, row_right=right, row_mask=mask,
                 capture=True)
    gold_vectors = {}
    wrong_vectors = {}
    for hop in range(hops):
        gold_ids = torch.tensor([path[hop] for path in symbols])
        wrong_ids = torch.tensor([path[hop] for path in wrong])
        gold_vectors[hop] = model.value_proj(model.symbol(gold_ids))
        wrong_vectors[hop] = model.value_proj(model.symbol(wrong_ids))
    outputs = {"clean": base}
    for condition in CONDITIONS[1:]:
        if condition == "identity":
            replacement = {f"read.{hop}.value": base.cache[f"read.{hop}.value"]
                           for hop in range(hops)}
        elif condition == "gold_first":
            replacement = {"read.0.value": gold_vectors[0]}
        elif condition == "gold_last":
            replacement = {f"read.{hops - 1}.value": gold_vectors[hops - 1]}
        elif condition == "gold_both":
            replacement = {f"read.{hop}.value": gold_vectors[hop]
                           for hop in range(hops)}
        else:
            replacement = {f"read.{hop}.value": wrong_vectors[hop]
                           for hop in range(hops)}
        outputs[condition] = model(ids, row_left=left, row_right=right,
                                   row_mask=mask, interventions=replacement)
    if not torch.allclose(base.logits, outputs["identity"].logits,
                          atol=1e-5, rtol=0):
        raise ValueError("Identity read-value intervention changed logits")
    predictions = {name: predict(output.logits)
                   for name, output in outputs.items()}
    rows = []
    for item, episode in enumerate(episodes):
        native = [int(base.cache[f"read.{hop}.attention"][item].argmax())
                  for hop in range(hops)]
        rows.append({
            "render_id": episode.render_id, "answer": episode.answer,
            "target_rows": targets[item], "target_symbols": symbols[item],
            "wrong_symbols": wrong[item], "native_argmax_rows": native,
            "native_target_probability": [
                float(base.cache[f"read.{hop}.attention"][item, targets[item][hop]])
                for hop in range(hops)],
            "predictions": {name: values[item]
                            for name, values in predictions.items()},
        })
    logits = {name: output.logits.float().tolist()
              for name, output in outputs.items()}
    return rows, logits


def run(root: Path) -> dict:
    result_dir = root / "results/TEACH-0021-oracle-reader"
    if result_dir.exists():
        raise FileExistsError("No automatic TEACH-0021 rerun")
    provenance_data = provenance(root)
    manifest_path = root / "outputs/TEACH-0016/teach14-74111.json"
    manifest = json.loads(manifest_path.read_text())
    if canonical(manifest) != SUITE_SHA:
        raise ValueError("Development suite differs")
    prior = json.loads((root / "results/TEACH-0020-interrupted-dev-v2/report.json").read_text())
    replay = json.loads((root / "results/TEACH-0020-interrupted-dev-v2/replay-audit.json").read_text())
    if replay.get("audit") != "pass" or prior.get("status") != "complete":
        raise ValueError("Prior checkpoint-derived screen not validated")
    for panel in PANELS:
        if len(manifest["panels"][panel]) != 128:
            raise ValueError("Panel size differs")
    result_dir.mkdir(parents=True)
    start = time.monotonic()
    status = {"experiment": "TEACH-0021", "status": "running",
              "scope": "exposed_development_only", "suite_sha256": SUITE_SHA,
              "suite_file_sha256": sha(manifest_path),
              "prior_report_sha256": sha(root / "results/TEACH-0020-interrupted-dev-v2/report.json"),
              "prior_replay_sha256": sha(root / "results/TEACH-0020-interrupted-dev-v2/replay-audit.json"),
              "checkpoint_sha256": {rep: prior["completed_checkpoint_files"][
                  "oracle_rows_workspace"][rep]["checkpoint_sha256"]
                                    for rep in ("0", "1")},
              **provenance_data}
    (result_dir / "status.json").write_text(json.dumps(status, indent=2,
                                                        sort_keys=True) + "\n")
    try:
        archive = {}
        with torch.no_grad():
            for rep in ("0", "1"):
                checkpoint = root / f"outputs/TEACH-0014-v3/rep{rep}-oracle_rows_workspace-step6000.pt"
                if sha(checkpoint) != status["checkpoint_sha256"][rep]:
                    raise ValueError("Oracle checkpoint differs")
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                model, optimizer = new_model(Config(), "oracle_rows_workspace",
                                             int(rep), "cpu")
                model.load_state_dict(saved["model"], strict=True)
                model.eval()
                archive[rep] = {}
                for panel in PANELS:
                    if time.monotonic() - start > MAX_SECONDS:
                        raise TimeoutError("TEACH-0021 wall cap")
                    source = manifest["panels"][panel]
                    episodes = [episode_from_json(item) for item in source]
                    hops = 2 if panel == "composed_confirm_confirm" else 1
                    rows = []
                    samples = {name: [] for name in CONDITIONS}
                    for offset in range(0, len(episodes), 32):
                        chunk_rows, logits = evaluate_batch(
                            model, episodes[offset:offset + 32], hops)
                        rows.extend(chunk_rows)
                        for local in range(len(chunk_rows)):
                            index = offset + local
                            if index in (0, len(episodes) // 2,
                                         len(episodes) - 1):
                                for name in CONDITIONS:
                                    samples[name].append({
                                        "index": index,
                                        "logits": logits[name][local]})
                    archive[rep][panel] = {"rows": rows, "samples": samples}
                del model, optimizer, saved
        payload = {**status, "status": "complete",
                   "elapsed_seconds": time.monotonic() - start,
                   "panels": archive}
        output = result_dir / "rows.json"
        output.write_text(json.dumps(payload, separators=(",", ":"),
                                     allow_nan=False) + "\n")
        if output.stat().st_size > MAX_BYTES:
            raise ValueError("TEACH-0021 artifact cap")
        status.update({"status": "complete", "elapsed_seconds": payload[
            "elapsed_seconds"], "rows_sha256": sha(output),
                       "artifact_bytes": output.stat().st_size})
        (result_dir / "status.json").write_text(json.dumps(status, indent=2,
                                                            sort_keys=True) + "\n")
        return status
    except Exception as exc:
        status.update({"status": "stopped",
                       "reason": f"{type(exc).__name__}: {exc}"})
        (result_dir / "status.json").write_text(json.dumps(status, indent=2,
                                                            sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2, sort_keys=True))
