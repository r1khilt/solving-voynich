"""One-shot fresh-suite prediction archive for all TEACH-0022 arms."""

import gzip
import json
from pathlib import Path
import time

import torch

from scripts.teacher0020_dev_run import episode_from_json
from scripts.teacher0022_train import (
    ARMS, _resource, new_arm, provenance, sha, write_json,
)
from voynich.workspace.teacher14_models import public_row_tensors
from voynich.workspace.teacher14_tasks import SYMBOL_START
from voynich.workspace.teacher14_train import Config as OldConfig
from voynich.workspace.teacher14_train import new_model
from voynich.workspace.teacher14_objectives import padded_tokens


ALL_ARMS = ("oracle_rows_workspace",) + ARMS
SEEDS = (84411, 84511)


def _path_indices(left: torch.Tensor, right: torch.Tensor,
                  mask: torch.Tensor, queries: list[int],
                  hops: list[int], answers: list[int]) -> list[list[int]]:
    results = []
    for item, (query, hop_count, answer) in enumerate(zip(
            queries, hops, answers, strict=True)):
        mapping = {int(left[item, j]): (j, int(right[item, j]))
                   for j in range(left.shape[1]) if bool(mask[item, j])}
        value = query
        path = []
        for _ in range(min(hop_count, 2)):
            if value not in mapping:
                raise ValueError("Visible path missing")
            index, value = mapping[value]
            path.append(index)
        if hop_count <= 2 and value != answer:
            raise ValueError("Visible answer mismatch")
        results.append(path)
    return results


def _panel(model, source: list[dict], device: str) -> dict:
    episodes = [episode_from_json(item) for item in source]
    sample_indices = (0, len(episodes) // 2, len(episodes) - 1)
    rows = []
    samples = []
    with torch.no_grad():
        for offset in range(0, len(episodes), 32):
            chunk = episodes[offset:offset + 32]
            ids = padded_tokens(chunk, device)
            left, right, mask = public_row_tensors(ids)
            result = model(ids, row_left=left, row_right=right, row_mask=mask,
                           capture=True)
            predictions = (result.logits[:, SYMBOL_START:].argmax(dim=-1) +
                           SYMBOL_START).tolist()
            paths = _path_indices(left, right, mask,
                                  [episode.query for episode in chunk],
                                  [episode.hops for episode in chunk],
                                  [episode.answer for episode in chunk])
            for local, (episode, prediction, path) in enumerate(zip(
                    chunk, predictions, paths, strict=True)):
                native = [int(result.cache[
                    f"read.{hop}.attention"][local].argmax())
                    for hop in range(len(path))]
                rows.append({"render_id": episode.render_id,
                             "prediction": prediction,
                             "native_argmax_rows": native})
            for index in sample_indices:
                if offset <= index < offset + len(chunk):
                    local = index - offset
                    samples.append({
                        "index": index,
                        "render_id": chunk[local].render_id,
                        "logits": result.logits[local].float().tolist()})
    return {"rows": rows, "samples": sorted(samples, key=lambda row: row[
        "index"])}


def score(root: Path) -> dict:
    source = provenance(root)
    result_dir = root / "results/TEACH-0022"
    status_path = result_dir / "score-status.json"
    if status_path.exists():
        raise FileExistsError("No automatic TEACH-0022 score rerun")
    trained = json.loads((result_dir / "status.json").read_text())
    if (trained["status"] != "complete" or trained[
            "source_sha256"] != source["source_sha256"]):
        raise ValueError("Training or source is incomplete")
    suite_audit = json.loads((result_dir / "suite-audit.json").read_text())
    if suite_audit["audit"] != "pass":
        raise ValueError("Fresh suite audit failed")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required")
    manifests = {seed: json.loads((root / f"outputs/TEACH-0022/suite-{seed}.json").read_text())
                 for seed in SEEDS}
    start = time.monotonic()
    status = {"experiment": "TEACH-0022", "status": "running",
              "scope": "fresh_84411_84511_once", "source_head": source[
                  "source_head"], "source_sha256": source["source_sha256"],
              "training_status_sha256": sha(result_dir / "status.json"),
              "suite_audit_sha256": sha(result_dir / "suite-audit.json"),
              "archives": {}}
    write_json(status_path, status)
    try:
        with torch.no_grad():
            for arm in ALL_ARMS:
                status["archives"][arm] = {}
                for rep in (0, 1):
                    if arm == "oracle_rows_workspace":
                        model, optimizer = new_model(
                            OldConfig(), arm, rep, "mps")
                        checkpoint = root / f"outputs/TEACH-0014-v3/rep{rep}-{arm}-step6000.pt"
                        expected = trained["baseline"]["files"][str(rep)][
                            "checkpoint_sha256"]
                    else:
                        model, optimizer = new_arm(arm, rep, "mps")
                        checkpoint = root / f"outputs/TEACH-0022/rep{rep}-{arm}-step6000.pt"
                        expected = trained["runs"][arm][str(rep)][
                            "checkpoint_sha256"]
                    if sha(checkpoint) != expected:
                        raise ValueError("Score checkpoint differs")
                    saved = torch.load(checkpoint, map_location="cpu",
                                       weights_only=True)
                    model.load_state_dict(saved["model"], strict=True)
                    model.eval()
                    for seed in SEEDS:
                        panels = {}
                        for name, episodes in manifests[seed]["panels"].items():
                            panels[name] = _panel(model, episodes, "mps")
                            _resource(root, start)
                        path = result_dir / f"predictions-{arm}-rep{rep}-seed{seed}.json.gz"
                        path.write_bytes(gzip.compress(json.dumps({
                            "arm": arm, "replicate": rep, "seed": seed,
                            "panels": panels}, separators=(",", ":"),
                            allow_nan=False).encode(), compresslevel=9))
                        status["archives"][arm][f"{rep}/{seed}"] = {
                            "sha256": sha(path), "bytes": path.stat().st_size}
                        status["progress"] = {"arm": arm, "replicate": rep,
                                              "seed": seed}
                        status.update(_resource(root, start))
                        write_json(status_path, status)
                    del model, optimizer, saved
                    torch.mps.empty_cache()
        status.update({"status": "complete", **_resource(root, start)})
        write_json(status_path, status)
        return status
    except Exception as exc:
        status.update({"status": "stopped",
                       "reason": f"{type(exc).__name__}: {exc}"})
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    print(json.dumps(score(Path.cwd()), indent=2, sort_keys=True))
