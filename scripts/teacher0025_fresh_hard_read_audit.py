"""No-generator suite/exposure and no-model intervention audits for TEACH-0025."""

import hashlib
import json
import math
from pathlib import Path
import subprocess

from scripts.teacher0014_suite_audit import audit_manifest
from scripts.teacher0022_suite_audit import identities
from scripts.teacher0024_soft_read_audit import _score, entropy, visible_path


SEEDS = (84611, 84621)
PANEL = "composed_confirm_confirm"
CONDITIONS = ("clean", "identity", "hard_first", "hard_second",
              "hard_both", "gold_first", "gold_last", "wrong_first")
PRIOR = ("outputs/TEACH-0016/teach14-74111.json",
         "results/TEACH-0014-v3/suite.json",
         "outputs/TEACH-0022/suite-84411.json",
         "outputs/TEACH-0022/suite-84511.json")
SOURCES = (
    "docs/experiments/TEACH-0025-fresh-native-hard-read.md",
    "scripts/teacher0025_fresh_hard_read.py",
    "scripts/teacher0025_fresh_hard_read_audit.py",
    "scripts/teacher0025_fresh_hard_read_replay.py",
    "tests/test_teacher0025_fresh_hard_read.py",
    "scripts/teacher0024_soft_read_run.py",
    "scripts/teacher0024_soft_read_audit.py",
    "src/voynich/workspace/teacher14_tasks.py",
    "src/voynich/workspace/teacher14_models.py",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def suite_audit(root: Path) -> dict:
    entries = {}
    for name in PRIOR + tuple(f"outputs/TEACH-0025/suite-{seed}.json"
                              for seed in SEEDS):
        path = root / name
        manifest = json.loads(path.read_text())
        checked = audit_manifest(manifest)
        if name.startswith("outputs/TEACH-0025/"):
            expected_seed = int(path.stem.split("-")[-1])
            if (manifest["seed"] != expected_seed or
                    manifest["group_count"] != 128 or
                    len(manifest["panels"][PANEL]) != 128):
                raise ValueError("Fresh suite configuration differs")
        entries[name] = {"file_sha256": sha(path),
                         "canonical_sha256": checked["manifest_sha256"],
                         "identities": identities(manifest)}
    names = list(entries)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            for identity in ("graph_id", "logical_id", "render_id"):
                if entries[first]["identities"][identity] & entries[
                        second]["identities"][identity]:
                    raise ValueError(f"{identity} overlap: {first}/{second}")
    return {"audit": "pass", "seeds": list(SEEDS),
            "entries": {name: {key: value for key, value in row.items()
                               if key != "identities"}
                        for name, row in entries.items()},
            "exact_graph_logical_render_overlap": 0}


def provenance(root: Path) -> dict:
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *SOURCES],
                           cwd=root, check=True, capture_output=True,
                           text=True).stdout.strip()
    if dirty:
        raise RuntimeError("Commit TEACH-0025 sources before execution")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()
    return {"source_head": head,
            "source_sha256": {name: sha(root / name) for name in SOURCES}}


def _decision(scores: dict) -> dict:
    cells = {}
    for seed in SEEDS:
        for rep in ("0", "1"):
            row = scores[str(seed)][rep]
            clean = row["conditions"]["clean"]["correct"]
            hard = row["conditions"]["hard_first"]["correct"]
            controls = (
                row["conditions"]["identity"]["correct"] == clean and
                row["conditions"]["wrong_first"]["correct"] < clean and
                row["conditions"]["gold_last"]["correct"] >= clean)
            cells[f"{seed}/{rep}"] = {
                "clean_correct": clean, "hard_first_correct": hard,
                "net_gain_items": hard - clean,
                "pass": hard - clean >= 13 and controls}
    return {"cells": cells,
            "fresh_replicated": all(row["pass"] for row in cells.values())}


def audit(root: Path) -> dict:
    result_dir = root / "results/TEACH-0025"
    suite_path = result_dir / "suite-audit.json"
    stored_suite = json.loads(suite_path.read_text())
    checked_suite = suite_audit(root)
    if any(stored_suite.get(key) != value
           for key, value in checked_suite.items()):
        raise ValueError("Suite audit differs")
    status = json.loads((result_dir / "status.json").read_text())
    archive_path = result_dir / "rows.json"
    archive = json.loads(archive_path.read_text())
    if (status["status"] != "complete" or archive["status"] != "complete" or
            status["suite_audit_sha256"] != sha(suite_path) or
            status["rows_sha256"] != sha(archive_path) or
            archive_path.stat().st_size != status["rows_bytes"] or
            status["total_new_bytes"] > 100 * 1024**2 or
            sum(path.stat().st_size for folder in (
                root / "outputs/TEACH-0025", result_dir)
                for path in folder.rglob("*") if path.is_file()) >
            100 * 1024**2):
        raise ValueError("Result provenance/resource differs")
    if (archive["source_sha256"] != status["source_sha256"] or
            archive["source_head"] != status["source_head"] or
            stored_suite["source_head"] != status["source_head"] or
            stored_suite["source_sha256"] != status["source_sha256"] or
            set(status["source_sha256"]) != set(SOURCES)):
        raise ValueError("Source list differs")
    for name in SOURCES:
        committed = subprocess.run(["git", "show", f"{status['source_head']}:{name}"],
                                   cwd=root, check=True, capture_output=True).stdout
        digest = hashlib.sha256(committed).hexdigest()
        if digest != status["source_sha256"][name] or sha(root / name) != digest:
            raise ValueError(f"Source differs: {name}")
    prior = json.loads((root / "results/TEACH-0024-soft-read/status.json").read_text())
    if (status["prior_audit_sha256"] != sha(root / "results/TEACH-0024-soft-read/audit.json") or
            status["prior_replay_sha256"] != sha(root / "results/TEACH-0024-soft-read/replay-audit.json") or
            status["checkpoint_sha256"] != prior["checkpoint_sha256"]):
        raise ValueError("Prior causal result/checkpoints differ")
    scores = {}
    for seed in SEEDS:
        manifest_path = root / f"outputs/TEACH-0025/suite-{seed}.json"
        if status["suite_file_sha256"][str(seed)] != sha(manifest_path):
            raise ValueError("Fresh suite file differs")
        manifest = json.loads(manifest_path.read_text())
        source = manifest["panels"][PANEL]
        scores[str(seed)] = {}
        for rep in ("0", "1"):
            rows = archive["cells"][str(seed)][rep]["rows"]
            if len(rows) != 128:
                raise ValueError("Incomplete cell")
            checkpoint = root / f"outputs/TEACH-0014-v3/rep{rep}-oracle_rows_workspace-step6000.pt"
            if sha(checkpoint) != status["checkpoint_sha256"][rep]:
                raise ValueError("Checkpoint differs")
            for episode, row in zip(source, rows, strict=True):
                visible, target = visible_path(episode)
                if (row["render_id"] != episode["render_id"] or
                        row["answer"] != episode["answer"] or
                        row["target_rows"] != target or
                        row["target_symbols"] != [visible[i][1]
                                                  for i in target] or
                        len(row["native_top_rows"]) != 2 or
                        len(row["attention"]) != 2 or
                        set(row["predictions"]) != set(CONDITIONS) or
                        row["predictions"]["identity"] != row[
                            "predictions"]["clean"]):
                    raise ValueError("Visible row/prediction differs")
                if (row["wrong_first_row"] not in range(len(visible)) or
                        visible[row["wrong_first_row"]][1] == visible[
                            target[0]][1]):
                    raise ValueError("Wrong-row control differs")
                for hop in (0, 1):
                    attention = row["attention"][hop]
                    if len(attention) != len(visible):
                        raise ValueError("Attention length differs")
                    entropy(attention)
                    if max(range(len(visible)), key=lambda i: attention[i]) != (
                            row["native_top_rows"][hop]):
                        raise ValueError("Native top row differs")
                if any(type(value) is not int or not 16 <= value < 2064
                       for value in row["predictions"].values()):
                    raise ValueError("Prediction outside symbols")
            samples = archive["cells"][str(seed)][rep]["samples"]
            if set(samples) != set(CONDITIONS):
                raise ValueError("Sample condition set differs")
            for name, selected in samples.items():
                if [item["index"] for item in selected] != [0, 64, 127]:
                    raise ValueError("Sample positions differ")
                for item in selected:
                    logits = item["logits"]
                    if (len(logits) != 2064 or any(not math.isfinite(v)
                                                  for v in logits) or
                            16 + max(range(2048), key=lambda i: logits[
                                16 + i]) != rows[item["index"]][
                                    "predictions"][name]):
                        raise ValueError("Sample logits differ")
            scores[str(seed)][rep] = _score(rows)
    decision = _decision(scores)
    return {"audit": "pass", "scope": "fresh_native_hard_read",
            "rows_sha256": sha(archive_path), "scores": scores,
            "decision": decision}


if __name__ == "__main__":
    root = Path.cwd()
    result = audit(root)
    path = root / "results/TEACH-0025/audit.json"
    if path.exists():
        raise FileExistsError("No automatic audit overwrite")
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
