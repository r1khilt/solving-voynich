"""Independent all-item visible-row/attention/score audit for TEACH-0024."""

import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


CONDITIONS = ("clean", "identity", "hard_first", "hard_second",
              "hard_both", "gold_first", "gold_last", "wrong_first")
PANEL = "composed_confirm_confirm"
SUITE_SHA = "09930778461abea7618db8d31eb9aa82c610be0e05ca4d87493a7ff9834c98af"
SOURCES = ("docs/experiments/TEACH-0024-soft-read-localization.md",
           "scripts/teacher0024_soft_read_run.py",
           "scripts/teacher0024_soft_read_audit.py",
           "scripts/teacher0024_soft_read_replay.py",
           "tests/test_teacher0024_soft_read.py",
           "scripts/teacher0021_reader_run.py",
           "src/voynich/workspace/teacher14_models.py")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def visible_path(episode: dict) -> tuple[list[tuple[int, int]], list[int]]:
    body = [token for token in episode["tokens"][1:-3] if token >= 16]
    if len(body) % 2 or not body:
        raise ValueError("Malformed visible rows")
    rows = list(zip(body[::2], body[1::2], strict=True))
    mapping = {left: (i, right) for i, (left, right) in enumerate(rows)}
    if len(mapping) != len(rows):
        raise ValueError("Ambiguous visible mapping")
    value = episode["tokens"][-2]
    target = []
    for _ in range(2):
        if value not in mapping:
            raise ValueError("Visible target missing")
        index, value = mapping[value]
        target.append(index)
    if value != episode["answer"]:
        raise ValueError("Visible answer differs")
    return rows, target


def entropy(probabilities: list[float]) -> tuple[float, float]:
    if not probabilities or any(not math.isfinite(p) or p < 0 or p > 1
                                for p in probabilities):
        raise ValueError("Invalid attention probabilities")
    if abs(sum(probabilities) - 1) > 1e-5:
        raise ValueError("Attention does not sum to one")
    value = -sum(p * math.log(p) for p in probabilities if p)
    return value / math.log(len(probabilities)), math.exp(value)


def _score(rows: list[dict]) -> dict:
    clean_errors = sum(row["predictions"]["clean"] != row["answer"]
                       for row in rows)
    first_correct_errors = sum(
        row["predictions"]["clean"] != row["answer"] and
        row["native_top_rows"][0] == row["target_rows"][0]
        for row in rows)
    scores = {}
    for name in CONDITIONS:
        scores[name] = {
            "correct": sum(row["predictions"][name] == row["answer"]
                           for row in rows),
            "rescued_clean_errors": sum(
                row["predictions"]["clean"] != row["answer"] and
                row["predictions"][name] == row["answer"] for row in rows),
            "damaged_clean_correct": sum(
                row["predictions"]["clean"] == row["answer"] and
                row["predictions"][name] != row["answer"] for row in rows),
            "rescued_first_correct_errors": sum(
                row["predictions"]["clean"] != row["answer"] and
                row["native_top_rows"][0] == row["target_rows"][0] and
                row["predictions"][name] == row["answer"]
                for row in rows),
        }
    conditional = {}
    for label, subset in (
            ("clean_correct", [row for row in rows if row[
                "predictions"]["clean"] == row["answer"]]),
            ("clean_wrong", [row for row in rows if row[
                "predictions"]["clean"] != row["answer"]])):
        conditional[label] = {}
        for hop in (0, 1):
            masses = [row["attention"][hop][row["target_rows"][hop]]
                      for row in subset]
            normalized = [entropy(row["attention"][hop])[0]
                          for row in subset]
            effective = [entropy(row["attention"][hop])[1]
                         for row in subset]
            conditional[label][str(hop)] = {
                "items": len(subset), "target_mass_mean": statistics.mean(masses),
                "target_mass_median": statistics.median(masses),
                "normalized_entropy_mean": statistics.mean(normalized),
                "effective_rows_mean": statistics.mean(effective)}
    return {"items": len(rows), "clean_errors": clean_errors,
            "first_correct_clean_errors": first_correct_errors,
            "conditions": scores, "attention_by_clean": conditional}


def audit(root: Path) -> dict:
    result_dir = root / "results/TEACH-0024-soft-read"
    status = json.loads((result_dir / "status.json").read_text())
    path = result_dir / "rows.json"
    data = json.loads(path.read_text())
    if status["status"] != "complete" or data["status"] != "complete" or (
            status["rows_sha256"] != sha(path) or
            status["artifact_bytes"] != path.stat().st_size):
        raise ValueError("TEACH-0024 archive/status differs")
    suite_path = root / "outputs/TEACH-0016/teach14-74111.json"
    suite = json.loads(suite_path.read_text())
    if (canonical(suite) != SUITE_SHA or data["suite_sha256"] != SUITE_SHA
            or data["suite_file_sha256"] != sha(suite_path)):
        raise ValueError("Exposed suite differs")
    prior_audit_path = root / "results/TEACH-0021-oracle-reader/audit.json"
    prior_replay_path = root / "results/TEACH-0021-oracle-reader/replay-audit.json"
    prior = json.loads(prior_audit_path.read_text())
    if (data["prior_audit_sha256"] != sha(prior_audit_path) or
            data["prior_replay_sha256"] != sha(prior_replay_path) or
            prior["audit"] != "pass" or
            json.loads(prior_replay_path.read_text())["audit"] != "pass"):
        raise ValueError("Prior causal audit differs")
    head = data["source_head"]
    if set(data["source_sha256"]) != set(SOURCES):
        raise ValueError("Source list differs")
    for name in SOURCES:
        committed = subprocess.run(["git", "show", f"{head}:{name}"],
                                   cwd=root, check=True,
                                   capture_output=True).stdout
        digest = hashlib.sha256(committed).hexdigest()
        if digest != data["source_sha256"][name] or sha(root / name) != digest:
            raise ValueError(f"Source changed: {name}")
    if set(data["data"]) != {"0", "1"}:
        raise ValueError("Seed set differs")
    original = json.loads((root / "results/TEACH-0021-oracle-reader/rows.json").read_text())
    if sha(root / "results/TEACH-0021-oracle-reader/rows.json") != prior[
            "rows_sha256"]:
        raise ValueError("Prior rows differ")
    report = json.loads((root / "results/TEACH-0020-interrupted-dev-v2/report.json").read_text())
    scores = {}
    for rep in ("0", "1"):
        checkpoint = root / f"outputs/TEACH-0014-v3/rep{rep}-oracle_rows_workspace-step6000.pt"
        expected = report["completed_checkpoint_files"][
            "oracle_rows_workspace"][rep]["checkpoint_sha256"]
        if data["checkpoint_sha256"][rep] != expected or sha(checkpoint) != expected:
            raise ValueError("Checkpoint identity differs")
        rows = data["data"][rep]["rows"]
        source = suite["panels"][PANEL]
        baseline = original["panels"][rep][PANEL]["rows"]
        if len(rows) != 128 or len(baseline) != 128 or len(source) != 128:
            raise ValueError("Episode count differs")
        for episode, row, older in zip(source, rows, baseline, strict=True):
            visible, target = visible_path(episode)
            if (row["render_id"] != episode["render_id"] or
                    row["answer"] != episode["answer"] or
                    row["target_rows"] != target or
                    row["target_symbols"] != [visible[index][1]
                                              for index in target] or
                    len(row["attention"]) != 2 or
                    len(row["native_top_rows"]) != 2 or
                    set(row["predictions"]) != set(CONDITIONS)):
                raise ValueError("Visible row/identity differs")
            if (row["wrong_first_row"] not in range(len(visible)) or
                    visible[row["wrong_first_row"]][1] == visible[target[0]][1]):
                raise ValueError("Wrong-row control differs")
            for hop in (0, 1):
                probabilities = row["attention"][hop]
                if len(probabilities) != len(visible):
                    raise ValueError("Attention row count differs")
                entropy(probabilities)
                if max(range(len(visible)), key=lambda j: probabilities[j]) != (
                        row["native_top_rows"][hop]):
                    raise ValueError("Native top row differs")
            if any(type(guess) is not int or not 16 <= guess < 2064
                   for guess in row["predictions"].values()):
                raise ValueError("Prediction outside symbols")
            if (row["predictions"]["clean"] != older["predictions"]["clean"] or
                    row["predictions"]["identity"] != row["predictions"]["clean"] or
                    row["predictions"]["gold_first"] != older["predictions"][
                        "gold_first"] or
                    row["predictions"]["gold_last"] != older["predictions"][
                        "gold_last"]):
                raise ValueError("Clean/gold cross-check differs")
        samples = data["data"][rep]["samples"]
        if set(samples) != set(CONDITIONS):
            raise ValueError("Sample conditions differ")
        for name, selected in samples.items():
            if [item["index"] for item in selected] != [0, 64, 127]:
                raise ValueError("Sample positions differ")
            for item in selected:
                logits = item["logits"]
                if (len(logits) != 2064 or any(
                        not math.isfinite(value) for value in logits) or
                        16 + max(range(2048), key=lambda j: logits[
                            16 + j]) != rows[item["index"]]["predictions"][
                                name]):
                    raise ValueError("Sample logits differ")
        scores[rep] = _score(rows)
    return {"audit": "pass", "scope": "exposed_development_only",
            "rows_sha256": sha(path), "scores": scores}


if __name__ == "__main__":
    root = Path.cwd()
    result = audit(root)
    (root / "results/TEACH-0024-soft-read/audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
