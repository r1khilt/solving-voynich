"""Independent visible-token and all-row score audit for TEACH-0021."""

import hashlib
import json
import math
from pathlib import Path
import subprocess


PANELS = ("first_hop_confirm", "direct_confirm", "composed_confirm_confirm")
CONDITIONS = ("clean", "identity", "gold_first", "gold_last", "gold_both",
              "wrong_both")
SUITE_SHA = "09930778461abea7618db8d31eb9aa82c610be0e05ca4d87493a7ff9834c98af"
SYMBOL_START = 16
VOCAB_SIZE = 2064
SOURCES = ("docs/experiments/TEACH-0021-oracle-reader-localization.md",
           "scripts/teacher0021_reader_run.py",
           "scripts/teacher0021_reader_audit.py",
           "scripts/teacher0021_reader_replay.py",
           "tests/test_teacher0021_reader.py")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def visible_path(tokens: list[int], hops: int) -> tuple[list[int], list[int]]:
    if tokens[0] != 1 or tokens[-1] != 8:
        raise ValueError("Malformed visible envelope")
    body = [token for token in tokens[1:-3] if token >= SYMBOL_START]
    if len(body) % 2 or not body:
        raise ValueError("Malformed visible row operands")
    rows = list(zip(body[::2], body[1::2], strict=True))
    if len({left for left, _ in rows}) != len(rows):
        raise ValueError("Visible mapping is not a function")
    value = tokens[-2]
    indices, symbols = [], []
    for _ in range(hops):
        candidates = [(i, right) for i, (left, right) in enumerate(rows)
                      if left == value]
        if len(candidates) != 1:
            raise ValueError("Visible path missing or ambiguous")
        index, value = candidates[0]
        indices.append(index)
        symbols.append(value)
    return indices, symbols


def _score(rows: list[dict], hops: int) -> dict:
    scores = {name: {"correct": 0, "rescue_clean_wrong": 0,
                     "damage_clean_correct": 0}
              for name in CONDITIONS}
    address = {str(hop): {"hits": 0, "clean_correct_hits": 0,
                          "clean_wrong_hits": 0}
               for hop in range(hops)}
    clean_correct = 0
    for row in rows:
        right = row["answer"]
        clean = row["predictions"]["clean"] == right
        clean_correct += clean
        for hop in range(hops):
            if row["native_argmax_rows"][hop] == row["target_rows"][hop]:
                address[str(hop)]["hits"] += 1
                address[str(hop)]["clean_correct_hits" if clean else
                                  "clean_wrong_hits"] += 1
        for name, prediction in row["predictions"].items():
            hit = prediction == right
            scores[name]["correct"] += hit
            scores[name]["rescue_clean_wrong"] += hit and not clean
            scores[name]["damage_clean_correct"] += clean and not hit
    return {"items": len(rows), "clean_correct": clean_correct,
            "conditions": scores, "address": address}


def audit(root: Path) -> dict:
    result_dir = root / "results/TEACH-0021-oracle-reader"
    status = json.loads((result_dir / "status.json").read_text())
    path = result_dir / "rows.json"
    data = json.loads(path.read_text())
    if status.get("status") != "complete" or data.get("status") != "complete":
        raise ValueError("TEACH-0021 is incomplete")
    if sha(path) != status["rows_sha256"] or path.stat().st_size != status[
            "artifact_bytes"]:
        raise ValueError("Rows archive differs")
    manifest_path = root / "outputs/TEACH-0016/teach14-74111.json"
    manifest = json.loads(manifest_path.read_text())
    if canonical(manifest) != SUITE_SHA or data["suite_sha256"] != SUITE_SHA or (
            data["suite_file_sha256"] != sha(manifest_path)):
        raise ValueError("Development suite differs")
    if set(data["panels"]) != {"0", "1"} or data["scope"] != (
            "exposed_development_only"):
        raise ValueError("Panel/seed scope differs")
    prior_path = root / "results/TEACH-0020-interrupted-dev-v2/report.json"
    prior = json.loads(prior_path.read_text())
    if data["prior_report_sha256"] != sha(prior_path):
        raise ValueError("Prior score provenance differs")
    replay_path = root / "results/TEACH-0020-interrupted-dev-v2/replay-audit.json"
    if (data["prior_replay_sha256"] != sha(replay_path) or
            json.loads(replay_path.read_text())["audit"] != "pass"):
        raise ValueError("Prior replay provenance differs")
    for rep in ("0", "1"):
        expected = prior["completed_checkpoint_files"][
            "oracle_rows_workspace"][rep]["checkpoint_sha256"]
        checkpoint = root / f"outputs/TEACH-0014-v3/rep{rep}-oracle_rows_workspace-step6000.pt"
        if data["checkpoint_sha256"][rep] != expected or sha(checkpoint) != expected:
            raise ValueError("Checkpoint identity differs")
    head = data["source_head"]
    if set(data["source_sha256"]) != set(SOURCES):
        raise ValueError("Source list differs")
    for name in SOURCES:
        committed = subprocess.run(["git", "show", f"{head}:{name}"],
                                   cwd=root, check=True, capture_output=True).stdout
        digest = hashlib.sha256(committed).hexdigest()
        if digest != data["source_sha256"][name] or sha(root / name) != digest:
            raise ValueError(f"Source changed: {name}")
    from gzip import open as gzip_open
    scores = {}
    for rep in ("0", "1"):
        scores[rep] = {}
        with gzip_open(root / f"results/TEACH-0020-interrupted-dev-v2/rows-oracle_rows_workspace-rep{rep}.json.gz",
                       "rt") as source:
            prior_rows = json.load(source)["panels"]
        if set(data["panels"][rep]) != set(PANELS):
            raise ValueError("Panel set differs")
        for panel in PANELS:
            source = manifest["panels"][panel]
            payload = data["panels"][rep][panel]
            rows = payload["rows"]
            hops = 2 if panel == "composed_confirm_confirm" else 1
            if len(source) != 128 or len(rows) != 128 or len(prior_rows[panel]) != 128:
                raise ValueError("Panel length differs")
            if set(payload["samples"]) != set(CONDITIONS):
                raise ValueError("Sample condition set differs")
            for index, (episode, row) in enumerate(zip(source, rows, strict=True)):
                target_indices, target_symbols = visible_path(
                    episode["tokens"], hops)
                if (row["render_id"] != episode["render_id"] or
                        row["answer"] != episode["answer"] or
                        target_symbols[-1] != episode["answer"] or
                        row["target_rows"] != target_indices or
                        row["target_symbols"] != target_symbols):
                    raise ValueError("Visible path or row identity differs")
                operands = [token for token in episode["tokens"][1:-3]
                            if token >= SYMBOL_START]
                right_values = operands[1::2]
                if (len(row["wrong_symbols"]) != hops or
                        any(wrong not in right_values or wrong == gold
                            for wrong, gold in zip(row["wrong_symbols"],
                                                   target_symbols, strict=True))):
                    raise ValueError("Wrong-value control differs")
                if (row["predictions"]["clean"] != prior_rows[panel][index][
                        "prediction"] or
                        set(row["predictions"]) != set(CONDITIONS) or
                        row["predictions"]["identity"] != row[
                            "predictions"]["clean"]):
                    raise ValueError("Prediction/prior/identity differs")
                if hops == 1 and (row["predictions"]["gold_first"] != row[
                        "predictions"]["gold_last"] or row["predictions"][
                            "gold_first"] != row["predictions"]["gold_both"]):
                    raise ValueError("One-hop gold conditions differ")
                if len(row["native_argmax_rows"]) != hops or any(
                        not 0 <= j < len(right_values)
                        for j in row["native_argmax_rows"]):
                    raise ValueError("Native attention index differs")
                if len(row["native_target_probability"]) != hops or any(
                        not math.isfinite(p) or not 0 <= p <= 1
                        for p in row["native_target_probability"]):
                    raise ValueError("Native attention probability invalid")
                for prediction in row["predictions"].values():
                    if not SYMBOL_START <= prediction < VOCAB_SIZE:
                        raise ValueError("Prediction outside symbol range")
            for name, samples in payload["samples"].items():
                if [sample["index"] for sample in samples] != [0, 64, 127]:
                    raise ValueError("Sample indices differ")
                for sample in samples:
                    vector = sample["logits"]
                    if len(vector) != VOCAB_SIZE or any(
                            not math.isfinite(value) for value in vector):
                        raise ValueError("Sample logits invalid")
                    predicted = SYMBOL_START + max(range(VOCAB_SIZE - SYMBOL_START),
                                                   key=lambda j: vector[
                                                       SYMBOL_START + j])
                    if predicted != rows[sample["index"]]["predictions"][name]:
                        raise ValueError("Sample argmax differs")
            scores[rep][panel] = _score(rows, hops)
    return {"audit": "pass", "scope": "exposed_development_only",
            "rows_sha256": sha(path), "scores": scores}


if __name__ == "__main__":
    root = Path.cwd()
    result = audit(root)
    (root / "results/TEACH-0021-oracle-reader/audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
