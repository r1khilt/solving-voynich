"""Independent no-model TEACH-0022 trace, prediction and decision audit."""

import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

from scripts.teacher0014_behavior_audit import _score_panel
from scripts.teacher0014_suite_audit import audit_manifest
from scripts.teacher0022_suite_audit import audit as audit_suites


ARMS = ("oracle_rows_workspace", "unlinked_route",
        "linked_answer", "linked_route")
NEW_ARMS = ARMS[1:]
SEEDS = (84411, 84511)
STEPS = 6000
SOURCE_PATHS = (
    "docs/experiments/TEACH-0022-linked-reader.md",
    "src/voynich/workspace/teacher14_tasks.py",
    "src/voynich/workspace/teacher14_models.py",
    "src/voynich/workspace/teacher14_objectives.py",
    "src/voynich/workspace/teacher14_train.py",
    "src/voynich/workspace/teacher22_models.py",
    "scripts/teacher0014_suite_audit.py",
    "scripts/teacher0014_behavior_audit.py",
    "scripts/teacher0014_artifact_audit.py",
    "scripts/teacher0020_dev_run.py",
    "scripts/teacher0022_suite.py",
    "scripts/teacher0022_suite_audit.py",
    "scripts/teacher0022_train.py",
    "scripts/teacher0022_score.py",
    "scripts/teacher0022_audit.py",
    "scripts/teacher0022_replay.py",
    "tests/test_teacher0022_models.py",
    "tests/test_teacher0022_campaign.py",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _visible_path(episode: dict) -> list[int]:
    body = [token for token in episode["tokens"][1:-3] if token >= 16]
    if not body or len(body) % 2:
        raise ValueError("Malformed visible row stream")
    rows = list(zip(body[::2], body[1::2], strict=True))
    mapping = {left: (index, right) for index, (left, right) in enumerate(rows)}
    if len(mapping) != len(rows):
        raise ValueError("Visible mapping is ambiguous")
    value = episode["tokens"][-2]
    path = []
    for _ in range(min(episode["hops"], 2)):
        if value not in mapping:
            raise ValueError("Visible target row missing")
        index, value = mapping[value]
        path.append(index)
    if episode["hops"] <= 2 and value != episode["answer"]:
        raise ValueError("Visible answer differs")
    return path


def _trace(path: Path, baseline: list[dict], arm: str, rep: int) -> dict:
    rows = json.loads(gzip.decompress(path.read_bytes()))
    if len(rows) != STEPS or len(baseline) != STEPS:
        raise ValueError("Training trace incomplete")
    first_loss, final_loss = None, None
    for step, (row, control) in enumerate(zip(rows, baseline, strict=True)):
        if (row["step"] != step or row["arm"] != arm or
                row["replicate"] != rep or
                row["answer_input_sha256"] != control[
                    "answer_input_sha256"] or
                row["answer_label_sha256"] != control[
                    "answer_label_sha256"] or
                set(row["losses"]) != ({"answer", "route"} if arm.endswith(
                    "_route") else {"answer"})):
            raise ValueError("Training trace/exposure differs")
        if not all(math.isfinite(value) and value >= 0
                   for value in row["losses"].values()) or not math.isfinite(
                       row["gradient_norm"]):
            raise ValueError("Nonfinite training trace")
        if step == 0:
            first_loss = row["losses"]["answer"]
        final_loss = row["losses"]["answer"]
    return {"steps": len(rows), "initial_answer_loss": first_loss,
            "final_answer_loss": final_loss}


def _competent(panels: dict) -> bool:
    def item(name: str) -> float:
        return panels[name]["items"]["accuracy"]
    return (item("first_hop_confirm") >= .90 and
            item("direct_confirm") >= .90 and
            item("copy_confirm") >= .95 and
            panels["factorial"]["groups"]["accuracy"] >= .70)


def audit(root: Path) -> dict:
    result_dir = root / "results/TEACH-0022"
    trained_path = result_dir / "status.json"
    score_path = result_dir / "score-status.json"
    benchmark_path = result_dir / "benchmark.json"
    suite_path = result_dir / "suite-audit.json"
    trained = json.loads(trained_path.read_text())
    score = json.loads(score_path.read_text())
    benchmark = json.loads(benchmark_path.read_text())
    suite = json.loads(suite_path.read_text())
    if (trained["status"] != "complete" or score["status"] != "complete" or
            benchmark["status"] != "pass" or suite["audit"] != "pass" or
            suite != audit_suites(root)):
        raise ValueError("Campaign gate incomplete")
    if (score["training_status_sha256"] != sha(trained_path) or
            score["suite_audit_sha256"] != sha(suite_path) or
            trained["benchmark_sha256"] != sha(benchmark_path) or
            trained["suite_audit_sha256"] != sha(suite_path) or
            score["scope"] != "fresh_84411_84511_once"):
        raise ValueError("Campaign provenance differs")
    if (benchmark["config"]["steps"] != STEPS or
            benchmark["config"]["batch"] != 32 or
            benchmark["config"]["route_weight"] != .2 or
            benchmark["conservative_projected_seconds"] >= 4 * 3600 or
            trained["artifact_bytes"] > 2 * 1024**3 or
            trained["sampled_mps_bytes"] > 12 * 1024**3):
        raise ValueError("Resource/config gate differs")
    head = trained["source_head"]
    if set(trained["source_sha256"]) != set(SOURCE_PATHS) or (
            trained["source_sha256"] != score["source_sha256"] or
            trained["source_sha256"] != benchmark["source_sha256"]):
        raise ValueError("Source list differs")
    for name in SOURCE_PATHS:
        committed = subprocess.run(["git", "show", f"{head}:{name}"],
                                   cwd=root, check=True, capture_output=True).stdout
        digest = hashlib.sha256(committed).hexdigest()
        if digest != trained["source_sha256"][name] or sha(root / name) != digest:
            raise ValueError(f"Campaign source differs: {name}")
    trace_results = {}
    for arm in NEW_ARMS:
        if set(trained["runs"][arm]) != {"0", "1"}:
            raise ValueError("Arm/replicate incomplete")
        trace_results[arm] = {}
        for rep in (0, 1):
            baseline_path = root / f"results/TEACH-0014-v3/losses-rep{rep}-oracle_rows_workspace.json.gz"
            if sha(baseline_path) != trained["baseline"]["files"][str(rep)][
                    "loss_sha256"]:
                raise ValueError("Baseline trace differs")
            baseline = json.loads(gzip.decompress(baseline_path.read_bytes()))
            path = result_dir / f"losses-rep{rep}-{arm}.json.gz"
            checkpoint = root / f"outputs/TEACH-0022/rep{rep}-{arm}-step6000.pt"
            row = trained["runs"][arm][str(rep)]
            if (row["status"] != "complete" or sha(path) != row[
                    "loss_sha256"] or sha(checkpoint) != row[
                        "checkpoint_sha256"]):
                raise ValueError("Training files differ")
            trace_results[arm][str(rep)] = _trace(path, baseline, arm, rep)
    if set(score["archives"]) != set(ARMS):
        raise ValueError("Score arms differ")
    scores = {}
    for arm in ARMS:
        scores[arm] = {}
        if set(score["archives"][arm]) != {
                f"{rep}/{seed}" for rep in (0, 1) for seed in SEEDS}:
            raise ValueError("Score split/replicate set differs")
        for rep in (0, 1):
            scores[arm][str(rep)] = {}
            for seed in SEEDS:
                manifest = json.loads((root / f"outputs/TEACH-0022/suite-{seed}.json").read_text())
                checked = audit_manifest(manifest)
                if checked["manifest_sha256"] != suite["new"][str(seed)][
                        "canonical_sha256"]:
                    raise ValueError("Score suite differs")
                path = result_dir / f"predictions-{arm}-rep{rep}-seed{seed}.json.gz"
                entry = score["archives"][arm][f"{rep}/{seed}"]
                if sha(path) != entry["sha256"] or path.stat().st_size != entry[
                        "bytes"]:
                    raise ValueError("Prediction file differs")
                payload = json.loads(gzip.decompress(path.read_bytes()))
                if (payload["arm"] != arm or payload["replicate"] != rep or
                        payload["seed"] != seed or set(payload[
                            "panels"]) != set(manifest["panels"])):
                    raise ValueError("Prediction identity differs")
                panel_scores = {}
                for name, episodes in manifest["panels"].items():
                    panel = payload["panels"][name]
                    rows = panel["rows"]
                    if len(rows) != len(episodes) or [
                            sample["index"] for sample in panel[
                                "samples"]] != [0, len(episodes) // 2,
                                                len(episodes) - 1]:
                        raise ValueError("Prediction panel length/samples differ")
                    predictions = []
                    first_hits, second_hits = 0, 0
                    for episode, row in zip(episodes, rows, strict=True):
                        path_indices = _visible_path(episode)
                        body = [token for token in episode["tokens"][1:-3]
                                if token >= 16]
                        if (row["render_id"] != episode["render_id"] or
                                type(row["prediction"]) is not int or
                                not 16 <= row["prediction"] < 2064 or
                                len(row["native_argmax_rows"]) != len(
                                    path_indices) or any(
                                        not 0 <= index < len(body) // 2
                                        for index in row[
                                            "native_argmax_rows"])):
                            raise ValueError("Prediction row differs")
                        predictions.append(row["prediction"])
                        first_hits += bool(path_indices and row[
                            "native_argmax_rows"][0] == path_indices[0])
                        second_hits += bool(len(path_indices) > 1 and row[
                            "native_argmax_rows"][1] == path_indices[1])
                    for sample in panel["samples"]:
                        vector = sample["logits"]
                        index = sample["index"]
                        if (sample["render_id"] != episodes[index][
                                "render_id"] or len(vector) != 2064 or
                                any(not math.isfinite(value)
                                    for value in vector) or
                                16 + max(range(2048), key=lambda j: vector[
                                    16 + j]) != rows[index]["prediction"]):
                            raise ValueError("Sample logits differ")
                    panel_scores[name] = {
                        **_score_panel(name, [episode["answer"]
                                             for episode in episodes],
                                       predictions),
                        "first_target_row_hits": first_hits,
                        "second_target_row_hits": second_hits}
                scores[arm][str(rep)][str(seed)] = {
                    "panels": panel_scores,
                    "competent": _competent(panel_scores)}
    decisions = {}
    for rep in ("0", "1"):
        def cc(arm: str) -> float:
            return scores[arm][rep]["84511"]["panels"][
                "composed_confirm_confirm"]["items"]["accuracy"]
        decisions[rep] = {
            "linked_answer_minus_baseline": cc("linked_answer") -
                                            cc("oracle_rows_workspace"),
            "linked_route_minus_unlinked_route": cc("linked_route") -
                                                 cc("unlinked_route"),
            "competent": {arm: scores[arm][rep]["84511"][
                "competent"] for arm in ARMS}}
    return {"audit": "pass", "scope": "fresh_confirmation_once",
            "trace": trace_results, "scores": scores,
            "decisions": decisions,
            "both_seed_linked_answer_margin": all(decisions[rep][
                "linked_answer_minus_baseline"] >= .15 for rep in ("0", "1")),
            "both_seed_linked_route_margin": all(decisions[rep][
                "linked_route_minus_unlinked_route"] >= .15
                for rep in ("0", "1")),
            "score_status_sha256": sha(score_path)}


if __name__ == "__main__":
    root = Path.cwd()
    result = audit(root)
    path = root / "results/TEACH-0022/audit.json"
    if path.exists():
        raise FileExistsError("No automatic TEACH-0022 audit overwrite")
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
