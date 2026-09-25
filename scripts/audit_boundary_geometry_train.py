"""Independent source-parser replay for the train-only physical gap alignment.

Extracts only four data functions from the pinned external paper source by AST;
this avoids a Matplotlib dependency and keeps the comparison separate from our
producer's ZL and box parsers. It never scores validation/test text.
"""

from __future__ import annotations

import ast
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = ROOT / "data/raw/external/voynich-units"
AUTHOR_PARSER_SHA = "457b5cb34009577fe842cd1825d30df16fc656f87ae1dc42af67113a63762fee"
EXTRA_AUTHOR_PAIRS = {("f103r", 59, ".", 42), ("f2r", 71, ".", 6),
                      ("f55r", 55, ".", 12)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_functions(path: Path) -> dict:
    if sha(path) != AUTHOR_PARSER_SHA:
        raise AssertionError("External author parser changed")
    syntax = ast.parse(path.read_text())
    names = {"collapse", "clean_body", "zl_folio_tokens", "vy_tokens"}
    nodes = [node for node in syntax.body if
             (isinstance(node, ast.FunctionDef) and node.name in names) or
             (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
              and target.id == "SUBSTITUTIONS" for target in node.targets))]
    if len(nodes) != 5:
        raise AssertionError("External parser AST shape changed")
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = {"json": json, "re": re, "Path": Path}
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace


def run() -> dict:
    manifest_path = ROOT / "data/manifests/boundary_geometry_train.json"
    rows_path = ROOT / "data/processed/boundary_geometry/train_pairs.jsonl"
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    if manifest["rows_sha256"] != sha(rows_path) or manifest["rows"] != len(rows):
        raise AssertionError("Row hash/count mismatch")
    if manifest["external_revision"] != "956a7c4fc39981f4d116fa3f4edfccce6d065571":
        raise AssertionError("External revision mismatch")
    split_path = ROOT / "data/manifests/zl3b_split.json"
    if sha(split_path) != manifest["split_sha256"]:
        raise AssertionError("Split changed")
    allowed = {leaf for leaf, side in json.loads(split_path.read_text())["leaf_assignments"].items()
               if side == "train"}
    if any(row["leaf"] not in allowed for row in rows):
        raise AssertionError("Nontraining leaf in derived rows")
    if len({(row["folio"], row["visual_left_index"]) for row in rows}) != len(rows):
        raise AssertionError("Duplicate physical boundary")
    if set(manifest["by_folio"]) != {row["folio"] for row in rows}:
        raise AssertionError("Manifest/row folio mismatch")
    original = source_functions(EXTERNAL / "analysis/reproduce_headlines.py")
    root = EXTERNAL / "data/voynich-units"
    author: set[tuple[str, int, str, int]] = set()
    all_median_width: dict[str, float] = {}
    for folio in sorted(manifest["by_folio"]):
        if re.match(r"^f[0-9]+", folio).group() not in allowed:
            raise AssertionError("Nontraining box requested")
        box_path = root / "morphometry_voynichese/voynichese_boxes" / (folio + ".js")
        if sha(box_path) != manifest["box_sha256_by_folio"][folio]:
            raise AssertionError(f"Box changed: {folio}")
        words, labels = original["zl_folio_tokens"](root / "ZL3b.txt", folio)
        boxes = original["vy_tokens"](box_path)
        all_median_width[folio] = statistics.median(box["w"] for box in boxes) or 1
        matcher = SequenceMatcher(
            a=[original["collapse"](box["word"]) for box in boxes],
            b=[original["collapse"](word) for word in words], autojunk=False)
        for visual_start, text_start, length in matcher.get_matching_blocks():
            for offset in range(length - 1):
                vi, ti = visual_start + offset, text_start + offset
                left, right = boxes[vi], boxes[vi + 1]
                if left["line"] == right["line"] and labels[ti] in (".", ","):
                    author.add((folio, vi, labels[ti], right["x"] - left["x"] - left["w"]))
    produced = {(row["folio"], row["visual_left_index"], row["label"], row["gap_px"])
                for row in rows}
    if produced != author - EXTRA_AUTHOR_PAIRS:
        raise AssertionError(f"Physical rows disagree: producer-only {len(produced-author)}, "
                             f"author-only {len(author-produced)}")
    for row in rows:
        exact = row["gap_px"] / all_median_width[row["folio"]]
        if abs(exact - row["gap_over_median_word_width"]) > 1e-12:
            raise AssertionError("Gap normalization mismatch")
    labels = Counter(row["label"] for row in rows)
    if dict(labels) != manifest["labels"]:
        raise AssertionError("Label totals mismatch")
    for label in (".", ","):
        value = statistics.median(row["gap_over_median_word_width"] for row in rows
                                  if row["label"] == label)
        if abs(value - manifest["gap_median_by_label"][label]) > 1e-12:
            raise AssertionError("Median mismatch")
    result = {"status": "pass", "producer_rows": len(produced),
              "author_rows": len(author), "conservative_exclusions": len(EXTRA_AUTHOR_PAIRS),
              "folios": len(manifest["by_folio"]), "row_sha256": sha(rows_path),
              "manifest_sha256": sha(manifest_path)}
    path = ROOT / "results/BOUNDARY-CHANNEL-0001/audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
