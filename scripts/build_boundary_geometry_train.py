"""Build a train-only image-gap / transcription-boundary alignment.

The external coordinate archive is read only for folios whose physical leaf is
assigned to training. The project test and validation text is not parsed here.
Derived rows are stored under gitignored data/processed; the tracked manifest
contains counts, hashes, and aggregate geometry, never candidate readings.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path


EXTERNAL_REV = "956a7c4fc39981f4d116fa3f4edfccce6d065571"
SOURCE_SHA = "bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc"
EXTERNAL_ZL_SHA = "8384ef2572444d076a9dfb22bb069808e331546832e0b456a4507adb376e7657"
SPLIT_SHA = "9fc80cb4b000fdd5d952b7c2416b37b6b92f07bc56486a63309142953ed6634e"
SUBSTITUTIONS = (("cth", "T"), ("ckh", "K"), ("cph", "P"), ("cfh", "F"),
                 ("ch", "C"), ("sh", "S"), ("iin", "N"), ("in", "I"), ("ee", "E"))
BAD = set("?*<>{}[]()|@;:.,0123456789'")
LOCUS = re.compile(r"^<(f[0-9]+[rv][0-9]*|fRos[0-9]*)\.\d+,([@+*=~])([A-Za-z])[^>]*>")
FOLIO_LEAF = re.compile(r"^f[0-9]+")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def leaf_id(folio: str) -> str:
    match = FOLIO_LEAF.match(folio)
    return match.group() if match else folio


def collapse(token: str) -> str:
    for old, new in SUBSTITUTIONS:
        token = token.replace(old, new)
    return token


def clean_body(raw: str) -> str:
    body = raw[raw.index(">") + 1 :].strip()
    body = re.sub(r"<[^>]*>", "", body)
    body = re.sub(r"\[([^:\]]*):[^\]]*\]", r"\1", body)
    return re.sub(r"\{[^}]*\}", "", body)


def parse_line(raw: str) -> list[tuple[str, str]]:
    """Return clean tokens with the *original adjacent* separator or L/X.

    X means a candidate boundary crossed malformed/uncertain material; it may
    align visually, but is not eligible for a physical-gap row.
    """
    parts = re.split(r"([.,])", clean_body(raw))
    words = [part.strip() for part in parts[::2]]
    marks = parts[1::2]
    output: list[tuple[str, str]] = []
    for index, word in enumerate(words):
        if not word or set(word) & BAD:
            continue
        if index < len(marks):
            following = words[index + 1]
            label = marks[index] if following and not set(following) & BAD else "X"
        else:
            label = "L"
        output.append((word, label))
    return output


def read_train_zl(path: Path, allowed_leaves: set[str]) -> dict[str, list[tuple[str, str]]]:
    folios: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            match = LOCUS.match(raw)
            if not match or match.group(3) != "P":
                continue
            folio = match.group(1)
            if leaf_id(folio) not in allowed_leaves:
                continue
            folios[folio].extend(parse_line(raw))
    return folios


def read_boxes(path: Path) -> list[dict]:
    vocabulary, entries = json.loads(path.read_text())
    words = [item[0] for item in vocabulary]
    output: list[dict] = []
    line = 0
    previous_x: int | None = None
    for entry in entries:
        x = int(entry[1])
        if previous_x is not None and x < previous_x - 3:
            line += 1
        output.append({"word": words[entry[0]], "x": x, "y": int(entry[2]),
                       "w": int(entry[3]), "h": int(entry[4]), "line": line})
        previous_x = x
    return output


def align_folio(folio: str, textual: list[tuple[str, str]], visual: list[dict]) -> tuple[list[dict], dict]:
    if not textual or not visual:
        return [], {"visual_tokens": len(visual), "text_tokens": len(textual),
                    "matched_tokens": 0, "eligible_pairs": 0}
    median_width = statistics.median(item["w"] for item in visual) or 1
    matcher = SequenceMatcher(a=[collapse(item["word"]) for item in visual],
                              b=[collapse(word) for word, _ in textual], autojunk=False)
    rows: list[dict] = []
    matched = 0
    for visual_start, text_start, length in matcher.get_matching_blocks():
        matched += length
        for offset in range(length - 1):
            vi, ti = visual_start + offset, text_start + offset
            left, right = visual[vi], visual[vi + 1]
            label = textual[ti][1]
            if left["line"] != right["line"] or label not in (".", ","):
                continue
            gap_px = right["x"] - left["x"] - left["w"]
            rows.append({"folio": folio, "leaf": leaf_id(folio),
                         "visual_line": left["line"], "visual_left_index": vi,
                         "text_left_index": ti, "label": label,
                         "gap_px": gap_px, "gap_over_median_word_width": gap_px / median_width,
                         "left_word": textual[ti][0], "right_word": textual[ti + 1][0]})
    return rows, {"visual_tokens": len(visual), "text_tokens": len(textual),
                  "matched_tokens": matched, "eligible_pairs": len(rows)}


def check_transcription_loci(project: Path, external: Path,
                             allowed_leaves: set[str]) -> int:
    def keyed(path: Path) -> dict[str, str]:
        rows = {}
        with path.open(encoding="utf-8", errors="replace") as stream:
            for raw in stream:
                if not raw.startswith("<") or ">" not in raw:
                    continue
                locator = raw[1:raw.index(">")]
                folio = locator.split(".", 1)[0]
                if leaf_id(folio) not in allowed_leaves:
                    continue
                rows[locator] = raw.rstrip("\n")
        return rows

    first, second = keyed(project), keyed(external)
    if first != second:
        raise AssertionError("Project and external ZL locus content differs")
    return len(first)


def run(root: Path, external: Path) -> dict:
    raw = root / "data/raw/ZL3b-n.txt"
    split = root / "data/manifests/zl3b_split.json"
    ext_raw = external / "data/voynich-units/ZL3b.txt"
    if sha(raw) != SOURCE_SHA or sha(split) != SPLIT_SHA or sha(ext_raw) != EXTERNAL_ZL_SHA:
        raise AssertionError("Frozen input checksum mismatch")
    revision = subprocess.check_output(["git", "-C", str(external), "rev-parse", "HEAD"], text=True).strip()
    if revision != EXTERNAL_REV:
        raise AssertionError("External archive revision mismatch")
    assignments = json.loads(split.read_text())["leaf_assignments"]
    allowed = {leaf for leaf, side in assignments.items() if side == "train"}
    same_loci = check_transcription_loci(raw, ext_raw, allowed)
    tokens = read_train_zl(raw, allowed)
    box_root = external / "data/voynich-units/morphometry_voynichese/voynichese_boxes"
    all_rows: list[dict] = []
    by_folio = {}
    box_hashes = {}
    for box in sorted(box_root.glob("*.js")):
        if leaf_id(box.stem) not in allowed:
            continue
        visual = read_boxes(box)
        rows, counts = align_folio(box.stem, tokens.get(box.stem, []), visual)
        all_rows.extend(rows)
        if counts["text_tokens"]:
            by_folio[box.stem] = counts
            box_hashes[box.stem] = sha(box)
    output = root / "data/processed/boundary_geometry/train_pairs.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows))
    labels = Counter(row["label"] for row in all_rows)
    gap = {label: [row["gap_over_median_word_width"] for row in all_rows if row["label"] == label]
           for label in (".", ",")}
    summary = {"schema_version": 1, "external_revision": revision,
               "external_zl_sha256": EXTERNAL_ZL_SHA, "project_zl_sha256": SOURCE_SHA,
               "split_sha256": SPLIT_SHA, "same_locus_lines": same_loci,
               "selection": "train-assigned physical leaves only; P loci; same-line exact collapsed-form alignments",
               "rows_sha256": sha(output), "rows": len(all_rows),
               "folios": len(by_folio), "by_folio": by_folio,
               "box_sha256_by_folio": box_hashes, "labels": dict(labels),
               "gap_median_by_label": {label: statistics.median(values) if values else None
                                       for label, values in gap.items()}}
    manifest = root / "data/manifests/boundary_geometry_train.json"
    manifest.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--external", type=Path,
                        default=Path("data/raw/external/voynich-units"))
    args = parser.parse_args()
    report = run(args.root, args.external)
    print(json.dumps({key: report[key] for key in
                      ("rows", "folios", "labels", "gap_median_by_label", "rows_sha256")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
