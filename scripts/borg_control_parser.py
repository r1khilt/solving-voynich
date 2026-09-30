"""Lossless Borg format quarantine, deliberately not a cipher-glyph decoder.

Only byte offsets, hashes, page labels and aggregate counts leave this module's
CLI. Original cleartext and annotated material are never emitted as solver
records. ``unresolved_body`` is a lexical category, not a validated cipher unit.
The original bytes plus the offset partition reproduce every input byte.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/borg/transcription-0001r-0204v.txt"
PINNED_SHA256 = "79950123a2760e92f3f27287aca94d16168169651eac8a4f9c15af37e349dde3"
HEADER = re.compile(r"#(page|pahe)\s+(\d{1,4})([rv])(\.\d+)?\s*")
CLEARTEXT = re.compile(r"<CLEARTEXT-(?:LA|AR)>")
# Scope/meaning is unknown. Quarantine the entire physical line, not merely
# delete the marker and join its two neighbors into a falsely certain string.
UNRESOLVED_MARKERS = frozenset("?*/")


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class Page:
    block_index: int
    raw_label: str
    normalized_label_only: str
    header_tag: str
    start_line: int
    end_line: int
    start_char: int
    end_char: int
    explicit_cleartext: bool


@dataclass(frozen=True)
class Span:
    kind: str
    start_char: int
    end_char: int
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    block_index: int | None
    sha256: str


@dataclass(frozen=True)
class Parse:
    source_sha256: str
    source_bytes: int
    source_chars: int
    physical_lines: int
    pages: tuple[Page, ...]
    spans: tuple[Span, ...]
    annotations: tuple[Span, ...]
    marker_lines: tuple[int, ...]


def _lines(text: str) -> tuple[list[str], list[int]]:
    """Preserve CRLF, bare CR, bare LF and the missing final newline exactly."""
    # Other Unicode line separators are not part of this source's documented
    # physical-line syntax; rejecting them avoids two competing line indices.
    if any(char in text for char in "\v\f\x1c\x1d\x1e\x85\u2028\u2029"):
        raise ValueError("Unsupported physical-line separator")
    rows = text.splitlines(keepends=True)
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    return rows, offsets


def parse_bytes(blob: bytes, *, expected_sha256: str | None = None) -> Parse:
    """Return a complete source-bound offset partition; never infer plaintext.

    ``expected_sha256=None`` is for artificial fixtures. The command line fixes
    the published source hash and offers no switch to bypass that check.
    """
    actual = digest(blob)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("Source SHA-256 differs from the pinned bytes")
    text = blob.decode("utf-8", errors="strict")
    rows, line_offsets = _lines(text)
    byte_offsets = [0]
    for char in text:
        byte_offsets.append(byte_offsets[-1] + len(char.encode("utf-8")))
    labels = ["unresolved_body"] * len(text)
    page_ids: list[int | None] = [None] * len(text)
    headers = []
    for line, row in enumerate(rows, 1):
        body = row.rstrip("\r\n")
        match = HEADER.fullmatch(body)
        if match:
            headers.append((line, match))
        elif re.match(r"\s*#(?:page|pahe)\b", body):
            raise ValueError(f"Malformed page header at line {line}")
    if not headers:
        raise ValueError("No page headers")

    pages = []
    for index, (line, match) in enumerate(headers):
        end_line = headers[index + 1][0] - 1 if index + 1 < len(headers) else len(rows)
        start, end = line_offsets[line - 1], line_offsets[end_line]
        clear = any(CLEARTEXT.fullmatch(row.strip()) for row in rows[line:end_line])
        page = Page(index, match[2] + match[3] + (match[4] or ""),
                    f"{int(match[2]):04}{match[3]}" + (match[4] or ""), match[1],
                    line, end_line, start, end, clear)
        pages.append(page)
        page_ids[start:end] = [index] * (end - start)
        if clear:
            labels[start:end] = ["explicit_cleartext_page"] * (end - start)
    labels[:pages[0].start_char] = ["preamble"] * pages[0].start_char
    header_lines = {page.start_line for page in pages}
    comment_lines = set()
    for line, row in enumerate(rows, 1):
        if row.startswith("#"):
            start, end = line_offsets[line - 1:line + 1]
            kind = "page_header" if line in header_lines else "hash_comment"
            labels[start:end] = [kind] * (end - start)
            comment_lines.add(line)

    def span(kind: str, start: int, end: int) -> Span:
        return Span(kind, start, end, byte_offsets[start], byte_offsets[end],
                    bisect.bisect_right(line_offsets, start),
                    bisect.bisect_right(line_offsets, end - 1), page_ids[start],
                    digest(blob[byte_offsets[start]:byte_offsets[end]]))

    # Structural balancing only: no assumption about the bracket content's
    # language or whether it is cleartext, an uncertain glyph, or an editor note.
    annotations = []
    stack: list[tuple[str, int]] = []
    for offset, char in enumerate(text):
        if labels[offset] in {"page_header", "hash_comment"}:
            if stack:
                raise ValueError("Annotation crosses a page header or hash comment")
            continue
        if char in "[<":
            stack.append((char, offset))
        elif char in "]>":
            if not stack or stack[-1][0] != {"]": "[", ">": "<"}[char]:
                raise ValueError(f"Mismatched annotation delimiter at character {offset}")
            left, start = stack.pop()
            if not stack:
                if page_ids[start] != page_ids[offset]:
                    raise ValueError("Annotation crosses a page boundary")
                kind = "square_annotation" if left == "[" else "angle_annotation"
                annotations.append(span(kind, start, offset + 1))
                # Keep whole explicitly cleartext pages quarantined as pages.
                for position in range(start, offset + 1):
                    if labels[position] == "unresolved_body":
                        labels[position] = kind
    if stack:
        raise ValueError("Unclosed annotation delimiter")

    marker_lines = []
    for line, row in enumerate(rows, 1):
        if line in comment_lines:
            continue
        start, end = line_offsets[line - 1:line + 1]
        if any(labels[i] == "unresolved_body" and text[i] in UNRESOLVED_MARKERS
               for i in range(start, end)):
            marker_lines.append(line)
            for position in range(start, end):
                if labels[position] == "unresolved_body":
                    labels[position] = "unresolved_marker_line"
    for offset, char in enumerate(text):
        if char.isspace() and labels[offset] == "unresolved_body":
            labels[offset] = "whitespace"

    # A record cannot silently bridge a line/page, annotation, or excluded span.
    # Split at every physical line as well as every lexical-category change.
    spans = []
    line_boundaries = set(line_offsets)
    start = 0
    for end in range(1, len(text) + 1):
        if (end == len(text) or end in line_boundaries or labels[end] != labels[start]
                or page_ids[end] != page_ids[start]):
            spans.append(span(labels[start], start, end))
            start = end
    result = Parse(actual, len(blob), len(text), len(rows), tuple(pages), tuple(spans),
                   tuple(annotations), tuple(marker_lines))
    validate_partition(blob, result)
    return result


def validate_partition(blob: bytes, parsed: Parse) -> None:
    """Verify contiguous coverage, raw-to-Unicode offsets and each source slice."""
    if digest(blob) != parsed.source_sha256 or len(blob) != parsed.source_bytes:
        raise ValueError("Partition is for different source bytes")
    char_cursor = byte_cursor = 0
    for item in parsed.spans:
        if (item.start_char != char_cursor or item.start_byte != byte_cursor
                or item.end_char <= item.start_char or item.end_byte <= item.start_byte):
            raise ValueError("Offset partition has a gap, overlap or empty interval")
        part = blob[item.start_byte:item.end_byte]
        if (digest(part) != item.sha256
                or len(part.decode("utf-8")) != item.end_char - item.start_char):
            raise ValueError("Source slice hash or Unicode offset differs")
        char_cursor, byte_cursor = item.end_char, item.end_byte
    if byte_cursor != len(blob) or char_cursor != parsed.source_chars:
        raise ValueError("Offset partition does not cover the entire input")


def redacted_summary(blob: bytes, parsed: Parse) -> dict:
    """Count literal unresolved codepoints only; do not print source snippets."""
    validate_partition(blob, parsed)
    chars_by_kind, bytes_by_kind, spans_by_kind = Counter(), Counter(), Counter()
    unresolved_counts = Counter()
    unresolved_lines, unresolved_pages = set(), set()
    for item in parsed.spans:
        chars_by_kind[item.kind] += item.end_char - item.start_char
        bytes_by_kind[item.kind] += item.end_byte - item.start_byte
        spans_by_kind[item.kind] += 1
        if item.kind == "unresolved_body":
            unresolved_counts.update(blob[item.start_byte:item.end_byte].decode("utf-8"))
            unresolved_lines.add(item.start_line)
            unresolved_pages.add(item.block_index)
    page_labels = Counter(page.normalized_label_only for page in parsed.pages)
    return {
        "schema_version": 1,
        "status": "lossless_structural_quarantine_only_not_solver_ready",
        "source_sha256": parsed.source_sha256,
        "source_bytes": parsed.source_bytes,
        "source_chars": parsed.source_chars,
        "physical_lines": parsed.physical_lines,
        "page_blocks": len(parsed.pages),
        "partition_spans": len(parsed.spans),
        "partition_verified": True,
        "bytes_by_kind": dict(sorted(bytes_by_kind.items())),
        "chars_by_kind": dict(sorted(chars_by_kind.items())),
        "spans_by_kind": dict(sorted(spans_by_kind.items())),
        "annotation_spans": dict(sorted(Counter(x.kind for x in parsed.annotations).items())),
        "multiline_annotations": sum(x.start_line != x.end_line for x in parsed.annotations),
        "explicit_cleartext_blocks": [x.block_index for x in parsed.pages if x.explicit_cleartext],
        "duplicate_normalized_labels": {k: v for k, v in page_labels.items() if v > 1},
        "suffixed_block_indices": [x.block_index for x in parsed.pages if "." in x.raw_label],
        "typo_header_block_indices": [x.block_index for x in parsed.pages if x.header_tag != "page"],
        "unresolved_marker_lines": len(parsed.marker_lines),
        "unresolved_body": {
            "nonspace_codepoints": sum(unresolved_counts.values()),
            "codepoint_types": len(unresolved_counts),
            "physical_lines": len(unresolved_lines),
            "page_blocks": len(unresolved_pages),
            "codepoint_counts": {f"U+{ord(char):04X}": count
                                 for char, count in sorted(unresolved_counts.items())},
        },
        "solver_records_produced": 0,
        "glyph_unit_semantics_verified": False,
        "remaining_body_proven_ciphertext_only": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RAW)
    parser.add_argument("--ledger", type=Path, help="Optional offset-only ledger; refuses overwrite")
    args = parser.parse_args()
    blob = args.source.read_bytes()
    parsed = parse_bytes(blob, expected_sha256=PINNED_SHA256)
    summary = redacted_summary(blob, parsed)
    if args.ledger:
        # No original source content or cipher strings are serialized.
        ledger = {"summary": summary, "pages": [asdict(x) for x in parsed.pages],
                  "partition": [asdict(x) for x in parsed.spans],
                  "annotations": [asdict(x) for x in parsed.annotations],
                  "marker_lines": parsed.marker_lines}
        payload = (json.dumps(ledger, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with args.ledger.open("xb") as stream:
            stream.write(payload)
        summary["ledger_sha256"] = digest(payload)
        summary["ledger_bytes"] = len(payload)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
