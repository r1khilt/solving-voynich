"""Extract pinned P1/P2/D Latin bodies; never fetch or open final authors.

Raw editions and derived text/maps stay ignored. The tracked manifest contains
hashes, exact byte spans, exclusions and counts, not bulk literary text.
"""
from __future__ import annotations

import hashlib
import json
import re
import string
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALPHABET = "".join(c for c in string.ascii_lowercase if c not in "jvw")
CAP = 50_000
DUPLICATE_WINDOW = 64
SPECS = {
    "caesar": {
        "role": "P1", "pg": 218, "author": "Julius Caesar",
        "path": "data/raw/external/voynich-units/voynich_decipherment_repro_bundle/decipherment_attack_v6/lm_corpora/caesar_la.txt",
        "sha256": "84ac8411841a4d8f5f4a49b6a2cd1f466917c6a5af72916d5e0b2b1ecb2f659c",
        "heading": r"^C\. IULI CAESARIS DE BELLO GALLICO COMMENTARIUS [A-Z]+\r?$",
        "expected_bodies": 4,
    },
    "virgil": {
        "role": "P2", "pg": 227, "author": "Virgil",
        "path": "data/raw/blind-channel-development/pg227.txt",
        "sha256": "2620ba82c00964d1a1f0458c8cf171946f158a06de0d0e764594170602c4c7b8",
        "heading": r"^  LIBER [IVX]+\r?$", "expected_bodies": 12,
    },
    "cicero": {
        "role": "D", "pg": 226, "author": "Marcus Tullius Cicero",
        "path": "data/raw/blind-channel-development/pg226.txt",
        "sha256": "d35576eee076806a236f7f9edf8c1a07462a4102084e025076f6227d8847afdb",
        "heading": r"^M\. TULLI CICERONIS\r?$", "expected_bodies": 4,
        "openings": ["Quo usque tandem abutere,", "Tandem aliquando, Quirites,",
                     "Rem publicam, Quirites,", "Video, patres conscripti,"],
    },
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def bracket_intervals(text: str) -> list[tuple[int, int]]:
    """Exclude balanced square brackets including their contents; fail closed."""
    depth, start = 0, None
    result = []
    for i, char in enumerate(text):
        if char == "[":
            if depth == 0:
                start = i
            depth += 1
        elif char == "]":
            if depth == 0:
                raise ValueError("Unmatched closing bracket")
            depth -= 1
            if depth == 0:
                result.append((start, i + 1))
    if depth:
        raise ValueError("Unmatched opening bracket")
    return result


def normalize_segment(text: str, offset: int = 0) -> tuple[str, list[dict], list[dict]]:
    """Retain a normalized-token map to original Unicode character intervals.

    Punctuation/whitespace delimit tokens; digits or mixed alphanumerics are
    excluded whole. Bracketed contents are excluded without joining pieces of
    a word. Latin ligatures expand, combining marks drop, j->i and v->u.
    Unknown alphabetic characters fail instead of silently disappearing.
    """
    excluded = []
    masked = list(text)
    for start, end in bracket_intervals(text):
        masked[start:end] = " " * (end - start)
        excluded.append({"reason": "square_bracket_span", "start": offset + start,
                         "end": offset + end, "sha256": digest(text[start:end].encode())})
    tokens, pieces, position = [], [], 0
    for match in re.finditer(r"[^\W_]+", "".join(masked)):
        token = match.group()
        if any(char.isdigit() for char in token):
            excluded.append({"reason": "numeric_or_mixed_token", "start": offset + match.start(),
                             "end": offset + match.end(), "sha256": digest(token.encode())})
            continue
        expanded = token.lower().replace("æ", "ae").replace("œ", "oe")
        expanded = "".join(c for c in unicodedata.normalize("NFKD", expanded)
                           if not unicodedata.combining(c)).replace("j", "i").replace("v", "u")
        if not expanded or set(expanded) - set(ALPHABET):
            raise ValueError(f"Unsupported alphabetic token at {offset + match.start()}")
        tokens.append({"text_start": position, "text_end": position + len(expanded),
                       "raw_character_start": offset + match.start(),
                       "raw_character_end": offset + match.end()})
        pieces.append(expanded)
        position += len(expanded)
    return "".join(pieces), tokens, excluded


def body_intervals(text: str, spec: dict) -> list[tuple[int, int]]:
    end_markers = list(re.finditer(r"^\*\*\* END OF THE PROJECT GUTENBERG EBOOK", text, re.M))
    if len(end_markers) != 1 or "Project Gutenberg License" not in text:
        raise ValueError("Missing/ambiguous ebook boundary or license notice")
    end = end_markers[0].start()
    headings = list(re.finditer(spec["heading"], text[:end], re.M))
    if len(headings) != spec["expected_bodies"]:
        raise ValueError("Edition headings changed")
    intervals = []
    for i, heading in enumerate(headings):
        stop = headings[i + 1].start() if i + 1 < len(headings) else end
        start = heading.end()
        if "openings" in spec:
            opening = spec["openings"][i]
            candidates = list(re.finditer(r"^" + re.escape(opening), text[start:stop], re.M))
            if len(candidates) != 1:
                raise ValueError("Missing/ambiguous authorial opening")
            start += candidates[0].start()
        intervals.append((start, stop))
    return intervals


def extract(raw: bytes, spec: dict, cap: int = CAP) -> tuple[dict, dict]:
    if len(raw) > 2 * 1024 * 1024 or digest(raw) != spec["sha256"]:
        raise ValueError("Raw edition exceeds cap or differs from pinned bytes")
    text = raw.decode("utf-8", errors="strict")
    byte_offsets = [0]
    for char in text:
        byte_offsets.append(byte_offsets[-1] + len(char.encode("utf-8")))
    pieces, token_map, exclusions, spans, normalized_position = [], [], [], [], 0
    for start, end in body_intervals(text, spec):
        clean, tokens, excluded = normalize_segment(text[start:end], start)
        spans.append({"raw_byte_start": byte_offsets[start], "raw_byte_end": byte_offsets[end],
                      "normalized_start": normalized_position,
                      "normalized_end": normalized_position + len(clean),
                      "raw_sha256": digest(raw[byte_offsets[start]:byte_offsets[end]])})
        for token in tokens:
            token["text_start"] += normalized_position
            token["text_end"] += normalized_position
            token["raw_byte_start"] = byte_offsets[token.pop("raw_character_start")]
            token["raw_byte_end"] = byte_offsets[token.pop("raw_character_end")]
        for item in excluded:
            item["raw_byte_start"] = byte_offsets[item.pop("start")]
            item["raw_byte_end"] = byte_offsets[item.pop("end")]
        pieces.append(clean)
        token_map.extend(tokens)
        exclusions.extend(excluded)
        normalized_position += len(clean)
    full = "".join(pieces)
    if len(full) < cap:
        raise ValueError("Insufficient eligible body text; do not borrow final-author text")
    selected = full[:cap]
    payload = {"schema_version": 1, "text": selected,
               "tokens": [t for t in token_map if t["text_start"] < cap],
               "body_boundaries": [min(span["normalized_end"], cap) for span in spans
                                   if span["normalized_start"] < cap]}
    # Final token may extend past the cap: original span retained and flagged.
    for token in payload["tokens"]:
        token["selected_end"] = min(token["text_end"], cap)
    summary = {"role": spec["role"], "author": spec["author"], "raw_path": spec["path"],
               "catalogue_url": f"https://www.gutenberg.org/ebooks/{spec['pg']}",
               "advertised_text_url": f"https://www.gutenberg.org/ebooks/{spec['pg']}.txt.utf-8",
               "raw_sha256": digest(raw), "raw_bytes": len(raw),
               "eligible_characters": len(full), "selected_characters": len(selected),
               "full_normalized_sha256": digest(full.encode()),
               "selected_text_sha256": digest(selected.encode()), "body_spans": spans,
               "excluded_spans": exclusions, "excluded_counts": dict(Counter(x["reason"] for x in exclusions)),
               "observed_codepoints": dict(sorted(Counter(c for c in text if ord(c) > 127).items())),
               "edition": "Pinned PG electronic edition; source print edition/editor not established",
               "rights_screen": "PG catalogue public domain in USA; ebook license notice retained in raw"}
    return payload, summary


def overlap_counts(texts: dict[str, str], width: int = DUPLICATE_WINDOW) -> list[dict]:
    """All sliding exact windows in selected normalized prefixes; no fuzzy claim."""
    if width < 1:
        raise ValueError("Window must be positive")
    windows = {name: Counter(text[i:i + width] for i in range(len(text) - width + 1))
               for name, text in texts.items()}
    result = []
    names = sorted(texts)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            shared = windows[first].keys() & windows[second].keys()
            result.append({"first": first, "second": second, "window": width,
                           "shared_unique_windows": len(shared),
                           "first_positions": sum(windows[first][s] for s in shared),
                           "second_positions": sum(windows[second][s] for s in shared)})
    return result


def main() -> None:
    output = ROOT / "data/processed/blind-channel-development"
    output.mkdir(parents=True, exist_ok=True)
    sources, texts = {}, {}
    for name, spec in SPECS.items():
        payload, summary = extract((ROOT / spec["path"]).read_bytes(), spec)
        path = output / f"{name}.json"
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        path.write_bytes(encoded)
        summary.update(derived_path=str(path.relative_to(ROOT)), derived_sha256=digest(encoded))
        sources[name], texts[name] = summary, payload["text"]
    overlaps = overlap_counts(texts)
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "development_only", "source_code_sha256": digest(Path(__file__).read_bytes()),
                "alphabet": ALPHABET, "prefix_cap": CAP, "duplicate_screen_window": DUPLICATE_WINDOW,
                "normalization": "NFKD; lower; ae/oe ligatures; remove combining marks; j->i,v->u; omit spaces; reject unknown letters; remove square bracket spans and entire numeric/mixed tokens",
                "sources": sources, "exact_overlap_screen": overlaps,
                "final_authors_downloaded": False, "paid_cost_usd": 0}
    path = ROOT / "data/manifests/blind_channel_development_corpora.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(path.relative_to(ROOT)),
                      "counts": {n: s["eligible_characters"] for n, s in sources.items()},
                      "overlap": overlaps}, indent=2))
    if any(row["shared_unique_windows"] for row in overlaps):
        raise SystemExit("Exact cross-author overlap found; review before fitting")


if __name__ == "__main__":
    main()
