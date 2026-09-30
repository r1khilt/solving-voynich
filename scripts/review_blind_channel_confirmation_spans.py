"""Edition-specific bibliographic span ledger; no normalization or cipher scores.

Written after acquisition, as allowed by the pre-frozen corpus policy. The
original-Latin boundaries are identified by edition headings and chapter starts.
This is an evaluator's explicit source audit, not a language classification model.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from scripts.prepare_blind_channel_confirmation_corpora import ACQUISITION, digest

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "data/manifests/blind_channel_confirmation_spans.json"
AUDIT = ROOT / "results/BLIND-CHANNEL-CONFIRM-001/corpus_span_review.json"


def span(raw, start, end, **fields):
    return {"raw_byte_start": start, "raw_byte_end": end,
            "sha256": digest(raw[start:end]), **fields}


def unique(raw, text, *, after=0):
    matches = list(re.finditer(re.escape(text.encode()), raw[after:]))
    if len(matches) != 1:
        raise ValueError("Missing or ambiguous reviewed edition boundary")
    return after + matches[0].start()


def footnotes(raw, start, end):
    """Explicit notes: [number] at line start plus indented continuation lines.

    Blank separators and immediately following notes are within the same block.
    The next nonindented non-note line resumes the authorial body. No original
    body paragraph is removed based on vocabulary, statistics or recoverability.
    """
    lines = raw[start:end].splitlines(keepends=True)
    offset, pending, excluded, ids = start, None, [], []
    for line in lines:
        match = re.match(rb"^\[(\d+)\]", line)
        if match:
            ids.append(int(match[1]))
            if pending is None:
                pending = offset
        elif pending is not None and line.strip() and not line.startswith(b"    "):
            excluded.append(span(raw, pending, offset, reason="Numbered editorial footnote block with indented continuations"))
            pending = None
        offset += len(line)
    if pending is not None:
        excluded.append(span(raw, pending, end, reason="Numbered editorial footnote block with indented continuations"))
    return excluded, ids


def main():
    if PLAN.exists() or AUDIT.exists():
        raise FileExistsError("Reviewed span plan already exists; preserve any revision separately")
    acquisition = json.loads((ROOT / ACQUISITION).read_text())
    if acquisition["status"] != "downloaded_pending_body_review":
        raise ValueError("Successful pre-text acquisition is required")
    plan = {"schema_version": 1, "status": "mechanical_body_review_complete",
            "prefix_cap": 50000, "duplicate_window": 64, "sources": {}}
    audit = {"scope": "Bibliographic body and apparatus audit only; no normalization/source/cipher scores",
             "review_script_sha256": digest(Path(__file__).read_bytes()),
             "acquisition_sha256": digest((ROOT / ACQUISITION).read_bytes()), "sources": {}}
    for name in ("sallust", "tacitus"):
        item = acquisition["sources"][name]
        raw = (ROOT / item["raw"]["path"]).read_bytes()
        if digest(raw) != item["raw_sha256"]:
            raise ValueError("Acquired edition changed")
        if b"Project Gutenberg License" not in raw:
            raise ValueError("License notice missing")
        if name == "sallust":
            start1 = unique(raw, "1. Omnes[1] homines")
            second_heading = unique(raw, "C. SALLUSTII CRISPI", after=start1)
            start2 = unique(raw, "1. Falso queritur de natura sua")
            end2 = unique(raw, "*** END OF THE PROJECT GUTENBERG EBOOK")
            boundaries = [("bellum_catilinarium", start1, second_heading, 61),
                          ("bellum_jugurthinum", start2, end2, 114)]
            edition = "PG7402; Schmitz/Zumpt edition identified in header; Latin chapters with interleaved numbered English notes; no spelling emendation"
        else:
            start1 = unique(raw, "I. Germania omnis a Gallis")
            end1 = unique(raw, "CN. JULII AGRICOLAE VITA.", after=start1)
            start2 = unique(raw, "I. Clarorum virorum facta moresque")
            end2 = unique(raw, "\r\nNOTES\r\n", after=start2)
            boundaries = [("germania", start1, end1, 46), ("agricola", start2, end2, 46)]
            edition = "PG9090; W. S. Tyler college-notes edition identified in title matter; separate original Germania/Agricola bodies; Latin breviaria and English notes outside spans"
        bodies, observations = [], []
        for body_id, start, end, expected in boundaries:
            exclusions, ids = footnotes(raw, start, end) if name == "sallust" else ([], [])
            if name == "sallust":
                chapters = [int(x[1]) for x in re.finditer(rb"^(\d+)\. ", raw[start:end], re.M)]
                if chapters != list(range(1, expected + 1)):
                    raise ValueError("Latin chapter sequence differs from reviewed edition")
            else:
                chapters = [x[1].decode() for x in re.finditer(rb"^([IVXL]+)\. ", raw[start:end], re.M)]
                if len(chapters) != expected or chapters[0] != "I" or chapters[-1] != "XLVI":
                    raise ValueError("Tacitus chapter boundaries differ")
            bodies.append(span(raw, start, end, body_id=body_id, editorial_exclusions=exclusions))
            counts = Counter(ids)
            observations.append({"body_id": body_id, "first_line": raw[:start].count(b"\n") + 1,
                "exclusive_end_line": raw[:end].count(b"\n") + 1, "chapter_count": len(chapters),
                "editorial_blocks": len(exclusions), "editorial_note_headers": len(ids),
                "repeated_note_numbers_preserved": {str(k): v for k, v in counts.items() if v > 1},
                "bytes_excluded": sum(s["raw_byte_end"] - s["raw_byte_start"] for s in exclusions)})
        plan["sources"][name] = {"raw_sha256": digest(raw), "rights_review": "public_domain_usa_notice_checked",
            "edition_review": edition, "reviewer": "Root assistant: frozen-policy evaluator, not an independent philologist",
            "bodies": bodies}
        audit["sources"][name] = {"observations": observations, "catalogue": item["catalogue_url"],
            "catalogue_rights_checked": "2026-09-30: public domain in USA; raw license notice retained",
            "scope_limit": "Mechanical bibliographic extraction, no manuscript collation or certification of authorial/editorial readings"}
    for path, payload in ((PLAN, plan), (AUDIT, audit)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
