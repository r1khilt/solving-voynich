"""Evaluator-only acquisition and extraction; never generate keys or fit models.

Running this module with --help does not access corpora or the network. Actual
acquisition and preparation require an already committed source/protocol freeze.
Body spans are supplied only after an explicit edition/rights audit; this module
does not guess which Latin-looking text is authorial rather than editorial.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from scripts.build_blind_channel_development_corpora import (
    ALPHABET, CAP, DUPLICATE_WINDOW, SPECS, digest, extract, normalize_segment,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_LIMIT = 2 * 1024 * 1024
RECORD_LENGTH = 224
WINDOW_STRIDE = 256
KEY_COUNT = 8
KEY_WINDOW_BLOCK = 16
DEVELOPMENT = "data/manifests/blind_channel_development_corpora.json"
ACQUISITION = "data/manifests/blind_channel_confirmation_acquisition.json"
MANIFEST = "data/manifests/blind_channel_confirmation_corpora.json"
FINAL = {
    "sallust": {"role": "F", "author": "Sallust", "pg": 7402},
    "tacitus": {"role": "T", "author": "Cornelius Tacitus", "pg": 9090},
}
SOURCE_PATHS = [
    "scripts/prepare_blind_channel_confirmation_corpora.py",
    "scripts/build_blind_channel_development_corpora.py",
    "tests/test_blind_channel_confirmation_corpora.py",
    "docs/research/final-corpus-provenance-plan.md", DEVELOPMENT,
]


def encoded_json(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def save_new(path: Path, raw: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(raw), "bytes": len(raw)}


def require_freeze(commit: str, paths: list[str]) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Need a complete immutable Git commit")
    for name in paths:
        current = (ROOT / name).read_bytes()
        frozen = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        if current != frozen:
            raise ValueError(f"File differs from the registered freeze: {name}")


def allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    return (parsed.scheme == "https" and parsed.hostname in {"www.gutenberg.org", "gutenberg.org"}
            and parsed.username is None and parsed.password is None and parsed.port in {None, 443})


class GutenbergRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not allowed_url(newurl):
            raise ValueError("Download redirect left the registered HTTPS source")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url: str, *, opener=None) -> tuple[bytes, dict]:
    """At most 2 MiB; no fallback edition, retry, text logging or silent truncation."""
    if not allowed_url(url):
        raise ValueError("Unregistered download host")
    opener = opener or build_opener(GutenbergRedirect())
    request = Request(url, headers={"User-Agent": "VoynichCorpusResearch/1.0", "Accept-Encoding": "identity"})
    with opener.open(request, timeout=60) as response:
        if not allowed_url(response.geturl()):
            raise ValueError("Response left the registered source")
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise ValueError("Unexpected compressed transfer encoding")
        length = response.headers.get("Content-Length")
        if length is not None and (not length.isdigit() or int(length) > RAW_LIMIT):
            raise ValueError("Advertised download exceeds cap or has invalid length")
        raw = response.read(RAW_LIMIT + 1)
        if not raw or len(raw) > RAW_LIMIT:
            raise ValueError("Empty or oversized download")
        if length is not None and len(raw) != int(length):
            raise ValueError("Incomplete download")
        text = raw.decode("utf-8", errors="strict")
        if "Project Gutenberg License" not in text:
            raise ValueError("Raw ebook license notice missing")
        metadata = {"requested_url": url, "resolved_url": response.geturl(),
                    "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "last_modified": response.headers.get("Last-Modified"),
                    "etag": response.headers.get("ETag"), "raw_bytes": len(raw), "raw_sha256": digest(raw)}
    return raw, metadata


def acquire(commit: str, registration: str) -> None:
    require_freeze(commit, [*SOURCE_PATHS, registration])
    targets = [ROOT / f"data/raw/blind-channel-confirmation/pg{spec['pg']}.txt" for spec in FINAL.values()]
    if any(path.exists() for path in [*targets, ROOT / ACQUISITION, ROOT / MANIFEST]):
        raise FileExistsError("Final acquisition/preparation already exists; do not replace it")
    # Sequential acquisition, with each successful raw file preserved if a later request fails.
    sources = {}
    for (name, spec), path in zip(FINAL.items(), targets, strict=True):
        url = f"https://www.gutenberg.org/ebooks/{spec['pg']}.txt.utf-8"
        try:
            raw, row = download(url)
        except Exception as error:
            save_new(ROOT / ACQUISITION, encoded_json({"schema_version": 1,
                "status": "acquisition_failed", "source_freeze": commit, "registration": registration,
                "sources": sources, "failed_source": name, "exception_type": type(error).__name__,
                "paid_cost_usd": 0, "keys_generated": False, "models_fitted": False}))
            raise
        row.update(spec, raw=save_new(path, raw), catalogue_url=f"https://www.gutenberg.org/ebooks/{spec['pg']}",
                   rights_status="Raw license retained; file-specific rights/body review pending")
        sources[name] = row
    result = {"schema_version": 1, "status": "downloaded_pending_body_review", "source_freeze": commit,
              "registration": registration, "sources": sources, "paid_cost_usd": 0,
              "keys_generated": False, "models_fitted": False}
    save_new(ROOT / ACQUISITION, encoded_json(result))
    print(json.dumps({"status": result["status"], "sources": sources}, sort_keys=True))


def checked_span(raw: bytes, span: dict, *, minimum: int = 0, maximum: int | None = None) -> tuple[int, int]:
    start, end = span["raw_byte_start"], span["raw_byte_end"]
    maximum = len(raw) if maximum is None else maximum
    if type(start) is not int or type(end) is not int or not minimum <= start < end <= maximum:
        raise ValueError("Invalid or out-of-parent byte span")
    if digest(raw[start:end]) != span["sha256"]:
        raise ValueError("Body/exclusion bytes differ from the reviewed span")
    # Both endpoints must be Unicode boundaries, not merely a decodable substring.
    raw[:start].decode("utf-8", errors="strict")
    raw[start:end].decode("utf-8", errors="strict")
    return start, end


def extract_reviewed(raw: bytes, plan: dict, *, cap: int = CAP) -> tuple[dict, dict, list[str]]:
    """Apply reviewed body and editorial spans; retain every original byte map."""
    if not raw or len(raw) > RAW_LIMIT or digest(raw) != plan["raw_sha256"]:
        raise ValueError("Raw edition changed or exceeds cap")
    text = raw.decode("utf-8", errors="strict")
    beginnings = list(re.finditer(r"^\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\n]*", text, re.M))
    endings = list(re.finditer(r"^\*\*\* END OF THE PROJECT GUTENBERG EBOOK", text, re.M))
    if (len(beginnings) != 1 or len(endings) != 1 or beginnings[0].end() >= endings[0].start()
            or "Project Gutenberg License" not in text):
        raise ValueError("Ambiguous ebook boundary or missing license notice")
    start_byte = len(text[:beginnings[0].end()].encode())
    end_byte = len(text[:endings[0].start()].encode())
    if (type(cap) is not int or cap < 1 or not plan["bodies"]
            or plan.get("rights_review") != "public_domain_usa_notice_checked"
            or not plan.get("edition_review") or not plan.get("reviewer")):
        raise ValueError("Missing edition/body/rights review or invalid cap")
    pieces, maps, exclusions, spans, offset, previous_end = [], [], [], [], 0, start_byte
    body_ids = set()
    for body in plan["bodies"]:
        if not body.get("body_id") or body["body_id"] in body_ids:
            raise ValueError("Body identifiers must be nonempty and unique")
        body_ids.add(body["body_id"])
        start, end = checked_span(raw, body, minimum=previous_end, maximum=end_byte)
        original = raw[start:end].decode("utf-8")
        offsets = [start]
        for char in original:
            offsets.append(offsets[-1] + len(char.encode()))
        byte_to_character = {byte: i for i, byte in enumerate(offsets)}
        masked = list(original)
        exclusion_end = start
        for excluded in body.get("editorial_exclusions", []):
            a, b = checked_span(raw, excluded, minimum=exclusion_end, maximum=end)
            if a not in byte_to_character or b not in byte_to_character or not excluded.get("reason"):
                raise ValueError("Invalid editorial exclusion boundary or reason")
            first, last = byte_to_character[a], byte_to_character[b]
            masked[first:last] = " " * (last - first)
            exclusions.append({**excluded, "body_id": body["body_id"], "type": "reviewed_editorial"})
            exclusion_end = b
        clean, tokens, automatic = normalize_segment("".join(masked))
        if not clean:
            raise ValueError("Reviewed body contains no eligible Latin letters")
        for token in tokens:
            token["text_start"] += offset
            token["text_end"] += offset
            token["raw_byte_start"] = offsets[token.pop("raw_character_start")]
            token["raw_byte_end"] = offsets[token.pop("raw_character_end")]
            token["body_id"] = body["body_id"]
        for excluded in automatic:
            a, b = offsets[excluded.pop("start")], offsets[excluded.pop("end")]
            excluded.update(raw_byte_start=a, raw_byte_end=b, body_id=body["body_id"], type="normalizer")
            # The normalized-mask hash can differ after a reviewed exclusion inside brackets.
            excluded["sha256"] = digest(raw[a:b])
        spans.append({"body_id": body["body_id"], "raw_byte_start": start, "raw_byte_end": end,
                      "raw_sha256": digest(raw[start:end]), "normalized_start": offset,
                      "normalized_end": offset + len(clean), "normalized_sha256": digest(clean.encode())})
        pieces.append(clean)
        maps.extend(tokens)
        exclusions.extend(automatic)
        offset += len(clean)
        previous_end = end
    full = "".join(pieces)
    if len(full) < cap:
        raise ValueError("Insufficient eligible body text; no replacement or adaptive window shift")
    payload = {"schema_version": 1, "text": full[:cap],
               "tokens": [{**t, "selected_end": min(t["text_end"], cap)} for t in maps if t["text_start"] < cap],
               "body_boundaries": [min(s["normalized_end"], cap) for s in spans if s["normalized_start"] < cap]}
    summary = {"raw_sha256": digest(raw), "raw_bytes": len(raw), "eligible_characters": len(full),
               "selected_characters": cap, "full_normalized_sha256": digest(full.encode()),
               "selected_text_sha256": digest(payload["text"].encode()), "body_spans": spans,
               "excluded_spans": exclusions, "rights_review": plan["rights_review"],
               "edition_review": plan["edition_review"], "reviewer": plan["reviewer"],
               "observed_codepoints": dict(sorted(Counter(c for c in text if ord(c) > 127).items()))}
    return payload, summary, pieces


def exact_overlap_screen(bodies: dict[str, list[str]], width: int = DUPLICATE_WINDOW) -> dict:
    """Literal string equality, every position, never cross an authorial body boundary."""
    if type(width) is not int or width < 1:
        raise ValueError("Duplicate window must be positive")
    counts = {name: Counter(body[i:i + width] for body in rows for i in range(len(body) - width + 1))
              for name, rows in bodies.items()}
    within = {name: {"repeated_unique_windows": sum(n > 1 for n in counter.values()),
                     "repeated_positions": sum(n for n in counter.values() if n > 1),
                     "total_positions": sum(counter.values())} for name, counter in counts.items()}
    between = []
    for i, first in enumerate(sorted(counts)):
        for second in sorted(counts)[i + 1:]:
            shared = counts[first].keys() & counts[second].keys()
            between.append({"first": first, "second": second, "shared_unique_windows": len(shared),
                            "first_positions": sum(counts[first][s] for s in shared),
                            "second_positions": sum(counts[second][s] for s in shared)})
    return {"window": width, "scope": "all_eligible_authorial_bodies", "within": within, "between": between}


def selected_window_metadata(payload: dict, role: str) -> list[dict]:
    """Publication-order body-aware windows; never seek a convenient passage.

    All 224-letter candidates at a body's 0, 256, ... offsets are enumerated,
    then key k takes candidate 16*k+[0..3] for F or 16*k+[0..1] for T.
    """
    if role not in {"F", "T"}:
        raise ValueError("Unknown final-author role")
    boundaries = payload["body_boundaries"]
    if (not boundaries or boundaries[-1] != len(payload["text"])
            or any(a >= b for a, b in zip([0, *boundaries[:-1]], boundaries, strict=True))):
        raise ValueError("Malformed body boundaries")
    candidates = []
    for body_index, (a, b) in enumerate(zip([0, *boundaries[:-1]], boundaries, strict=True)):
        for start in range(a, b - RECORD_LENGTH + 1, WINDOW_STRIDE):
            candidates.append({"candidate_index": len(candidates), "body_index": body_index,
                               "body_offset": start - a, "normalized_start": start,
                               "normalized_end": start + RECORD_LENGTH})
    per_key = 4 if role == "F" else 2
    if len(candidates) <= (KEY_COUNT - 1) * KEY_WINDOW_BLOCK + per_key - 1:
        raise ValueError("Insufficient fixed windows; no replacement or adaptive window shift")
    result, seen = [], set()
    for key in range(KEY_COUNT):
        for record in range(per_key):
            row = candidates[KEY_WINDOW_BLOCK * key + record]
            selected = payload["text"][row["normalized_start"]:row["normalized_end"]]
            if selected in seen:
                raise ValueError("Duplicate fixed records; do not replace or shift them")
            seen.add(selected)
            result.append({**row, "key_index": key, "record_index": record,
                           "sha256": digest(selected.encode())})
    return result


def fixed_windows(payload: dict, role: str, key_index: int) -> list[dict]:
    """Evaluator/generator interface; this return value contains private plaintext."""
    if role not in {"fit", "transfer"} or type(key_index) is not int or not 0 <= key_index < KEY_COUNT:
        raise ValueError("Need a registered fit/transfer role and key index")
    rows = selected_window_metadata(payload, "F" if role == "fit" else "T")
    return [{"text": payload["text"][row["normalized_start"]:row["normalized_end"]],
             "start": row["normalized_start"], "end": row["normalized_end"],
             "body_index": row["body_index"], "body_offset": row["body_offset"],
             "candidate_index": row["candidate_index"]}
            for row in rows if row["key_index"] == key_index]


def prepare(commit: str, registration: str, plan_path: str) -> None:
    require_freeze(commit, [*SOURCE_PATHS, registration, ACQUISITION, plan_path])
    targets = [ROOT / f"data/processed/blind-channel-confirmation/{name}.json" for name in FINAL]
    if any(path.exists() for path in [*targets, ROOT / MANIFEST]):
        raise FileExistsError("Final prepared corpus already exists")
    plan_raw = (ROOT / plan_path).read_bytes()
    plan = json.loads(plan_raw)
    if (plan.get("schema_version") != 1 or plan.get("status") != "mechanical_body_review_complete"
            or set(plan["sources"]) != set(FINAL) or plan["prefix_cap"] != CAP
            or plan["duplicate_window"] != DUPLICATE_WINDOW):
        raise ValueError("Preparation plan differs from the registered extraction policy")
    acquisition_raw = (ROOT / ACQUISITION).read_bytes()
    acquisition = json.loads(acquisition_raw)
    if acquisition.get("status") != "downloaded_pending_body_review" or set(acquisition["sources"]) != set(FINAL):
        raise ValueError("Acquisition did not complete; no partial final-corpus preparation")
    if acquisition["registration"] != registration:
        raise ValueError("Preparation must use the original acquisition registration")
    # A later commit adds reviewed spans, not a silently changed policy after text exposure.
    require_freeze(acquisition["source_freeze"], [*SOURCE_PATHS, registration])
    development_raw = (ROOT / DEVELOPMENT).read_bytes()
    development = json.loads(development_raw)
    full_bodies, payloads, sources = {}, {}, {}
    for name, spec in SPECS.items():
        old = development["sources"][name]
        payload, summary = extract((ROOT / spec["path"]).read_bytes(), spec, cap=old["eligible_characters"])
        if summary["full_normalized_sha256"] != old["full_normalized_sha256"]:
            raise ValueError("Development full-body normalization changed")
        boundaries = [0, *payload["body_boundaries"]]
        full_bodies[name] = [payload["text"][a:b] for a, b in zip(boundaries, boundaries[1:])]
    for name, spec in FINAL.items():
        row = acquisition["sources"][name]
        raw = (ROOT / row["raw"]["path"]).read_bytes()
        if digest(raw) != row["raw"]["sha256"] or row["raw_sha256"] != row["raw"]["sha256"]:
            raise ValueError("Acquired raw artifact changed")
        payload, summary, bodies = extract_reviewed(raw, plan["sources"][name], cap=CAP)
        summary.update(spec, raw_path=row["raw"]["path"], requested_url=row["requested_url"],
                       resolved_url=row["resolved_url"], retrieved_utc=row["retrieved_utc"])
        summary["fixed_windows"] = selected_window_metadata(payload, spec["role"])
        payloads[name], sources[name], full_bodies[name] = payload, summary, bodies
    screen = exact_overlap_screen(full_bodies)
    failed = any(row["shared_unique_windows"] and ({row["first"], row["second"]} & FINAL.keys())
                 for row in screen["between"])
    result = {"schema_version": 1, "status": "blocked_exact_cross_author_overlap" if failed else "prepared",
              "source_freeze": commit, "registration": registration, "extraction_plan": plan_path,
              "extraction_plan_sha256": digest(plan_raw), "acquisition_sha256": digest(acquisition_raw),
              "development_manifest_sha256": digest(development_raw), "alphabet": ALPHABET,
              "prefix_cap": CAP, "sources": sources, "exact_overlap_screen": screen,
              "window_policy": {"record_length": RECORD_LENGTH, "stride": WINDOW_STRIDE,
                                "key_count": KEY_COUNT, "candidate_block_per_key": KEY_WINDOW_BLOCK,
                                "fit_records_per_key": 4, "transfer_records_per_key": 2,
                                "candidate_order": "publication body order; reset stride at each body"},
              "keys_generated": False, "models_fitted": False, "paid_cost_usd": 0}
    if not failed:
        for name, path in zip(FINAL, targets, strict=True):
            sources[name]["derived"] = save_new(path, encoded_json(payloads[name]))
            sources[name]["derived_path"] = sources[name]["derived"]["path"]
            sources[name]["derived_sha256"] = sources[name]["derived"]["sha256"]
    save_new(ROOT / MANIFEST, encoded_json(result))
    print(json.dumps({"status": result["status"], "manifest": MANIFEST,
                      "eligible_characters": {n: s["eligible_characters"] for n, s in sources.items()},
                      "exact_overlap_screen": screen}, sort_keys=True))
    if failed:
        raise ValueError("Exact cross-author overlap found; preserve failure before any generation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("acquire", "prepare"))
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--registration", required=True, help="Tracked experiment registration path")
    parser.add_argument("--plan", help="Tracked hashed body-span plan; required for prepare")
    args = parser.parse_args()
    if args.stage == "prepare":
        if not args.plan:
            parser.error("prepare requires --plan")
        prepare(args.freeze, args.registration, args.plan)
    else:
        if args.plan:
            parser.error("acquire does not use an extraction plan")
        acquire(args.freeze, args.registration)


if __name__ == "__main__":
    main()
