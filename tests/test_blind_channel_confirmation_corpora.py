"""Artificial-only qualification: no real final-author path is read or fetched."""
import io
import json
import random

import pytest

from scripts import prepare_blind_channel_confirmation_corpora as corpus


def ebook(*bodies):
    raw = ("Project Gutenberg License\n*** START OF THE PROJECT GUTENBERG EBOOK FIXTURE ***\n"
           + "\nEDITORIAL HEADING\n".join(bodies)
           + "\n*** END OF THE PROJECT GUTENBERG EBOOK FIXTURE ***\n").encode()
    plan = {"raw_sha256": corpus.digest(raw), "rights_review": "public_domain_usa_notice_checked",
            "edition_review": "artificial edition", "reviewer": "artificial evaluator", "bodies": []}
    position = 0
    for index, body in enumerate(bodies):
        start = raw.index(body.encode(), position)
        end = start + len(body.encode())
        plan["bodies"].append({"body_id": f"book{index}", "raw_byte_start": start,
                               "raw_byte_end": end, "sha256": corpus.digest(raw[start:end]),
                               "editorial_exclusions": []})
        position = end
    return raw, plan


def exclude(raw, plan, text, body=0):
    start = raw.index(text.encode())
    end = start + len(text.encode())
    plan["bodies"][body]["editorial_exclusions"].append(
        {"raw_byte_start": start, "raw_byte_end": end, "sha256": corpus.digest(raw[start:end]),
         "reason": "artificial English apparatus"})


def test_exact_shared_normalizer_with_original_utf8_offsets_and_editorial_exclusion():
    raw, plan = ebook("Jūlius [nota] vīvit æther. English note! amor.", "Cœlum 12 1o bonum.")
    exclude(raw, plan, "English note!")
    payload, summary, bodies = corpus.extract_reviewed(raw, plan, cap=27)
    assert bodies == ["iuliusuiuitaetheramor", "coelumbonum"]
    assert payload["text"] == "iuliusuiuitaetheramorcoelum"
    assert payload["body_boundaries"] == [21, 27]
    assert payload["tokens"][0]["text_start"] == 0
    assert raw[payload["tokens"][0]["raw_byte_start"]:payload["tokens"][0]["raw_byte_end"]] == "Jūlius".encode()
    assert summary["eligible_characters"] == 32
    assert summary["selected_text_sha256"] == corpus.digest(payload["text"].encode())
    assert summary["full_normalized_sha256"] == corpus.digest("".join(bodies).encode())
    assert {x["reason"] for x in summary["excluded_spans"]} == {
        "artificial English apparatus", "square_bracket_span", "numeric_or_mixed_token"}
    for row in summary["body_spans"]:
        assert corpus.digest(raw[row["raw_byte_start"]:row["raw_byte_end"]]) == row["raw_sha256"]
    for row in summary["excluded_spans"]:
        assert corpus.digest(raw[row["raw_byte_start"]:row["raw_byte_end"]]) == row["sha256"]


def test_partial_token_is_flagged_and_does_not_change_raw_span():
    raw, plan = ebook("arma bonum")
    payload, _, _ = corpus.extract_reviewed(raw, plan, cap=6)
    last = payload["tokens"][-1]
    assert payload["text"] == "armabo"
    assert (last["text_end"], last["selected_end"]) == (9, 6)
    assert raw[last["raw_byte_start"]:last["raw_byte_end"]] == b"bonum"


@pytest.mark.parametrize("mutation", ["raw", "bodyhash", "exclusionhash", "overlap", "order", "missingrights",
                                     "bodyid", "startheader", "endfooter", "splitutf8", "emptybody"])
def test_bad_spans_editions_and_reviews_fail_closed(mutation):
    raw, plan = ebook("Jūlius note arma", "terra bona")
    exclude(raw, plan, "note")
    if mutation == "raw":
        raw += b"changed"
    elif mutation == "bodyhash":
        plan["bodies"][0]["sha256"] = "0" * 64
    elif mutation == "exclusionhash":
        plan["bodies"][0]["editorial_exclusions"][0]["sha256"] = "0" * 64
    elif mutation == "overlap":
        plan["bodies"][0]["editorial_exclusions"] *= 2
    elif mutation == "order":
        plan["bodies"].reverse()
    elif mutation == "missingrights":
        plan["rights_review"] = "catalogue_only"
    elif mutation == "bodyid":
        plan["bodies"][1]["body_id"] = plan["bodies"][0]["body_id"]
    elif mutation == "startheader":
        body = plan["bodies"][0]
        body["raw_byte_start"] = 0
        body["sha256"] = corpus.digest(raw[:body["raw_byte_end"]])
    elif mutation == "endfooter":
        body = plan["bodies"][-1]
        body["raw_byte_end"] = len(raw)
        body["sha256"] = corpus.digest(raw[body["raw_byte_start"]:])
    elif mutation == "splitutf8":
        body = plan["bodies"][0]
        body["raw_byte_start"] += 2
        body["sha256"] = corpus.digest(raw[body["raw_byte_start"]:body["raw_byte_end"]])
    else:
        exclude(raw, plan, "Jūlius note arma")
        plan["bodies"][0]["editorial_exclusions"] = plan["bodies"][0]["editorial_exclusions"][-1:]
    with pytest.raises((ValueError, UnicodeDecodeError)):
        corpus.extract_reviewed(raw, plan, cap=5)


@pytest.mark.parametrize("body", ["arma λογος", "arma [nota", "arma ]nota"])
def test_foreign_letters_and_unbalanced_apparatus_are_not_silently_removed(body):
    raw, plan = ebook(body)
    with pytest.raises(ValueError):
        corpus.extract_reviewed(raw, plan, cap=1)


def test_insufficient_text_is_not_replaced_or_truncated():
    raw, plan = ebook("arma")
    with pytest.raises(ValueError, match="Insufficient"):
        corpus.extract_reviewed(raw, plan, cap=5)


def test_duplicate_audit_is_exact_counts_positions_and_respects_body_boundaries():
    report = corpus.exact_overlap_screen({"a": ["ab", "cabc"], "b": ["abcabc"], "c": ["xyzyz"]}, 3)
    assert report["within"]["b"] == {"repeated_unique_windows": 1, "repeated_positions": 2, "total_positions": 4}
    first = report["between"][0]
    assert first == {"first": "a", "second": "b", "shared_unique_windows": 2,
                     "first_positions": 2, "second_positions": 3}
    assert report["within"]["a"]["total_positions"] == 2  # No fabricated abc across bodies.
    assert all(row["shared_unique_windows"] == 0 for row in report["between"][1:])


def artificial_payload(length=50_000, boundaries=None, seed=30):
    rng = random.Random(seed)
    return {"text": "".join(rng.choices(corpus.ALPHABET, k=length)),
            "body_boundaries": boundaries or [length]}


def test_registered_eight_key_fixed_windows_and_body_reset():
    payload = artificial_payload(boundaries=[1000, 50_000])
    fit = corpus.selected_window_metadata(payload, "F")
    transfer = corpus.selected_window_metadata(payload, "T")
    assert len(fit) == 32 and len(transfer) == 16
    assert [row["normalized_start"] for row in fit[:4]] == [0, 256, 512, 768]
    assert fit[4] == {"candidate_index": 16, "body_index": 1, "body_offset": 3072,
                      "normalized_start": 4072, "normalized_end": 4296, "key_index": 1,
                      "record_index": 0, "sha256": corpus.digest(payload["text"][4072:4296].encode())}
    assert [row["candidate_index"] for row in fit[-4:]] == [112, 113, 114, 115]
    assert [row["candidate_index"] for row in transfer[-2:]] == [112, 113]
    for row in fit:
        assert row["normalized_end"] - row["normalized_start"] == 224
        assert not row["normalized_start"] < 1000 < row["normalized_end"]
        assert "text" not in row


def test_generator_interface_returns_exact_registered_text_and_offsets():
    payload = artificial_payload(boundaries=[1000, 50_000])
    rows = corpus.fixed_windows(payload, "fit", 1)
    assert len(rows) == 4
    assert rows[0] == {"text": payload["text"][4072:4296], "start": 4072, "end": 4296,
                       "body_index": 1, "body_offset": 3072, "candidate_index": 16}
    assert len(corpus.fixed_windows(payload, "transfer", 7)) == 2


@pytest.mark.parametrize("role,key", [("F", 0), ("fit", -1), ("transfer", 8), ("fit", True)])
def test_generator_interface_rejects_unregistered_inputs(role, key):
    with pytest.raises(ValueError, match="registered"):
        corpus.fixed_windows(artificial_payload(), role, key)


@pytest.mark.parametrize("failure", ["insufficient", "duplicate", "boundary", "role"])
def test_bad_fixed_windows_fail_without_replacement(failure):
    payload = artificial_payload()
    role = "F"
    if failure == "insufficient":
        payload = artificial_payload(length=1000)
    elif failure == "duplicate":
        payload["text"] = "a" * 50_000
    elif failure == "boundary":
        payload["body_boundaries"] = [900, 800, 50_000]
    else:
        role = "D"
    with pytest.raises(ValueError):
        corpus.selected_window_metadata(payload, role)


class Response(io.BytesIO):
    def __init__(self, raw, headers=None, url="https://www.gutenberg.org/cache/epub/0/pg0.txt"):
        super().__init__(raw)
        self.headers = headers or {}
        self.url = url

    def geturl(self):
        return self.url


class Opener:
    def __init__(self, response):
        self.response = response

    def open(self, request, timeout):
        assert timeout == 60
        assert request.get_header("Accept-encoding") == "identity"
        return self.response


def test_download_preserves_raw_bytes_metadata_without_corpus_output():
    raw, _ = ebook("arma")
    response = Response(raw, {"Content-Length": str(len(raw)), "Last-Modified": "fixture", "ETag": "fixture"})
    actual, metadata = corpus.download("https://www.gutenberg.org/ebooks/0.txt.utf-8", opener=Opener(response))
    assert actual == raw and metadata["raw_sha256"] == corpus.digest(raw)
    assert metadata["raw_bytes"] == len(raw) and metadata["last_modified"] == "fixture"
    assert "arma" not in json.dumps(metadata)


@pytest.mark.parametrize("failure", ["oversize", "truncated", "encoding", "redirect", "no_license", "badutf8"])
def test_download_fails_closed_without_fallback(failure):
    raw, _ = ebook("arma")
    headers, url = {}, "https://www.gutenberg.org/cache/epub/0/pg0.txt"
    if failure == "oversize":
        raw = b"x" * (corpus.RAW_LIMIT + 1)
    elif failure == "truncated":
        headers["Content-Length"] = str(len(raw) + 1)
    elif failure == "encoding":
        headers["Content-Encoding"] = "gzip"
    elif failure == "redirect":
        url = "https://example.com/alternative.txt"
    elif failure == "no_license":
        raw = b"just text"
    else:
        raw += b"\xff"
    with pytest.raises(ValueError):
        corpus.download("https://www.gutenberg.org/ebooks/0.txt.utf-8", opener=Opener(Response(raw, headers, url)))


@pytest.mark.parametrize("url", ["http://www.gutenberg.org/a", "https://user:pass@www.gutenberg.org/a",
                                 "https://www.gutenberg.org.evil.com/a", "https://www.gutenberg.org:8443/a"])
def test_unregistered_download_origins_rejected(url):
    with pytest.raises(ValueError, match="host"):
        corpus.download(url)


def test_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "ROOT", tmp_path)
    path = tmp_path / "result.json"
    artifact = corpus.save_new(path, b"first")
    assert artifact["sha256"] == corpus.digest(b"first")
    with pytest.raises(FileExistsError):
        corpus.save_new(path, b"second")
    assert path.read_bytes() == b"first"


def test_failed_second_acquisition_preserves_first_and_records_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "ROOT", tmp_path)
    monkeypatch.setattr(corpus, "require_freeze", lambda *_: None)
    raw, _ = ebook("arma")
    calls = []

    def fake_download(url):
        calls.append(url)
        if len(calls) == 2:
            raise TimeoutError("artificial timeout")
        return raw, {"raw_sha256": corpus.digest(raw), "raw_bytes": len(raw)}

    monkeypatch.setattr(corpus, "download", fake_download)
    with pytest.raises(TimeoutError):
        corpus.acquire("0" * 40, "fixture.md")
    result = json.loads((tmp_path / corpus.ACQUISITION).read_bytes())
    assert result["status"] == "acquisition_failed" and result["failed_source"] == "tacitus"
    assert result["sources"]["sallust"]["raw_sha256"] == corpus.digest(raw)
    assert (tmp_path / result["sources"]["sallust"]["raw"]["path"]).read_bytes() == raw
    with pytest.raises(FileExistsError):
        corpus.acquire("0" * 40, "fixture.md")
    assert len(calls) == 2
    assert calls == ["https://www.gutenberg.org/cache/epub/7402/pg7402.txt",
                     "https://www.gutenberg.org/cache/epub/9090/pg9090.txt"]


def test_freeze_checks_actual_committed_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "ROOT", tmp_path)
    (tmp_path / "fixture").write_bytes(b"frozen")
    calls = []

    def show(argv, cwd):
        calls.append((argv, cwd))
        return b"frozen"

    monkeypatch.setattr(corpus.subprocess, "check_output", show)
    corpus.require_freeze("a" * 40, ["fixture"])
    assert calls[0][0] == ["git", "show", f"{'a' * 40}:fixture"]
    (tmp_path / "fixture").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        corpus.require_freeze("a" * 40, ["fixture"])
    with pytest.raises(ValueError, match="immutable"):
        corpus.require_freeze("main", ["fixture"])


@pytest.mark.parametrize("overlap", [False, True])
def test_complete_artificial_preparation_and_overlap_failure(tmp_path, monkeypatch, overlap):
    monkeypatch.setattr(corpus, "ROOT", tmp_path)
    monkeypatch.setattr(corpus, "require_freeze", lambda *_: None)
    monkeypatch.setattr(corpus, "CAP", 100)
    monkeypatch.setattr(corpus, "RECORD_LENGTH", 4)
    monkeypatch.setattr(corpus, "WINDOW_STRIDE", 6)
    monkeypatch.setattr(corpus, "KEY_COUNT", 2)
    monkeypatch.setattr(corpus, "KEY_WINDOW_BLOCK", 4)
    specs, development, sources, plans = {}, {"sources": {}}, {}, {}
    raw_bodies = {}
    for index, name in enumerate(["caesar", "virgil", "cicero", "sallust", "tacitus"]):
        text = artificial_payload(length=120, seed=index + 10)["text"]
        if name == "sallust" and overlap:
            text = raw_bodies["caesar"][:64] + text[64:]
        raw_bodies[name] = text
        raw, body_plan = ebook(text)
        relative = f"data/raw/{name}.txt"
        corpus.save_new(tmp_path / relative, raw)
        if name in {"caesar", "virgil", "cicero"}:
            spec = {"sha256": corpus.digest(raw), "path": relative, "role": "D", "author": name,
                    "pg": 0, "heading": r"^\*\*\* START OF THE PROJECT GUTENBERG EBOOK FIXTURE \*\*\*$",
                    "expected_bodies": 1}
            _, summary = corpus.extract(raw, spec, cap=120)
            specs[name] = spec
            development["sources"][name] = summary
        else:
            plans[name] = body_plan
            sources[name] = {"raw": {"path": relative, "sha256": corpus.digest(raw)},
                             "raw_sha256": corpus.digest(raw), "requested_url": "fixture",
                             "resolved_url": "fixture", "retrieved_utc": "fixture"}
    monkeypatch.setattr(corpus, "SPECS", specs)
    corpus.save_new(tmp_path / corpus.DEVELOPMENT, corpus.encoded_json(development))
    corpus.save_new(tmp_path / corpus.ACQUISITION, corpus.encoded_json(
        {"status": "downloaded_pending_body_review", "sources": sources,
         "source_freeze": "0" * 40, "registration": "registration.md"}))
    plan = {"schema_version": 1, "status": "mechanical_body_review_complete", "prefix_cap": 100,
            "duplicate_window": 64, "sources": plans}
    corpus.save_new(tmp_path / "plan.json", corpus.encoded_json(plan))
    if overlap:
        with pytest.raises(ValueError, match="cross-author"):
            corpus.prepare("0" * 40, "registration.md", "plan.json")
    else:
        corpus.prepare("0" * 40, "registration.md", "plan.json")
    result = json.loads((tmp_path / corpus.MANIFEST).read_bytes())
    assert result["status"] == ("blocked_exact_cross_author_overlap" if overlap else "prepared")
    assert result["keys_generated"] is result["models_fitted"] is False
    assert len(result["sources"]["sallust"]["fixed_windows"]) == 8
    assert len(result["sources"]["tacitus"]["fixed_windows"]) == 4
    if not overlap:
        for row in result["sources"].values():
            derived = (tmp_path / row["derived"]["path"]).read_bytes()
            assert corpus.digest(derived) == row["derived"]["sha256"]
            assert len(json.loads(derived)["text"]) == 100
    else:
        assert not (tmp_path / "data/processed").exists()
    with pytest.raises(FileExistsError):
        corpus.prepare("0" * 40, "registration.md", "plan.json")
