from dataclasses import replace
import json
import math
import time

import pytest

from voynich import paragraph_carry as pc
from scripts import audit_paragraph_carry001 as auditor
from scripts import run_paragraph_carry001 as runner


def fixture_lines(seed=9901):
    return pc.controls("iid_terminal", seed, "paragraph_carry", n_leaves=10, n_lines=12)


def test_normalized_transport_unknown_and_unobserved_state():
    rows = fixture_lines()
    model = pc.fit(rows)
    for kind in ("continuation", "paragraph"):
        base, carry = pc.distributions(model, rows[0].context, kind, "a")
        assert math.fsum(base.values()) == pytest.approx(1)
        assert math.fsum(carry.values()) == pytest.approx(1)
        assert min(carry.values()) > 0
        assert pc.distributions(model, rows[0].context, kind, "unobserved")[0] == pytest.approx(
            pc.distributions(model, rows[0].context, kind, "unobserved")[1])
        event = {"context": rows[0].context, "kind": kind, "last": "a", "target": "unseen"}
        assert pc.gain(model, event) == pytest.approx(auditor.alternate_gain(auditor.alternate_model(rows), event))


def test_full_crossfit_and_alternate_arithmetic():
    rows = fixture_lines()
    ledger, manifest = pc.cross_fit(rows)
    assert len(ledger) == 110
    for fold in manifest["models"]:
        assert not set(fold["training_leaves"]) & set(fold["test_leaves"])
    for event in ledger:
        training = [row for row in rows if manifest["leaf_fold"][row.leaf] != event["fold"]]
        assert abs(event["gain"] - auditor.alternate_gain(auditor.alternate_model(training), event)) < 1e-12
    assert pc.junctions(rows) == auditor.independent_junctions(rows)
    null = pc.permutation_null(rows, ledger, 9911, n_perm=5)
    cell = {"name": "fixture", "manifest": manifest, "ledger_sha": pc.json_sha(ledger),
            "bootstrap_seed": 9913, "permutation_seed": 9911,
            "summary": pc.summarize(ledger, 9913), "null": null}
    # All five production permutations get independently rescored.
    original = pc.N_PERM
    try:
        pc.N_PERM = 5
        assert auditor.replay_cell(rows, cell, ledger, time.monotonic(), time.process_time()) == (110, 5)
        broken = json.loads(json.dumps(cell))
        broken["manifest"]["models"][0]["count_digest"] = "corrupt"
        with pytest.raises(AssertionError):
            auditor.replay_cell(rows, broken, ledger, time.monotonic(), time.process_time())
    finally:
        pc.N_PERM = original


def test_junction_conservativity_and_leaf_assignments():
    a = pc.Line("f1", "f1r", 1, ("H", "A", "1"), (("a", "b"), ("b", "a")), True, False, 0)
    b = replace(a, number=2, start=False, end=True, depth=1, locator="+")
    assert len(pc.junctions([a, b])) == 1
    for changed in (replace(b, page="f1v"), replace(b, number=3), replace(b, start=True),
                    replace(b, context=("H", "A", "2")), replace(b, leaf="f2")):
        assert pc.junctions([a, changed]) == []
    for locator in ("@", "=", "&", "~", "/", "!"):
        assert pc.junctions([a, replace(b, locator=locator)]) == []
        assert auditor.independent_junctions([a, replace(b, locator=locator)]) == []
    assert pc.junctions([a, replace(b, locator="*")]) == auditor.independent_junctions([a, b])
    with pytest.raises(ValueError):
        pc.junctions([a, a])
    with pytest.raises(ValueError):
        pc.folds([a, b])


def zl_page(lines):
    return {"page_id": "f1r", "leaf_id": "f1", "split": "train",
            "metadata": {"page_variables": {"I": "H", "L": "A", "H": "1"}}, "loci": lines}


def locus(number=1, start=True, end=False, locator=None):
    text = "ab ba ab"
    marks = [{"kind": "paragraph_start", "start": 0, "end": 0}] if start else []
    if end:
        marks.append({"kind": "paragraph_end", "start": len(text), "end": len(text)})
    return {"locus_type": "P0", "locator": locator or ("@" if number == 1 else "*" if start else "+"),
            "locus_id": f"f1r.{number}",
            "text": text, "text_tags": {}, "annotations": marks}


def test_training_only_parser_markers_uncertainty_and_rosettes(tmp_path):
    path = tmp_path / "train.jsonl"
    page = zl_page([locus(), locus(2, False, True)])
    path.write_text(json.dumps(page) + "\n")
    lines, _ = pc.load_zl_train(path)
    assert len(lines) == 2 and [row.depth for row in lines] == [0, 1]
    gc = tmp_path / "GC.txt"
    gc.write_text("#=IVTFF v101 2.0 M 6\n<f1r> <! $I=H $L=A $H=1 >\n"
                  "<f1r.1,@P0> <%>ab.ba.ab\n<f1r.2,+P0> ab.ba.ab<$>\n"
                  "<f2r> <! $I=H $L=A $H=1 >\nMALFORMED TEST GLYPHS MUST NOT BE PARSED\n")
    parsed, _ = pc.load_gc_train(gc, {"f1": "train", "f2": "test"})
    views, counts = pc.matched_views(lines, parsed)
    assert counts["matched_lines"] == 2 and len(views["gc"]) == 2
    assert [row.locator for row in lines] == ["@", "+"]
    assert len(pc.junctions(views["zl"])) == len(pc.junctions(views["gc"])) == 1
    page["loci"][0]["annotations"][0]["start"] = 1
    path.write_text(json.dumps(page))
    assert pc.load_zl_train(path)[1]["excluded_paragraph_markers"] == 1
    page["split"] = "validation"
    path.write_text(json.dumps(page))
    with pytest.raises(ValueError, match="Only train"):
        pc.load_zl_train(path)
    gc.write_text("#=IVTFF v101 2.0 M 6\n<f85r1> <! $I=C $H=4 >\n<f85r1.1,@P0> ab.ba.ab\n")
    assert pc.load_gc_train(gc, {"f85-f86-Ros": "train"})[0][0].leaf == "f85-f86-Ros"
    gc.write_text("#=IVTFF v101 2.0 M 6\n<f1r> <! $I=H $L=A $H=1 >\n<f1r.1,@P0> ab<%>.ba.ab\n")
    assert pc.load_gc_train(gc, {"f1": "train"})[1]["excluded_paragraph_markers"] == 1


def test_real_ivtff_locator_sequence_and_complete_exclusion_accounting(tmp_path):
    # Primary IVTFF §§6.4/7.2: @ first/unrelated, + below numbered predecessor,
    # * lower line at the left margin, = same line; ;G is transcriber metadata.
    rows = [locus(1), locus(2, False, True), locus(3, True), locus(4, False, True),
            locus(5, False, False, "="), locus(6, False, False), locus(7, False, False),
            locus(8, False, False), locus(9, False, False), locus(10, False, False)]
    rows[5]["locus_type"] = "Pt"
    rows[6]["annotations"].append({"kind": "uncertain_space", "start": 2, "end": 3})
    rows[7]["text_tags"]["H"] = "?"
    rows[8]["text"] = "ab"
    rows[9]["annotations"].append({"kind": "paragraph_start", "start": 1, "end": 1})
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(zl_page(rows)))
    zl, counts = pc.load_zl_train(path)
    assert len(zl) == 4 and counts["retained_lines"] == 4
    assert all(counts[name] == 1 for name in pc.EXCLUSIONS)
    assert counts["loci_seen"] == len(zl) + sum(counts[name] for name in pc.EXCLUSIONS)
    gc = tmp_path / "GC.txt"
    gc.write_text("#=IVTFF v101 2.0 M 6\n<f1r> <! $I=H $L=A $H=1 >\n"
                  "<f1r.1,@P0;G> <%><!@252;>ab.ba.ab\n<f1r.2,+P0;G> ab.ba.ab<$>\n"
                  "<f1r.3,*P0;G> <%>ab.ba.ab\n<f1r.4,+P0;G> ab.ba.ab<$>\n"
                  "<f1r.5,=P0> ab.ba.ab\n<f1r.6,=Pt> ab.ba.ab<$>\n"
                  "<f1r.7,+P0> ab,ba.ab\n<f1r.8,+P0> ab\n"
                  "<f1r.9,+P0> ab<%>.ba.ab\n<f1r.10,+P0> <@H=9>ab.ba.ab\n")
    parsed, gc_counts = pc.load_gc_train(gc, {"f1": "train"})
    assert len(parsed) == 4 and all(gc_counts[name] == 1 for name in pc.EXCLUSIONS)
    views, match = pc.matched_views(zl, parsed)
    assert match["matched_lines"] == 4
    for view in views.values():
        assert [e["kind"] for e in pc.junctions(view)] == ["continuation", "paragraph", "continuation"]
        assert pc.junctions(view) == auditor.independent_junctions(view)


def test_common_cause_is_preserved_as_nonidentification_control():
    rows = pc.controls("markov_terminal", 9909, "paragraph_common_cause", n_leaves=10, n_lines=8)
    assert len(rows) == 80
    assert [row.depth for row in rows[:8]] == [0, 1, 2, 3, 0, 1, 2, 3]
    with pytest.raises(ValueError):
        pc.controls("not_a_source", 9901, "iid")
    with pytest.raises(ValueError):
        runner.decisions([])


def test_full_artificial_controller_audit_and_exclusivity(tmp_path, monkeypatch):
    root = tmp_path
    for path in runner.PATHS:
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fixture")
    pages, gc_lines, assignments = [], ["#=IVTFF v101 2.0 M 6"], {}
    for index in range(10):
        leaf, page_id = f"f{index + 1}", f"f{index + 1}r"
        assignments[leaf] = "train"
        page = zl_page([locus(i + 1, i % 4 == 0, i % 4 == 3) for i in range(8)])
        page.update(page_id=page_id, leaf_id=leaf)
        gc_lines.append(f"<{page_id}> <! $I=H $L=A $H=1 >")
        for row in page["loci"]:
            n = int(row["locus_id"].split(".")[1])
            row["locus_id"] = f"{page_id}.{n}"
            gc_lines.append(f"<{page_id}.{n},{row['locator']}P0> " + ("<%>" if (n - 1) % 4 == 0 else "")
                            + "ab.ba.ab" + ("<$>" if n % 4 == 0 else ""))
        pages.append(page)
    (root / "data/processed/zl3b/train.jsonl").write_text("\n".join(json.dumps(page) for page in pages))
    (root / "data/raw/v101/GC2a-n.txt").write_text("\n".join(gc_lines))
    (root / "data/manifests/zl3b_split.json").write_text(json.dumps({"leaf_assignments": assignments}))
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(auditor, "ROOT", root)
    monkeypatch.setattr(runner, "OUT", root / "results" / pc.EXPERIMENT)
    monkeypatch.setattr(runner, "BULK", root / "outputs" / pc.EXPERIMENT)
    monkeypatch.setattr(runner, "EXPECTED", {path: pc.sha(root / path) for path in runner.EXPECTED})
    monkeypatch.setattr(runner, "require_frozen", lambda *_: None)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *_args, **_kwargs: "fixture")
    monkeypatch.setattr(runner.platform, "platform", lambda: "fixture-platform")
    monkeypatch.setattr(pc, "N_PERM", 3)
    monkeypatch.setattr(pc, "CONTROL_ALLOCATION", (("iid_terminal", 9921),))
    original_controls = pc.controls
    monkeypatch.setattr(pc, "controls", lambda family, seed, mode: original_controls(family, seed, mode, 10, 8))
    result = runner.run("fixture")
    assert len(result["cells"]) == 8 and result["complete"]
    assert result["decisions"]["causal_carry_identified"] is False
    receipt = auditor.audit()
    assert receipt["pass"] and receipt["permutations_recomputed"] == 24
    with pytest.raises(FileExistsError):
        runner.run("fixture")
    with pytest.raises(FileExistsError):
        auditor.audit()
