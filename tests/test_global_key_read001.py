"""Artificial-only end-to-end staging, independent audits, and retained failures."""

import copy
import gzip
import hashlib
import json
import math
from types import SimpleNamespace

import pytest

from scripts import audit_global_key_read001 as auditor
from scripts import global_key_read001_common as common
from scripts import evaluate_global_key_read001 as evaluator
from scripts import predict_global_key_read001 as predictor
from scripts.run_blind_channel_dev001 import fit_glyph_baseline
from voynich.compact_suffix_source import fit_compact
from voynich.dense_suffix_adapter import DenseSuffixAdapter
from voynich.finite_state_channel import SourceModel
from voynich.higher_order_unit_channel import MarkovSource, contexts_for
from voynich.native_suffix_marginal import marginal_python


class PythonScorer:
    def __init__(self, source, *_):
        self.source, self.alphabet = source, source.alphabet

    def score(self, units, record, rho, **kwargs):
        return marginal_python(self.source, units, record, rho, **kwargs)


def initial_sources():
    alphabet = ("a", "b")
    probabilities = {h: {"a": 0.75, "b": 0.25} for h in contexts_for(alphabet, 3)}
    first = SourceModel(alphabet, 1, {h: probabilities[h] for h in ("", "a", "b")})
    third = MarkovSource(alphabet, 3, probabilities)
    return first, third, {"order1": first.to_dict(), "order3": third.to_dict()}


def setup_files(tmp_path, monkeypatch):
    out, bulk = tmp_path / "results" / common.EXP, tmp_path / "outputs" / common.EXP
    opened, allowed = [], [False]

    def identity(path):
        raw = path.read_bytes()
        return {
            "path": str(path.relative_to(tmp_path)),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    def save(path, value, compressed=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value, allow_nan=False).encode()
        with path.open("xb") as stream:
            stream.write(gzip.compress(raw, mtime=0) if compressed else raw)
        return identity(path)

    def checked(spec):
        opened.append(spec["path"])
        if "answer" in spec["path"]:
            assert allowed[0], "Gold opened before audited prediction freeze"
        assert identity(tmp_path / spec["path"]) == spec
        return json.loads((tmp_path / spec["path"]).read_text())

    def archive(spec):
        assert identity(tmp_path / spec["path"]) == spec
        return json.loads(gzip.decompress((tmp_path / spec["path"]).read_bytes()))

    def verify(spec):
        assert identity(tmp_path / spec["path"]) == spec

    compact = fit_compact(["a"] * 7 + ["b"] * 3, "ab", 12, 4)
    path = tmp_path / "counts.npz"
    compact.save(path)
    source = DenseSuffixAdapter(compact, 64.0)
    raw = {"alphabet": ("a", "b"), "order": 12, "tau": 64.0, "counts": {"": {"a": 7, "b": 3}}}
    save(
        tmp_path / "results/LATIN-SOURCE-COMPACT-001/large.json",
        {"counts": identity(path), "selected": {"tau": 64.0}},
    )
    save(tmp_path / "results/NATIVE-SUFFIX-001/result.json", {"build": {}})
    save(tmp_path / "scripts/audit_global_key_read001.py", {"artificial": True})
    for module in (predictor, auditor, evaluator, common):
        replacements = {
            "ROOT": tmp_path,
            "OUT": out,
            "BULK": bulk,
            "FIT": tmp_path / "search",
            "artifact": identity,
            "save_new": save,
            "checked_artifact": checked,
            "load_archive": archive,
            "verify": verify,
            "load_source": lambda: source,
            "sources": initial_sources,
            "literal_source": lambda: raw,
            "limit_resources": lambda *_: None,
            "NativeMarginal": PythonScorer,
        }
        for key, value in replacements.items():
            if hasattr(module, key):
                monkeypatch.setattr(module, key, value)
    return SimpleNamespace(
        out=out,
        bulk=bulk,
        save=save,
        checked=checked,
        identity=identity,
        archive=archive,
        opened=opened,
        allowed=allowed,
        raw=raw,
        source=source,
    )


@pytest.mark.parametrize("late_failure", [False, True])
def test_32_case_prediction_audit_evaluation_flow_without_gold(tmp_path, monkeypatch, late_failure):
    f = setup_files(tmp_path, monkeypatch)
    panel, admitted, processes = {"cases": {}}, {}, []
    fit_campaign = {"status": "complete"}
    context = {"source_alphabet": ["a", "b"], "stop_probability": 1 / 225}
    bank = {
        "bank": [
            {"units": ["X", "Y"], "log_weight": -math.log(2)},
            {"units": ["Y", "X"], "log_weight": -math.log(2)},
        ],
        "best_index": 0,
        "best_units": ["X", "Y"],
    }
    for i, name in enumerate(common.NAMES):
        spec = f.save(tmp_path / f"{name}-transfer.json", {"records": ["X" * 224] * 2, "context": context})
        positive = i < 16
        answer = f.save(
            tmp_path / f"{name}-answer.json",
            {
                "positive": positive,
                "pair_index": i % 16,
                "plaintext": {"transfer": ["a" * 224] * 2, "fit": ["a" * 224] * 4} if positive else None,
                "units": ["X", "Y"],
            },
        )
        panel["cases"][name] = {
            "transfer": spec,
            "answer": answer,
            "positive": positive,
            "pair_index": i % 16,
        }
        parent = f.save(
            tmp_path / f"{name}-parent.json",
            {
                "units": ["X", "Y"],
                "frequency_units": ["Y", "X"],
                "baseline": fit_glyph_baseline(["XXY", "YYX"], ("X", "Y")),
            },
        )
        admitted[name] = {
            "parent": parent,
            "bank": f.save(tmp_path / f"{name}-bank.gz", bank, compressed=True),
        }
    for module in (predictor, auditor, evaluator):
        monkeypatch.setattr(module, "fit_status", lambda *_: (panel, {"build": {}}, admitted, fit_campaign))
    original_read = predictor.read_case

    def fail_after_points(*args, **kwargs):
        keep = kwargs["save_points"]

        def save_then_fail(value):
            keep(value)
            raise RuntimeError("Artificial fixed cap")

        kwargs["save_points"] = save_then_fail
        return original_read(*args, **kwargs)

    for i, name in enumerate(common.NAMES):
        fail = late_failure and i == 0
        monkeypatch.setattr(predictor, "read_case", fail_after_points if fail else original_read)
        if fail:
            with pytest.raises(RuntimeError, match="fixed cap"):
                predictor.predict(name, "fit-freeze")
        else:
            predictor.predict(name, "fit-freeze")
        processes.append({"case": name, "returncode": 1 if fail else 0})
    assert not any("answer" in p for p in f.opened)
    f.save(f.out / "campaign.json", {"freeze": "fit-freeze", "results": processes})
    f.save(tmp_path / "search/campaign.json", fit_campaign)
    auditor.audit_predictions()
    assert not any("answer" in p for p in f.opened)
    with pytest.raises(FileExistsError):
        predictor.predict(common.NAMES[0], "fit-freeze")

    def freeze(_commit, paths):
        assert (f.out / "audit.json").exists()
        assert any(p.endswith("/audit.json") for p in paths)
        assert not any("answer" in p for p in f.opened)
        f.allowed[0] = True

    monkeypatch.setattr(evaluator, "require_frozen", freeze)
    monkeypatch.setattr(evaluator, "observed", lambda spec, count: f.checked(spec))
    # An artificial prior report tests comparison binding without opening answers.
    original = gate_fixture()
    for i, name in enumerate(common.NAMES):
        original[name]["pair_index"] = i % 16
        if i < 16:
            original[name]["arms"]["mixture"].update(exact_records=0)
    previous = {
        "cases": original,
        "totals": {"mixture": {"edits": 16, "gold_characters": 7168}},
        "recovery_gate": "FAIL",
        "iid_screen_gate": "FAIL",
    }
    prior = f.save(tmp_path / "results/BLIND-CHANNEL-CONFIRM-002/evaluation.json", previous)
    f.save(
        tmp_path / "results/BLIND-CHANNEL-CONFIRM-002/evaluation-audit.json",
        {"status": "PASS", "evaluation": prior},
    )
    evaluator.evaluate("prediction-freeze")
    result = json.loads((f.out / "evaluation.json").read_text())
    assert result["exposed_development_not_fresh_qualification"] is True
    assert result["original_fresh_fail_unchanged"] is True
    assert result["comparison_to_original_fresh_run"]["old_primary_edits"] == 16
    assert result["comparison_to_original_fresh_run"]["new_primary_edits"] == (448 if late_failure else 0)
    assert result["totals"]["mixture"]["gold_characters"] == 7168
    assert result["totals"]["mixture"]["edits"] == (448 if late_failure else 0)
    assert result["completed_case_metrics_conditional"]["mixture"]["cases"] == (15 if late_failure else 16)
    assert result["independent_record_edit_checks"] == 192 and result["independent_oracle_records"] == 32
    assert result["recovery_gate"] == ("FAIL" if late_failure else "PASS")
    assert not result["screen_clauses"]["no_shuffle_beats_frozen_iid"]
    assert sum("answer" in p for p in f.opened) == 16


def gate_fixture():
    reports = {}
    for i, name in enumerate(common.NAMES):
        reports[name] = {
            "positive": i < 16,
            "fit_complete": True,
            "prediction_complete": True,
            "evidence": {"gain_bits_vs_iid": 1 if i < 16 else -1},
        }
        if i < 16:
            reports[name]["arms"] = {
                arm: {
                    "edits": 1 if arm == "mixture" else 100 if arm == "frequency_order3" else 0,
                    "gold_characters": 448,
                    "supported_records": 2,
                }
                for arm in evaluator.ALL_ARMS
            }
    return reports


@pytest.mark.parametrize(
    "fault",
    [
        "null_fit_failure",
        "null_prediction_failure",
        "noise_false_positive",
        "positive_not_above_iid",
        "missing_evidence",
        "individual_cer",
        "mean_cer",
        "oracle_gap",
        "frequency_exact",
        "unsupported",
    ],
)
def test_fixed_gates_fail_for_all_failure_types(fault):
    reports = gate_fixture()
    assert all(evaluator.decisions(reports)[0].values())
    assert all(evaluator.decisions(reports)[1].values())
    positive, negative = reports[common.NAMES[0]], reports[common.NAMES[16]]
    if fault == "null_fit_failure":
        negative["fit_complete"] = False
    elif fault == "null_prediction_failure":
        negative["prediction_complete"] = False
    elif fault == "noise_false_positive":
        negative["evidence"]["gain_bits_vs_iid"] = 0.001
    elif fault == "positive_not_above_iid":
        positive["evidence"]["gain_bits_vs_iid"] = 0
    elif fault == "missing_evidence":
        negative["evidence"] = None
    elif fault == "individual_cer":
        positive["arms"]["mixture"]["edits"] = 23
    elif fault == "mean_cer":
        for row in list(reports.values())[:16]:
            row["arms"]["mixture"]["edits"] = 9
            row["arms"]["oracle_large"]["edits"] = 9
    elif fault == "oracle_gap":
        positive["arms"]["mixture"]["edits"] = 9
    elif fault == "frequency_exact":
        positive["arms"]["frequency_order3"]["edits"] = 0
    else:
        positive["arms"]["mixture"]["supported_records"] = 1
    recovery, screen = evaluator.decisions(reports)
    assert not all(recovery.values()) or not all(screen.values())


def test_oracle_and_point_audits_catch_score_and_encoding_corruption(tmp_path, monkeypatch):
    f = setup_files(tmp_path, monkeypatch)
    rows, delta = evaluator.oracle_rows(f.source, f.raw, ["X", "YY"], ["XXX", "XYY"], 0.2)
    assert delta < 1e-12
    parent, bank = {"units": ["X", "YY"]}, {"best_units": ["X", "YY"]}
    points = {"original": rows, "fit_selected": copy.deepcopy(rows)}
    assert auditor.audit_points(f.raw, ["XXX", "XYY"], parent, bank, points, 0.2) == 4
    points["fit_selected"][0]["joint_log_probability"] += 0.01
    with pytest.raises(ValueError):
        auditor.audit_points(f.raw, ["XXX", "XYY"], parent, bank, points, 0.2)


def terminal_fixture():
    rows = [{"case": name, "fit_returncode": 0, "audit_returncode": 0} for name in common.NAMES]
    return (
        {"freeze": "source", "status": "complete", "results": rows},
        {"freeze": "source", "workers": 8, "cases": list(common.NAMES)},
    )


@pytest.mark.parametrize(
    "fault",
    ["missing", "duplicate", "reorder", "freeze", "status", "nonterminal", "boolcode", "failed_fit_audit"],
)
def test_terminal_inventory_rejects_partial_or_inconsistent_fit_checkpoint(fault):
    campaign, started = terminal_fixture()
    common.terminal_inventory(campaign, started)
    if fault == "missing":
        campaign["results"].pop()
    elif fault == "duplicate":
        campaign["results"][-1]["case"] = common.NAMES[0]
    elif fault == "reorder":
        campaign["results"].reverse()
    elif fault == "freeze":
        started["freeze"] = "different"
    elif fault == "status":
        campaign["status"] = "failures_retained"
    elif fault == "nonterminal":
        campaign["results"][0]["fit_returncode"] = None
    elif fault == "boolcode":
        campaign["results"][0]["fit_returncode"] = False
    else:
        campaign["results"][0]["fit_returncode"] = 1
    with pytest.raises(ValueError):
        common.terminal_inventory(campaign, started)


def test_terminal_failed_fit_and_audit_timeout_are_retained():
    campaign, started = terminal_fixture()
    campaign["status"] = "failures_retained"
    campaign["results"][0].update(fit_returncode=1, audit_returncode=None)
    campaign["results"][1].update(audit_returncode=None, audit_status="outer_timeout")
    common.terminal_inventory(campaign, started)


@pytest.mark.parametrize(
    "fault", [None, "audit_hash", "archive_bytes", "access_claim", "case_input", "missing_process"]
)
def test_published_global_bank_admission_checks_all_bindings_without_loading_answers(
    tmp_path, monkeypatch, fault
):
    root, fit = tmp_path, tmp_path / "search"
    fit.mkdir()

    def identity(path):
        raw = path.read_bytes()
        return {
            "path": str(path.relative_to(root)),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return identity(path)

    campaign, started = terminal_fixture()
    write(fit / "campaign.json", campaign)
    write(fit / "campaign-started.json", started)
    fit_panel, panel = {"cases": {}}, {"cases": {}}
    write(root / "scripts/audit_global_key_search001.py", {"artificial": True})
    for process in campaign["results"]:
        name = process["case"]
        spec = write(root / (name + "-fit-input.json"), {"fitting_only": True})
        fit_panel["cases"][name] = {"fit": spec}
        panel["cases"][name] = {"fit": spec, "answer": {"path": "FORBIDDEN-answer"}}
        old = write(root / f"results/BLIND-CHANNEL-CONFIRM-002/{name}-fit.json", {"original": True})
        parent = write(root / f"results/BLIND-CHANNEL-CONFIRM-002/{name}-parent.json", {"original": True})
        fields = {
            k: write(root / f"{name}-{k}.json", {"artificial": k})
            for k in ("reader_bank", "old_bank", "warm", "search", "progress")
        }
        result = write(
            fit / (name + ".json"),
            {
                **fields,
                "case": name,
                "freeze": "source",
                "status": "complete_fit_only",
                "fit": spec,
                "old_fit": old,
                "parent": parent,
                "transfer_or_answer_files_opened": False,
                "unique_full_fit_scores_including_warm_and_neighborhood": 20,
            },
        )
        write(
            fit / (name + "-audit.json"),
            {
                "case": name,
                "status": "PASS",
                "result": result,
                "auditor": identity(root / "scripts/audit_global_key_search001.py"),
                "full_fit_scores_accounted": 20,
                "transfer_or_answer_files_opened": False,
            },
        )
        write(fit / (name + "-process.json"), process)
    freezes = []
    monkeypatch.setattr(common, "ROOT", root)
    monkeypatch.setattr(common, "FIT", fit)
    monkeypatch.setattr(common, "artifact", identity)
    monkeypatch.setattr(common, "require_frozen", lambda freeze, paths: freezes.append((freeze, paths)))
    monkeypatch.setattr(common, "search_admission", lambda freeze: (fit_panel, {"build": {}}))
    monkeypatch.setattr(common, "panel_inputs", lambda freeze: (panel, {}))
    name = common.NAMES[0]
    if fault == "audit_hash":
        p = fit / (name + "-audit.json")
        v = json.loads(p.read_text())
        v["result"]["sha256"] = "bad"
        p.write_text(json.dumps(v))
    elif fault == "archive_bytes":
        (root / (name + "-reader_bank.json")).write_text("changed")
    elif fault == "access_claim":
        p = fit / (name + ".json")
        v = json.loads(p.read_text())
        v["transfer_or_answer_files_opened"] = True
        p.write_text(json.dumps(v))
    elif fault == "case_input":
        panel["cases"][name]["fit"] = {"path": "wrong"}
    elif fault == "missing_process":
        (fit / (name + "-process.json")).unlink()
    if fault:
        with pytest.raises((ValueError, FileNotFoundError)):
            common.fit_status("published-fit")
    else:
        _, _, admitted, actual = common.fit_status("published-fit")
        assert actual == campaign and len(admitted) == 32
        assert admitted[name]["bank"] == admitted[name]["reader_bank"]
        assert len(freezes) == 2 and all(f == "published-fit" for f, _ in freezes)
        assert all("FORBIDDEN" not in p for _, paths in freezes for p in paths)
        assert any(p.endswith(name + "-audit.json") for p in freezes[-1][1])


def test_original_comparison_counts_regressions_and_rejects_reduced_denominators():
    reports = gate_fixture()
    for i, name in enumerate(common.NAMES):
        reports[name]["pair_index"] = i % 16
        if i < 16:
            reports[name]["arms"]["mixture"]["exact_records"] = 0
    previous = {
        "cases": copy.deepcopy(reports),
        "totals": {"mixture": {"edits": 16, "gold_characters": 7168}},
        "recovery_gate": "FAIL",
        "iid_screen_gate": "FAIL",
    }
    reports[common.NAMES[0]]["arms"]["mixture"]["edits"] = 0
    reports[common.NAMES[1]]["arms"]["mixture"]["edits"] = 3
    result = evaluator.original_comparison(reports, previous)
    assert (result["cases_improved"], result["cases_tied"], result["cases_regressed"]) == (1, 14, 1)
    assert result["edits_reduced"] == -1
    previous["cases"][common.NAMES[0]]["arms"]["mixture"]["gold_characters"] = 224
    with pytest.raises(ValueError, match="denominator"):
        evaluator.original_comparison(reports, previous)
