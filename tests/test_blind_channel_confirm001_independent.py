"""Artificial-only independent qualification orchestration checks.

No fixture loads a corpus, saved experimental source, key, answer or prediction.
Real inference and trace auditors operate only on two-letter hand fixtures.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import itertools
import json
import math
from dataclasses import asdict, replace
from fractions import Fraction
from types import SimpleNamespace

import pytest

from scripts import build_blind_channel_confirm001 as builder
from scripts import run_blind_channel_confirm001 as runner
from voynich import higher_order_unit_refine as refine_module
from voynich.finite_state_channel import SourceModel
from voynich.finite_state_channel_fit import CodingContext
from voynich.higher_order_unit_channel import MarkovSource, contexts_for
from voynich.unit_channel_search import channel_from_units


def artificial_sources():
    alphabet = ("a", "b")
    probabilities = {
        h: {"a": .75 if not h or h[-1] == "a" else .25,
            "b": .25 if not h or h[-1] == "a" else .75}
        for h in contexts_for(alphabet, 3)
    }
    source3 = MarkovSource(alphabet, 3, probabilities)
    source1 = SourceModel(alphabet, 1, {h: probabilities[h] for h in ("", "a", "b")})
    context = CodingContext(alphabet, ("X", "Y"), 8, 2, 2, 3, stop_probability=.25)
    return source1, source3, context


def enumerated_score(source, units, records, context):
    """Full plaintext enumeration with rational multiplication; no engine helper."""
    likelihood = 0.
    for record in records:
        mass = Fraction()
        for length in range(len(record) + 1):
            for chars in itertools.product(source.alphabet, repeat=length):
                if "".join(units[source.alphabet.index(a)] for a in chars) != record:
                    continue
                value, history = Fraction(1, 4), ""
                for a in chars:
                    value *= Fraction(3, 4) * Fraction(source.probabilities[history][a])
                    history = (history + a)[-source.order:]
                mass += value
        if not mass:
            return None
        likelihood += math.log(float(mass))
    def width(n):
        return (n - 1).bit_length()
    bits = (width(context.source_count) + width(context.max_states)
            + len(units) * (width(context.max_alternatives) + width(context.max_emission_length))
            + width(len(context.glyph_alphabet)) * sum(map(len, units)))
    return {"model_bits": bits, "log_likelihood": likelihood,
            "data_bits": -likelihood / math.log(2), "total_bits": bits - likelihood / math.log(2)}


def disable_process_limits(monkeypatch):
    monkeypatch.setattr(runner.resource, "setrlimit", lambda *_: None)
    monkeypatch.setattr(runner.signal, "signal", lambda *_: None)
    monkeypatch.setattr(runner.signal, "alarm", lambda *_: None)


def test_fit_reads_only_fit_artifact_and_real_trace_metadata_passes(tmp_path, monkeypatch):
    source1, source3, context = artificial_sources()
    disable_process_limits(monkeypatch)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "SOURCE_PATHS", ["fixture.py"])
    (tmp_path / "fixture.py").write_text("# Artificial frozen source\n")
    source_path = tmp_path / runner.SELECTION
    source_path.parent.mkdir(parents=True)
    source_path.write_text("{}")
    name = runner.CASE_NAMES[0]
    records = ["XXX", "XY", "", "X"]
    case = {"search_seed": 63101, "artifacts": {
        role: {"path": role, "sha256": "f" * 64} for role in ("fit", "transfer", "answer")}}
    models = {"order1": source1.to_dict(), "order3": source3.to_dict()}
    monkeypatch.setattr(runner, "inputs", lambda _: (
        {"pipeline_freeze": "a" * 40, "cases": {name: case}}, source1, source3, models))
    opened = []

    def checked(artifact):
        opened.append(artifact["path"])
        assert artifact["path"] == "fit", "Fitting touched forbidden transfer/answer input"
        return {"records": records, "context": asdict(context)}

    monkeypatch.setattr(runner, "checked_artifact", checked)
    original_search, original_refine = runner.search_unit_channel, runner.refine_units

    def tiny_search(source, observed, coding, *, config):
        assert (config.seed, config.restarts, config.max_seconds, config.max_sweeps) == (63101, 16, 300., 80)
        return original_search(source, observed, coding,
                               config=replace(config, restarts=2, max_sweeps=1, max_seconds=10.))

    def tiny_refine(source, observed, coding, initial, **kwargs):
        assert kwargs == {"max_sweeps": 20, "max_seconds": 300., "tolerance": 1e-8}
        return original_refine(source, observed, coding, initial, max_sweeps=2, max_seconds=10.)

    monkeypatch.setattr(runner, "search_unit_channel", tiny_search)
    monkeypatch.setattr(runner, "refine_units", tiny_refine)
    runner.fit_case(name, "b" * 40)
    frozen = json.loads(runner.selected_path(name).read_text())
    assert opened == ["fit"]
    for field in ("stage1_trace", "stage2_trace"):
        artifact = frozen[field]
        packed = (tmp_path / artifact["path"]).read_bytes()
        assert len(packed) == artifact["bytes"]
        assert hashlib.sha256(packed).hexdigest() == artifact["sha256"]
    for source, key, score_key in ((source1, "stage1_units", "stage1_score"),
                                   (source3, "units", "score")):
        expected = enumerated_score(source, frozen[key], records, context)
        assert expected is not None
        assert frozen[score_key] == pytest.approx(expected, abs=1e-11)
    assert max(frozen["independent_final_score_deltas"]) < 1e-11
    with pytest.raises((FileExistsError, ValueError)):
        runner.fit_case(name, "b" * 40)


@pytest.mark.parametrize("failed", [set(), {"B-key1", "B-key2-shuffle"}, set(runner.CASE_NAMES)])
def test_complete_evaluation_retains_failed_cases_and_checks_control_arithmetic(tmp_path, monkeypatch, failed):
    source1, source3, context = artificial_sources()
    disable_process_limits(monkeypatch)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    directory = tmp_path / f"results/{runner.EXPERIMENT}"
    directory.mkdir(parents=True)
    source_path = tmp_path / runner.SELECTION
    source_path.parent.mkdir(parents=True)
    source_path.write_text("{}")
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    records = {"fit": ["X", "XX", "", "XY"], "transfer": ["X", "XY"]}
    plain = {"fit": ["a", "aa", "", "ab"], "transfer": ["a", "ab"]}
    artifacts, cases, jobs = {}, {}, []
    baseline = runner.fit_glyph_baseline(records["fit"], context.glyph_alphabet)
    selected_score = enumerated_score(source3, ("X", "Y"), records["fit"], context)
    for index, name in enumerate(runner.CASE_NAMES):
        positive = not name.endswith("-shuffle")
        case = {"positive": positive, "artifacts": {}}
        for split in ("fit", "transfer"):
            key = f"{name}/{split}"
            case["artifacts"][split] = {"path": key, "sha256": str(index) * 64}
            artifacts[key] = {"records": records[split], "context": asdict(context)}
        key = f"{name}/answer"
        case["artifacts"]["answer"] = {"path": key}
        artifacts[key] = {"plaintext": plain if positive else None,
                          "gold_channel": channel_from_units(source3, context, ("X", "Y")).to_dict()
                          if positive else None}
        cases[name] = case
        job = {"case": name, "returncode": 1 if name in failed else 0}
        jobs.append(job)
        (directory / f"{name}_process.json").write_text(json.dumps(job))
        # Even an existing key from a failed process must not be treated as success.
        frozen = {"fit_sha256": case["artifacts"]["fit"]["sha256"],
                  "source_selection_sha256": source_hash, "units": ["X", "Y"],
                  "stage1_units": ["X", "Y"], "score": selected_score, "baseline": baseline}
        runner.selected_path(name).write_text(json.dumps(frozen))
    (directory / "campaign.json").write_text(json.dumps({"results": jobs}))
    monkeypatch.setattr(runner, "inputs", lambda _: (
        {"cases": cases}, source1, source3, {"order1": source1.to_dict(), "order3": source3.to_dict()}))
    events = []
    monkeypatch.setattr(runner, "require_frozen", lambda commit, paths: events.append(("freeze", commit, paths)))

    def checked(artifact):
        events.append(("read", artifact["path"]))
        assert events[0][0] == "freeze"
        return copy.deepcopy(artifacts[artifact["path"]])

    monkeypatch.setattr(runner, "checked_artifact", checked)
    runner.evaluate("c" * 40)
    result = json.loads((directory / "evaluation.json").read_text())
    assert set(result["cases"]) == set(runner.CASE_NAMES)
    failed_positive = sum(not name.endswith("-shuffle") for name in failed)
    assert result["audit"]["record_predictions_replayed"] == 336 - 12 * len(failed)
    assert result["audit"]["floor_records_replayed"] == 96 - 6 * failed_positive
    assert result["decision"]["recovery_gates"]["all_jobs_complete"] == (not failed)
    assert not result["decision"]["semantic_rejection_qualified"]
    for name in runner.CASE_NAMES:
        row = result["cases"][name]
        if name in failed:
            assert row["learned"]["transfer"]["log_likelihood"] is None
            assert row["learned"]["transfer"]["model_bits"] is None
            assert not row["iid_diagnostic"]["flag"]
            if not name.endswith("-shuffle"):
                assert row["learned"]["transfer"]["edits"] == 3
                assert row["learned"]["transfer"]["dictionary_floor"]["edits"] is None
        else:
            selected = enumerated_score(source3, ("X", "Y"), records["fit"], context)
            rho = baseline["stop_count"] / baseline["stop_denominator"]
            ps = dict(zip(baseline["glyph_alphabet"], baseline["counts"], strict=True))

            def iid_ll(strings):
                return sum(math.log(rho) + len(text) * math.log1p(-rho)
                           + sum(math.log(ps[g] / baseline["denominator"]) for g in text)
                           for text in strings)

            fit_margin = baseline["model_bits"] - iid_ll(records["fit"]) / math.log(2) - selected["total_bits"] - 1
            transfer = enumerated_score(source3, ("X", "Y"), records["transfer"], context)
            gain = (transfer["log_likelihood"] - iid_ll(records["transfer"])) / (3 * math.log(2))
            assert row["iid_diagnostic"]["fit_margin_bits"] == pytest.approx(fit_margin, abs=1e-11)
            assert row["iid_diagnostic"]["transfer_gain_bits_per_glyph"] == pytest.approx(gain, abs=1e-11)
    artifact = result["predictions"]
    packed = (tmp_path / artifact["path"]).read_bytes()
    assert hashlib.sha256(packed).hexdigest() == artifact["sha256"]
    assert set(json.loads(gzip.decompress(packed))) == set(runner.CASE_NAMES)
    events.clear()
    with pytest.raises(FileExistsError):
        runner.evaluate("c" * 40)
    assert not any(event[0] == "read" for event in events)


def test_campaign_preserves_timeout_nonzero_and_success_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "inputs", lambda _: ({"pipeline_freeze": "p"},))
    attempts = []

    def run(command, **kwargs):
        name = command[command.index("--case") + 1]
        attempts.append(name)
        assert kwargs["timeout"] == 960
        assert all(kwargs["env"][key] == "1" for key in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"))
        if name == runner.CASE_NAMES[0]:
            raise runner.subprocess.TimeoutExpired(command, 960)
        return SimpleNamespace(returncode=2 if name == runner.CASE_NAMES[1] else 0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", run)
    runner.campaign("d" * 40)
    path = tmp_path / f"results/{runner.EXPERIMENT}/campaign.json"
    campaign = json.loads(path.read_text())
    assert sorted(attempts) == sorted(runner.CASE_NAMES)
    assert [job["case"] for job in campaign["results"]] == list(runner.CASE_NAMES)
    assert campaign["status"] == "failures_preserved"
    assert campaign["results"][0]["returncode"] is None
    assert campaign["results"][1]["returncode"] == 2
    with pytest.raises(FileExistsError):
        runner.campaign("d" * 40)
    assert len(attempts) == len(runner.CASE_NAMES)


def test_complete_tolerance_stop_does_not_certify_different_returned_neighbor():
    _, source, context = artificial_sources()
    result = refine_module.refine_units(source, ["XXX"], context, ("Y", "X"),
                                       max_sweeps=1, max_seconds=10., tolerance=1e6)
    sweep = result["trace"][0]
    assert sweep["complete"] and not sweep["accepted"]
    assert result["units"] != sweep["parent_units"]
    assert not result["best_is_certified_local_optimum"]
    all_scores = [enumerated_score(source, result["initial_units"], ["XXX"], context)]
    for candidate in sweep["neighbors"]:
        expected = enumerated_score(source, candidate["units"], ["XXX"], context)
        if expected is None:
            assert candidate["score"] is None
        else:
            assert candidate["score"] == pytest.approx(expected, abs=1e-11)
            all_scores.append(expected)
    assert result["score"]["total_bits"] == pytest.approx(min(row["total_bits"] for row in all_scores))
    runner.trace_accounting(result, source.to_dict(), context)
    corrupt = copy.deepcopy(result)
    corrupt["best_is_certified_local_optimum"] = True
    with pytest.raises(ValueError):
        runner.trace_accounting(corrupt, source.to_dict(), context)


def test_builder_stops_on_duplicate_key_without_redraw(monkeypatch):
    alphabet = tuple("abcdefghiklmnopqrstuxyz")
    original = builder.make_channel
    calls = []

    def repeated_key(source_alphabet, family, seed):
        calls.append(seed)
        return original(source_alphabet, family, 60129)

    def windows(_payload, role, index):
        return [{"text": "a" * 224, "start": index * 4096 + j * 256}
                for j in range(4 if role == "fit" else 2)]

    monkeypatch.setattr(builder, "make_channel", repeated_key)
    with pytest.raises(ValueError, match="Duplicate"):
        builder.construct({"sallust": {}, "tacitus": {}}, alphabet, windows)
    assert calls == [60129, 60129 + 104729]


def test_gate_average_is_equal_key_weighted_and_rates_are_not_capped():
    cases = {}
    for index, name in enumerate(runner.CASE_NAMES):
        denominator = 100 if index < 2 else 10_000
        edits = 2 if index < 2 else 0
        cases[name] = {arm: {"transfer": {"edits": count, "gold_characters": denominator}}
                       for arm, count in (("learned", edits), ("oracle", 0), ("frequency", denominator))}
        cases[name]["iid_diagnostic"] = {"flag": not name.endswith("-shuffle")}
    decision = runner.decision_gates(cases, True)
    assert decision["macro_transfer_cer"] == pytest.approx(.02 / 8)
    assert decision["recovery_pass"]
    cases["B-key1"]["learned"]["transfer"]["edits"] = 200
    assert runner.decision_gates(cases, True)["macro_transfer_cer"] == pytest.approx(2 / 8)


def test_incomplete_null_searches_cannot_pass_screening():
    cases = {name: {**{arm: {"transfer": {"edits": edits, "gold_characters": 448}}
                      for arm, edits in (("learned", 0), ("oracle", 0), ("frequency", 100))},
                    "iid_diagnostic": {"flag": not name.endswith("-shuffle")}}
             for name in runner.CASE_NAMES}
    assert runner.decision_gates(cases, True)["iid_screen_pass"]
    result = runner.decision_gates(cases, False)
    assert not result["recovery_pass"]
    assert not result["iid_screen_pass"], "Unflagged failed nulls are not successful negative controls"


def test_data_guard_checks_registered_seed_before_loading_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    corpus = tmp_path / runner.CORPUS
    corpus.parent.mkdir(parents=True)
    corpus.write_text("{}")
    manifest = {"experiment": runner.EXPERIMENT, "pipeline_freeze": "a" * 40,
                "final_author_roles": {"fit": "sallust", "transfer": "tacitus"},
                "corpus_manifest_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                "cases": {name: {"search_seed": 63101 + 257 * index, "positive": index % 2 == 0}
                          for index, name in enumerate(runner.CASE_NAMES)}}
    manifest["cases"]["B-key8"]["search_seed"] += 1
    (tmp_path / runner.MANIFEST).write_text(json.dumps(manifest))
    frozen = []
    monkeypatch.setattr(runner, "require_frozen", lambda commit, paths: frozen.append((commit, paths)))
    monkeypatch.setattr(runner, "sources", lambda: pytest.fail("Must reject seed before source load"))
    with pytest.raises(ValueError, match="seed"):
        runner.inputs("b" * 40)
    assert frozen == [("a" * 40, runner.SOURCE_PATHS), ("b" * 40, [runner.MANIFEST, runner.CORPUS])]


def test_freeze_rejection_precedes_fit_or_answer_access(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = tmp_path / runner.MANIFEST
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"experiment": runner.EXPERIMENT, "pipeline_freeze": "a" * 40,
                               "cases": dict.fromkeys(runner.CASE_NAMES),
                               "final_author_roles": {"fit": "sallust", "transfer": "tacitus"}}))
    monkeypatch.setattr(runner, "require_frozen", lambda *_: (_ for _ in ()).throw(ValueError("Unfrozen fixture")))
    monkeypatch.setattr(runner, "checked_artifact", lambda *_: pytest.fail("Freeze failed before data access"))
    monkeypatch.setattr(runner, "sources", lambda: pytest.fail("Freeze failed before source access"))
    with pytest.raises(ValueError, match="Unfrozen"):
        runner.inputs("b" * 40)


def test_failed_direct_fit_cannot_repeat_search(tmp_path, monkeypatch):
    source1, source3, context = artificial_sources()
    disable_process_limits(monkeypatch)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    name = runner.CASE_NAMES[0]
    case = {"search_seed": 63101, "artifacts": {"fit": {"path": "fit"}}}
    monkeypatch.setattr(runner, "inputs", lambda _: (
        {"pipeline_freeze": "a" * 40, "cases": {name: case}}, source1, source3,
        {"order1": source1.to_dict(), "order3": source3.to_dict()}))
    monkeypatch.setattr(runner, "checked_artifact", lambda _: (
        {"records": ["X", "XY", "XX", "Y"], "context": asdict(context)}))
    attempts = []

    def broken_search(*_args, **_kwargs):
        attempts.append(1)
        raise ArithmeticError("Artificial interrupted fit")

    monkeypatch.setattr(runner, "search_unit_channel", broken_search)
    with pytest.raises(ArithmeticError, match="interrupted"):
        runner.fit_case(name, "b" * 40)
    with pytest.raises((FileExistsError, ValueError)):
        runner.fit_case(name, "b" * 40)
    assert attempts == [1]


def test_failed_evaluation_cannot_reopen_answers(tmp_path, monkeypatch):
    source1, source3, context = artificial_sources()
    disable_process_limits(monkeypatch)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    directory = tmp_path / f"results/{runner.EXPERIMENT}"
    directory.mkdir(parents=True)
    jobs = [{"case": name, "returncode": 1} for name in runner.CASE_NAMES]
    (directory / "campaign.json").write_text(json.dumps({"results": jobs}))
    manifest = {"cases": {name: {"positive": not name.endswith("-shuffle"),
                                "artifacts": {role: {"path": role} for role in ("fit", "answer")}}
                          for name in runner.CASE_NAMES}}
    monkeypatch.setattr(runner, "inputs", lambda _: (
        manifest, source1, source3, {"order1": source1.to_dict(), "order3": source3.to_dict()}))
    monkeypatch.setattr(runner, "require_frozen", lambda *_: None)
    answers = []

    def checked(artifact):
        if artifact["path"] == "answer":
            answers.append(1)
            raise ArithmeticError("Artificial answer-phase interruption")
        return {"records": ["X"], "context": asdict(context)}

    monkeypatch.setattr(runner, "checked_artifact", checked)
    with pytest.raises(ArithmeticError, match="answer-phase"):
        runner.evaluate("c" * 40)
    with pytest.raises(FileExistsError):
        runner.evaluate("c" * 40)
    assert answers == [1]


def test_failed_builder_cannot_regenerate_panel(tmp_path, monkeypatch):
    disable_process_limits(monkeypatch)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "require_frozen", lambda *_: None)
    path = tmp_path / builder.CORPUS
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status": "prepared", "alphabet": list("abcdefghiklmnopqrstuxyz"),
                               "sources": {name: {"derived_path": name, "derived_sha256": "f" * 64,
                                                  "role": role}
                                           for name, role in (("sallust", "F"), ("tacitus", "T"))}}))
    monkeypatch.setattr(builder, "checked_artifact", lambda _: {})
    monkeypatch.setattr(runner.sys, "argv", ["build", "--pipeline-freeze", "a" * 40,
                                           "--corpus-freeze", "b" * 40])
    attempts = []

    def failed_construct(*_args):
        attempts.append(1)
        raise ArithmeticError("Artificial construction interruption")

    monkeypatch.setattr(builder, "construct", failed_construct)
    with pytest.raises(ArithmeticError, match="construction"):
        builder.main()
    with pytest.raises(FileExistsError):
        builder.main()
    assert attempts == [1]


def test_builder_rejects_failed_corpus_screen_before_derived_read(tmp_path, monkeypatch):
    disable_process_limits(monkeypatch)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "require_frozen", lambda *_: None)
    path = tmp_path / builder.CORPUS
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status": "blocked_exact_cross_author_overlap",
                               "alphabet": list("abcdefghiklmnopqrstuxyz"),
                               "sources": {name: {"derived_path": name, "derived_sha256": "f" * 64}
                                           for name in ("sallust", "tacitus")}}))
    monkeypatch.setattr(runner.sys, "argv", ["build", "--pipeline-freeze", "a" * 40,
                                           "--corpus-freeze", "b" * 40])
    monkeypatch.setattr(builder, "checked_artifact", lambda _: pytest.fail("Blocked corpus must stay unopened"))
    with pytest.raises(ValueError):
        builder.main()
