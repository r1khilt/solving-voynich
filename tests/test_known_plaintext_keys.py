from itertools import product
from collections import Counter
import hashlib
import json

import pytest

from voynich.known_plaintext_keys import oracle_statistics, solve_known_plaintext


def brute(source, cipher, rows, glyphs):
    pool = [(g,) for g in range(glyphs)]+list(product(range(glyphs), repeat=2))
    used = {r for s in source for r in s}
    return {tuple(k[r] if r in used else None for r in range(rows))
            for k in product(pool, repeat=rows)
            if all(tuple(g for r in s for g in k[r]) == tuple(c) for s, c in zip(source, cipher, strict=True))}


@pytest.mark.parametrize("source", [((0, 1),), ((0, 1, 0),), ((1, 1, 0), (0, 1)), ((0, 0), (0,))])
def test_all_tiny_inputs_against_independent_full_dictionary_enumeration(source):
    # Two glyphs, two rows, all observed strings of lengths1..2n, including
    # impossible ones. Brute reference ranges over units rather than lengths.
    options = [list(c for n in range(1, 2*len(s)+1) for c in product(range(2), repeat=n)) for s in source]
    for cipher in product(*options):
        solved = solve_known_plaintext(source, cipher, rows=2, glyphs=2)
        assert solved["complete"]
        assert set(solved["solutions"]) == brute(source, cipher, 2, 2)


def test_ambiguous_boundaries_duplicate_units_unused_rows_and_exact_prior():
    solved = solve_known_plaintext(((0, 1),), ((0, 0, 0),), rows=3, glyphs=1)
    assert set(solved["solutions"]) == {((0,), (0, 0), None), ((0, 0), (0,), None)}
    value = oracle_statistics(solved)
    assert value["full_key_support_size"] == 4
    assert value["used_key_map_probability"] == "1/2"
    assert value["full_key_map_probability"] == "1/4"
    assert value["determined_used_rows"] == []
    assert value["bayes_expected_used_row_matches"] == "1"
    unique = solve_known_plaintext(((0, 1, 0),), ((0, 0, 0),), rows=2, glyphs=1)
    assert unique["solutions"] == (((0,), (0,)),)


def test_second_record_disambiguates_same_shared_dictionary_and_resets_offset():
    one = solve_known_plaintext(((0, 1),), ((0, 0, 0),), rows=2, glyphs=1)
    two = solve_known_plaintext(((0, 1), (0,)), ((0, 0, 0), (0,)), rows=2, glyphs=1)
    assert len(one["solutions"]) == 2 and len(two["solutions"]) == 1
    assert two["solutions"][0] == ((0,), (0, 0))


def test_caps_never_claim_exhaustive_or_compute_posterior_from_partial_support():
    for caps in ({"node_cap": 1}, {"solution_cap": 1}):
        solved = solve_known_plaintext(((0, 1),), ((0, 0, 0),), rows=2, glyphs=1, **caps)
        assert not solved["complete"] and solved["cutoff"] in caps
        assert oracle_statistics(solved) is None
    impossible = solve_known_plaintext(((0, 0),), ((0, 1),), rows=1, glyphs=2)
    assert impossible["complete"] and not impossible["solutions"]
    assert oracle_statistics(impossible) is None


def test_canonical_names_with_unseen_glyphs_do_not_tilt_uniform_literal_oracle():
    pool = [(g,) for g in range(3)]+list(product(range(3), repeat=2))
    counts = Counter()
    for key in product(pool, repeat=3):
        observed = tuple(g for r in (0, 1) for g in key[r])
        names = tuple(dict.fromkeys(observed))
        names += tuple(g for g in range(3) if g not in names)
        canonical = {g: i for i, g in enumerate(names)}
        if tuple(canonical[g] for g in observed) == (0, 1, 0):
            counts[tuple(tuple(canonical[g] for g in u) for u in key)] += 1
    # Each compatible canonical full dictionary has the same raw multiplicity,
    # including arbitrary unseen-glyph values in the unused third row.
    assert set(counts.values()) == {6}
    solved = solve_known_plaintext(((0, 1),), ((0, 1, 0),), rows=3, glyphs=3)
    assert oracle_statistics(solved)["full_key_support_size"] == len(counts) == 24
    assert {k[:2] for k in counts} == {k[:2] for k in solved["solutions"]}


@pytest.mark.parametrize("source,cipher,kwargs", [([], [], {}), ([(0,)], [(2,)], {}),
    ([(True,)], [(0,)], {}), ([(0,)], [(0,)], {"rows": 0}),
    ([(0,)], [(0,)], {"solution_cap": False})])
def test_invalid_inputs(source, cipher, kwargs):
    with pytest.raises(ValueError):
        solve_known_plaintext(source, cipher, rows=kwargs.pop("rows", 2), glyphs=2, **kwargs)


def test_real_runner_and_auditor_transport_artificial_grid(tmp_path, monkeypatch):
    from scripts import audit_key_observability001 as checker
    from scripts import run_key_observability001 as runner
    from scripts import run_blind_channel_dev004 as storage

    monkeypatch.setattr(storage, "ROOT", tmp_path)

    source, cipher = ((0, 1), (0,)), ((0, 0, 0), (0,))
    truth = ((0,), (0, 0))+((0,),)*21
    manifest = {"validation_source": {"role": "artificial"}, "validation_episodes": {"count": 64}}

    def local_artifact(path):
        raw = path.read_bytes()
        return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    for module in (runner, checker):
        monkeypatch.setattr(module, "OUT", tmp_path/"result")
        monkeypatch.setattr(module, "BULK", tmp_path/"bulk")
        monkeypatch.setattr(module, "require_frozen", lambda *_: None)
        monkeypatch.setattr(module, "limit_resources", lambda *_: None)
        monkeypatch.setattr(module, "fixed_inputs", lambda: (manifest, [(source, cipher, truth)]*64))
        monkeypatch.setattr(module, "artifact", local_artifact)
    runner.run("artificial-test-only")
    checker.audit()
    value = json.loads((tmp_path/"result/result.json").read_text())
    assert value["summary"]["complete_cases"] == 64
    assert value["summary"]["determined_used_rows"] == 128
    assert (tmp_path/"result/audit.json").exists()
    assert local_artifact(tmp_path/"bulk/cases.jsonl") == value["records"]
