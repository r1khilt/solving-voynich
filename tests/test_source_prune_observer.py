"""Gold observers preserve every ordinary output and distinguish path aliases."""
import itertools
import math

import numpy as np
import pytest

from tests.test_source_state_lattice import source
from voynich.native_source_state import NativeStateLattice, build_source_state_native
from voynich.source_prune_observer import (
    BASE_SHA, NativePruneObserver, build_prune_observer, golden_path, instrument,
)


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    return (build_source_state_native(tmp_path_factory.mktemp("prune-original")),
            build_prune_observer(tmp_path_factory.mktemp("prune-observer")))


def literal(texts, key, glyphs=2):
    pool = [(g,) for g in range(glyphs)]+list(itertools.product(range(glyphs), repeat=2))
    return tuple(tuple(g for row in text for g in pool[key[row]]) for text in texts)


@pytest.mark.parametrize("schedule", ["balanced", "sequential"])
@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize("guidance", ["none", "iid"])
def test_all36_observed_literal_pairs_match_untouched_search(builds, schedule, merge, guidance):
    original, instrumented = builds
    baseline, observer = NativeStateLattice(*source(), original), NativePruneObserver(*source(), instrumented)
    observations = [(g,) for g in range(2)]+list(itertools.product(range(2), repeat=2))
    keys = list(itertools.product(range(6), repeat=2))
    texts = [(0,), (1,)]+list(itertools.product(range(2), repeat=2))
    for cipher in itertools.product(observations, repeat=2):
        answer = next((pair, key) for key in keys for pair in itertools.product(texts, repeat=2) if literal(pair, key) == cipher)
        for width in (2, 100_000):
            cfg = {"glyphs": 2, "rho": .25, "schedule": schedule, "merge": merge,
                "guidance": guidance, "width": width, "max_seconds": 30.}
            expected = baseline.search(cipher, **cfg)
            actual, trace = observer.observe(cipher, *answer, **cfg)
            assert actual == expected
            processed = [r for r in trace if r["processed"]]
            assert processed and processed[0]["alive_before"]
            dead = False
            for r in processed:
                if dead:
                    assert not r["alive_before"] and not r["alive_after"]
                if r["alive_before"]:
                    assert r["path_present"] and r["state_log_mass"] >= r["prefix_log_mass"]-1e-10
                    if r["target_glyph_layer"] < sum(map(len, cipher)):
                        assert r["pre_rank"] >= 1
                if not r["alive_after"]:
                    dead = True
            if width == 100_000:
                assert trace[-1]["alive_after"] and trace[-1]["terminal_group_rank"] >= 1
                assert trace[-1]["terminal_used_mapping_returned"]
            # Thread-local observer pointers have been cleared; no dangling gold.
            assert observer.search(cipher, **cfg) == expected


def test_gold_geometry_literal_factors_and_source_contexts():
    p, goto = source()
    texts, key = ((0, 1, 0), (1, 0)), (0, 2)
    cipher = literal(texts, key)
    path, masses = golden_path(cipher, texts, key, p, goto, glyphs=2, rho=.25)
    valid = np.flatnonzero(path[:, 29])
    assert len(valid) == 6 and valid[0] == 0 and valid[-1] == sum(map(len, cipher))
    assert set(np.diff(valid)) <= {1, 2}
    assert list(path[-1, 2:4]) == [0, 0] and list(path[-1, 4:6]) == list(key)
    expected = -2*math.log(6)+2*math.log(.25)+5*math.log(.75)
    for text in texts:
        context = 0
        for row in text:
            expected += math.log(p[context, row])
            context = int(goto[context, row])
    assert masses[-1] == pytest.approx(expected, abs=1e-13)
    with pytest.raises(ValueError):
        golden_path(((1,),), ((0,),), (0, 2), p, goto, glyphs=2)
    with pytest.raises(ValueError):
        golden_path(((0, 0),), ((1,),), (0, 2), *source(zero=True), glyphs=2)
    with pytest.raises(ValueError):
        instrument("changed source")
    assert len(BASE_SHA) == 64


def test_terminal_truncation_and_dead_path_alias_are_different(builds):
    p = np.array([[.8, .2]], dtype=np.float64)
    goto = np.zeros((1, 2), dtype=np.uint32)
    observer = NativePruneObserver(p, goto, builds[1])
    # Exhaustive search retains the known rare path; its group can be cut
    # only at output. This must not be reported as an early search loss.
    texts, key = ((1,),), (0, 0)
    actual, trace = observer.observe(literal(texts, key), texts, key,
        glyphs=2, rho=.25, width=100_000, max_terminals=1)
    assert actual["terminal_output_truncated"] and trace[-1]["alive_after"]
    assert trace[-1]["terminal_group_rank"] > 1 and not trace[-1]["terminal_group_returned"]
    # Losing a literal early path does not erase an equivalent abstract
    # terminal that another history reaches later.
    found_alias = False
    for text in itertools.product(range(2), repeat=4):
        cipher = literal((text,), (0, 0))
        _, traced = observer.observe(cipher, (text,), (0, 0), glyphs=2, rho=.25, width=2)
        found_alias |= any(r["processed"] and not r["alive_before"] and r["aliases"] for r in traced)
    assert found_alias


def test_preliminary_guide_prune_and_cap_are_not_confused(builds):
    observer = NativePruneObserver(*source(), builds[1])
    losses = []
    for text in itertools.product(range(2), repeat=3):
        cipher = literal((text,), (0, 1))
        _, trace = observer.observe(cipher, (text,), (0, 1), glyphs=2, rho=.25,
            width=2, guide_prewidth=1, guidance="iid")
        losses += [r for r in trace if r["alive_before"] and not r["alive_after"]]
    assert any(not r["pre_kept"] for r in losses)
    actual, trace = observer.observe(((0,),), ((0,),), (0, 1), glyphs=2, rho=.25, max_expanded=1)
    assert actual["stop_reason"] in ("frontier_exhausted", "expansion_cap")
    assert sum(r["path_expanded"] for r in trace) <= 1
