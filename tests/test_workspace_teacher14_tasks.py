"""Generator invariants required before any TEACH-0014 model training."""

import random

import pytest

from voynich.workspace.teacher14_tasks import (
    ANSWER, BOS, COMPOSE, EDGE, GAP, HOP3, HOP4, SYMBOL_START,
    RenderSpec, make_episode, rows_oracle, sample_episode, strip_pair_rows,
    visible_oracle,
)


def test_visible_grammar_and_oracle_hold_across_hops_aliases_and_marker_dropout():
    rng = random.Random(140014)
    checked = 0
    for hops in (1, 2, 3, 4):
        task = "first_hop" if hops == 1 else "composed"
        for alias in ("none", "terminal", "inner"):
            if alias == "inner" and hops == 1:
                continue
            for dropout in (0.0, .25, .5, 1.0):
                for _ in range(10):
                    episode = sample_episode(
                        rng, signal_hops=hops, task=task, distractors=8,
                        spec=RenderSpec(dropout, 2, ("prefix", "infix", "suffix")),
                        alias_mode=alias,
                    )
                    assert strip_pair_rows(episode.tokens) == episode.serialized_rows
                    assert visible_oracle(episode.tokens) == episode.answer
                    assert rows_oracle(episode.rows, episode.query, episode.hops) == (
                        episode.answer)
                    assert len(episode.rows) == 4 * hops + 16
                    assert len(episode.tokens) <= 192
                    assert len(set(left for left, _ in episode.rows)) == len(episode.rows)
                    assert len(episode.row_positions) == len(episode.rows)
                    for (left_pos, right_pos), (left, right) in zip(
                            episode.row_positions, episode.serialized_rows, strict=True):
                        assert episode.tokens[left_pos] == left
                        assert episode.tokens[right_pos] == right
                    checked += 1
    assert checked == 440


def test_family_assignment_is_graph_based_and_unchanged_by_rerender():
    rng = random.Random(140015)
    base = sample_episode(
        rng, signal_hops=2, task="composed", distractors=4,
        spec=RenderSpec(0.0, 0, ("prefix",)), stage_partitions=("confirm", "train"))
    rerendered = make_episode(
        base.signal_paths, base.distractor_paths, task=base.task, query=base.query,
        rng=random.Random(500), spec=RenderSpec(1.0, 2, ("infix",)),
        alias_mode=base.alias_mode)
    assert base.stage_partitions == rerendered.stage_partitions == ("confirm", "train")
    assert base.graph_partition == rerendered.graph_partition
    assert base.graph_id == rerendered.graph_id
    assert base.logical_id == rerendered.logical_id
    assert base.render_id != rerendered.render_id
    assert base.answer == rerendered.answer
    assert EDGE in base.tokens and EDGE not in rerendered.tokens


def test_public_parser_rejects_malformed_streams_and_oracle_rejects_nonfunctions():
    with pytest.raises(ValueError, match="Odd number"):
        strip_pair_rows((BOS, SYMBOL_START, COMPOSE, SYMBOL_START, ANSWER))
    with pytest.raises(ValueError, match="Unknown body"):
        strip_pair_rows((BOS, 15, SYMBOL_START, COMPOSE, SYMBOL_START, ANSWER))
    with pytest.raises(ValueError, match="Rows are not a function"):
        rows_oracle(((16, 17), (16, 18)), 16, 1)
    with pytest.raises(ValueError, match="incomplete"):
        rows_oracle(((16, 17),), 16, 2)
    assert strip_pair_rows(
        (BOS, EDGE, 16, 17, GAP, 18, EDGE, 19, HOP3, 16, ANSWER)
    ) == ((16, 17), (18, 19))
    with pytest.raises(ValueError, match="incomplete"):
        visible_oracle((BOS, 16, 17, HOP4, 16, ANSWER))


def test_alias_metadata_and_signal_uniqueness_are_enforced():
    signal_paths = ((16, 17, 18), (19, 20, 21), (22, 23, 24), (25, 26, 27))
    distractor_paths = ((28, 29, 18),)
    spec = RenderSpec(1.0, 0, ("prefix",))
    with pytest.raises(ValueError, match="Unregistered signal alias"):
        make_episode(signal_paths, distractor_paths, task="composed", query=16,
                     rng=random.Random(1), spec=spec, alias_mode="none")
    with pytest.raises(ValueError, match="symbol-disjoint"):
        bad_signals = (signal_paths[0], (19, 20, 18), *signal_paths[2:])
        make_episode(bad_signals, (), task="composed", query=16,
                     rng=random.Random(1), spec=spec)
