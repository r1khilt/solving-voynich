"""Generator invariants required before any TEACH-0014 model training."""

import random

import pytest

from voynich.workspace.teacher14_tasks import (
    ANSWER, BOS, COMPOSE, EDGE, GAP, HOP3, HOP4, SYMBOL_START,
    RenderSpec, causal_training_batch, evaluation_suite, make_episode, rows_oracle,
    sample_episode, strip_pair_rows, training_batch, visible_oracle,
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


def test_fresh_suite_has_complete_factorial_and_long_hop_panels():
    suite = evaluation_suite(74111, size=2)
    assert len(suite) == 19
    assert sum(len(panel) for panel in suite.values()) == 74
    assert all(visible_oracle(item.tokens) == item.answer
               for panel in suite.values() for item in panel)
    for index in range(0, len(suite["factorial"]), 4):
        group = suite["factorial"][index:index + 4]
        assert len({item.answer for item in group}) == 4
        assert all(item.stage_partitions == ("confirm", "confirm")
                   for item in group)
    for name in ("order_groups", "boundary_groups"):
        for index in range(0, len(suite[name]), 4):
            group = suite[name][index:index + 4]
            assert len({item.graph_id for item in group}) == 1
            assert len({item.answer for item in group}) == 1
    assert all(item.graph_partition == "confirm" and len(item.rows) == 32
               for item in suite["hop_4_long_ood"])


def test_training_stream_is_arm_matched_and_null_changes_only_labels():
    episodes, labels = training_batch(74444, 64, step=0)
    null_episodes, null_labels = training_batch(
        74444, 64, step=0, null_composed=True)
    assert [item.render_id for item in episodes] == [
        item.render_id for item in null_episodes]
    assert all(item.stage_partitions == ("train", "train") for item in episodes)
    assert {item.task for item in episodes} == {
        "first_hop", "direct", "composed", "copy"}
    assert all(label == item.answer for label, item in zip(labels, episodes, strict=True))
    assert all(label == item.answer for label, item in zip(
        null_labels, null_episodes, strict=True) if item.task != "composed")
    assert any(a != b for a, b in zip(labels, null_labels, strict=True))
    assert [item.render_id for item in training_batch(74444, 64, step=0)[0]] == [
        item.render_id for item in episodes]


def test_causal_pair_requires_recipient_specific_second_read():
    pairs = causal_training_batch(74611, 16, step=0)
    assert len({pair.pair_id for pair in pairs}) == 16
    for pair in pairs:
        donor_key = rows_oracle(pair.donor.rows, pair.donor.query, 1)
        assert pair.targets == tuple(rows_oracle(base.rows, donor_key, 1)
                                     for base in pair.bases)
        assert pair.targets[0] == pair.fixed_donor_answer
        assert pair.targets[1] != pair.fixed_donor_answer
        assert pair.bases[0].answer != pair.targets[0]
        assert pair.bases[1].answer != pair.targets[1]
        assert all(item.stage_partitions == ("train", "train")
                   for item in (pair.donor, *pair.bases))
