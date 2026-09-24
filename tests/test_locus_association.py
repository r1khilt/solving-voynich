from collections import Counter
import math

import numpy as np

from voynich.locus_association import (
    Association, edit_distance, fit_readout, losses, match_pairs, margins,
    page_seed, shared_edge, shuffled_p0,
)


def test_edit_and_shared_form_features():
    assert edit_distance("chol", "chor") == 1
    assert edit_distance("abc", "xyz") == 3
    assert shared_edge("chol", "chor") == 0.75


def test_known_topic_association_positive_control():
    loci = [["aa", "bb"] for _ in range(20)] + [["cc", "dd"] for _ in range(20)]
    model = Association(loci)
    assert model.score("aa", ("bb",)) > model.score("cc", ("bb",))
    assert model.score("cc", ("dd",)) > model.score("aa", ("dd",))


def test_page_shuffle_preserves_locus_lengths_and_page_inventory():
    page = {"page_id": "f1r", "loci": [
        {"locus_id": "f1r.1", "locus_type": "P0", "text": "chol chor daiin"},
        {"locus_id": "f1r.2", "locus_type": "L0", "text": "label"},
        {"locus_id": "f1r.3", "locus_type": "P0", "text": "shedy chedy"},
    ]}
    altered = shuffled_p0(page, 290229)
    assert [len(x) for x in altered] == [3, 2]
    assert Counter(w for locus in altered for w in locus) == Counter(
        "chol chor daiin shedy chedy".split())
    assert page_seed(290229, "f1r") == page_seed(290229, "f1r")


def test_matched_pair_and_logistic_direction():
    page = {"page_id": "f1r", "leaf_id": "f1", "loci": [
        {"locus_id": "f1r.1", "locus_type": "P0", "text": "chol chor daiin chedy"},
        {"locus_id": "f1r.2", "locus_type": "P0", "text": "chot shor dain shedy"},
    ]}
    pairs, stats = match_pairs([page], Counter({w: 5 for w in
                                               "chol chor daiin chedy chot shor dain shedy".split()}))
    assert stats["selected"] == len(pairs)
    assert pairs
    for pair in pairs:
        assert len(pair["target"]) == len(pair["negative"])
        assert pair["target"][0] == pair["negative"][0] or pair["target"][-1] == pair["negative"][-1]
        assert min(edit_distance(pair["target"], x) for x in pair["context"]) == min(
            edit_distance(pair["negative"], x) for x in pair["context"])
    x = np.array([[2.0], [1.0], [3.0]])
    model = fit_readout(x)
    margin = margins(x, model)
    assert np.all(margin > 0)
    assert np.mean(losses(margin)) < 1
    assert math.isfinite(model["weights"][0])
