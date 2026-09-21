import json

import pytest

from voynich.communication.grounding import GroundingObservation, Relation, align_relations


def graph(prefix):
    return (Relation(f"{prefix}0", "part_of", f"{prefix}1"),
            Relation(f"{prefix}1", "touches", f"{prefix}2"))


def observation(relations=None, **changes):
    values = {"source_group": "scan-A", "relations": graph("o") if relations is None else relations,
              "provenance": {"image_sha256": "test-fixture", "annotation_version": 1},
              "annotation_method": "human anonymous relation labeling"}
    values.update(changes)
    return GroundingObservation(**values)


def test_alignment_is_invariant_to_entity_renaming_and_relation_order():
    first = align_relations(graph("c"), graph("o"))
    renamed = (Relation("beta", "part_of", "alpha"), Relation("alpha", "touches", "gamma"))
    second = align_relations(tuple(reversed(renamed)), tuple(reversed(graph("o"))))
    for key in ("best_match_count", "best_alignment_count", "edge_precision", "edge_recall", "edge_f1"):
        assert first[key] == second[key]
    assert first["best_match_count"] == 2 and first["edge_f1"] == 1
    assert first["best_alignment_count"] == 1 and first["identifiable_alignment"]
    assert second["alignments"][0]["mapping"] == {"alpha": "o1", "beta": "o0", "gamma": "o2"}


def test_symmetric_graph_retains_every_best_alignment_and_reports_output_cap():
    candidate = (Relation("a", "touches", "b"), Relation("b", "touches", "a"))
    observed = (Relation("x", "touches", "y"), Relation("y", "touches", "x"))
    all_ties = align_relations(candidate, observed)
    assert all_ties["best_alignment_count"] == 2
    assert all_ties["returned_alignment_count"] == 2
    assert not all_ties["identifiable_alignment"]
    assert not all_ties["alignments_truncated"]
    capped = align_relations(candidate, observed, max_alignments=1)
    assert capped["best_alignment_count"] == 2 and capped["returned_alignment_count"] == 1
    assert capped["alignments_truncated"]
    assert capped["alignments"] == all_ties["alignments"][:1]


def test_labeled_relation_disagreement_reports_unsupported_and_missing_edges():
    candidate = (Relation("a", "contains", "b"), Relation("a", "touches", "b"))
    observed = (Relation("x", "contains", "y"), Relation("x", "not_touches", "y"))
    report = align_relations(candidate, observed)
    assert report["best_match_count"] == 1
    assert report["edge_precision"] == .5 and report["edge_recall"] == .5
    best = report["alignments"][0]
    assert best["violations"] == {"unsupported_candidate_count": 1, "unmatched_observed_count": 1}
    assert best["unsupported_candidate_edges"] == [Relation("x", "touches", "y").to_dict()]
    assert best["unmatched_observed_edges"] == [Relation("x", "not_touches", "y").to_dict()]


def test_injective_matching_keeps_extra_observed_entities_and_bijection_is_explicit():
    candidate = (Relation("a", "p", "b"),)
    observed = (Relation("x", "p", "y"), Relation("y", "q", "z"))
    report = align_relations(candidate, observed)
    assert report["evaluated_alignments"] == 6
    assert report["edge_precision"] == 1 and report["edge_recall"] == .5
    assert report["edge_f1"] == pytest.approx(2 / 3)
    invalid_bijection = align_relations(candidate, observed, require_bijection=True)
    assert invalid_bijection["status"] == "infeasible_node_counts"
    assert invalid_bijection["alignments"] == [] and invalid_bijection["edge_f1"] is None
    too_many_candidate_nodes = align_relations(observed, candidate)
    assert too_many_candidate_nodes["status"] == "infeasible_node_counts"


def test_empty_graphs_and_isolated_nodes_never_create_fake_perfect_evidence():
    empty = align_relations((), ())
    assert empty["status"] == "no_relational_information"
    assert empty["best_match_count"] == 0 and not empty["identifiable_alignment"]
    assert empty["edge_precision"] is None and empty["edge_recall"] is None and empty["edge_f1"] is None
    ambiguous = align_relations((), (), candidate_nodes=["a", "b"], observed_nodes=["x", "y", "z"])
    assert ambiguous["best_alignment_count"] == 6 and len(ambiguous["alignments"]) == 6
    assert ambiguous["has_relational_information"] is False
    assert ambiguous["identifiable_alignment"] is False
    json.dumps(ambiguous, allow_nan=False)
    missing = align_relations((), (Relation("x", "p", "y"),), candidate_nodes=["a", "b"])
    assert missing["edge_precision"] is None and missing["edge_recall"] == 0


def test_self_relations_and_cycles_are_allowed_without_imposed_relation_semantics():
    candidate = (Relation("a", "loops", "a"),)
    observed = (Relation("x", "loops", "x"),)
    report = align_relations(candidate, observed)
    assert report["best_match_count"] == 1
    # Even a domain-forced single mapping is not evidence if predicates disagree.
    absent = align_relations(candidate, (Relation("x", "different", "x"),))
    assert absent["status"] == "no_matching_relations" and not absent["identifiable_alignment"]


def test_source_group_must_be_independent_of_declared_text_and_previous_evidence():
    observed = observation()
    with pytest.raises(ValueError, match="dependent provenance"):
        align_relations(graph("c"), observed, text_source_group="scan-A")
    with pytest.raises(ValueError, match="dependent provenance"):
        observed.check_independence(used_source_groups=["caption-on-unrelated-image", "scan-A"])
    with pytest.raises(ValueError, match="duplicate provenance"):
        observed.check_independence(used_source_groups=["already-counted", "already-counted"])
    report = align_relations(graph("c"), observed, text_source_group="text-transcription", used_source_groups=["scan-B"])
    assert report["source_audit"]["status"] == "no_declared_overlap"
    assert report["source_audit"]["likelihood_assigned"] is False
    with pytest.raises(ValueError, match="source_group is required"):
        align_relations(graph("c"), graph("o"), text_source_group="text")
    with pytest.raises(ValueError, match="conflicts with observation provenance"):
        align_relations(graph("c"), observed, source_group="invented-independent-group")


def test_annotation_round_trip_is_immutable_and_confidence_is_not_a_score_multiplier():
    metadata = {"source": "image", "version": [1]}
    observed = observation(provenance=metadata, confidence=0.2)
    metadata["version"][0] = 99
    assert observed.provenance["version"] == (1,)
    restored = GroundingObservation.from_dict(json.loads(json.dumps(observed.to_dict(), allow_nan=False)))
    assert restored == observed
    assert align_relations(graph("c"), restored)["edge_f1"] == 1
    with pytest.raises(TypeError):
        restored.provenance["source"] = "changed"


def test_duplicate_edges_and_nodes_and_missing_endpoints_are_rejected():
    edge = Relation("a", "p", "b")
    for action in (
        lambda: observation((edge, edge)),
        lambda: align_relations((edge, edge), (edge,)),
        lambda: align_relations((edge,), (edge, edge)),
        lambda: align_relations((edge,), (edge,), candidate_nodes=["a", "a", "b"]),
        lambda: align_relations((edge,), (edge,), observed_nodes=["a"]),
    ):
        with pytest.raises(ValueError):
            action()


@pytest.mark.parametrize("kwargs", [{"max_nodes": 0}, {"max_nodes": True}, {"max_nodes": 9},
                                    {"max_nodes": 1}, {"max_alignments": 0}, {"max_alignments": 40_321},
                                    {"require_bijection": "yes"}])
def test_resource_bounds_and_parameter_types_are_enforced(kwargs):
    with pytest.raises(ValueError):
        align_relations(graph("c"), graph("o"), **kwargs)


def test_default_bound_rejects_seven_nodes_before_permutation_search():
    with pytest.raises(ValueError, match="resource bound"):
        align_relations((), (), candidate_nodes=[str(i) for i in range(7)], observed_nodes=[str(i) for i in range(7)])


@pytest.mark.parametrize("changes", [{"source_group": ""}, {"annotation_method": ""}, {"provenance": ""},
                                    {"confidence": float("nan")}, {"confidence": 1.1}, {"confidence": True}])
def test_invalid_annotation_metadata_is_rejected(changes):
    with pytest.raises(ValueError):
        observation(**changes)


def test_serialization_does_not_accept_extra_fields_or_invalid_predicates():
    value = observation().to_dict()
    value["independent_by_magic"] = True
    with pytest.raises(ValueError):
        GroundingObservation.from_dict(value)
    with pytest.raises(ValueError):
        Relation("a", " ", "b")
    with pytest.raises(ValueError):
        Relation.from_dict({"subject": "a", "predicate": "p", "object": "b", "guessed_name": "plant"})
