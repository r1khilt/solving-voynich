"""Bounded alignment of independently supplied anonymous relational annotations.

No vision model or guessed object/plant names are supplied by this module. Predicates
are explicit annotation labels; node names are opaque identities and may be renamed.
Cycles and self-relations are permitted because no universal relation algebra is
assumed. Graph overlap is a descriptive compatibility score, never a likelihood.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import itertools
import math

from .evidence import _freeze_json, _name, _number, _thaw


@dataclass(frozen=True, order=True)
class Relation:
    subject: str
    predicate: str
    object: str

    def __post_init__(self):
        for name in ("subject", "predicate", "object"):
            _name(getattr(self, name), name)

    def to_dict(self):
        return {"subject": self.subject, "predicate": self.predicate, "object": self.object}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or set(value) != {"subject", "predicate", "object"}:
            raise ValueError("relations require exactly subject, predicate, object")
        return cls(**value)


def _relations(values, name):
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of Relation objects") from exc
    if any(not isinstance(value, Relation) for value in values):
        raise ValueError(f"{name} must contain Relation objects")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate edges")
    return tuple(sorted(values))


def _nodes(values, relations, name):
    endpoints = {endpoint for edge in relations for endpoint in (edge.subject, edge.object)}
    if values is None:
        return tuple(sorted(endpoints))
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of node identifiers")
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of node identifiers") from exc
    for node in values:
        _name(node, "node id")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate nodes")
    if not endpoints.issubset(values):
        raise ValueError(f"{name} must include every relation endpoint")
    return tuple(sorted(values))


@dataclass(frozen=True)
class GroundingObservation:
    """Versionable human or external annotations, kept separate from text evidence.

    ``source_group`` identifies the underlying observation (for example one image),
    including all captions/annotations derived from it. A second annotation method
    does not create a new independent source. Confidence is descriptive metadata,
    not a calibrated likelihood or a multiplier on graph overlap.
    """

    source_group: str
    relations: tuple[Relation, ...]
    provenance: object
    annotation_method: str
    confidence: float = 1.0
    nodes: tuple[str, ...] | None = None

    def __post_init__(self):
        _name(self.source_group, "source_group")
        _name(self.annotation_method, "annotation_method")
        if not isinstance(self.provenance, (str, Mapping)) or not self.provenance:
            raise ValueError("provenance must be a nonempty description or JSON object")
        if isinstance(self.provenance, str):
            _name(self.provenance, "provenance")
        confidence = _number(self.confidence, "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must lie in [0, 1]")
        relations = _relations(self.relations, "relations")
        nodes = _nodes(self.nodes, relations, "nodes")
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "provenance", _freeze_json(self.provenance))
        object.__setattr__(self, "confidence", confidence)

    def check_independence(self, *, text_source_group=None, used_source_groups=()):
        return _source_audit(self.source_group, text_source_group, used_source_groups)

    def to_dict(self):
        return {"schema_version": 1, "source_group": self.source_group,
                "relations": [edge.to_dict() for edge in self.relations], "nodes": list(self.nodes),
                "provenance": _thaw(self.provenance), "annotation_method": self.annotation_method,
                "confidence": self.confidence}

    @classmethod
    def from_dict(cls, value):
        fields = {"schema_version", "source_group", "relations", "nodes", "provenance", "annotation_method", "confidence"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("invalid grounding-observation fields")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported grounding-observation schema")
        if not isinstance(value["relations"], list) or not isinstance(value["nodes"], list):
            raise ValueError("serialized relations and nodes must be lists")
        return cls(value["source_group"], tuple(Relation.from_dict(edge) for edge in value["relations"]),
                   value["provenance"], value["annotation_method"], value["confidence"], tuple(value["nodes"]))


def _source_audit(source_group, text_source_group, used_source_groups):
    if source_group is not None:
        _name(source_group, "source_group")
    if text_source_group is not None:
        _name(text_source_group, "text_source_group")
    if isinstance(used_source_groups, (str, bytes)):
        raise ValueError("used_source_groups must be a sequence, not a single string")
    try:
        used = tuple(used_source_groups)
    except TypeError as exc:
        raise ValueError("used_source_groups must be a sequence") from exc
    for group in used:
        _name(group, "used source group")
    if len(used) != len(set(used)):
        raise ValueError("used_source_groups contains duplicate provenance groups")
    if (text_source_group is not None or used) and source_group is None:
        raise ValueError("source_group is required to check evidence independence")
    if source_group is not None and (source_group == text_source_group or source_group in used):
        raise ValueError("dependent provenance: grounding source was already counted as text or other evidence")
    checked = sorted(set(used) | ({text_source_group} if text_source_group is not None else set()))
    return {"source_group": source_group, "checked_against": checked,
            "status": "no_declared_overlap" if checked else "independence_not_checked",
            "limitation": "Distinct identifiers do not by themselves establish statistical independence",
            "likelihood_assigned": False}


def align_relations(
    candidate_relations, observed_relations, *, candidate_nodes=None, observed_nodes=None,
    max_nodes=6, max_alignments=720, require_bijection=False, source_group=None,
    text_source_group=None, used_source_groups=(),
) -> dict:
    """Exhaustively maximize directed, predicate-preserving edge overlap.

    Candidate nodes map injectively into observed nodes; ``require_bijection`` also
    requires equal node counts. All assignments within the declared domain are
    searched, including isolated nodes. All optimal ties are counted, and up to
    ``max_alignments`` are returned in deterministic order with explicit truncation.
    Resource bounds permit at most eight nodes per graph and 40,320 assignments;
    the default is six nodes (at most 720). There is no approximate fallback.

    Node identities may be permuted, predicate labels may not. An unmatched edge
    reports disagreement without asserting a universal logical contradiction.
    Empty-edge graphs carry no relational identification evidence; undefined ratios
    are null, not perfect scores. No result is converted to Bayesian evidence.
    """
    for name, value, upper in (("max_nodes", max_nodes, 8), ("max_alignments", max_alignments, 40_320)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
            raise ValueError(f"{name} must be an integer in [1, {upper}]")
    if not isinstance(require_bijection, bool):
        raise ValueError("require_bijection must be boolean")
    if isinstance(observed_relations, GroundingObservation):
        observation = observed_relations
        if source_group is not None and source_group != observation.source_group:
            raise ValueError("source_group conflicts with observation provenance")
        source_group = observation.source_group
        if observed_nodes is not None and _nodes(observed_nodes, observation.relations, "observed_nodes") != observation.nodes:
            raise ValueError("observed_nodes cannot override the observation's node set")
        observed_nodes, observed_relations = observation.nodes, observation.relations
    audit = _source_audit(source_group, text_source_group, used_source_groups)
    candidate = _relations(candidate_relations, "candidate_relations")
    observed = _relations(observed_relations, "observed_relations")
    candidate_nodes = _nodes(candidate_nodes, candidate, "candidate_nodes")
    observed_nodes = _nodes(observed_nodes, observed, "observed_nodes")
    if max(len(candidate_nodes), len(observed_nodes)) > max_nodes:
        raise ValueError("graph exceeds max_nodes; exact-search resource bound must be respected")
    if max(len(candidate), len(observed)) > 256:
        raise ValueError("graph exceeds the 256-edge exact-search resource bound")
    informative = bool(candidate and observed)
    result = {
        "method": "exact_directed_labeled_edge_alignment",
        "mapping_assumption": "bijection" if require_bijection else "injection",
        "candidate_nodes": list(candidate_nodes), "observed_nodes": list(observed_nodes),
        "candidate_edges": len(candidate), "observed_edges": len(observed),
        "has_relational_information": informative, "source_audit": audit,
        "claim": "Compatibility of supplied relations only; no inferred likelihood, object naming, or image perception",
    }
    if len(candidate_nodes) > len(observed_nodes) or (require_bijection and len(candidate_nodes) != len(observed_nodes)):
        return {**result, "status": "infeasible_node_counts", "evaluated_alignments": 0,
                "best_match_count": None, "best_alignment_count": 0, "returned_alignment_count": 0,
                "alignments_truncated": False, "identifiable_alignment": False, "alignments": [],
                "edge_precision": None, "edge_recall": None, "edge_f1": None}
    observed_set = set(observed)
    best, ties, alignments = -1, 0, []
    for permutation in itertools.permutations(observed_nodes, len(candidate_nodes)):
        mapping = dict(zip(candidate_nodes, permutation))
        mapped = {Relation(mapping[edge.subject], edge.predicate, mapping[edge.object]) for edge in candidate}
        matched = mapped & observed_set
        if len(matched) < best:
            continue
        if len(matched) > best:
            best, ties, alignments = len(matched), 0, []
        ties += 1
        if len(alignments) < max_alignments:
            unsupported, missed = mapped - observed_set, observed_set - mapped
            alignments.append({
                "mapping": mapping, "matched_edges": [edge.to_dict() for edge in sorted(matched)],
                "unsupported_candidate_edges": [edge.to_dict() for edge in sorted(unsupported)],
                "unmatched_observed_edges": [edge.to_dict() for edge in sorted(missed)],
                "violations": {"unsupported_candidate_count": len(unsupported), "unmatched_observed_count": len(missed)},
            })
    precision = best / len(candidate) if candidate else None
    recall = best / len(observed) if observed else None
    f1 = 2 * best / (len(candidate) + len(observed)) if candidate and observed else None
    status = "aligned" if best > 0 else "no_matching_relations"
    return {**result, "status": status if informative else "no_relational_information",
            "evaluated_alignments": math.perm(len(observed_nodes), len(candidate_nodes)),
            "best_match_count": best, "best_alignment_count": ties, "returned_alignment_count": len(alignments),
            "alignments_truncated": ties > len(alignments), "identifiable_alignment": informative and best > 0 and ties == 1,
            "alignments": alignments, "edge_precision": precision, "edge_recall": recall, "edge_f1": f1}
