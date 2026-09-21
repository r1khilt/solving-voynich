"""Auditable finite-hypothesis evidence and active-observation accounting.

All likelihoods and possible query outcomes are supplied by the caller. This module
does not obtain observations, estimate historical likelihoods, or make proposal
scores into evidence. Independence is an explicit source-group assumption: reusing
one group is refused, not silently counted twice. Distinct group names alone do not
establish real independence. Entropies and information gains use natural logarithms.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType


def _name(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _number(value, name, *, negative_infinity=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) and not (negative_infinity and value == -math.inf):
        raise ValueError(f"{name} must be finite or an explicitly permitted negative infinity")
    return value


def _freeze_json(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("payload object keys must be strings")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    raise ValueError("payload must contain only finite JSON values")


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _encode_log(value):
    return None if value == -math.inf else value


def _decode_log(value):
    return -math.inf if value is None else _number(value, "serialized log weight")


@dataclass(frozen=True)
class Hypothesis:
    id: str
    log_prior: float
    payload: object = None

    def __post_init__(self):
        _name(self.id, "hypothesis id")
        object.__setattr__(self, "log_prior", _number(self.log_prior, "log_prior", negative_infinity=True))
        object.__setattr__(self, "payload", _freeze_json(self.payload))


@dataclass(frozen=True)
class Evidence:
    id: str
    source_group: str
    log_likelihoods: Mapping[str, float]
    description: str = ""

    def __post_init__(self):
        _name(self.id, "evidence id")
        _name(self.source_group, "source_group")
        if not isinstance(self.description, str):
            raise ValueError("description must be a string")
        if not isinstance(self.log_likelihoods, Mapping) or not self.log_likelihoods:
            raise ValueError("log_likelihoods must be a nonempty hypothesis mapping")
        values = {
            _name(key, "hypothesis id"): _number(value, "log_likelihood", negative_infinity=True)
            for key, value in self.log_likelihoods.items()
        }
        object.__setattr__(self, "log_likelihoods", MappingProxyType(values))

    def to_dict(self) -> dict:
        return {"id": self.id, "source_group": self.source_group,
                "log_likelihoods": {key: _encode_log(value) for key, value in self.log_likelihoods.items()},
                "description": self.description}

    @classmethod
    def from_dict(cls, value) -> "Evidence":
        if not isinstance(value, Mapping) or set(value) != {"id", "source_group", "log_likelihoods", "description"}:
            raise ValueError("invalid serialized evidence")
        if not isinstance(value["log_likelihoods"], Mapping):
            raise ValueError("serialized log_likelihoods must be a mapping")
        return cls(value["id"], value["source_group"],
                   {key: _decode_log(weight) for key, weight in value["log_likelihoods"].items()},
                   value["description"])


def _normalize(log_values):
    maximum = max(log_values)
    if maximum == -math.inf:
        raise ValueError("impossible evidence: every hypothesis has zero weight")
    shifted = [value - maximum for value in log_values]
    normalizer = math.log(math.fsum(math.exp(value) for value in shifted))
    return tuple(value - normalizer for value in shifted)


@dataclass(frozen=True)
class BeliefState:
    """An immutable ledger, recomputable from prior weights and independent evidence.

    Priors may be unnormalized log weights. ``log_weights`` are normalized posterior
    log probabilities, retaining tails such as -10000 without exponentiating them
    during updates. Zero-prior hypotheses cannot be revived by later evidence.
    """

    hypotheses: tuple[Hypothesis, ...]
    ledger: tuple[Evidence, ...] = ()
    _normalized: tuple[float, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        try:
            hypotheses, ledger = tuple(self.hypotheses), tuple(self.ledger)
        except TypeError as exc:
            raise ValueError("hypotheses and ledger must be sequences") from exc
        if not hypotheses or any(not isinstance(item, Hypothesis) for item in hypotheses):
            raise ValueError("hypotheses must be a nonempty sequence of Hypothesis objects")
        ids = [hypothesis.id for hypothesis in hypotheses]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate hypothesis id")
        weights = _normalize([hypothesis.log_prior for hypothesis in hypotheses])
        evidence_ids, source_groups = set(), set()
        for evidence in ledger:
            if not isinstance(evidence, Evidence):
                raise ValueError("ledger entries must be Evidence objects")
            if evidence.id in evidence_ids:
                raise ValueError(f"duplicate evidence id: {evidence.id}")
            if evidence.source_group in source_groups:
                raise ValueError(f"dependent source_group already used: {evidence.source_group}")
            if set(evidence.log_likelihoods) != set(ids):
                raise ValueError("evidence must cover exactly every hypothesis")
            # Remove common large offsets before combining; otherwise even equal
            # finite likelihoods near -1e308 can overflow after repeated updates.
            likelihoods = [evidence.log_likelihoods[key] for key in ids]
            surviving = [value for value, weight in zip(likelihoods, weights) if weight != -math.inf]
            maximum = max(surviving)
            if maximum == -math.inf:
                raise ValueError("impossible evidence: every surviving hypothesis has zero likelihood")
            weights = _normalize([
                -math.inf if weight == -math.inf else weight + (likelihood - maximum)
                for weight, likelihood in zip(weights, likelihoods)
            ])
            evidence_ids.add(evidence.id)
            source_groups.add(evidence.source_group)
        object.__setattr__(self, "hypotheses", hypotheses)
        object.__setattr__(self, "ledger", ledger)
        object.__setattr__(self, "_normalized", weights)

    @property
    def log_weights(self) -> dict[str, float]:
        return dict(zip((hypothesis.id for hypothesis in self.hypotheses), self._normalized))

    @property
    def probabilities(self) -> dict[str, float]:
        return {hypothesis.id: math.exp(weight) for hypothesis, weight in zip(self.hypotheses, self._normalized)}

    @property
    def entropy(self) -> float:
        return -math.fsum(math.exp(weight) * weight for weight in self._normalized if weight != -math.inf)

    @property
    def effective_sample_size(self) -> float:
        return 1.0 / math.fsum(math.exp(2 * weight) for weight in self._normalized)

    def update(self, evidence: Evidence) -> "BeliefState":
        if not isinstance(evidence, Evidence):
            raise ValueError("update requires Evidence")
        return BeliefState(self.hypotheses, (*self.ledger, evidence))

    def to_dict(self) -> dict:
        """JSON-safe ledger; null encodes log(0), never NaN or Infinity tokens."""
        return {
            "schema_version": 1,
            "hypotheses": [
                {"id": item.id, "log_prior": _encode_log(item.log_prior), "payload": _thaw(item.payload)}
                for item in self.hypotheses
            ],
            "ledger": [item.to_dict() for item in self.ledger],
        }

    @classmethod
    def from_dict(cls, value) -> "BeliefState":
        if not isinstance(value, Mapping) or set(value) != {"schema_version", "hypotheses", "ledger"}:
            raise ValueError("invalid belief-state fields")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported belief-state schema_version")
        hypotheses, ledger = [], []
        if not isinstance(value["hypotheses"], list) or not isinstance(value["ledger"], list):
            raise ValueError("serialized hypotheses and ledger must be lists")
        for item in value["hypotheses"]:
            if not isinstance(item, Mapping) or set(item) != {"id", "log_prior", "payload"}:
                raise ValueError("invalid serialized hypothesis")
            hypotheses.append(Hypothesis(item["id"], _decode_log(item["log_prior"]), item["payload"]))
        for item in value["ledger"]:
            ledger.append(Evidence.from_dict(item))
        return cls(tuple(hypotheses), tuple(ledger))


@dataclass(frozen=True)
class Query:
    id: str
    cost: float
    outcome_probabilities: Mapping[str, Mapping[str, float]]

    def __post_init__(self):
        _name(self.id, "query id")
        cost = _number(self.cost, "cost")
        if cost <= 0:
            raise ValueError("cost must be positive")
        object.__setattr__(self, "cost", cost)
        if not isinstance(self.outcome_probabilities, Mapping) or not self.outcome_probabilities:
            raise ValueError("outcome_probabilities must be a nonempty hypothesis mapping")
        outcomes, values = None, {}
        for hypothesis, row in self.outcome_probabilities.items():
            _name(hypothesis, "hypothesis id")
            if not isinstance(row, Mapping) or not row:
                raise ValueError("each hypothesis needs a nonempty outcome mapping")
            checked = {_name(key, "outcome id"): _number(value, "outcome probability") for key, value in row.items()}
            if any(not 0 <= value <= 1 for value in checked.values()):
                raise ValueError("outcome probabilities must lie in [0, 1]")
            if not math.isclose(math.fsum(checked.values()), 1.0, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("outcome probabilities must be normalized for each hypothesis")
            if outcomes is not None and set(checked) != outcomes:
                raise ValueError("every hypothesis must use the same outcome set, including explicit zeros")
            outcomes = set(checked)
            values[hypothesis] = MappingProxyType(checked)
        object.__setattr__(self, "outcome_probabilities", MappingProxyType(values))


def expected_information_gain(state: BeliefState, query: Query) -> float:
    """Exact finite-outcome expected entropy reduction, conditional on input models."""
    if not isinstance(state, BeliefState) or not isinstance(query, Query):
        raise ValueError("expected_information_gain requires a BeliefState and Query")
    probabilities = state.probabilities
    if set(query.outcome_probabilities) != set(probabilities):
        raise ValueError("query must cover exactly every hypothesis")
    outcomes = next(iter(query.outcome_probabilities.values()))
    expected_entropy = 0.0
    for outcome in outcomes:
        joint = [prior * query.outcome_probabilities[key][outcome] for key, prior in probabilities.items()]
        marginal = math.fsum(joint)
        if marginal == 0:
            continue
        posterior = [weight / marginal for weight in joint if weight > 0]
        entropy = -math.fsum(weight * math.log(weight) for weight in posterior)
        expected_entropy += marginal * entropy
    # Roundoff at deterministic/uninformative endpoints must not create negative IG.
    return max(0.0, min(state.entropy, state.entropy - expected_entropy))


def rank_queries(state: BeliefState, queries) -> list[dict]:
    if not isinstance(state, BeliefState):
        raise ValueError("state must be a BeliefState")
    results, ids = [], set()
    for query in queries:
        if not isinstance(query, Query):
            raise ValueError("queries must contain Query objects")
        if query.id in ids:
            raise ValueError("duplicate query id")
        gain = expected_information_gain(state, query)
        results.append({"query_id": query.id, "expected_information_gain": gain, "cost": query.cost,
                        "information_per_cost": gain / query.cost})
        ids.add(query.id)
    return sorted(results, key=lambda item: (-item["information_per_cost"], -item["expected_information_gain"],
                                            item["query_id"]))
