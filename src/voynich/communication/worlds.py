"""Small, executable document-producing worlds with an explicit observation channel.

These are deliberately restrictive synthetic worlds, not models of medieval botany.
The gold record is an audit artifact.  Only ``observation_dict`` is an unanchored
inference input; the seed, language, key, states and alignments are supervision.

Symbols 0/1 are reserved for padding/masking.  A key entry at index i maps
observed i+2 to a canonical symbol, or 0 for a globally licensed null.  Nulls
are never arbitrary per-position deletions.  Boundaries mark word STARTS, with
False at null positions.  Homophones are total variants per canonical symbol.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Mapping, Sequence


FAMILIES = ("procedure", "taxonomy", "copy")
GRAMMARS = ("SOV", "SVO", "VSO", "VOS", "OSV", "OVS")
MORPHOLOGIES = ("suffix", "prefix", "none")
PROCEDURE_OPERATORS = ("transfer", "heat", "cool")
TAXONOMY_OPERATORS = ("parent", "ancestor", "sibling")
ROLES = ("subject", "object", "predicate", "subject_marker", "object_marker", "null")


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _flags(values: Sequence[bool], name: str, length: int) -> tuple[bool, ...]:
    result = tuple(values)
    if len(result) != length or any(type(x) is not bool for x in result):
        raise ValueError(f"{name} must contain exactly {length} booleans")
    return result


def _rng(seed: int, domain: str) -> random.Random:
    # Separate semantic, key, and channel draws; no process-dependent hash().
    digest = hashlib.sha256(f"worlds-v1:{seed}:{domain}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


@dataclass(frozen=True)
class WorldConfig:
    seed: int
    family: str = "procedure"
    alphabet_size: int = 24
    entities: int = 4
    events: int = 8
    null_rate: float = 0.15
    homophones: int = 1
    grammar: str = "SOV"
    morphology: str = "suffix"
    stateful: bool = False

    def __post_init__(self) -> None:
        _integer(self.seed, "seed")
        _integer(self.alphabet_size, "alphabet_size", 3)
        _integer(self.entities, "entities", 3 if self.family == "procedure" else 2)
        _integer(self.events, "events", 1)
        _integer(self.homophones, "homophones", 1)
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        if self.grammar not in GRAMMARS or self.morphology not in MORPHOLOGIES:
            raise ValueError("unsupported grammar or morphology")
        if type(self.stateful) is not bool or self.stateful:
            raise ValueError("stateful channels are not implemented; use stateful=False")
        _validate_rate(self.null_rate)
        required = self.canonical_count * self.homophones + int(self.null_rate > 0)
        if required > self.alphabet_size:
            raise ValueError(f"channel requires {required} symbols, alphabet has {self.alphabet_size}")

    @property
    def canonical_count(self) -> int:
        return self.entities + 3 + (0 if self.morphology == "none" else 2)


def _validate_rate(value: float) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("null_rate must be finite and in [0, 1)")


@dataclass(frozen=True)
class Entity:
    id: int
    kind: str

    def __post_init__(self) -> None:
        _integer(self.id, "entity id")
        if self.kind not in ("agent", "vessel", "taxon"):
            raise ValueError("unknown entity type")


@dataclass(frozen=True)
class EntityState:
    entity: int
    full: bool
    hot: bool

    def __post_init__(self) -> None:
        _integer(self.entity, "state entity")
        if type(self.full) is not bool or type(self.hot) is not bool or (self.hot and not self.full):
            raise ValueError("invalid material-state properties")


@dataclass(frozen=True)
class Action:
    kind: str
    subject: int
    object: int

    def __post_init__(self) -> None:
        _integer(self.subject, "action subject")
        _integer(self.object, "action object")
        if self.kind not in PROCEDURE_OPERATORS:
            raise ValueError("unknown procedure operator")


@dataclass(frozen=True)
class ActionStep:
    action: Action
    before: tuple[EntityState, ...]
    after: tuple[EntityState, ...]


@dataclass(frozen=True)
class Relation:
    kind: str
    subject: int
    object: int

    def __post_init__(self) -> None:
        _integer(self.subject, "relation subject")
        _integer(self.object, "relation object")
        if self.kind not in TAXONOMY_OPERATORS:
            raise ValueError("unknown taxonomy operator")


@dataclass(frozen=True)
class SurfaceClause:
    """A realization instruction; in the copy family this has NO referent meaning."""

    operator: int
    subject: int
    object: int

    def __post_init__(self) -> None:
        _integer(self.operator, "clause operator")
        _integer(self.subject, "clause subject")
        _integer(self.object, "clause object")
        if self.operator >= 3:
            raise ValueError("clause operator must be in [0, 3)")


@dataclass(frozen=True)
class DecodedSequence:
    plaintext: tuple[int, ...]
    keep: tuple[bool, ...]
    alignment: tuple[int, ...]
    word_boundary: tuple[bool, ...]


@dataclass(frozen=True)
class EncodedSequence:
    observation: tuple[int, ...]
    keep: tuple[bool, ...]
    alignment: tuple[int, ...]
    word_boundary: tuple[bool, ...]
    roles: tuple[str, ...]


def _validate_key(inverse_key: Sequence[int]) -> tuple[int, ...]:
    key = tuple(inverse_key)
    if not key:
        raise ValueError("inverse_key cannot be empty")
    if any(type(x) is not int or x == 1 or x < 0 or x > len(key) + 1 for x in key):
        raise ValueError("key values must be 0 or canonical symbols in [2, alphabet_size+1]")
    return key


def decode(
    observation: Sequence[int],
    inverse_key: Sequence[int],
    keep: Sequence[bool] | None = None,
    word_boundary: Sequence[bool] | None = None,
) -> DecodedSequence:
    """Execute a global candidate key, independently of any generator/gold record.

    Optional keep labels are assertions, not permission to discard inconsistent
    symbols.  With no boundary hypothesis the returned boundaries are unknown
    (all False); this function does not infer linguistic segmentation.
    """
    key = _validate_key(inverse_key)
    obs = tuple(observation)
    if any(type(x) is not int or not 2 <= x <= len(key) + 1 for x in obs):
        raise ValueError("observation contains out-of-alphabet or reserved symbols")
    expected_keep = tuple(key[x - 2] != 0 for x in obs)
    if keep is not None and _flags(keep, "keep", len(obs)) != expected_keep:
        raise ValueError("keep contradicts the global key; unlicensed deletion")
    boundaries = ((False,) * len(obs) if word_boundary is None else
                  _flags(word_boundary, "word_boundary", len(obs)))
    if any(boundary and not retained for boundary, retained in zip(boundaries, expected_keep)):
        raise ValueError("null positions cannot start words")
    if word_boundary is not None and any(expected_keep):
        if not boundaries[expected_keep.index(True)]:
            raise ValueError("the first retained symbol must start a word")
    plain: list[int] = []
    alignment: list[int] = []
    for symbol, retained in zip(obs, expected_keep):
        alignment.append(len(plain) if retained else -1)
        if retained:
            plain.append(key[symbol - 2])
    return DecodedSequence(tuple(plain), expected_keep, tuple(alignment), boundaries)


def encode(
    plaintext: Sequence[int],
    inverse_key: Sequence[int],
    *,
    seed: int = 0,
    null_rate: float = 0.0,
    word_boundary: Sequence[bool] | None = None,
    roles: Sequence[str] | None = None,
) -> EncodedSequence:
    """Render a canonical stream through a constrained stochastic channel.

    Exactly round(N*r/(1-r)) nulls are inserted, so even r close to 1 has a
    declared finite count rather than a stochastic unbounded insertion loop.
    A hard safety cap rejects accidental requests above one million positions.
    The homophone and insertion choices are represented by the full alignment.
    """
    key = _validate_key(inverse_key)
    _integer(seed, "seed")
    _validate_rate(null_rate)
    plain = tuple(plaintext)
    variants: dict[int, list[int]] = {}
    for index, value in enumerate(key):
        variants.setdefault(value, []).append(index + 2)
    if any(type(x) is not int or x < 2 or x not in variants for x in plain):
        raise ValueError("plaintext contains symbols unsupported by the key")
    boundaries = ((True,) * len(plain) if word_boundary is None else
                  _flags(word_boundary, "word_boundary", len(plain)))
    if plain and not boundaries[0]:
        raise ValueError("the first plaintext symbol must start a word")
    source_roles = tuple(roles) if roles is not None else ("predicate",) * len(plain)
    if len(source_roles) != len(plain) or any(role not in ROLES[:-1] for role in source_roles):
        raise ValueError("invalid canonical role labels")
    count = round(len(plain) * null_rate / (1 - null_rate))
    if len(plain) + count > 1_000_000:
        raise ValueError("channel output exceeds the one-million-position safety cap")
    if null_rate > 0 and 0 not in variants:
        raise ValueError("null insertion requires at least one globally null symbol")
    randomizer = _rng(seed, "encoding")
    gaps = [0] * (len(plain) + 1)
    for _ in range(count):
        gaps[randomizer.randrange(len(gaps))] += 1
    observation: list[int] = []
    alignment: list[int] = []
    aligned_boundary: list[bool] = []
    aligned_roles: list[str] = []
    for index, gap in enumerate(gaps):
        for _ in range(gap):
            observation.append(randomizer.choice(variants[0]))
            alignment.append(-1)
            aligned_boundary.append(False)
            aligned_roles.append("null")
        if index < len(plain):
            observation.append(randomizer.choice(variants[plain[index]]))
            alignment.append(index)
            aligned_boundary.append(boundaries[index])
            aligned_roles.append(source_roles[index])
    return EncodedSequence(tuple(observation), tuple(i >= 0 for i in alignment),
                           tuple(alignment), tuple(aligned_boundary), tuple(aligned_roles))


def reencode(
    plaintext: Sequence[int],
    inverse_key: Sequence[int],
    alignment: Sequence[int],
    *,
    symbols: Sequence[int],
    word_boundary: Sequence[bool] | None = None,
) -> tuple[int, ...]:
    """Verify/replay observed homophone choices and accountable null positions.

    ``symbols`` supplies channel choices, not an exception table: every retained
    position must map to the next plaintext symbol and every -1 must be globally
    null.  No reorder, duplication, omission, or unexplained glyph is accepted.
    """
    result = decode(symbols, inverse_key, word_boundary=word_boundary)
    if any(type(x) is not int or x < 2 for x in plaintext):
        raise ValueError("plaintext must contain canonical integer symbols")
    supplied = tuple(alignment)
    if any(type(x) is not int for x in supplied) or supplied != result.alignment:
        raise ValueError("alignment does not explain every observed position in order")
    if tuple(plaintext) != result.plaintext:
        raise ValueError("channel choices do not re-encode the supplied plaintext")
    return tuple(symbols)


def _validate_entities(entities: Sequence[Entity]) -> dict[int, Entity]:
    table = {entity.id: entity for entity in entities}
    if len(table) != len(entities) or set(table) != set(range(len(entities))):
        raise ValueError("entities must have unique contiguous integer ids")
    for entity in entities:
        _integer(entity.id, "entity id")
        if entity.kind not in ("agent", "vessel", "taxon"):
            raise ValueError("unknown entity type")
    return table


def _validate_state(entities: Sequence[Entity], state: Sequence[EntityState]) -> None:
    table = _validate_entities(entities)
    if tuple(row.entity for row in state) != tuple(range(len(table))):
        raise ValueError("state must contain each entity once, in id order")
    for row in state:
        _integer(row.entity, "state entity")
        if type(row.full) is not bool or type(row.hot) is not bool:
            raise ValueError("state properties must be booleans")
        if row.hot and not row.full:
            raise ValueError("empty entities cannot contain hot material")
        if table[row.entity].kind != "vessel" and (row.full or row.hot):
            raise ValueError("only vessels can hold material")


def execute_action(
    entities: Sequence[Entity], state: Sequence[EntityState], action: Action,
) -> tuple[EntityState, ...]:
    """Independent typed transition evaluator; raises on failed preconditions."""
    _validate_state(entities, state)
    _integer(action.subject, "action subject")
    _integer(action.object, "action object")
    table = {entity.id: entity for entity in entities}
    if action.subject not in table or action.object not in table:
        raise ValueError("action refers to an absent entity")
    if action.subject == action.object:
        raise ValueError("action arguments must be distinct")
    source, target = state[action.subject], state[action.object]
    result = list(state)
    if action.kind == "transfer":
        if table[action.subject].kind != "vessel" or table[action.object].kind != "vessel":
            raise ValueError("transfer requires two vessels")
        if not source.full or target.full:
            raise ValueError("transfer needs a full source and empty target")
        result[action.subject] = EntityState(action.subject, False, False)
        result[action.object] = EntityState(action.object, True, source.hot)
    elif action.kind in ("heat", "cool"):
        if table[action.subject].kind != "agent" or table[action.object].kind != "vessel":
            raise ValueError("heat/cool requires an agent and a vessel")
        if not target.full or target.hot == (action.kind == "heat"):
            raise ValueError("heat/cool requires material in the opposite thermal state")
        result[action.object] = EntityState(action.object, True, action.kind == "heat")
    else:
        raise ValueError("unknown procedure operator")
    return tuple(result)


def state_projection(state: Sequence[EntityState]) -> tuple[int, ...]:
    """Finite observable state code per entity: empty=0, cold=1, hot=2."""
    if any(type(row.full) is not bool or type(row.hot) is not bool or
           (row.hot and not row.full) for row in state):
        raise ValueError("invalid material state")
    return tuple(int(row.full) + int(row.hot) for row in state)


def _realize(
    config: WorldConfig, clauses: Sequence[SurfaceClause],
) -> tuple[tuple[int, ...], tuple[bool, ...], tuple[str, ...]]:
    plain: list[int] = []
    boundaries: list[bool] = []
    roles: list[str] = []
    for clause in clauses:
        for value in (clause.subject, clause.object):
            if type(value) is not int or not 0 <= value < config.entities:
                raise ValueError("surface clause entity is out of range")
        if type(clause.operator) is not int or not 0 <= clause.operator < 3:
            raise ValueError("surface clause operator is out of range")
        stems = {"S": clause.subject + 2, "O": clause.object + 2,
                 "V": config.entities + 2 + clause.operator}
        role_names = {"S": "subject", "O": "object", "V": "predicate"}
        for slot in config.grammar:
            word = [(stems[slot], role_names[slot])]
            if slot != "V" and config.morphology != "none":
                marker = (config.entities + 5 + int(slot == "O"), role_names[slot] + "_marker")
                word = [marker, *word] if config.morphology == "prefix" else [*word, marker]
            for index, (symbol, role) in enumerate(word):
                plain.append(symbol)
                boundaries.append(index == 0)
                roles.append(role)
    return tuple(plain), tuple(boundaries), tuple(roles)


def parse_document(
    plaintext: Sequence[int],
    word_boundary: Sequence[bool],
    *,
    entities: int,
    grammar: str,
    morphology: str,
    family: str,
) -> tuple[SurfaceClause, ...]:
    """Parse candidate symbols/boundaries using only the declared grammar.

    Boundaries here are CANONICAL-length (filter observed boundaries by keep).
    This checks word morphology, lexical classes and complete three-word clauses;
    it does not consult a seed, key, gold example, or reference action trace.
    Semantic satisfiability is a separate execution check.
    """
    _integer(entities, "entities", 2)
    if grammar not in GRAMMARS or morphology not in MORPHOLOGIES or family not in FAMILIES:
        raise ValueError("unsupported document family/grammar/morphology")
    plain = tuple(plaintext)
    boundaries = _flags(word_boundary, "word_boundary", len(plain))
    if not plain or not boundaries[0] or any(type(x) is not int or x < 2 for x in plain):
        raise ValueError("document must contain a valid first word")
    starts = [i for i, boundary in enumerate(boundaries) if boundary] + [len(plain)]
    words = [plain[left:right] for left, right in zip(starts, starts[1:])]
    if len(words) % 3:
        raise ValueError("document has an incomplete three-word clause")
    clauses: list[SurfaceClause] = []
    for offset in range(0, len(words), 3):
        slots: dict[str, int] = {}
        for slot, word in zip(grammar, words[offset:offset + 3]):
            if slot == "V":
                if len(word) != 1 or not entities + 2 <= word[0] < entities + 5:
                    raise ValueError("predicate is not a declared operator word")
                slots[slot] = word[0] - entities - 2
            else:
                if morphology == "none":
                    stem = word[0]
                    valid = len(word) == 1
                else:
                    marker = entities + 5 + int(slot == "O")
                    stem = word[-1] if morphology == "prefix" else word[0]
                    valid = len(word) == 2 and (word[0] if morphology == "prefix" else word[-1]) == marker
                if not valid or not 2 <= stem < entities + 2:
                    raise ValueError("nominal word violates stem/case morphology")
                slots[slot] = stem - 2
        clauses.append(SurfaceClause(slots["V"], slots["S"], slots["O"]))
    return tuple(clauses)


@dataclass(frozen=True)
class ProcedureInterpretation:
    entities: tuple[Entity, ...]
    initial_state: tuple[EntityState, ...]
    action_trace: tuple[ActionStep, ...]


def infer_procedure_trace(
    clauses: Sequence[SurfaceClause], entities: int, *, initial_hot: bool | None = False,
) -> ProcedureInterpretation:
    """Find a typed execution without gold state, under the one-unit world prior.

    Search is finite: entities possible agent identities times entities-1 initial
    holders (optionally two thermal states). The first witness is returned, not a
    claim of unique interpretation. Default cold initial material matches the
    generator; ``None`` deliberately broadens the candidate prior.
    """
    _integer(entities, "entities", 3)
    if initial_hot is not None and type(initial_hot) is not bool:
        raise ValueError("initial_hot must be a bool or None")
    if not clauses:
        raise ValueError("cannot infer an empty procedure")
    actions = []
    for clause in clauses:
        if not isinstance(clause, SurfaceClause) or max(clause.subject, clause.object) >= entities:
            raise ValueError("invalid candidate clause")
        actions.append(Action(PROCEDURE_OPERATORS[clause.operator], clause.subject, clause.object))
    for agent in range(entities):
        table = tuple(Entity(i, "agent" if i == agent else "vessel") for i in range(entities))
        for holder in range(entities):
            if holder == agent:
                continue
            for hot in (False, True) if initial_hot is None else (initial_hot,):
                initial = tuple(EntityState(i, i == holder, hot and i == holder) for i in range(entities))
                state = initial
                trace: list[ActionStep] = []
                try:
                    for action in actions:
                        after = execute_action(table, state, action)
                        trace.append(ActionStep(action, state, after))
                        state = after
                except ValueError:
                    continue
                return ProcedureInterpretation(table, initial, tuple(trace))
    raise ValueError("no typed initial state satisfies the complete procedure")


@dataclass(frozen=True)
class WorldExample:
    config: WorldConfig
    observation: tuple[int, ...]
    plaintext: tuple[int, ...]
    inverse_key: tuple[int, ...]
    keep: tuple[bool, ...]
    word_boundary: tuple[bool, ...]
    roles: tuple[str, ...]
    alignment: tuple[int, ...]
    entities: tuple[Entity, ...]
    initial_state: tuple[EntityState, ...]
    action_trace: tuple[ActionStep, ...]
    relations: tuple[Relation, ...]
    clauses: tuple[SurfaceClause, ...]
    lexical_inventory: tuple[str, ...]

    def observation_dict(self) -> dict[str, Any]:
        """The entire permitted input for unanchored inference. No gold metadata."""
        return {"observation": list(self.observation), "alphabet_size": self.config.alphabet_size}

    def anonymous_trace(self) -> tuple[tuple[int, int, int], ...]:
        """Anonymous *gold* semantic labels; never included in observation export."""
        if self.config.family == "copy":
            return ()
        return tuple((row.operator, row.subject, row.object) for row in self.clauses)

    def to_dict(self) -> dict[str, Any]:
        """Full gold audit record (NOT a solver input)."""
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WorldExample:
        _exact_fields(cls, data)
        config_data = data["config"]
        _exact_fields(WorldConfig, config_data)
        config = WorldConfig(**config_data)
        steps = []
        for row in data["action_trace"]:
            _exact_fields(ActionStep, row)
            steps.append(ActionStep(_record(Action, row["action"]),
                                    tuple(_record(EntityState, x) for x in row["before"]),
                                    tuple(_record(EntityState, x) for x in row["after"])))
        result = cls(config=config,
                     **{name: tuple(data[name]) for name in (
                         "observation", "plaintext", "inverse_key", "keep", "word_boundary",
                         "roles", "alignment", "lexical_inventory")},
                     entities=tuple(_record(Entity, x) for x in data["entities"]),
                     initial_state=tuple(_record(EntityState, x) for x in data["initial_state"]),
                     action_trace=tuple(steps),
                     relations=tuple(_record(Relation, x) for x in data["relations"]),
                     clauses=tuple(_record(SurfaceClause, x) for x in data["clauses"]))
        result.validate()
        return result

    def validate(self) -> None:
        """Check the channel, realization, graph, types, and complete state trace."""
        cfg = self.config
        if len(self.inverse_key) != cfg.alphabet_size:
            raise ValueError("key length differs from alphabet size")
        canonical = set(range(2, cfg.canonical_count + 2))
        key = _validate_key(self.inverse_key)
        if set(key) - {0} != canonical or any(key.count(x) != cfg.homophones for x in canonical):
            raise ValueError("key violates the declared inventory/homophone budget")
        if len(self.lexical_inventory) != cfg.canonical_count or any(
            type(x) is not str for x in self.lexical_inventory
        ):
            raise ValueError("invalid lexical inventory")
        if len(self.clauses) != cfg.events:
            raise ValueError("wrong number of clauses")
        plain, boundaries, roles = _realize(cfg, self.clauses)
        if self.plaintext != plain or any(type(x) is not int for x in self.plaintext):
            raise ValueError("plaintext does not realize the declared clauses")
        decoded = decode(self.observation, key, keep=self.keep, word_boundary=self.word_boundary)
        if decoded.plaintext != plain or decoded.alignment != self.alignment:
            raise ValueError("channel decode or alignment disagrees with realization")
        reencode(plain, key, self.alignment, symbols=self.observation)
        if self.word_boundary != tuple(False if i < 0 else boundaries[i] for i in self.alignment):
            raise ValueError("boundaries disagree with the linguistic realization")
        if self.roles != tuple("null" if i < 0 else roles[i] for i in self.alignment):
            raise ValueError("roles disagree with the linguistic realization")
        if len(self.observation) - len(plain) != round(len(plain) * cfg.null_rate / (1 - cfg.null_rate)):
            raise ValueError("null count does not match the declared channel")
        if cfg.family == "copy":
            if self.entities or self.initial_state or self.action_trace or self.relations:
                raise ValueError("copy control must not claim referent semantics")
            return
        table = _validate_entities(self.entities)
        if len(table) != cfg.entities:
            raise ValueError("wrong number of persistent entities")
        if cfg.family == "procedure":
            if self.relations or len(self.action_trace) != cfg.events:
                raise ValueError("procedure requires a complete action trace")
            if sorted(x.kind for x in self.entities) != ["agent"] + ["vessel"] * (cfg.entities - 1):
                raise ValueError("procedure requires exactly one agent and remaining vessels")
            _validate_state(self.entities, self.initial_state)
            if sum(x.full for x in self.initial_state) != 1:
                raise ValueError("procedure conserves exactly one material unit")
            state = self.initial_state
            for step, clause in zip(self.action_trace, self.clauses):
                if step.before != state:
                    raise ValueError("action trace state is discontinuous")
                state = execute_action(self.entities, state, step.action)
                if step.after != state or clause != SurfaceClause(
                    PROCEDURE_OPERATORS.index(step.action.kind), step.action.subject, step.action.object
                ):
                    raise ValueError("action effects/realization disagree")
        else:
            if self.initial_state or self.action_trace or any(x.kind != "taxon" for x in self.entities):
                raise ValueError("taxonomy requires static taxa, not a procedure trace")
            truth = _taxonomy_truth(cfg.entities, self.relations)
            for clause in self.clauses:
                if Relation(TAXONOMY_OPERATORS[clause.operator], clause.subject, clause.object) not in truth:
                    raise ValueError("taxonomy statement is false in the relation graph")


def _exact_fields(record_type: type, data: Mapping[str, Any]) -> None:
    if not isinstance(data, Mapping) or set(data) != {field.name for field in fields(record_type)}:
        raise ValueError(f"invalid fields for {record_type.__name__}")


def _record(record_type: type, data: Mapping[str, Any]) -> Any:
    _exact_fields(record_type, data)
    return record_type(**data)


def _taxonomy_truth(count: int, relations: Sequence[Relation]) -> set[Relation]:
    parents: dict[int, int] = {}
    for row in relations:
        if (row.kind != "parent" or type(row.subject) is not int or type(row.object) is not int or
                not 0 <= row.subject < count or not 0 <= row.object < count or row.subject == row.object):
            raise ValueError("taxonomy base graph needs valid parent edges")
        if row.object in parents:
            raise ValueError("taxonomy child has duplicate/multiple parents")
        parents[row.object] = row.subject
    if len(parents) != count - 1:
        raise ValueError("taxonomy graph must be a connected rooted tree")
    truth = set(relations)
    for child in range(count):
        visited = {child}
        node = child
        while node in parents:
            node = parents[node]
            if node in visited:
                raise ValueError("taxonomy graph is cyclic")
            visited.add(node)
            truth.add(Relation("ancestor", node, child))
    for left, left_parent in parents.items():
        for right, right_parent in parents.items():
            if left != right and left_parent == right_parent:
                truth.add(Relation("sibling", left, right))
    return truth


def infer_taxonomy_graph(clauses: Sequence[SurfaceClause], entities: int) -> tuple[Relation, ...]:
    """Find a rooted-tree witness using only candidate relational statements.

    At most six anonymous taxa are supported by this exact finite search.  The
    cap is explicit: callers must not interpret an unsupported size as a failed
    semantic hypothesis.  Parent/ancestor are irreflexive; sibling is symmetric.
    As with the procedure witness, success need not identify a unique graph.
    """
    _integer(entities, "entities", 2)
    if entities > 6:
        raise ValueError("exact taxonomy search supports at most 6 entities")
    if not clauses:
        raise ValueError("cannot infer an empty taxonomy")
    required: set[Relation] = set()
    fixed: dict[int, int] = {}
    for clause in clauses:
        if not isinstance(clause, SurfaceClause) or max(clause.subject, clause.object) >= entities:
            raise ValueError("invalid candidate clause")
        if clause.subject == clause.object:
            raise ValueError("no taxonomy relation is reflexive")
        relation = Relation(TAXONOMY_OPERATORS[clause.operator], clause.subject, clause.object)
        required.add(relation)
        if relation.kind == "parent":
            if relation.object in fixed and fixed[relation.object] != relation.subject:
                raise ValueError("taxonomy child is assigned conflicting parents")
            fixed[relation.object] = relation.subject
    for root in range(entities):
        if root in fixed:
            continue
        children = tuple(i for i in range(entities) if i != root)
        choices = [((fixed[child],) if child in fixed else tuple(i for i in range(entities) if i != child))
                   for child in children]
        for parents in itertools.product(*choices):
            graph = tuple(Relation("parent", parent, child) for child, parent in zip(children, parents))
            try:
                truth = _taxonomy_truth(entities, graph)
            except ValueError:
                continue
            if required.issubset(truth):
                return graph
    raise ValueError("no rooted taxonomy satisfies the complete document")


def generate_world(config: WorldConfig) -> WorldExample:
    """Generate one reproducible, fully audited example, with no external I/O."""
    if not isinstance(config, WorldConfig):
        raise TypeError("config must be WorldConfig")
    randomizer = _rng(config.seed, "semantics")
    entities: tuple[Entity, ...] = ()
    initial: tuple[EntityState, ...] = ()
    trace: list[ActionStep] = []
    relations: tuple[Relation, ...] = ()
    clauses: list[SurfaceClause] = []
    if config.family == "procedure":
        entities = tuple(Entity(i, "agent" if i == 0 else "vessel") for i in range(config.entities))
        occupied = randomizer.randrange(1, config.entities)
        initial = tuple(EntityState(i, i == occupied, False) for i in range(config.entities))
        state = initial
        for _ in range(config.events):
            holder = next(row.entity for row in state if row.full)
            actions = [Action("cool" if state[holder].hot else "heat", 0, holder)]
            actions += [Action("transfer", holder, i) for i in range(1, config.entities) if i != holder]
            action = randomizer.choice(actions)
            after = execute_action(entities, state, action)
            trace.append(ActionStep(action, state, after))
            clauses.append(SurfaceClause(PROCEDURE_OPERATORS.index(action.kind), action.subject, action.object))
            state = after
    elif config.family == "taxonomy":
        entities = tuple(Entity(i, "taxon") for i in range(config.entities))
        relations = tuple(Relation("parent", randomizer.randrange(i), i) for i in range(1, config.entities))
        facts = sorted(_taxonomy_truth(config.entities, relations), key=lambda x: (x.kind, x.subject, x.object))
        for _ in range(config.events):
            fact = randomizer.choice(facts)
            clauses.append(SurfaceClause(TAXONOMY_OPERATORS.index(fact.kind), fact.subject, fact.object))
    else:
        # A writer copies/revises surface slots; no world state or referents exist.
        # Same order/morphology renderer keeps this a serious structured control.
        for index in range(config.events):
            if index and randomizer.random() < 0.8:
                source = randomizer.choice(clauses[max(0, index - 4):])
                slot = randomizer.randrange(3)
                values = [source.operator, source.subject, source.object]
                values[slot] = randomizer.randrange(3 if slot == 0 else config.entities)
                clauses.append(SurfaceClause(*values))
            else:
                clauses.append(SurfaceClause(randomizer.randrange(3), randomizer.randrange(config.entities),
                                             randomizer.randrange(config.entities)))
    plain, boundaries, roles = _realize(config, clauses)
    key = [value for value in range(2, config.canonical_count + 2) for _ in range(config.homophones)]
    key += [0] * (config.alphabet_size - len(key))
    _rng(config.seed, "key").shuffle(key)
    encoded = encode(plain, key, seed=config.seed, null_rate=config.null_rate,
                     word_boundary=boundaries, roles=roles)
    inventory = tuple([f"entity_{i}" for i in range(config.entities)] +
                      [f"operator_{i}" for i in range(3)] +
                      ([] if config.morphology == "none" else ["subject_marker", "object_marker"]))
    result = WorldExample(config, encoded.observation, plain, tuple(key), encoded.keep,
                          encoded.word_boundary, encoded.roles, encoded.alignment, entities, initial,
                          tuple(trace), relations, tuple(clauses), inventory)
    result.validate()
    return result


def rename_observation(example: WorldExample, permutation: Sequence[int]) -> WorldExample:
    """Exact alphabet-renaming control, preserving semantics and channel behavior.

    permutation[i] is the new glyph for old glyph i+2. No re-generation occurs.
    """
    perm = tuple(permutation)
    size = example.config.alphabet_size
    if any(type(x) is not int for x in perm) or sorted(perm) != list(range(2, size + 2)):
        raise ValueError("renaming must be an observed-alphabet permutation")
    key = [0] * size
    for old_index, new_symbol in enumerate(perm):
        key[new_symbol - 2] = example.inverse_key[old_index]
    result = replace(example, observation=tuple(perm[x - 2] for x in example.observation), inverse_key=tuple(key))
    result.validate()
    return result


def equivalent_entity_renaming(example: WorldExample, permutation: Sequence[int]) -> WorldExample:
    """Construct an observationally identical alternative gold interpretation.

    Entity identities are arbitrary without anchors. This explicit automorphism
    demonstrates why exact lexical-id recovery is stronger than structural recovery.
    """
    perm = tuple(permutation)
    count = example.config.entities
    if any(type(x) is not int for x in perm) or sorted(perm) != list(range(count)):
        raise ValueError("entity renaming must be a permutation")

    def symbol(value: int) -> int:
        return perm[value - 2] + 2 if 2 <= value < count + 2 else value

    def state(rows: Sequence[EntityState]) -> tuple[EntityState, ...]:
        return tuple(sorted((replace(row, entity=perm[row.entity]) for row in rows), key=lambda x: x.entity))

    result = replace(
        example,
        plaintext=tuple(symbol(x) for x in example.plaintext),
        inverse_key=tuple(symbol(x) for x in example.inverse_key),
        entities=tuple(sorted((replace(row, id=perm[row.id]) for row in example.entities), key=lambda x: x.id)),
        initial_state=state(example.initial_state),
        action_trace=tuple(ActionStep(replace(row.action, subject=perm[row.action.subject],
                                             object=perm[row.action.object]), state(row.before), state(row.after))
                           for row in example.action_trace),
        relations=tuple(replace(row, subject=perm[row.subject], object=perm[row.object]) for row in example.relations),
        clauses=tuple(replace(row, subject=perm[row.subject], object=perm[row.object]) for row in example.clauses),
    )
    result.validate()
    return result
