"""Prospective counterfactual suites for raw binding mechanism experiments.

This module does not inspect trained models or TEACH-0012 results.  It builds paired
programs whose physical serialization is held fixed while F bindings or G recipients
change, and it reconstructs semantic token positions from generator metadata.
"""

from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import random

from .teacher12_tasks import (
    ANSWER,
    BOS,
    EDGE,
    GAP,
    SYMBOL_START,
    SYMBOLS,
    TASK_MARKERS,
    Episode,
    RenderSpec,
    _make_episode,
    symbolic_oracle,
    table_partitions,
)


SPLIT_NAMESPACE = "TEACH-0013-v1"
MARKED_SPEC = RenderSpec(0.0, 2, ("prefix", "infix", "suffix"))
MARKER_FREE_SPEC = RenderSpec(1.0, 2, ("prefix", "infix", "suffix"))


def _digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def surface_skeleton(episode: Episode) -> tuple[int, ...]:
    """Erase opaque symbol identities while retaining every structural token."""
    return tuple(token if token < SYMBOL_START else -1 for token in episode.tokens)


@dataclass(frozen=True)
class SemanticLayout:
    """A role and logical-row index for every serialized token position."""

    roles: tuple[str, ...]
    labels: tuple[str, ...]
    row_indices: tuple[int, ...]

    def positions(self, role: str) -> tuple[int, ...]:
        return tuple(index for index, candidate in enumerate(self.roles) if candidate == role)

    def label_position(self, label: str) -> int:
        positions = tuple(index for index, candidate in enumerate(self.labels)
                          if candidate == label)
        if len(positions) != 1:
            raise ValueError(f"Label {label!r} has {len(positions)} positions, expected one")
        return positions[0]


@dataclass(frozen=True)
class PhysicalShortcutPrediction:
    """One frozen physical-slot prediction and its scoring eligibility.

    ``structurally_valid`` says the requested shortcut is defined by the two
    renderings.  ``adversarially_eligible`` additionally requires its target to
    differ from the logical answer.  Confirmation accuracy must use the latter
    as its denominator; a coincidentally correct shortcut is recorded as a
    collision and is never counted as shortcut evidence.
    """

    target: int | None
    structurally_valid: bool
    semantic_collision: bool
    adversarially_eligible: bool
    invalid_reason: str | None


@dataclass(frozen=True)
class PhysicalOrderShortcutOracle:
    """Predeclared shortcut targets induced by a complete-row reorder.

    The F-slot oracle reads the right endpoint now occupying the original
    queried-F slot and, only when that endpoint is a G key, follows the frozen G
    table once.  The G-slot oracle reads the right endpoint now occupying the
    original matched-G slot.  The joint oracle is defined only when those two
    occupying rows form a coherent edge pair.  All three are reported; the
    registered primary kind is selected separately from the frozen mediator.
    """

    logical_id: str
    original_render_id: str
    reordered_render_id: str
    semantic_target: int
    queried_f_slot: int
    matched_g_slot: int
    queried_f_original_row: tuple[int, int]
    matched_g_original_row: tuple[int, int]
    queried_f_slot_occupant: tuple[int, int]
    matched_g_slot_occupant: tuple[int, int]
    f_slot: PhysicalShortcutPrediction
    g_slot: PhysicalShortcutPrediction
    joint: PhysicalShortcutPrediction
    target_agreements: tuple[str, ...]

    def prediction(self, kind: str) -> PhysicalShortcutPrediction:
        if kind not in ("f_slot", "g_slot", "joint"):
            raise ValueError(f"Unknown physical-order oracle kind: {kind}")
        return getattr(self, kind)

    def compact_fields(self, kind: str) -> dict:
        """Flatten one frozen oracle family into JSON-safe confirmation fields.

        ``oracle_valid`` is the scoring-denominator flag: the target is both
        structurally defined and different from the semantic answer.  The two
        constituent flags remain explicit so an auditor can reproduce it.
        """
        prediction = self.prediction(kind)
        return {
            "oracle_kind": kind,
            "oracle_target": prediction.target,
            "oracle_valid": prediction.adversarially_eligible,
            "oracle_structurally_valid": prediction.structurally_valid,
            "oracle_semantic_collision": prediction.semantic_collision,
            "oracle_invalid_reason": (
                "semantic_target_collision" if prediction.semantic_collision
                else prediction.invalid_reason),
            "physical_f_slot_index": self.queried_f_slot,
            "physical_g_slot_index": self.matched_g_slot,
            "physical_f_slot_key": self.queried_f_slot_occupant[1],
            "physical_g_slot_key": self.matched_g_slot_occupant[0],
            "physical_g_slot_value": self.matched_g_slot_occupant[1],
            "physical_slots_compose": (
                self.queried_f_slot_occupant[1]
                == self.matched_g_slot_occupant[0]),
        }


def _relation_role(episode: Episode, row: tuple[int, int]) -> str:
    queried_f = next((candidate for candidate in episode.f_rows
                      if candidate[0] == episode.query), None)
    if queried_f is not None and row == queried_f:
        return "queried_f"
    if row in episode.f_rows:
        return "other_f"
    if row in episode.g_rows and (
            (queried_f is not None and row[0] == queried_f[1])
            or (episode.task == "direct" and row[0] == episode.query)):
        return "matched_g"
    if row in episode.g_rows:
        return "other_g"
    index = episode.distractor_rows.index(row)
    if index < 4:
        return "false_path_first" if index % 2 == 0 else "false_path_terminal"
    return "distractor_first" if index % 2 == 0 else "distractor_second"


def _relation_label(episode: Episode, row: tuple[int, int], role: str) -> str:
    if role in ("queried_f", "matched_g"):
        return role
    if role == "other_f":
        candidates = tuple(sorted(episode.f_rows, key=lambda candidate: candidate[0]))
    elif role == "other_g":
        candidates = tuple(sorted(episode.g_rows, key=lambda candidate: candidate[0]))
    else:
        indices = {
            "false_path_first": (0, 2),
            "false_path_terminal": (1, 3),
            "distractor_first": (4, 6),
            "distractor_second": (5, 7),
        }[role]
        candidates = tuple(sorted((episode.distractor_rows[index] for index in indices),
                                  key=lambda candidate: candidate[0]))
    return f"{role}.{candidates.index(row)}"


def semantic_layout(episode: Episode) -> SemanticLayout:
    """Recover generated row boundaries and semantic roles without neural activations."""
    tokens = episode.tokens
    if len(tokens) < 4 or tokens[0] != BOS or tokens[-1] != ANSWER:
        raise ValueError("Malformed serialized episode")
    roles = ["unassigned"] * len(tokens)
    labels = ["unassigned"] * len(tokens)
    rows = [-1] * len(tokens)
    roles[0] = "bos"
    labels[0] = "bos"
    gap_index = 0
    cursor, body_end = 1, len(tokens) - 3
    for row_index, expected in enumerate(episode.serialized_rows):
        while cursor < body_end and tokens[cursor] == GAP:
            roles[cursor] = "gap"
            labels[cursor] = f"gap.{gap_index}"
            gap_index += 1
            cursor += 1
        if cursor >= body_end:
            raise ValueError("Serialized rows ended early")
        marker = None
        if tokens[cursor] == EDGE:
            marker, left, right = cursor, cursor + 1, cursor + 2
            cursor += 3
        elif cursor + 1 < body_end and tokens[cursor + 1] == EDGE:
            left, marker, right = cursor, cursor + 1, cursor + 2
            cursor += 3
        else:
            left, right = cursor, cursor + 1
            cursor += 2
            if cursor < body_end and tokens[cursor] == EDGE:
                marker = cursor
                cursor += 1
        if right >= body_end or (tokens[left], tokens[right]) != expected:
            raise ValueError("Tokens do not match stored serialized rows")
        relation = _relation_role(episode, expected)
        relation_label = _relation_label(episode, expected, relation)
        roles[left], roles[right] = relation + ".left", relation + ".right"
        labels[left], labels[right] = relation_label + ".left", relation_label + ".right"
        rows[left] = rows[right] = row_index
        if marker is not None:
            roles[marker] = relation + ".marker"
            labels[marker] = relation_label + ".marker"
            rows[marker] = row_index
    while cursor < body_end and tokens[cursor] == GAP:
        roles[cursor] = "gap"
        labels[cursor] = f"gap.{gap_index}"
        gap_index += 1
        cursor += 1
    if cursor != body_end or tokens[-3] != TASK_MARKERS[episode.task] \
            or tokens[-2] != episode.query:
        raise ValueError("Malformed task/query suffix")
    roles[-3:] = ["task", "query", "answer"]
    labels[-3:] = ["task", "query", "answer"]
    if "unassigned" in roles or "unassigned" in labels:
        raise ValueError("Some token positions were not assigned semantic roles")
    if len(set(labels)) != len(labels):
        raise ValueError("Semantic labels are not unique")
    return SemanticLayout(tuple(roles), tuple(labels), tuple(rows))


def _physical_prediction(target: int | None, semantic_target: int, *,
                         invalid_reason: str | None = None
                         ) -> PhysicalShortcutPrediction:
    valid = target is not None and invalid_reason is None
    collision = valid and target == semantic_target
    return PhysicalShortcutPrediction(
        target, valid, collision, valid and not collision,
        None if valid else invalid_reason,
    )


def _semantic_row_slot(layout: SemanticLayout, role: str) -> int:
    positions = layout.positions(f"{role}.left")
    if len(positions) != 1:
        raise ValueError(f"Expected one {role!r} left endpoint, found {len(positions)}")
    slot = layout.row_indices[positions[0]]
    if slot < 0:
        raise ValueError(f"Semantic role {role!r} is not assigned to a row")
    return slot


def physical_order_shortcut_oracle(
        original: Episode, reordered: Episode) -> PhysicalOrderShortcutOracle:
    """Compute frozen F-slot, G-slot and coherent-pair shortcut predictions.

    This function only accepts the exact complete-row cyclic reorder produced by
    :func:`_reorder_only`.  It does not inspect model outputs.  A target equal to
    the unchanged symbolic answer is retained as a semantic collision but is
    excluded from the adversarial shortcut denominator.
    """
    graph_fields = ("logical_id", "task", "query", "answer", "f_rows", "g_rows",
                    "distractor_rows")
    if any(getattr(original, name) != getattr(reordered, name) for name in graph_fields):
        raise ValueError("Physical-order oracle requires two renderings of one graph")
    if original.task != "composed":
        raise ValueError("Physical-order shortcut oracle is registered for composed items")
    if surface_skeleton(original) != surface_skeleton(reordered):
        raise ValueError("Physical-order pair does not preserve the serialized surface")
    expected_rows = original.serialized_rows[1:] + original.serialized_rows[:1]
    if reordered.serialized_rows != expected_rows:
        raise ValueError("Physical-order donor is not the registered one-slot cyclic reorder")

    layout = semantic_layout(original)
    queried_f_slot = _semantic_row_slot(layout, "queried_f")
    matched_g_slot = _semantic_row_slot(layout, "matched_g")
    queried_f_original = original.serialized_rows[queried_f_slot]
    matched_g_original = original.serialized_rows[matched_g_slot]
    f_occupant = reordered.serialized_rows[queried_f_slot]
    g_occupant = reordered.serialized_rows[matched_g_slot]

    g_mapping = dict(original.g_rows)
    if len(g_mapping) != len(original.g_rows):
        raise ValueError("Physical-order oracle requires a functional G table")
    f_key = f_occupant[1]
    if f_key in g_mapping:
        f_prediction = _physical_prediction(g_mapping[f_key], original.answer)
    else:
        f_prediction = _physical_prediction(
            None, original.answer, invalid_reason="queried_f_slot_rhs_is_not_a_g_key")
    g_prediction = _physical_prediction(g_occupant[1], original.answer)
    if f_key == g_occupant[0]:
        joint_prediction = _physical_prediction(g_occupant[1], original.answer)
    else:
        joint_prediction = _physical_prediction(
            None, original.answer, invalid_reason="physical_slot_rows_do_not_compose")

    predictions = {
        "f_slot": f_prediction, "g_slot": g_prediction, "joint": joint_prediction,
    }
    agreements = tuple(
        f"{left}={right}"
        for index, left in enumerate(predictions)
        for right in tuple(predictions)[index + 1:]
        if predictions[left].structurally_valid
        and predictions[right].structurally_valid
        and predictions[left].target == predictions[right].target
    )
    return PhysicalOrderShortcutOracle(
        original.logical_id, original.render_id, reordered.render_id, original.answer,
        queried_f_slot, matched_g_slot, queried_f_original, matched_g_original,
        f_occupant, g_occupant, f_prediction, g_prediction, joint_prediction,
        agreements,
    )


def registered_physical_order_oracle_kind(*, mediator_kind: str,
                                          label: str | None = None,
                                          source_label: str | None = None) -> str:
    """Freeze the primary shortcut family from the mediator, before predictions.

    A single site uses its selected label; a path uses its early source label.
    A selected matched-G occurrence receives the G-slot oracle.  Every other
    occurrence receives the F-slot oracle because TEACH-0013's primary mediator
    transports an intermediate key.  The joint oracle is always reported as a
    stricter coherence diagnostic and is never substituted after outcomes are
    observed.
    """
    if mediator_kind == "single":
        if not label or source_label is not None:
            raise ValueError("Single-site oracle selection requires only label")
        frozen_label = label
    elif mediator_kind == "path":
        if not source_label or label is not None:
            raise ValueError("Path oracle selection requires only source_label")
        frozen_label = source_label
    else:
        raise ValueError(f"Unknown mediator kind: {mediator_kind}")
    return "g_slot" if frozen_label.startswith("matched_g.") else "f_slot"


def physical_order_shortcut_denominators(
        oracles: tuple[PhysicalOrderShortcutOracle, ...] | list[PhysicalOrderShortcutOracle],
        *, kind: str, group_ids: tuple[str, ...] | list[str] | None = None) -> dict:
    """Return the exact prospective item and optional three-recipient denominators.

    Item accuracy divides by ``adversarially_eligible_items``.  When group IDs
    are supplied, group accuracy divides by ``adversarially_eligible_groups``:
    groups containing exactly three items and no invalid or semantic-collision
    cell.  Partial groups are rejected rather than silently changing the unit.
    """
    rows = tuple(oracles)
    if not rows:
        raise ValueError("Physical-order denominator requires at least one item")
    predictions = tuple(oracle.prediction(kind) for oracle in rows)
    result = {
        "kind": kind,
        "all_items": len(rows),
        "structurally_valid_items": sum(row.structurally_valid for row in predictions),
        "semantic_collision_items": sum(row.semantic_collision for row in predictions),
        "adversarially_eligible_items": sum(
            row.adversarially_eligible for row in predictions),
        "item_denominator_rule": (
            "structurally_valid_and_target_differs_from_semantic_answer"),
    }
    if group_ids is None:
        return result
    keys = tuple(group_ids)
    if len(keys) != len(rows):
        raise ValueError("Group IDs and physical-order items have unequal lengths")
    grouped: dict[str, list[PhysicalShortcutPrediction]] = {}
    for key, prediction in zip(keys, predictions, strict=True):
        grouped.setdefault(key, []).append(prediction)
    if any(len(group) != 3 for group in grouped.values()):
        raise ValueError("Every physical-order group must contain exactly three recipients")
    result.update({
        "all_groups": len(grouped),
        "adversarially_eligible_groups": sum(
            all(row.adversarially_eligible for row in group)
            for group in grouped.values()),
        "group_denominator_rule": (
            "exactly_three_recipients_and_every_item_adversarially_eligible"),
    })
    return result


@dataclass(frozen=True)
class CounterfactualGroup:
    """One F-binding swap crossed with three independently remapped G recipients."""

    group_id: str
    split: str
    base: tuple[Episode, Episode, Episode]
    donor: tuple[Episode, Episode, Episode]
    marker_free_base: tuple[Episode, Episode, Episode]
    marker_free_donor: tuple[Episode, Episode, Episode]
    reordered_base: tuple[Episode, Episode, Episode]
    reordered_donor: tuple[Episode, Episode, Episode]
    g_content_base: tuple[Episode, Episode, Episode]
    g_content_donor: tuple[Episode, Episode, Episode]
    binding_base: tuple[Episode, Episode, Episode]
    binding_donor: tuple[Episode, Episode, Episode]
    binding_marker_free_base: tuple[Episode, Episode, Episode]
    binding_marker_free_donor: tuple[Episode, Episode, Episode]
    binding_reordered_base: tuple[Episode, Episode, Episode]
    binding_reordered_donor: tuple[Episode, Episode, Episode]
    binding_format_base: tuple[Episode, Episode, Episode]
    binding_format_donor: tuple[Episode, Episode, Episode]
    binding_distractor_base: tuple[Episode, Episode, Episode]
    binding_distractor_donor: tuple[Episode, Episode, Episode]
    g_binding_base: tuple[Episode, Episode, Episode]
    g_binding_donor: tuple[Episode, Episode, Episode]
    format_base: tuple[Episode, Episode, Episode]
    format_donor: tuple[Episode, Episode, Episode]
    distractor_base: tuple[Episode, Episode, Episode]
    distractor_donor: tuple[Episode, Episode, Episode]
    marker_free_reordered_base: tuple[Episode, Episode, Episode]
    marker_free_reordered_donor: tuple[Episode, Episode, Episode]
    format_reordered_base: tuple[Episode, Episode, Episode]
    format_reordered_donor: tuple[Episode, Episode, Episode]
    distractor_reordered_base: tuple[Episode, Episode, Episode]
    distractor_reordered_donor: tuple[Episode, Episode, Episode]
    first_hop_base: Episode
    first_hop_donor: Episode
    direct_base: Episode
    direct_donor: Episode
    direct_format_donor: Episode
    direct_order_donor: Episode
    direct_distractor_donor: Episode
    copy_control: Episode
    copy_format_donor: Episode
    copy_order_donor: Episode
    copy_distractor_donor: Episode
    key_base: int
    key_donor: int
    base_answers: tuple[int, int, int]
    recipient_answers: tuple[int, int, int]
    fixed_donor_answer: int
    render_seed: int


def _partition(logical) -> str:
    bucket = int(_digest([SPLIT_NAMESPACE, logical])[:16], 16) % 10
    return "discovery" if bucket < 5 else "confirmation"


def _independent_distractors(values: list[int]) -> tuple[tuple[int, int], ...]:
    rows = []
    for offset in range(0, len(values), 3):
        name, key, obj = values[offset:offset + 3]
        rows.extend(((name, key), (key, obj)))
    return tuple(rows)


def _mixed_distractors(values: list[int], shared_objects: tuple[int, int]
                       ) -> tuple[tuple[int, int], ...]:
    """Two false paths sharing signal endpoints plus two independent chains."""
    if len(values) != 10:
        raise ValueError("Mixed distractors require ten fresh symbols")
    rows = []
    for index in range(2):
        name, key = values[2 * index:2 * index + 2]
        rows.extend(((name, key), (key, shared_objects[index])))
    rows.extend(_independent_distractors(values[4:]))
    return tuple(rows)


def _recipient_permutations(rng: random.Random, objects: tuple[int, ...],
                            key_base_index: int, key_donor_index: int
                            ) -> tuple[tuple[int, ...], ...]:
    candidates = list(itertools.permutations(objects))
    rng.shuffle(candidates)
    for selected in itertools.combinations(candidates, 3):
        base = tuple(row[key_base_index] for row in selected)
        donor = tuple(row[key_donor_index] for row in selected)
        if (all(a != b for a, b in zip(base, donor, strict=True))
                and len(set(base)) == 3 and len(set(donor)) == 3):
            return selected
    raise RuntimeError("Could not construct distinguishable recipient tables")


def _render(f_rows, g_rows, distractor_rows, query, seed, *, task="composed",
            spec=MARKED_SPEC) -> Episode:
    episode = _make_episode(tuple(f_rows), tuple(g_rows), tuple(distractor_rows),
                            task, query, random.Random(seed), spec)
    if symbolic_oracle(episode) != episode.answer:
        raise AssertionError("Generator/oracle disagreement")
    semantic_layout(episode)
    return episode


def _row_encodings(episode: Episode):
    """Return each row's visible encoding and the physical gap count after its slot."""
    tokens, cursor, body_end = episode.tokens, 1, len(episode.tokens) - 3
    encodings, gaps = [], []
    for expected in episode.serialized_rows:
        start = cursor
        if tokens[cursor] == EDGE or (cursor + 1 < body_end and tokens[cursor + 1] == EDGE):
            cursor += 3
        else:
            cursor += 2
            if cursor < body_end and tokens[cursor] == EDGE:
                cursor += 1
        encoding = list(tokens[start:cursor])
        need_symbols = [token for token in encoding if token >= SYMBOL_START]
        if need_symbols != list(expected):
            raise ValueError("Could not recover row encoding")
        gap_count = 0
        while cursor < body_end and tokens[cursor] == GAP:
            gap_count += 1
            cursor += 1
        encodings.append(encoding)
        gaps.append(gap_count)
    if cursor != body_end:
        raise ValueError("Could not consume serialized body")
    return encodings, gaps


def _rebuild_surface(episode: Episode, encodings, gaps, serialized_rows) -> Episode:
    body = [BOS]
    for encoding, gap_count in zip(encodings, gaps, strict=True):
        body.extend(encoding)
        body.extend([GAP] * gap_count)
    tokens = tuple(body + list(episode.tokens[-3:]))
    logical = episode.logical_id
    return replace(episode, tokens=tokens, serialized_rows=tuple(serialized_rows),
                   render_id=_digest({"logical": logical, "tokens": tokens}))


def _reorder_only(episode: Episode) -> Episode:
    """Move complete logical rows while preserving every physical format/gap slot."""
    encodings, gaps = _row_encodings(episode)
    count = len(encodings)
    permutation = tuple(range(1, count)) + (0,)
    new_rows = tuple(episode.serialized_rows[index] for index in permutation)
    rebuilt = []
    for slot_encoding, row in zip(encodings, new_rows, strict=True):
        iterator = iter(row)
        rebuilt.append([next(iterator) if token >= SYMBOL_START else token
                        for token in slot_encoding])
    return _rebuild_surface(episode, rebuilt, gaps, new_rows)


def _format_only(episode: Episode) -> Episode:
    """Rotate marker position while preserving rows, row order, gaps and length."""
    encodings, gaps = _row_encodings(episode)
    changed = []
    for encoding in encodings:
        if encoding[0] == EDGE:
            changed.append([encoding[1], EDGE, encoding[2]])
        elif encoding[1] == EDGE:
            changed.append([encoding[0], encoding[2], EDGE])
        elif encoding[-1] == EDGE:
            changed.append([EDGE, encoding[0], encoding[1]])
        else:
            raise ValueError("Format-only donor requires marked rows")
    return _rebuild_surface(episode, changed, gaps, episode.serialized_rows)


def _fresh_confirm_g_value(f_rows, g_rows, row_index: int, used: set[int],
                           rng: random.Random) -> int | None:
    """Choose a fresh replacement whose altered G family remains held-out confirm."""
    candidates = [symbol for symbol in SYMBOLS if symbol not in used]
    rng.shuffle(candidates)
    for candidate in candidates:
        changed = list(g_rows)
        left, _ = changed[row_index]
        changed[row_index] = (left, candidate)
        if table_partitions(tuple(f_rows), tuple(changed))[1] == "confirm":
            return candidate
    return None


def _candidate_group(rng: random.Random) -> CounterfactualGroup | None:
    symbols = rng.sample(SYMBOLS, 32)
    names = tuple(symbols[:4])
    keys = tuple(symbols[4:8])
    objects = tuple(symbols[8:12])
    shared = (objects[0], objects[1])
    distractors = _mixed_distractors(symbols[12:22], shared)
    nuisance_distractors = _mixed_distractors(symbols[22:32], shared)

    f_values = list(keys)
    rng.shuffle(f_values)
    f0 = tuple(zip(names, f_values, strict=True))
    query_index, other_index = rng.sample(range(4), 2)
    changed = list(f_values)
    changed[query_index] = changed[other_index]
    f1 = tuple(zip(names, changed, strict=True))
    binding_values = list(f_values)
    binding_values[query_index], binding_values[other_index] = (
        binding_values[other_index], binding_values[query_index])
    f_binding = tuple(zip(names, binding_values, strict=True))
    query = names[query_index]
    key_base, key_donor = f_values[query_index], changed[query_index]
    key_base_index, key_donor_index = keys.index(key_base), keys.index(key_donor)

    recipients = _recipient_permutations(rng, objects, key_base_index, key_donor_index)
    g_tables = tuple(tuple(zip(keys, values, strict=True)) for values in recipients)
    if table_partitions(f0, g_tables[0]) != ("confirm", "confirm") \
            or table_partitions(f1, g_tables[0])[0] != "confirm":
        return None

    render_seed = rng.randrange(2**63)
    base = tuple(_render(f0, g_rows, distractors, query, render_seed)
                 for g_rows in g_tables)
    donor = tuple(_render(f1, g_rows, distractors, query, render_seed)
                  for g_rows in g_tables)
    if len({surface_skeleton(ep) for ep in base + donor}) != 1:
        raise AssertionError("Counterfactual render skeletons diverged")
    marker_free_seed = render_seed
    marker_free_base = tuple(_render(
        f0, g_rows, distractors, query, marker_free_seed, spec=MARKER_FREE_SPEC)
        for g_rows in g_tables)
    marker_free_donor = tuple(_render(
        f1, g_rows, distractors, query, marker_free_seed, spec=MARKER_FREE_SPEC)
        for g_rows in g_tables)
    reordered_base = tuple(_reorder_only(episode) for episode in base)
    reordered_donor = tuple(_reorder_only(episode) for episode in donor)
    for variants in (marker_free_base + marker_free_donor,
                     reordered_base + reordered_donor):
        if len({surface_skeleton(ep) for ep in variants}) != 1:
            raise AssertionError("Paired render skeletons diverged")
    base_answers = tuple(ep.answer for ep in base)
    recipient_answers = tuple(ep.answer for ep in donor)
    if len(set(base_answers)) != 3 or len(set(recipient_answers)) != 3:
        raise AssertionError("Recipient outputs are not distinguishable")

    g_changed_tables, g_binding_tables = [], []
    used = set(symbols)
    for g_rows in g_tables:
        g_base_index = next(index for index, (left, _) in enumerate(g_rows)
                            if left == key_base)
        g_other_index = next(index for index, (left, _) in enumerate(g_rows)
                             if left not in (key_base, key_donor))
        fresh = _fresh_confirm_g_value(f0, g_rows, g_base_index, used, rng)
        if fresh is None:
            return None
        used.add(fresh)
        g_changed = list(g_rows)
        left0, _ = g_changed[g_base_index]
        g_changed[g_base_index] = (left0, fresh)
        g_changed_tables.append(tuple(g_changed))
        g_binding = list(g_rows)
        left0, value0 = g_binding[g_base_index]
        left1, value1 = g_binding[g_other_index]
        g_binding[g_base_index] = (left0, value1)
        g_binding[g_other_index] = (left1, value0)
        g_binding_tables.append(tuple(g_binding))
    g_changed_tables = tuple(g_changed_tables)
    g_binding_tables = tuple(g_binding_tables)
    g_content_seed = render_seed
    g_content_base = tuple(_render(f0, rows, distractors, query, g_content_seed)
                           for rows in g_tables)
    g_content_donor = tuple(_render(f0, rows, distractors, query, g_content_seed)
                            for rows in g_changed_tables)
    if any(base_episode.answer == donor_episode.answer or
           surface_skeleton(base_episode) != surface_skeleton(donor_episode)
           for base_episode, donor_episode in zip(
               g_content_base, g_content_donor, strict=True)):
        raise AssertionError("G-content counterfactual is not isolated")

    binding_base = tuple(_render(f0, rows, distractors, query, render_seed)
                         for rows in g_tables)
    binding_donor = tuple(_render(f_binding, rows, distractors, query, render_seed)
                          for rows in g_tables)
    binding_marker_free_base = tuple(_render(
        f0, rows, distractors, query, render_seed, spec=MARKER_FREE_SPEC)
        for rows in g_tables)
    binding_marker_free_donor = tuple(_render(
        f_binding, rows, distractors, query, render_seed, spec=MARKER_FREE_SPEC)
        for rows in g_tables)
    binding_reordered_base = tuple(_reorder_only(episode) for episode in binding_base)
    binding_reordered_donor = tuple(_reorder_only(episode) for episode in binding_donor)
    binding_format_base = tuple(_format_only(episode) for episode in binding_base)
    binding_format_donor = tuple(_format_only(episode) for episode in binding_donor)
    binding_distractor_base = tuple(_render(
        f0, rows, nuisance_distractors, query, render_seed) for rows in g_tables)
    binding_distractor_donor = tuple(_render(
        f_binding, rows, nuisance_distractors, query, render_seed) for rows in g_tables)
    g_binding_base = tuple(_render(f0, rows, distractors, query, render_seed)
                           for rows in g_tables)
    g_binding_donor = tuple(_render(f0, rows, distractors, query, render_seed)
                            for rows in g_binding_tables)

    format_base = tuple(_format_only(episode) for episode in base)
    format_donor = tuple(_format_only(episode) for episode in donor)
    distractor_base = tuple(_render(
        f0, rows, nuisance_distractors, query, render_seed) for rows in g_tables)
    distractor_donor = tuple(_render(
        f1, rows, nuisance_distractors, query, render_seed) for rows in g_tables)
    marker_free_reordered_base = tuple(
        _reorder_only(episode) for episode in marker_free_base)
    marker_free_reordered_donor = tuple(
        _reorder_only(episode) for episode in marker_free_donor)
    format_reordered_base = tuple(_reorder_only(episode) for episode in format_base)
    format_reordered_donor = tuple(_reorder_only(episode) for episode in format_donor)
    distractor_reordered_base = tuple(
        _reorder_only(episode) for episode in distractor_base)
    distractor_reordered_donor = tuple(
        _reorder_only(episode) for episode in distractor_donor)
    first_hop_base = _render(
        f0, g_tables[0], distractors, query, rng.randrange(2**63), task="first_hop")
    first_hop_donor = _render(
        f1, g_tables[0], distractors, query, rng.randrange(2**63), task="first_hop")
    direct_base = _render(
        f0, g_tables[0], distractors, key_base, rng.randrange(2**63), task="direct")
    direct_seed = rng.randrange(2**63)
    direct_donor = _render(
        f1, g_tables[0], distractors, key_donor, direct_seed, task="direct")
    direct_format_donor = _format_only(direct_donor)
    direct_order_donor = _reorder_only(direct_donor)
    direct_distractor_donor = _render(
        f1, g_tables[0], nuisance_distractors, key_donor, direct_seed, task="direct")
    copy_seed = rng.randrange(2**63)
    copy_control = _render(
        f0, g_tables[0], distractors, objects[3], copy_seed, task="copy")
    copy_format_donor = _format_only(copy_control)
    copy_order_donor = _reorder_only(copy_control)
    copy_distractor_donor = _render(
        f0, g_tables[0], nuisance_distractors, objects[3], copy_seed, task="copy")
    logical = {
        "f0": f0, "f1": f1, "f_binding": f_binding, "g": g_tables,
        "g_content": g_changed_tables, "g_binding": g_binding_tables,
        "d": distractors, "nuisance_d": nuisance_distractors,
        "query": query, "key_base": key_base, "key_donor": key_donor,
    }
    return CounterfactualGroup(
        _digest([SPLIT_NAMESPACE, logical]), _partition(logical), base, donor,
        marker_free_base, marker_free_donor, reordered_base, reordered_donor,
        g_content_base, g_content_donor, binding_base, binding_donor,
        binding_marker_free_base, binding_marker_free_donor,
        binding_reordered_base, binding_reordered_donor,
        binding_format_base, binding_format_donor,
        binding_distractor_base, binding_distractor_donor,
        g_binding_base, g_binding_donor, format_base, format_donor,
        distractor_base, distractor_donor,
        marker_free_reordered_base, marker_free_reordered_donor,
        format_reordered_base, format_reordered_donor,
        distractor_reordered_base, distractor_reordered_donor,
        first_hop_base, first_hop_donor,
        direct_base, direct_donor, direct_format_donor, direct_order_donor,
        direct_distractor_donor, copy_control, copy_format_donor, copy_order_donor,
        copy_distractor_donor, key_base, key_donor,
        base_answers, recipient_answers, donor[0].answer, render_seed)


def counterfactual_suite(seed: int, *, discovery_groups: int,
                         confirmation_groups: int, return_stats: bool = False):
    """Generate exact split counts without selecting on any model behavior."""
    if discovery_groups <= 0 or confirmation_groups <= 0:
        raise ValueError("Both split sizes must be positive")
    rng = random.Random(seed)
    target = {"discovery": discovery_groups, "confirmation": confirmation_groups}
    groups: dict[str, list[CounterfactualGroup]] = {name: [] for name in target}
    seen = set()
    stats = {"attempts": 0, "candidate_rejected": 0, "duplicate_id": 0,
             "split_full": 0, "accepted_discovery": 0, "accepted_confirmation": 0}
    for _ in range(1_000_000):
        if all(len(groups[name]) == count for name, count in target.items()):
            break
        stats["attempts"] += 1
        group = _candidate_group(rng)
        if group is None:
            stats["candidate_rejected"] += 1
            continue
        if group.group_id in seen:
            stats["duplicate_id"] += 1
            continue
        if len(groups[group.split]) >= target[group.split]:
            stats["split_full"] += 1
            continue
        seen.add(group.group_id)
        groups[group.split].append(group)
        stats[f"accepted_{group.split}"] += 1
    else:
        raise RuntimeError("Could not fill counterfactual suite")
    frozen = {name: tuple(rows) for name, rows in groups.items()}
    return (frozen, stats) if return_stats else frozen
