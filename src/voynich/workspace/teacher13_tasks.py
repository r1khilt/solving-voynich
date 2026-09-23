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
    g_binding_base: tuple[Episode, Episode, Episode]
    g_binding_donor: tuple[Episode, Episode, Episode]
    format_donor: tuple[Episode, Episode, Episode]
    distractor_donor: tuple[Episode, Episode, Episode]
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
    g_binding_base = tuple(_render(f0, rows, distractors, query, render_seed)
                           for rows in g_tables)
    g_binding_donor = tuple(_render(f0, rows, distractors, query, render_seed)
                            for rows in g_binding_tables)

    format_donor = tuple(_format_only(episode) for episode in donor)
    distractor_donor = tuple(_render(
        f1, rows, nuisance_distractors, query, render_seed) for rows in g_tables)
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
        g_binding_base, g_binding_donor, format_donor, distractor_donor,
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
