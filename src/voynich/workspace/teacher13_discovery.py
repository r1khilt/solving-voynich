"""Clean-panel evaluation and batched residual discovery for TEACH-0013."""

from collections.abc import Iterable
import gzip
import json
from pathlib import Path

import torch
from torch import nn

from .teacher12_tasks import SYMBOL_START, Episode
from .teacher13_intervene import position_patch, raw_forward, residual_sites
from .teacher13_tasks import SemanticLayout, semantic_layout


def episode_from_record(row: dict) -> Episode:
    """Reconstruct an immutable Episode from a JSON/asdict record."""
    return Episode(
        tuple(row["tokens"]),
        tuple(tuple(pair) for pair in row["f_rows"]),
        tuple(tuple(pair) for pair in row["g_rows"]),
        tuple(tuple(pair) for pair in row["distractor_rows"]),
        tuple(tuple(pair) for pair in row["serialized_rows"]),
        row["answer"], row["task"], row["query"], row["f_partition"],
        row["g_partition"], row["logical_id"], row["render_id"],
        row["distractor_chains"], row["marker_dropout"])


def load_suite(path: Path | str) -> dict:
    raw = gzip.decompress(Path(path).read_bytes())
    suite = json.loads(raw)
    if suite.get("experiment") != "TEACH-0013" or suite.get("namespace") != "TEACH-0013-v1":
        raise ValueError("Not a TEACH-0013 suite")
    if set(suite.get("splits", {})) != {"discovery", "confirmation"}:
        raise ValueError("Suite does not contain both frozen splits")
    return suite


def _variant(group: dict, name: str) -> tuple[Episode, ...]:
    value = group[name]
    rows = value if isinstance(value, (list, tuple)) else (value,)
    return tuple(episode_from_record(row) for row in rows)


def _predict(net: nn.Module, episodes: tuple[Episode, ...], *, device, batch_size=64):
    predictions, target_probabilities = [], []
    for offset in range(0, len(episodes), batch_size):
        chunk = episodes[offset:offset + batch_size]
        logits = raw_forward(net, chunk, device=device).logits[:, SYMBOL_START:].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite clean logits")
        probabilities = logits.softmax(-1)
        if not torch.isfinite(probabilities).all() or not torch.allclose(
                probabilities.sum(-1), torch.ones(len(chunk), device=logits.device),
                atol=1e-6, rtol=1e-6):
            raise FloatingPointError("Clean probabilities are nonfinite or unnormalized")
        targets = torch.tensor(
            [episode.answer - SYMBOL_START for episode in chunk], device=logits.device)
        rows = torch.arange(len(chunk), device=logits.device)
        predictions.extend(int(value + SYMBOL_START) for value in logits.argmax(-1).cpu())
        target_probabilities.extend(float(value) for value in probabilities[rows, targets].cpu())
    return tuple(predictions), tuple(target_probabilities)


def _correct(net, episodes, *, device):
    predictions, _ = _predict(net, tuple(episodes), device=device)
    return tuple(prediction == episode.answer
                 for prediction, episode in zip(predictions, episodes, strict=True))


def _exact_chunks(flags: tuple[bool, ...], width: int) -> float:
    if not flags or len(flags) % width:
        raise ValueError("Flags do not form complete groups")
    return sum(all(flags[offset:offset + width])
               for offset in range(0, len(flags), width)) / (len(flags) / width)


def fresh_panel_scores(net: nn.Module, groups: list[dict], *, device="cpu") -> dict:
    """Compute the registered Stage-A clean competence cells for one checkpoint."""
    if not groups:
        raise ValueError("Fresh panel is empty")
    base, donor, marker_base, marker_donor = [], [], [], []
    order_base, order_donor = [], []
    first_hop, direct, copy = [], [], []
    distractor_pairs = []
    for group in groups:
        base.extend(_variant(group, "base"))
        donor.extend(_variant(group, "donor"))
        marker_base.extend(_variant(group, "marker_free_base"))
        marker_donor.extend(_variant(group, "marker_free_donor"))
        order_base.extend(_variant(group, "reordered_base"))
        order_donor.extend(_variant(group, "reordered_donor"))
        first_hop.extend((_variant(group, "first_hop_base")[0],
                          _variant(group, "first_hop_donor")[0]))
        direct.extend((_variant(group, "direct_base")[0],
                       _variant(group, "direct_donor")[0]))
        copy.extend(_variant(group, "copy_control"))
        distractor_pairs.extend(
            episode for pair in zip(_variant(group, "donor"),
                                    _variant(group, "distractor_donor"), strict=True)
            for episode in pair)

    base_flags = _correct(net, base, device=device)
    donor_flags = _correct(net, donor, device=device)
    marker_flags = _correct(net, marker_base + marker_donor, device=device)
    order_flags = _correct(net, order_base + order_donor, device=device)
    composed_flags = base_flags + donor_flags
    # Pair corresponding marked/marker-free and original/reordered items, rather than
    # relying on the concatenated inference order.
    marked_flags = base_flags + donor_flags
    marker_pairs = tuple(value for pair in zip(
        marked_flags, marker_flags, strict=True) for value in pair)
    order_pairs = tuple(value for pair in zip(
        marked_flags, order_flags, strict=True) for value in pair)
    distractor_flags = _correct(net, distractor_pairs, device=device)
    first_flags = _correct(net, first_hop, device=device)
    direct_flags = _correct(net, direct, device=device)
    copy_flags = _correct(net, copy, device=device)
    return {
        "composed_items": sum(composed_flags) / len(composed_flags),
        "base_recipient_groups": _exact_chunks(base_flags, 3),
        "donor_recipient_groups": _exact_chunks(donor_flags, 3),
        "marker_pairs": _exact_chunks(marker_pairs, 2),
        "order_pairs": _exact_chunks(order_pairs, 2),
        "distractor_pairs": _exact_chunks(distractor_flags, 2),
        "first_hop": sum(first_flags) / len(first_flags),
        "direct": sum(direct_flags) / len(direct_flags),
        "copy": sum(copy_flags) / len(copy_flags),
    }


def _screen_batch(groups: list[dict], variant_base: str, variant_donor: str, *,
                  fixed_donor: bool, target_mode: str):
    bases, donors, metadata = [], [], []
    for group in groups:
        base = _variant(group, variant_base)
        donor_rows = _variant(group, variant_donor)
        if len(base) != 3 or len(donor_rows) != 3:
            raise ValueError("Recipient screen requires three base tables")
        for recipient, episode in enumerate(base):
            bases.append(episode)
            donor = donor_rows[0] if fixed_donor else donor_rows[recipient]
            donors.append(donor)
            target = donor.answer if target_mode == "donor" else episode.answer
            metadata.append({"group_id": group["group_id"], "recipient": recipient,
                             "target": target,
                             "fixed_donor_answer": donor_rows[0].answer})
    return tuple(bases), tuple(donors), tuple(metadata)


def screen_conditions(family: str):
    """Return prospectively registered base/donor pairings for one causal factor."""
    if family == "f_content":
        return (("marked", "base", "donor", True, "donor"),
                ("marker_free", "marker_free_base", "marker_free_donor", True, "donor"))
    if family == "f_binding":
        return (("marked", "binding_base", "binding_donor", False, "donor"),)
    if family == "g_content":
        return (("marked", "g_content_base", "g_content_donor", False, "donor"),)
    if family == "g_binding":
        return (("marked", "g_binding_base", "g_binding_donor", False, "donor"),)
    if family == "order":
        return (("f0", "base", "reordered_base", False, "base"),
                ("f1", "donor", "reordered_donor", False, "base"))
    if family == "format":
        return (("marked_to_rotated", "donor", "format_donor", False, "base"),)
    if family == "distractor":
        return (("original_to_nuisance", "donor", "distractor_donor", False, "base"),)
    raise ValueError(f"Unknown residual-screen family: {family}")


def residual_numerical_qualification(net: nn.Module, groups: list[dict], *, device="cpu",
                                     sites: Iterable[str] | None = None) -> dict:
    """Verify finite native outputs and identity replacement at every residual cut."""
    if not groups:
        raise ValueError("Numerical panel is empty")
    arm = "raw_looped" if net.__class__.__name__ == "LoopedRawClassifier" else (
        "raw_deep" if len(net.core.blocks) == 12 else "raw_shallow")
    sites = tuple(residual_sites(arm) if sites is None else sites)
    episodes = tuple(_variant(groups[0], "base"))
    native = raw_forward(net, episodes, device=device, cache_names=sites)
    if not torch.isfinite(native.logits).all() \
            or any(not torch.isfinite(value).all() for value in native.cache.values()):
        raise FloatingPointError("Nonfinite native residual qualification tensor")
    probability = native.logits[:, SYMBOL_START:].float().softmax(-1)
    normalized_error = float((probability.sum(-1) - 1).abs().max().item())
    errors = {}
    for site in sites:
        identity = native.cache[site].detach().clone()
        replay = raw_forward(net, episodes, device=device,
                             interventions={site: lambda _value, saved=identity: saved})
        if not torch.isfinite(replay.logits).all():
            raise FloatingPointError(f"Nonfinite identity logits at {site}")
        errors[site] = float((replay.logits.float() - native.logits.float()).abs().max().item())
    maximum = max(errors.values(), default=0.0)
    return {"identity_errors": errors, "maximum_identity_logit_error": maximum,
            "maximum_probability_normalization_error": normalized_error,
            "finite": True,
            "qualified": maximum < 1e-6 and normalized_error <= 1e-6}


def _has_label(layout: SemanticLayout, label: str) -> bool:
    return label in layout.labels


def residual_screen(net: nn.Module, groups: list[dict], *, replicate: int,
                    device="cpu", sites: Iterable[str] | None = None,
                    labels: Iterable[str] | None = None, resource_probe=None,
                    traffic_probe=None, family: str = "f_content",
                    logit_sink=None) -> list[dict]:
    """Screen full residual swaps at every requested cut and semantic label."""
    if not groups:
        raise ValueError("Discovery group list is empty")
    arm = "raw_looped" if net.__class__.__name__ == "LoopedRawClassifier" else (
        "raw_deep" if len(net.core.blocks) == 12 else "raw_shallow")
    sites = tuple(residual_sites(arm) if sites is None else sites)
    if not sites:
        raise ValueError("At least one residual site is required")
    conditions = screen_conditions(family)
    variant_names = {name for _, base_name, donor_name, _, _ in conditions
                     for name in (base_name, donor_name)}
    registered_labels = tuple(labels) if labels is not None else tuple(sorted({
        label for group in groups for name in variant_names
        for episode in _variant(group, name) for label in semantic_layout(episode).labels
    }))
    rows = []
    for render, base_name, donor_name, fixed_donor, target_mode in conditions:
        bases, donors, metadata = _screen_batch(
            groups, base_name, donor_name, fixed_donor=fixed_donor,
            target_mode=target_mode)
        base_layouts = tuple(semantic_layout(episode) for episode in bases)
        donor_layouts = tuple(semantic_layout(episode) for episode in donors)
        base_output = raw_forward(net, bases, device=device, cache_names=sites)
        donor_output = raw_forward(net, donors, device=device, cache_names=sites)
        if traffic_probe is not None:
            traffic_probe(sum(tensor.numel() * tensor.element_size()
                              for tensor in base_output.cache.values())
                          + sum(tensor.numel() * tensor.element_size()
                                for tensor in donor_output.cache.values())
                          + base_output.logits.numel() * base_output.logits.element_size()
                          + donor_output.logits.numel() * donor_output.logits.element_size())
        if resource_probe is not None:
            resource_probe()
        for cut_index, site in enumerate(sites):
            for label in registered_labels:
                selected = [index for index, (base_layout, donor_layout) in enumerate(zip(
                    base_layouts, donor_layouts, strict=True))
                    if _has_label(base_layout, label) and _has_label(donor_layout, label)]
                if not selected:
                    continue
                selected_bases = tuple(bases[index] for index in selected)
                selected_base_layouts = tuple(base_layouts[index] for index in selected)
                selected_donor_layouts = tuple(donor_layouts[index] for index in selected)
                donor_activation = donor_output.cache[site][selected]
                patch = position_patch(
                    donor_activation, selected_base_layouts, selected_donor_layouts,
                    base_role=label, semantic_label=True)
                patched = raw_forward(
                    net, selected_bases, device=device, interventions={site: patch})
                if traffic_probe is not None:
                    traffic_probe(donor_activation.numel() * donor_activation.element_size()
                                  + patched.logits.numel() * patched.logits.element_size())
                if resource_probe is not None:
                    resource_probe()
                symbol_logits = patched.logits[:, SYMBOL_START:].float()
                clean_logits = base_output.logits[selected, SYMBOL_START:].float()
                if not torch.isfinite(symbol_logits).all() or not torch.isfinite(clean_logits).all():
                    raise FloatingPointError("Nonfinite residual-screen logits")
                predictions = symbol_logits.argmax(-1) + SYMBOL_START
                probabilities = symbol_logits.softmax(-1)
                clean_probabilities = clean_logits.softmax(-1)
                if not torch.isfinite(probabilities).all() or not torch.allclose(
                        probabilities.sum(-1), torch.ones(len(selected), device=device),
                        atol=1e-6, rtol=1e-6):
                    raise FloatingPointError("Intervention probabilities are nonfinite or unnormalized")
                batch_metadata = []
                for local, original in enumerate(selected):
                    item = metadata[original]
                    recipient, target = item["recipient"], item["target"]
                    target_index = target - SYMBOL_START
                    position = selected_base_layouts[local].label_position(label)
                    prediction = int(predictions[local].item())
                    candidates = ({selected_bases[local].query}
                                  | {right for _, right in selected_bases[local].rows})
                    base_prediction = int(
                        clean_logits[local].argmax().item() + SYMBOL_START)
                    other_targets = {candidate["target"] for index, candidate in enumerate(metadata)
                                     if candidate["group_id"] == item["group_id"]
                                     and index != original}
                    if prediction == target:
                        destination = "target"
                    elif prediction == selected_bases[local].answer:
                        destination = "base_answer"
                    elif prediction == item["fixed_donor_answer"]:
                        destination = "fixed_donor_answer"
                    elif prediction in other_targets:
                        destination = "other_recipient_target"
                    elif prediction in candidates:
                        destination = "other_legal_candidate"
                    else:
                        destination = "outside_legal_candidates"
                    row = {
                        "family": family,
                        "replicate": replicate, "cut_index": cut_index, "site": site,
                        "semantic_label": label,
                        "group_id": f"{item['group_id']}:{render}",
                        "logical_group_id": item["group_id"], "recipient": recipient,
                        "render_stratum": render,
                        "position_stratum": ("first_half" if position <
                                             (len(selected_bases[local].tokens) - 3) / 2
                                             else "second_half"),
                        "target": target, "prediction": prediction,
                        "base_prediction": base_prediction,
                        "base_answer": selected_bases[local].answer,
                        "base_correct": base_prediction == selected_bases[local].answer,
                        "preserves_base_prediction": prediction == base_prediction,
                        "preserves_base_answer": prediction == selected_bases[local].answer,
                        "fixed_donor_answer": item["fixed_donor_answer"],
                        "candidate_member": prediction in candidates,
                        "wrong_destination": destination,
                        "target_probability": float(probabilities[local, target_index].item()),
                        "base_target_probability": float(
                            clean_probabilities[local, target_index].item()),
                        "target_logit": float(symbol_logits[local, target_index].item()),
                        "base_target_logit": float(clean_logits[local, target_index].item()),
                        "prediction_logit": float(
                            symbol_logits[local, prediction - SYMBOL_START].item()),
                    }
                    rows.append(row)
                    batch_metadata.append({key: row[key] for key in (
                        "family", "replicate", "cut_index", "site", "semantic_label",
                        "logical_group_id", "recipient", "render_stratum")})
                if logit_sink is not None:
                    logit_sink(batch_metadata, symbol_logits.detach().cpu(),
                               clean_logits.detach().cpu())
    return rows
