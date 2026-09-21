"""Explicit neural interventions and supervised finite transition-table extraction.

These operations concern a modern model or already-labeled synthetic states. They
do not discover semantic labels, prove a causal abstraction, or identify a historical
writing process. See MECHANISMS_AND_IDENTIFIABILITY.md in the 2026-09-21 dossier.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import json

import torch
from torch import Tensor

from .evidence import _freeze_json, _thaw


def _hidden_pair(base, donor):
    if (
        not isinstance(base, Tensor) or not isinstance(donor, Tensor) or base.ndim != 3
        or min(base.shape) == 0 or base.shape != donor.shape or base.device != donor.device
        or base.dtype != donor.dtype or not base.is_floating_point()
    ):
        raise ValueError("hidden states must be matching nonempty floating [batch, position, dimension] tensors")
    if not bool(torch.isfinite(base).all()) or not bool(torch.isfinite(donor).all()):
        raise ValueError("hidden states must be finite")


def _indices(values, size, name):
    if isinstance(values, Tensor):
        if values.dtype != torch.long or values.ndim != 1:
            raise ValueError(f"{name} indices must be a rank-1 torch.long tensor")
        values = values.detach().cpu().tolist()
    else:
        try:
            values = list(values)
        except TypeError as exc:
            raise ValueError(f"{name} must be indices or a boolean mask") from exc
    if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < size for value in values):
        raise ValueError(f"{name} contains an invalid index")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate indices")
    return values


def _selection(base, positions, dimensions):
    batch, length, width = base.shape
    if isinstance(positions, Tensor) and positions.dtype == torch.bool:
        if positions.shape not in ((length,), (batch, length)):
            raise ValueError("positions mask must be [length] or [batch, length]")
        selected_positions = positions.to(base.device).expand(batch, length)
    else:
        selected_positions = torch.zeros((batch, length), device=base.device, dtype=torch.bool)
        selected_positions[:, _indices(positions, length, "positions")] = True
    if dimensions is None:
        selected_dimensions = torch.ones(width, device=base.device, dtype=torch.bool)
    elif isinstance(dimensions, Tensor) and dimensions.dtype == torch.bool:
        if dimensions.shape != (width,):
            raise ValueError("dimensions mask must be [dimension]")
        selected_dimensions = dimensions.to(base.device)
    else:
        selected_dimensions = torch.zeros(width, device=base.device, dtype=torch.bool)
        selected_dimensions[_indices(dimensions, width, "dimensions")] = True
    return selected_positions[..., None] & selected_dimensions


def patch_hidden(base: Tensor, donor: Tensor, positions, dimensions=None) -> Tensor:
    """Replace exactly selected coordinates without mutating inputs or detaching gradients.

    Position indices apply to every batch row; a bool [batch, length] mask permits
    per-example selection. Dimension indices or bool [width] masks are shared.
    Empty selections are valid identity interventions; negative/duplicate indices
    are rejected to keep the declared intervention unambiguous.
    """
    _hidden_pair(base, donor)
    return torch.where(_selection(base, positions, dimensions), donor, base)


def intervention_controls(base: Tensor, donor: Tensor, positions, dimensions=None, seed=0) -> dict[str, Tensor]:
    """Identity, targeted donor, entire donor, and matched random-delta controls.

    The random intervention changes only the selected coordinates. Its L2 change
    from base matches the targeted change separately for each batch example. The
    random direction uses a private CPU generator and does not change global RNG.
    """
    _hidden_pair(base, donor)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    selected = _selection(base, positions, dimensions)
    targeted = torch.where(selected, donor, base)
    working_dtype = torch.float64 if base.dtype == torch.float64 else torch.float32
    target_delta = targeted.to(working_dtype) - base.to(working_dtype)
    random_delta = torch.randn(
        base.shape, generator=torch.Generator().manual_seed(seed), dtype=working_dtype,
    ).to(base.device)
    random_delta = random_delta.masked_fill(~selected, 0)
    target_norm = torch.linalg.vector_norm(target_delta.flatten(1), dim=-1)
    random_norm = torch.linalg.vector_norm(random_delta.flatten(1), dim=-1)
    scale = target_norm / random_norm.clamp_min(torch.finfo(random_delta.dtype).tiny)
    random_hidden = base + (random_delta * scale[:, None, None]).to(base.dtype)
    if not bool(torch.isfinite(random_hidden).all()):
        raise ValueError("norm-matched random intervention exceeds the floating point range")
    return {"identity": base.clone(), "targeted": targeted, "full_donor": donor.clone(),
            "norm_matched_random": random_hidden}


def _probabilities(values, input_kind):
    if not isinstance(values, Tensor) or not values.is_floating_point() or values.ndim < 1 or values.numel() == 0:
        raise ValueError("distributions must be nonempty floating tensors with categories on the last axis")
    values = values.detach().cpu().double()
    if input_kind == "logits":
        if bool(torch.isnan(values).any()) or bool(torch.isposinf(values).any()):
            raise ValueError("logits may be finite or negative infinity, never NaN or positive infinity")
        if not bool(torch.isfinite(values).any(-1).all()):
            raise ValueError("each logit row needs at least one finite category")
        return values.softmax(-1)
    if input_kind != "probabilities":
        raise ValueError("input_kind must be 'probabilities' or 'logits'")
    if not bool(torch.isfinite(values).all()) or bool(((values < 0) | (values > 1)).any()):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    if not torch.allclose(values.sum(-1), torch.ones_like(values.sum(-1)), rtol=1e-6, atol=1e-7):
        raise ValueError("probability rows must be normalized")
    return values


def _entropy(probabilities):
    return -torch.where(probabilities > 0, probabilities * probabilities.log(), 0).sum(-1)


def _kl(left, right):
    # Preserve infinite KL when positive mass meets zero support; epsilon smoothing
    # would turn a support violation into a silently different scientific quantity.
    terms = torch.where(left > 0, left * (left.log() - right.log()), 0)
    return terms.sum(-1)


def compare_distributions(base: Tensor, changed: Tensor, input_kind="probabilities") -> dict:
    """Compare matching categorical distributions, with KL/JS in nats.

    KL is genuinely infinite if support is lost; callers exporting such a result
    to strict JSON must explicitly represent that case. JS and total variation
    remain finite, including disjoint one-hot distributions.
    """
    if not isinstance(base, Tensor) or not isinstance(changed, Tensor) or base.shape != changed.shape:
        raise ValueError("base and changed distributions must have the same shape")
    left, right = _probabilities(base, input_kind), _probabilities(changed, input_kind)
    mixture = (left + right) / 2
    return {
        "distributions": left.numel() // left.shape[-1],
        "mean_kl_base_to_changed": float(_kl(left, right).mean()),
        "mean_kl_changed_to_base": float(_kl(right, left).mean()),
        "mean_js_divergence": float(((_kl(left, mixture) + _kl(right, mixture)) / 2).mean()),
        "mean_total_variation": float((0.5 * (left - right).abs().sum(-1)).mean()),
        "argmax_agreement": float((left.argmax(-1) == right.argmax(-1)).double().mean()),
        "mean_entropy_base": float(_entropy(left).mean()),
        "mean_entropy_changed": float(_entropy(right).mean()),
        "max_absolute_change": float((left - right).abs().max()),
    }


@torch.no_grad()
def causal_report(
    model, noisy, condition, noise_level, *, positions, dimensions=None,
    donor_condition=None, donor_noisy=None, latent_types=None, allowed=None, seed=0,
) -> dict:
    """Measure final-hidden interventions on one paired denoiser input batch.

    This is a deterministic, fixed-noise neural-behavior audit in evaluation mode,
    not a demonstration of historical semantics or repeated-seed stability. Donor
    inputs use the same noise level, slot types, and padding as base. Effects include
    all active output slots, with selected/unselected positions reported separately.
    """
    donor_condition = condition if donor_condition is None else donor_condition
    donor_noisy = noisy if donor_noisy is None else donor_noisy
    if donor_noisy.shape != noisy.shape or not torch.equal(donor_noisy == 0, noisy == 0):
        raise ValueError("donor_noisy must preserve base shape and structural padding")
    active = noisy != 0
    if not bool(active.any()):
        raise ValueError("causal_report needs at least one active latent position")
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        baseline, base_hidden = model(noisy, condition, noise_level, latent_types=latent_types, return_hidden=True)
        donor_logits, donor_hidden = model(
            donor_noisy, donor_condition, noise_level, latent_types=latent_types, return_hidden=True,
        )
        selection = _selection(base_hidden, positions, dimensions)
        selected_positions = selection.any(-1) & active
        if allowed is not None:
            if (
                not isinstance(allowed, Tensor) or allowed.dtype != torch.bool
                or allowed.shape != baseline.shape or allowed.device != baseline.device
                or bool((active & ~allowed.any(-1)).any())
            ):
                raise ValueError("allowed must provide nonempty boolean output support on each active slot")

        def supported(logits):
            return logits if allowed is None else logits.masked_fill(~allowed, -torch.inf)

        baseline, donor_logits = supported(baseline), supported(donor_logits)
        controls = intervention_controls(base_hidden, donor_hidden, positions, dimensions, seed)
        reports = {}
        for name, hidden in controls.items():
            changed = supported(model(
                noisy, condition, noise_level, latent_types=latent_types, hidden_patch=lambda _h: hidden,
            ))
            report = {"against_base": compare_distributions(baseline[active], changed[active], "logits"),
                      "against_donor": compare_distributions(donor_logits[active], changed[active], "logits")}
            for label, mask in (("selected_positions", selected_positions),
                                ("other_positions", active & ~selected_positions)):
                report[label] = compare_distributions(baseline[mask], changed[mask], "logits") if mask.any() else None
            reports[name] = report
        return {"claim": "paired model-behavior intervention only; not identified historical semantics",
                "seed": seed, "noise_level": noise_level.detach().cpu().tolist(),
                "active_positions": int(active.sum()), "selected_coordinates": int(selection[active].sum()),
                "base_to_donor": compare_distributions(baseline[active], donor_logits[active], "logits"),
                "controls": reports}
    finally:
        for module, training in modes:
            module.training = training


def _canonical(value):
    return json.dumps(_thaw(_freeze_json(value)), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _triples(examples):
    rows = []
    for example in examples:
        if not isinstance(example, (tuple, list)) or len(example) != 3:
            raise ValueError("transition examples must be (state, action, next_state) triples")
        rows.append(tuple(_freeze_json(value) for value in example))
    return rows


class TransitionConflictError(ValueError):
    def __init__(self, conflicts):
        self.conflicts = conflicts
        super().__init__(f"conflicting labeled transitions for {len(conflicts)} state/action pairs")


@dataclass(frozen=True)
class TransitionProgram:
    """A lookup executor over supplied state/action labels, with no induced semantics."""

    rules: tuple
    training_examples: int
    conflicts: tuple = ()

    def __post_init__(self):
        if (
            isinstance(self.training_examples, bool) or not isinstance(self.training_examples, int)
            or self.training_examples < 1
        ):
            raise ValueError("training_examples must be a positive integer")
        rules = tuple(_triples(self.rules))
        keys = [(_canonical(state), _canonical(action)) for state, action, _next_state in rules]
        if len(set(keys)) != len(keys):
            raise ValueError("transition program contains duplicate or conflicting rules")
        if len(rules) > self.training_examples:
            raise ValueError("more rules than training examples")
        conflicts = tuple(_freeze_json(conflict) for conflict in self.conflicts)
        conflict_keys, conflict_examples = set(), 0
        for conflict in conflicts:
            if not isinstance(conflict, Mapping) or set(conflict) != {"state", "action", "outcomes"}:
                raise ValueError("malformed transition conflict")
            if not isinstance(conflict["outcomes"], tuple) or len(conflict["outcomes"]) < 2:
                raise ValueError("a conflict requires at least two outcomes")
            key = (_canonical(conflict["state"]), _canonical(conflict["action"]))
            if key in keys:
                raise ValueError("conflicted state/action pairs cannot have executable rules")
            if key in conflict_keys:
                raise ValueError("duplicate transition conflict")
            conflict_keys.add(key)
            outcomes = set()
            for outcome in conflict["outcomes"]:
                if not isinstance(outcome, Mapping) or set(outcome) != {"next_state", "count"}:
                    raise ValueError("malformed conflict outcome")
                if isinstance(outcome["count"], bool) or not isinstance(outcome["count"], int) or outcome["count"] < 1:
                    raise ValueError("conflict outcome counts must be positive integers")
                encoded = _canonical(outcome["next_state"])
                if encoded in outcomes:
                    raise ValueError("duplicate outcome in transition conflict")
                outcomes.add(encoded)
                conflict_examples += outcome["count"]
        if len(rules) + conflict_examples > self.training_examples:
            raise ValueError("rules and conflicts exceed the declared training examples")
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "conflicts", conflicts)

    def predict(self, state, action) -> dict:
        key = (_canonical(state), _canonical(action))
        for source, operation, destination in self.rules:
            if (_canonical(source), _canonical(operation)) == key:
                return {"known": True, "next_state": _thaw(destination)}
        return {"known": False, "next_state": None}

    def to_dict(self) -> dict:
        return {"schema_version": 1, "kind": "observed_transition_table", "training_examples": self.training_examples,
                "rules": [{"state": _thaw(s), "action": _thaw(a), "next_state": _thaw(n)} for s, a, n in self.rules],
                "conflicts": [_thaw(conflict) for conflict in self.conflicts]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or set(value) != {"schema_version", "kind", "training_examples", "rules", "conflicts"}:
            raise ValueError("invalid transition-program fields")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["kind"] != "observed_transition_table":
            raise ValueError("unsupported transition-program schema")
        if not isinstance(value["rules"], list) or not isinstance(value["conflicts"], list):
            raise ValueError("rules and conflicts must be serialized lists")
        rules = []
        for item in value["rules"]:
            if not isinstance(item, Mapping) or set(item) != {"state", "action", "next_state"}:
                raise ValueError("invalid serialized transition rule")
            rules.append((item["state"], item["action"], item["next_state"]))
        return cls(tuple(rules), value["training_examples"], tuple(value["conflicts"]))


def extract_transition_program(examples, *, on_conflict="raise") -> TransitionProgram:
    """Compile labeled transitions; conflicts raise or remain explicitly unmodeled.

    Reporting conflicts never substitutes majority votes or invents transitions.
    A held-out evaluation must be supplied independently by the caller.
    """
    if on_conflict not in {"raise", "report"}:
        raise ValueError("on_conflict must be 'raise' or 'report'")
    rows = _triples(examples)
    if not rows:
        raise ValueError("at least one labeled transition is required")
    groups = {}
    for state, action, next_state in rows:
        key = (_canonical(state), _canonical(action))
        group = groups.setdefault(key, {"state": state, "action": action, "outcomes": {}})
        encoded = _canonical(next_state)
        outcome = group["outcomes"].setdefault(encoded, {"next_state": next_state, "count": 0})
        outcome["count"] += 1
    rules, conflicts = [], []
    for key in sorted(groups):
        group = groups[key]
        if len(group["outcomes"]) == 1:
            rules.append((group["state"], group["action"], next(iter(group["outcomes"].values()))["next_state"]))
        else:
            conflicts.append({"state": _thaw(group["state"]), "action": _thaw(group["action"]),
                              "outcomes": [_thaw(group["outcomes"][key]) for key in sorted(group["outcomes"])]})
    if conflicts and on_conflict == "raise":
        raise TransitionConflictError(conflicts)
    return TransitionProgram(tuple(rules), len(rows), tuple(conflicts))


def evaluate_transition_program(program: TransitionProgram, examples) -> dict:
    """Report held-out coverage and accuracy without treating unknown as a prediction."""
    if not isinstance(program, TransitionProgram):
        raise ValueError("program must be a TransitionProgram")
    rows = _triples(examples)
    if not rows:
        raise ValueError("evaluation requires at least one transition")
    predictions, known, correct = [], 0, 0
    for index, (state, action, destination) in enumerate(rows):
        prediction = program.predict(state, action)
        matches = _canonical(prediction["next_state"]) == _canonical(destination) if prediction["known"] else None
        known += int(prediction["known"])
        correct += int(matches is True)
        predictions.append({"index": index, "state": _thaw(state), "action": _thaw(action),
                            "expected": _thaw(destination), "status": "known" if prediction["known"] else "unknown",
                            "predicted": prediction["next_state"], "correct": matches})
    return {"examples": len(rows), "known": known, "unknown": len(rows) - known, "correct": correct,
            "coverage": known / len(rows), "accuracy_on_known": correct / known if known else None,
            "accuracy_overall": correct / len(rows), "predictions": predictions,
            "claim": "evaluation of a table extracted from supplied labels, not generic semantic discovery"}
