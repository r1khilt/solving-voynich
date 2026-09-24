"""No-model mathematical audit of one TEACH-0015 three-recipient surface.

This validates the actual archived replacement vectors and logits. It does
not evaluate a trained checkpoint, aggregate a split or issue a mechanism
label; those require a separately registered runner and numerical replay.
"""

import math

import torch

from scripts.teacher0015_suite_audit import audit_group


WIDTH = 512
VOCAB_SIZE = 2064
VECTOR_TOLERANCE = 2e-3
LOGIT_TOLERANCE = 2e-3


def _cell(group: dict, f: int, g: int, distractor: int,
          marked: bool, order: int) -> dict:
    found = [cell["episode"] for cell in group["cells"] if (
        cell["f"], cell["g"], cell["distractor"], cell["marked"],
        cell["order"], cell["task"]) == (
            f, g, distractor, marked, order, "composed")]
    if len(found) != 1:
        raise ValueError("Finite-audit counterfactual cell missing")
    return found[0]


def _vector(value: object) -> torch.Tensor:
    if (not isinstance(value, list) or len(value) != WIDTH or
            any(type(item) not in (int, float) or not math.isfinite(item)
                for item in value)):
        raise ValueError("Finite-audit replacement vector malformed")
    return torch.tensor(value, dtype=torch.float64)


def _logits(row: dict, name: str, prediction: int | None) -> list[float] | None:
    values = row.get(f"{name}_logits")
    if prediction is None:
        if values is not None:
            raise ValueError("Unrequested finite-audit logits present")
        return None
    if (not isinstance(values, list) or len(values) != VOCAB_SIZE or
            any(type(value) not in (int, float) or not math.isfinite(value)
                for value in values)):
        raise ValueError("Finite-audit full logits invalid")
    observed = max(range(16, VOCAB_SIZE), key=lambda token: values[token])
    if observed != prediction:
        raise ValueError("Finite-audit prediction/logits disagree")
    return values


def _close(actual: torch.Tensor, expected: torch.Tensor,
           *, label: str) -> None:
    if actual.shape != expected.shape or not bool(torch.allclose(
            actual, expected, atol=VECTOR_TOLERANCE,
            rtol=VECTOR_TOLERANCE)):
        raise ValueError(f"Finite-audit {label} vector differs")


def _scaled(base: torch.Tensor, source: torch.Tensor,
            desired_norm: float) -> torch.Tensor:
    delta = source - base
    return base + delta * desired_norm / max(float(delta.norm().item()), 1e-12)


def audit_surface(group: dict, rows: list[dict], *, distractor: int,
                  marked: bool, order: int,
                  wrong_group: dict | None = None,
                  deranged_group: dict | None = None) -> dict:
    audit_group(group, group["split"])
    for other in (wrong_group, deranged_group):
        if other is not None:
            audit_group(other, other["split"])
            if other["group_id"] == group["group_id"]:
                raise ValueError("Finite-audit control donor reuses target group")
            if other["key1"] == group["key1"]:
                raise ValueError("Finite-audit control donor key is not wrong")
    if (wrong_group is not None and deranged_group is not None and
            wrong_group["group_id"] == deranged_group["group_id"]):
        raise ValueError("Finite-audit wrong and deranged donors coincide")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("Three recipient rows required")
    donor = _cell(group, 1, 0, distractor, marked, order)
    nuisance = _cell(group, 1, 0, distractor, not marked, order)
    wrong_episode = (_cell(wrong_group, 1, 0, distractor, marked, order)
                     if wrong_group is not None else None)
    deranged_episode = (_cell(deranged_group, 1, 0, distractor, marked, order)
                        if deranged_group is not None else None)
    shared_donor = None
    shared_final = None
    shared_nuisance = None
    shared_wrong = None
    shared_deranged = None
    max_identity = 0.0
    max_final = 0.0
    for g, row in enumerate(rows):
        base = _cell(group, 0, g, distractor, marked, order)
        target = _cell(group, 1, g, distractor, marked, order)
        if (row.get("group_id") != group["group_id"] or row.get("g") != g or
                row.get("distractor") != distractor or
                row.get("marked") is not marked or row.get("order") != order or
                row.get("base_render_id") != base["render_id"] or
                row.get("target_render_id") != target["render_id"] or
                row.get("donor_render_id") != donor["render_id"] or
                row.get("same_key_donor_render_id") != nuisance["render_id"] or
                row.get("wrong_donor_render_id") != (
                    wrong_episode["render_id"] if wrong_episode else None) or
                row.get("deranged_donor_render_id") != (
                    deranged_episode["render_id"] if deranged_episode else None) or
                row.get("base_answer") != base["answer"] or
                row.get("target_answer") != target["answer"] or
                row.get("fixed_donor_answer") != donor["answer"]):
            raise ValueError("Finite-audit counterfactual identity/answer differs")
        for name in ("base", "target", "donor", "transfer", "same_key",
                     "reverse", "final_donor", "random", "wrong_key",
                     "deranged"):
            prediction = row.get(f"{name}_prediction")
            if name in ("wrong_key", "deranged") and (
                    (wrong_group if name == "wrong_key" else deranged_group)
                    is None):
                if prediction is not None:
                    raise ValueError("Unrequested finite-audit control prediction")
            elif type(prediction) is not int or not 16 <= prediction < VOCAB_SIZE:
                raise ValueError("Finite-audit ordinary prediction invalid")
            _logits(row, name, prediction)
        vectors = row.get("replacement_vectors")
        if not isinstance(vectors, dict) or set(vectors) != {
                "base_native", "target_native", "donor_native",
                "same_key_native", "reverse_base", "final_donor", "random",
                "wrong_key", "wrong_source_native", "deranged",
                "deranged_source_native"}:
            raise ValueError("Finite-audit replacement vector set incomplete")
        base_state = _vector(vectors["base_native"])
        donor_state = _vector(vectors["donor_native"])
        _vector(vectors["target_native"])
        nuisance_state = _vector(vectors["same_key_native"])
        final_state = _vector(vectors["final_donor"])
        _close(_vector(vectors["reverse_base"]), base_state,
               label="reverse base")
        if shared_donor is None:
            shared_donor = donor_state
            shared_final = final_state
            shared_nuisance = nuisance_state
        else:
            _close(donor_state, shared_donor, label="shared donor")
            _close(final_state, shared_final, label="shared final donor")
            _close(nuisance_state, shared_nuisance, label="shared nuisance")
        norm = float((donor_state - base_state).norm().item())
        if (type(row.get("donor_delta_norm")) not in (int, float) or
                not math.isfinite(row["donor_delta_norm"]) or
                abs(row["donor_delta_norm"] - norm) > VECTOR_TOLERANCE):
            raise ValueError("Finite-audit donor delta norm differs")
        if abs(float((_vector(vectors["random"]) - base_state).norm()) - norm
               ) > VECTOR_TOLERANCE:
            raise ValueError("Finite-audit random norm mismatch")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(group["group_id"][:16], 16) ^
                              (g << 16) ^ (distractor << 8) ^
                              (int(marked) << 4) ^ order)
        random_delta = torch.randn((1, WIDTH), generator=generator)[0].double()
        expected_random = base_state + random_delta * norm / random_delta.norm()
        _close(_vector(vectors["random"]), expected_random, label="random")
        for name, other in (("wrong_key", wrong_group),
                            ("deranged", deranged_group)):
            source_name = ("wrong_source_native" if name == "wrong_key" else
                           "deranged_source_native")
            if other is None:
                if vectors[name] is not None or vectors[source_name] is not None:
                    raise ValueError("Unrequested finite-audit replacement vector")
                continue
            source = _vector(vectors[source_name])
            replacement = _vector(vectors[name])
            _close(replacement, _scaled(base_state, source, norm), label=name)
            if name == "wrong_key":
                if shared_wrong is None:
                    shared_wrong = source
                else:
                    _close(source, shared_wrong, label="shared wrong source")
            elif shared_deranged is None:
                shared_deranged = source
            else:
                _close(source, shared_deranged, label="shared deranged source")
        identity = row.get("identity_max_abs_logit_error")
        final_error = row.get("final_donor_max_abs_logit_error")
        if (type(identity) not in (int, float) or
                type(final_error) not in (int, float) or
                not 0 <= identity <= LOGIT_TOLERANCE or
                not 0 <= final_error <= LOGIT_TOLERANCE):
            raise ValueError("Finite-audit identity/final donor control differs")
        actual_final = max(abs(a - b) for a, b in zip(
            row["final_donor_logits"], row["donor_logits"], strict=True))
        if (abs(actual_final - final_error) > LOGIT_TOLERANCE or
                row["final_donor_prediction"] != row["donor_prediction"]):
            raise ValueError("Finite-audit final donor logits differ")
        max_identity = max(max_identity, identity)
        max_final = max(max_final, final_error)
    return {"audit": "pass", "scope": "single_three_recipient_surface_math",
            "group_id": group["group_id"], "recipient_rows": len(rows),
            "max_identity_error": max_identity,
            "max_final_donor_error": max_final,
            "mechanism_decision": "not_evaluated"}
