"""Connect compiled procedure candidates to a separately trained action model.

This score is a learned synthetic-domain consistency prior. It is NOT independent
manuscript evidence, and it must never be inserted as another observation in the
Bayesian evidence ledger.
"""

from copy import deepcopy
import hashlib
from pathlib import Path

import torch

from .dynamics import ActionWorldModel, DynamicsConfig
from .worlds import PROCEDURE_OPERATORS


def load_action_model(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported action checkpoint")
    model = ActionWorldModel(DynamicsConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device), payload


def _state(nodes):
    ids = [node["entity"] for node in nodes]
    if ids != list(range(len(nodes))):
        raise ValueError("Witness states must follow contiguous entity order")
    return [0 if not node["full"] else 2 if node["hot"] else 1 for node in nodes]


@torch.no_grad()
def score_action_candidates(report, model):
    output = deepcopy(report)
    device = next(model.parameters()).device
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        for candidate in output["candidates"]:
            semantics = candidate["verification"]["semantics"]
            if semantics["status"] != "executable_procedure":
                candidate["action_model"] = {"status": "not_applicable"}
                continue
            witness = semantics["witness"]
            trace = witness["action_trace"]
            kinds = [0 if entity["kind"] == "agent" else 1 for entity in witness["entities"]]
            if len(kinds) != model.config.entities or not trace:
                raise ValueError("Action witness is incompatible with checkpoint")
            states = torch.tensor([_state(step["before"]) for step in trace], device=device)
            targets = torch.tensor([_state(step["after"]) for step in trace], device=device)
            actions = torch.tensor(
                [
                    [
                        PROCEDURE_OPERATORS.index(step["action"]["kind"]),
                        step["action"]["subject"],
                        step["action"]["object"],
                    ]
                    for step in trace
                ],
                device=device,
            )
            type_ids = torch.tensor([kinds] * len(trace), device=device)
            logits, validity = model(states, actions, type_ids)
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(validity).all()):
                raise FloatingPointError("Nonfinite action prior")
            candidate["action_model"] = {
                "status": "scored",
                "steps": len(trace),
                "mean_next_state_log_probability": float(
                    logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).mean()
                ),
                "mean_applicability_probability": float(validity.sigmoid().mean()),
                "exact_transition_prediction_fraction": float(
                    logits.argmax(-1).eq(targets).all(-1).float().mean()
                ),
                "role": "synthetic-domain prior diagnostic; not independent evidence; ranking unchanged",
            }
    finally:
        for module, training in modes:
            module.training = training
    return output


def attach_action_prior(report, checkpoint, device="cpu"):
    model, _ = load_action_model(checkpoint, device)
    result = score_action_candidates(report, model)
    result["action_checkpoint_sha256"] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    return result
