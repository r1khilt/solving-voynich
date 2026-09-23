"""CPU smoke and invariance checks for TEACH-0013 confirmation primitives."""

from dataclasses import asdict

import torch

from voynich.workspace.teacher12_models import model_for_arm
from voynich.workspace.teacher13_confirm import (
    MediatorSpec,
    bidirectional_sufficiency,
    identity_error,
    mediator_logits,
    nuisance_preservation,
    summarize_confirmation_rows,
)
from voynich.workspace.teacher13_discovery import episode_from_record
from voynich.workspace.teacher13_tasks import counterfactual_suite


def _groups():
    suite = counterfactual_suite(74111, discovery_groups=1, confirmation_groups=1)
    return [asdict(suite["confirmation"][0])]


def test_single_and_path_self_donors_are_exact_instrumented_identities():
    groups = _groups()
    episodes = tuple(episode_from_record(row) for row in groups[0]["base"])
    torch.manual_seed(41)
    net = model_for_arm("raw_shallow").eval()
    single = MediatorSpec(kind="single", site="blocks.0.resid_post", label="query")
    path = MediatorSpec(kind="path", early_site="embed", source_label="queried_f.right",
                        late_site="blocks.0.resid_post", destination_label="query")
    for spec in (single, path):
        result = identity_error(net, episodes, spec)
        assert result["maximum_identity_logit_error"] < 1e-6
        assert result["qualified"] == (
            result["maximum_fused_instrumented_logit_error"] < 1e-6)
        assert mediator_logits(net, episodes, episodes, spec).shape[0] == 3


def test_bidirectional_sufficiency_rows_keep_full_recipient_denominators():
    groups = _groups()
    torch.manual_seed(42)
    rows = bidirectional_sufficiency(
        model_for_arm("raw_shallow").eval(), groups,
        MediatorSpec(kind="single", site="blocks.0.resid_post", label="query"),
        episode_loader=episode_from_record)
    assert len(rows) == 2 * 3
    assert {(row["direction"], row["recipient"]) for row in rows} == {
        (direction, recipient) for direction in ("forward", "reverse")
        for recipient in range(3)}
    assert all(row["condition"] == "sufficiency_marked" for row in rows)
    assert all(row["fixed_donor_answer"] != row["target"]
               for row in rows if row["recipient"] != 0)
    for direction in ("forward", "reverse"):
        summary = summarize_confirmation_rows(rows, direction=direction)
        assert summary["items"] == 3 and summary["groups"] == 1


def test_nuisance_conditions_preserve_registered_targets_and_group_shape():
    groups = _groups()
    torch.manual_seed(43)
    net = model_for_arm("raw_shallow").eval()
    spec = MediatorSpec(kind="single", site="blocks.0.resid_post", label="query")
    for family in ("format", "order", "distractor"):
        rows = nuisance_preservation(
            net, groups, spec, family=family, episode_loader=episode_from_record)
        assert len(rows) == 3
        assert all(row["condition"] == f"same_key_{family}"
                   and row["target"] == row["base_answer"] for row in rows)
        assert summarize_confirmation_rows(rows)["groups"] == 1
