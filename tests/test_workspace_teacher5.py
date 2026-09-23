import torch

from voynich.workspace.teacher4_models import LearnedMemory
from voynich.workspace.teacher4_tasks import symbolic_oracle, table_partitions
from voynich.workspace.teacher5_intervene import Config, _states, build_groups, decision, summarize


def test_teacher5_groups_are_deterministic_heldout_and_distinct():
    first = build_groups(65111, 4)
    second = build_groups(65111, 4)
    assert first == second
    for group in first:
        assert len(set(group["base_answers"] + group["donor_answers"])) == 6
        for field, answers in (("base", group["base_answers"]),
                               ("donor", group["donor_answers"]),
                               ("direct", group["direct_answers"]),
                               ("copy", group["copy_answers"])):
            assert all(symbolic_oracle(tuple(tokens)) == answer
                       for tokens, answer in zip(group[field], answers, strict=True))
        for tokens in group["base"]:
            t = tuple(tokens)
            assert table_partitions((t[2], t[4]), (t[7], t[9]), (t[8], t[10])) == (
                "holdout", "holdout")
        assert symbolic_oracle(tuple(group["same_key"])) == group["base_key"]


def test_teacher5_base_state_restoration_is_exact():
    torch.manual_seed(7)
    net = LearnedMemory(True).eval()
    groups = build_groups(65111, 2)
    ids = torch.tensor([tokens for group in groups for tokens in group["base"]])
    with torch.no_grad():
        base = _states(net, ids)
        restored = _states(net, ids, first_override=base["first"])
    assert torch.equal(base["logits"], restored["logits"])


def _perfect_rows():
    rows = []
    for group in range(2):
        donor_reference = 40 + group
        for g_index in range(3):
            base, donor = 33 + g_index, 36 + g_index
            row = {"group": group, "g_index": g_index, "eligible_group": True,
                   "base_answer": base, "donor_answer": donor,
                   "donor_reference_answer": donor_reference,
                   "direct_answer": base, "copy_answer": base}
            targets = {
                "clean_base": base, "clean_donor": donor, "donor_first": donor,
                "random_first": base, "zero_first": donor, "same_key_first": base,
                "attention_only": base, "rescue_base_first": base,
                "full_second": donor_reference, "clean_direct": base,
                "patched_direct": base, "clean_copy": base, "patched_copy": base,
            }
            for name, prediction in targets.items():
                row[name] = {"prediction": prediction, "target_probability": .99}
            row["zero_first"]["prediction"] = donor
            row["clean_base"]["donor_target_probability"] = .01
            row["donor_first"]["donor_target_probability"] = .99
            rows.append(row)
    return rows


def test_teacher5_frozen_decision_can_pass_perfect_causal_rows():
    summary = summarize(_perfect_rows())
    verdict = decision({"0": summary, "1": summary}, {"0": 0.0, "1": 0.0})
    assert Config().groups == 128
    assert verdict["verdict"] == "causal_key_supported"
    assert all(all(row.values()) for row in verdict["clauses_by_seed"].values())
