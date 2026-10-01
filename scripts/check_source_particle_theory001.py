"""Finite rational algebra checks, not an empirical particle/recovery run."""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from fractions import Fraction as F

from scripts.benchmark_source_prefix_systems001 import (
    full_key_reference, rational_row,
)
from scripts.run_blind_channel_dev004 import save_new
from scripts.run_latin_source_model001 import ROOT, artifact


def enumerate_local_proposal(cipher, schedule):
    """All outcomes of one locally adapted particle, including extinction.

    Enumerates probabilities exactly; never samples or calls a trained model.
    The full-dictionary/string reference uses a different counting algorithm.
    """
    outcomes, proposals, leaves = defaultdict(F), F(0), set()
    nodes, maximum_actions = 0, 0
    horizon = sum(map(len, cipher))+len(cipher)

    def visit(texts, offsets, closed, key, proposal, weight, actions):
        nonlocal proposals, nodes, maximum_actions
        nodes += 1
        maximum_actions = max(maximum_actions, actions)
        assert actions <= horizon
        if all(closed):
            assert (texts, key) not in leaves
            leaves.add((texts, key))
            outcomes[texts] += proposal*weight
            proposals += proposal
            return
        opened = [i for i, done in enumerate(closed) if not done]
        record = (opened[0] if schedule == "sequential" else
                  min(opened, key=lambda i: (F(offsets[i], len(cipher[i])), i)))
        if offsets[record] == len(cipher[record]):
            changed = list(closed)
            changed[record] = True
            visit(texts, offsets, tuple(changed), key, proposal, weight*F(1, 4), actions+1)
            return
        choices = []
        for row, probability in enumerate(rational_row(texts[record])):
            existing = key[row]
            for length in ((len(existing),) if existing is not None else (1, 2)):
                end = offsets[record]+length
                if end > len(cipher[record]):
                    continue
                unit = cipher[record][offsets[record]:end]
                if existing is not None and existing != unit:
                    continue
                updated = list(key)
                updated[row] = unit
                new_texts, new_offsets = list(texts), list(offsets)
                new_texts[record] += (row,)
                new_offsets[record] = end
                coefficient = F(3, 4)*probability*(F(1, 6) if existing is None else 1)
                choices.append((tuple(new_texts), tuple(new_offsets), tuple(updated), coefficient))
        local = sum((choice[-1] for choice in choices), F(0))
        if not local:
            # Explicit failure outcome: q remains a normalized probability law.
            proposals += proposal
            return
        for new_texts, new_offsets, updated, coefficient in choices:
            visit(new_texts, new_offsets, closed, updated,
                  proposal*coefficient/local, weight*local, actions+1)

    visit(((),)*len(cipher), (0,)*len(cipher), (False,)*len(cipher),
          (None, None), F(1), F(1), 0)
    assert proposals == 1
    return dict(outcomes), len(leaves), nodes, maximum_actions


def check():
    observed = ((0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1))
    total_nodes, total_leaves = 0, 0
    for cipher in itertools.product(observed, repeat=2):
        reference = full_key_reference(cipher)
        values = []
        for schedule in ("sequential", "balanced"):
            values.append(enumerate_local_proposal(cipher, schedule))
            actual, leaves, nodes, _ = values[-1]
            assert actual == reference
            total_nodes += nodes
            total_leaves += leaves
        assert values[0][1] == values[1][1]
    # Fixed two-stage tree: root A/B coefficients 1/2, then 1/4 or 3/4.
    target_a, target_b = F(1, 8), F(3, 8)
    evidence = target_a+target_b
    assert evidence == F(1, 2)
    assert target_a/evidence == F(1, 4)
    assert F(1, 2)*F(1, 4)+F(1, 2)*F(3, 4) == evidence
    # Two particles: first-stage count of A has Binomial(2,1/2) law.
    expected_z, expected_a, weighted_a = F(0), F(0), F(0)
    for count, probability in enumerate((F(1, 4), F(1, 2), F(1, 4))):
        z = (count*F(1, 4)+(2-count)*F(3, 4))/2
        posterior_a = count*F(1, 4)/(2*z)
        expected_z += probability*z
        expected_a += probability*posterior_a
        weighted_a += probability*z*posterior_a
    assert (expected_z, expected_a, weighted_a) == (evidence, F(3, 8), target_a)
    result = {"scope": "Exact finite rational algebra, no empirical SMC or recovery",
              "cipher_pairs": 36, "schedules_per_pair": 2,
              "schedule_probability_support_checks": "PASS",
              "one_particle_weighted_reading_measure_checks": "PASS",
              "normalization_including_extinction": "PASS",
              "enumerated_nodes": total_nodes, "enumerated_leaves": total_leaves,
              "fixed_tree": {"evidence": str(evidence), "true_posterior_a": "1/4",
                  "uncorrected_local_a": "1/2", "mean_two_particle_posterior_a": str(expected_a),
                  "mean_two_particle_evidence": str(expected_z),
                  "mean_evidence_times_posterior_a": str(weighted_a)},
              "source": artifact(ROOT/"scripts/check_source_particle_theory001.py")}
    save_new(ROOT/"results/SOURCE-PARTICLE-THEORY-001/check.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    check()
