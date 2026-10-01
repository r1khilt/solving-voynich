"""Exact finite target/kernel/normalizer proofs, without corpus model calls."""

from __future__ import annotations

import argparse
import itertools
import json
from fractions import Fraction as F

from scripts.check_prefix_bridge001 import CONTEXTS, enumerated
from scripts.run_source_state_systems001 import ROOT, save_new


def proposal(a,b,free,kernel):
    if not free:
        return F(a==b)
    pieces = [(1,F(1))] if kernel=="row" else [(1,F(3,4)),(min(2,len(free)),F(3,16)),(len(free),F(1,16))]
    probability = F(0)
    for size,mix in pieces:
        subsets = tuple(itertools.combinations(free,size))
        probability += mix*sum((F(1,6**size) for subset in subsets
            if all(a[r]==b[r] for r in range(2) if r not in subset)),F(0))/len(subsets)
    return probability


def transition(keys,values,free,kernel):
    result = []
    for i,a in enumerate(keys):
        row = [F(0)]*len(keys)
        if values[i]:
            for j,b in enumerate(keys):
                if i!=j:
                    row[j] = proposal(a,b,free,kernel)*min(F(1),values[j]/values[i])
        row[i] = 1-sum(row,F(0))
        result.append(row)
    return result


def targets(key,cipher,stage,rows):
    cuts,closed = stage
    return enumerated(cipher[0][:cuts[0]],key,0,rows,closed=closed)*enumerated(
        cipher[1][:cuts[1]],key,0,rows,closed=closed)


def check():
    matrices = flows = transports = one_particle_laws = 0
    full_keys = tuple(itertools.product(range(6),repeat=2))
    stages = (((1,0),False),((1,1),False),((2,1),False),((2,2),False),((2,2),True))
    witnesses = []
    minimum_joint_supported_transition = F(1)
    for rows in (CONTEXTS,((F(1),F(0)),)*3):
        for cipher in (((0,1),(1,0)),((0,0),(0,0))):
            full_values = [[targets(k,cipher,stage,rows) for k in full_keys] for stage in stages]
            for partial in itertools.product(range(-1,6),repeat=2):
                free = tuple(r for r,k in enumerate(partial) if k<0)
                allowed = [i for i,k in enumerate(full_keys) if all(p<0 or k[r]==p for r,p in enumerate(partial))]
                keys = tuple(full_keys[i] for i in allowed)
                for kernel in ("row","joint"):
                    prior = F(1,len(keys))
                    previous = [F(1)]*len(keys)
                    first_transition = first_values = None
                    for step,full in enumerate(full_values):
                        values = [full[i] for i in allowed]
                        matrix = transition(keys,values,free,kernel)
                        for i in range(len(keys)):
                            assert sum(matrix[i],F(0))==1 and all(v>=0 for v in matrix[i])
                            assert values[i]==0 or previous[i]>0
                            for j in range(len(keys)):
                                assert proposal(keys[i],keys[j],free,kernel)==proposal(keys[j],keys[i],free,kernel)
                                assert values[i]*matrix[i][j]==values[j]*matrix[j][i]
                                flows += 1
                                if kernel=="joint" and values[i] and values[j]:
                                    assert matrix[i][j]>0
                                    minimum_joint_supported_transition = min(minimum_joint_supported_transition,matrix[i][j])
                        for j in range(len(keys)):
                            transported = sum((previous[i]*(values[i]/previous[i] if previous[i] else F(0))*matrix[i][j]
                                for i in range(len(keys))),F(0))
                            assert transported==values[j]
                            transports += 1
                        matrices += 1
                        if step==0:
                            first_transition,first_values = matrix,values
                        previous = values
                    # N=1, two stages: expectation includes every initial key,
                    # first-stage mutation and explicit zero/extinction paths.
                    final_values = [full_values[-1][i] for i in allowed]
                    expected_estimator = sum((prior*first_values[i]*first_transition[i][j]
                        * (final_values[j]/first_values[j] if first_values[j] else F(0))
                        for i in range(len(keys)) for j in range(len(keys))),F(0))
                    exact_evidence = prior*sum(final_values,F(0))
                    assert expected_estimator==exact_evidence
                    one_particle_laws += 1
                    if partial==(-1,-1):
                        wrong = sum((prior*first_transition[i][j]*first_values[j]
                            for i in range(len(keys)) for j in range(len(keys))),F(0))
                        correct = prior*sum(first_values,F(0))
                        if wrong!=correct:
                            witnesses.append({"cipher":cipher,"source":"contextual" if rows==CONTEXTS else "zero_transition",
                                "kernel":kernel,"correct_weight_before_move":str(correct),
                                "incorrect_move_before_weight":str(wrong)})
    assert witnesses
    return {"status":"PASS_exact_finite_dictionary_smc_laws","target_kernel_matrices":matrices,
        "detailed_balance_key_pairs":flows,"weighted_transport_coordinates":transports,
        "one_particle_two_stage_evidence_expectations":one_particle_laws,
        "minimum_joint_positive_support_transition":str(minimum_joint_supported_transition),
        "wrong_operation_order_witnesses":witnesses,
        "scope":"Exact two-row/six-unit finite models: all49conditionalkeys, two2recordobservations, fiveprefix/EOSstages, contextual/zero-transition sources, row and joint mixture kernels. No particle performance, Latin corpus, historical evidence or large-state mixing guarantee.",
        "empirical_panel_calls":0,"paid_spend_usd":0}


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save",action="store_true")
    args = parser.parse_args()
    result = check()
    if args.save:
        save_new(ROOT/"results/DICTIONARY-SMC-THEORY-001/result.json",result)
    print(json.dumps(result,indent=2,allow_nan=False))
