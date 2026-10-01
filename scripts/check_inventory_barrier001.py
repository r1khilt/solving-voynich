"""Exact witness: code swaps plus single replacements need not connect support."""

from __future__ import annotations

import argparse
import itertools
import json
from fractions import Fraction as F

from scripts.check_prefix_bridge001 import CONTEXTS,enumerated
from scripts.run_source_state_systems001 import ROOT,artifact,save_new


def components(matrix):
    remaining = set(range(len(matrix)))
    groups = []
    while remaining:
        pending = [min(remaining)]
        group = set()
        while pending:
            i = pending.pop()
            if i in group:
                continue
            group.add(i)
            pending.extend(j for j,p in enumerate(matrix[i]) if p and j not in group)
        assert group<=remaining
        remaining -= group
        groups.append(sorted(group))
    return groups


def check():
    all_keys = list(itertools.product(range(6),repeat=2))
    values = [enumerated((0,1),k,1,CONTEXTS,closed=True)*
              enumerated((1,0),k,2,CONTEXTS,closed=True) for k in all_keys]
    supported = [k for k,v in zip(all_keys,values,strict=True) if v]
    masses = [v for v in values if v]
    assert supported==[(0,1),(1,0),(3,4),(4,3)]
    kernels = {}
    for kind in ('row','row_plus_swap','whole_pair'):
        matrix = [[F(0) for _ in supported] for _ in supported]
        for i,a in enumerate(supported):
            for j,b in enumerate(supported):
                if i==j:
                    continue
                single = F(1,12) if sum(x!=y for x,y in zip(a,b,strict=True))==1 else F(0)
                swap = F(int(a[::-1]==b))
                q = single if kind=='row' else (single+swap)/2 if kind=='row_plus_swap' else F(1,36)
                matrix[i][j] = q*min(F(1),masses[j]/masses[i])
            matrix[i][i] = 1-sum(matrix[i])
        assert all(sum(row)==1 for row in matrix)
        for i in range(4):
            for j in range(4):
                assert masses[i]*matrix[i][j]==masses[j]*matrix[j][i]
        groups = components(matrix)
        kernels[kind] = {'components':[[list(supported[i]) for i in group] for group in groups],
            'matrix':[[str(p) for p in row] for row in matrix], 'exact_balance_pairs':16}
    assert len(kernels['row']['components'])==4
    assert len(kernels['row_plus_swap']['components'])==2
    assert len(kernels['whole_pair']['components'])==1
    return {'status':'PASS_exact_inventory_barrier_witness','cipher':[[0,1],[1,0]],
        'strictly_positive_source':True,'keys_enumerated':36,'supported_keys':[list(k) for k in supported],
        'supported_likelihoods':[str(v) for v in masses],'kernels':kernels,
        'inputs':[artifact(ROOT/'scripts/check_prefix_bridge001.py'),artifact(ROOT/'scripts/check_inventory_barrier001.py')],
        'scope':'Own finite rational mathematics, not a proposed sampler run or large-case disconnection claim',
        'empirical_panel_calls':0,'paid_spend_usd':0}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',action='store_true')
    args = parser.parse_args()
    result = check()
    if args.save:
        save_new(ROOT/'results/INVENTORY-BARRIER-THEORY-001/result.json',result)
    print(json.dumps(result,indent=2))
