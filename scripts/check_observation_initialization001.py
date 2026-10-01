"""Exact counting of first-prefix support, without a new sampler or corpus call."""

import argparse
import itertools
import json
from fractions import Fraction as F

from scripts.check_prefix_bridge001 import CONTEXTS,enumerated
from scripts.run_source_state_systems001 import ROOT,artifact,save_new


def support_counts(observed,units,free,known=()):
    """Count uniform dictionary completions supporting an unclosed prefix.

    Only units that match somewhere in the visible prefix matter. The last
    unit may cross the cut. Dense finite availability masks, no source scores.
    """
    if not observed or len(observed)>4 or not 0<=free<=23:
        raise ValueError('One-to-four visible glyphs and bounded free rows required')
    relevant = tuple(i for i,u in enumerate(units) if any(
        u[:min(len(u),len(observed)-pos)]==observed[pos:pos+min(len(u),len(observed)-pos)]
        for pos in range(len(observed))))
    if len(relevant)>13:
        raise MemoryError('Exact availability mask cap13; no approximation')
    bits = {code:1<<i for i,code in enumerate(relevant)}
    matches = [[(bits[code],min(len(observed),pos+len(units[code]))) for code in relevant
                if units[code][:min(len(units[code]),len(observed)-pos)]==
                    observed[pos:pos+min(len(units[code]),len(observed)-pos)]]
               for pos in range(len(observed))]
    def supported(mask):
        reach = {0}
        for pos in range(len(observed)):
            if pos in reach:
                reach.update(end for bit,end in matches[pos] if mask&bit)
        return int(len(observed) in reach)
    states = 1<<len(relevant)
    counts = [[supported(mask) for mask in range(states)]]
    other = len(units)-len(relevant)
    for _ in range(free):
        old = counts[-1]
        counts.append([other*old[mask]+sum(old[mask|(1<<j)] for j in range(len(relevant)))
                       for mask in range(states)])
    initial = 0
    for code in known:
        initial |= bits.get(code,0)
    return counts,bits,initial


def check():
    units = ((0,),(1,),(0,0),(0,1),(1,0),(1,1))
    counts_checked,path_checks,evidence_checks = 0,0,0
    for n in range(1,4):
        for observed in itertools.product(range(2),repeat=n):
            for partial in ((-1,-1),(0,-1),(3,-1),(0,1)):
                known = tuple(k for k in partial if k>=0)
                free = partial.count(-1)
                counts,bits,start = support_counts(observed,units,free,known)
                total = counts[free][start]
                keys,weights = [],[]
                for completion in itertools.product(range(6),repeat=free):
                    values = iter(completion)
                    key = tuple(next(values) if k<0 else k for k in partial)
                    mask = start
                    q = F(1)
                    for i,code in enumerate(completion):
                        following = mask|bits.get(code,0)
                        denominator = counts[free-i][mask]
                        if not denominator:
                            q = F(0)
                            break
                        q *= F(counts[free-i-1][following],denominator)
                        mask = following
                    valid = bool(enumerated(observed,key,0,CONTEXTS))
                    assert bool(counts[0][mask])==valid
                    assert q==(F(1,total) if valid else 0)
                    keys.append(key)
                    weights.append(q)
                    path_checks += 1
                assert total==sum(enumerated(observed,k,0,CONTEXTS)>0 for k in keys)
                counts_checked += 1
                if total:
                    assert sum(weights)==1
                    event_probability = F(total,6**free)
                    for context in range(3):
                        g = [enumerated(observed,k,context,CONTEXTS) for k in keys]
                        assert sum(q*event_probability*v for q,v in zip(weights,g,strict=True))==sum(g)/6**free
                        evidence_checks += 1
    # Mid-emission coverage retained; forcing a closed substring would fail.
    counts,_,start = support_counts((0,),units,0,(3,3))
    assert counts[0][start]==1
    assert enumerated((0,),(3,3),0,CONTEXTS)>0
    assert enumerated((0,),(3,3),0,CONTEXTS,closed=True)==0
    # Source positivity is essential for identifying inventory support with G>0.
    assert enumerated((1,),(0,1),0,((F(1),F(0)),)*3)==0
    large_units = tuple(u for length in (1,2) for u in itertools.product(range(6),repeat=length))
    large,bits,start = support_counts((0,1,2,3),large_units,23)
    assert len(bits)==13 and len(large)==24 and len(large[0])==8192
    probability = F(large[23][start],42**23)
    return {'status':'PASS_exact_observation_support_counts','count_profiles':counts_checked,
        'conditional_path_checks':path_checks,'source_evidence_checks':evidence_checks,
        'generic_four_glyph_example':{'prefix':[0,1,2,3],'free_rows':23,'units':42,
            'relevant_units':13,'availability_masks':8192,'count_layers':24,
            'supported_completion_count':str(large[23][start]),'prior_event_probability':str(probability)},
        'mid_emission_and_source_zero_controls':True,'empirical_model_or_sampler_calls':0,
        'scope':'Own finite ideal-law mathematics only; no new actual sampler, mixing or recovery claim',
        'inputs':[artifact(ROOT/p) for p in ('scripts/check_observation_initialization001.py',
            'scripts/check_prefix_bridge001.py')],'paid_spend_usd':0}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',action='store_true')
    args = parser.parse_args()
    result = check()
    if args.save:
        save_new(ROOT/'results/OBSERVATION-INITIALIZATION-THEORY-001/result.json',result)
    print(json.dumps(result,indent=2))
