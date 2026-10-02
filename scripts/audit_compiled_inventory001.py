"""One full compiler/sampler replay and alternate stored-graph counting audit."""

import gc
import json
import math
import signal
import time
from fractions import Fraction

import numpy as np

from scripts.audit_censored_context001 import logical
from scripts.run_compiled_inventory001 import (
    ORDERS,OUT,NativeMarginal,array_identity,artifact,evaluate,guard,inputs,
    limit_resources,load_bound,load_source,metrics,resource_report,save_new,summarize,
)
from tests.test_inventory_bdd import literal_support


def verify_graph(value,query):
    """Iterative polynomial reference, not the compiler's recursive profile."""
    if value['status']=='compiler_cap':
        assert value['no_partial_count_or_sampler']
        assert not any(k in value for k in ('nodes','root','dictionary_count','keys'))
        return 0
    nodes,order = value['nodes'],value['order']
    assert sorted(order)==list(range(42)) and nodes[:2]==[[42,0,0],[42,1,1]]
    assert len({tuple(n) for n in nodes[2:]})==len(nodes)-2
    polynomials = [(0,),(1,)]
    def extended(node,start):
        skip = nodes[node][0]-start
        assert skip>=0
        base = polynomials[node]
        out = [0]*(len(base)+skip)
        for j in range(skip+1):
            factor = math.comb(skip,j)
            for i,c in enumerate(base):
                out[i+j] += factor*c
        return out
    for index,(var,lo,hi) in enumerate(nodes[2:],2):
        assert 0<=var<42 and 0<=lo<index and 0<=hi<index and lo!=hi
        assert var<min(nodes[lo][0],nodes[hi][0])
        left,right = extended(lo,var+1),extended(hi,var+1)
        out = [0]*max(len(left),len(right)+1)
        for i,c in enumerate(left):
            out[i] += c
        for i,c in enumerate(right):
            out[i+1] += c
        polynomials.append(tuple(out))
    root = value['root']
    assert type(root) is int and 0<=root<len(nodes)
    coefficients = extended(root,0)
    assert coefficients==value['coefficients']
    # Inclusion-exclusion independently counts onto maps from 23 labelled rows.
    weights = [b*sum((-1)**j*math.comb(k,j)*(k-j)**23 for j in range(k+1))
               for k,b in enumerate(coefficients)]
    total = sum(weights)
    assert list(map(str,weights))==value['cardinality_weights']
    assert str(total)==value['dictionary_count']
    assert str(Fraction(total,42**23))==value['prior_support_probability']
    units = [(c,) for c in range(6)]+[(a,b) for a in range(6) for b in range(6)]
    def accept(key):
        assert len(key)==23 and all(type(c) is int and 0<=c<42 for c in key)
        present,node = set(key),root
        while node>1:
            var,lo,hi = nodes[node]
            node = hi if order[var] in present else lo
        return node==1
    assert len(value['control_keys'])==64 and len(value['keys'])==32
    for key,observed in zip(value['control_keys'],value['control_supported'],strict=True):
        assert observed==accept(key)==literal_support(query['cipher'],units,set(key))
    for key in value['keys']:
        assert accept(key) and literal_support(query['cipher'],units,set(key))
    return len(value['keys'])


def audit():
    result = json.loads((OUT/'result.json').read_text())
    parent,benchmark,queries = inputs(result['freeze'])
    assert result['numpy_version']==np.__version__ and result['native_build']==benchmark['build']
    assert result['queries']==queries
    save_new(OUT/'audit-started.json',{'freeze':result['freeze'],'start_unix':time.time()})
    limit_resources(600,500)
    wall,cpu,cells,bulk,checked = time.monotonic(),time.process_time(),[],0,0
    scored,max_delta = 0,0.
    try:
        source,chosen = load_source()
        identity = array_identity(source)
        assert parent['source']==result['source']==chosen['counts']
        assert parent['source_arrays']==result['source_arrays']==identity
        native = NativeMarginal(source,benchmark['build'])
        assert len(result['workloads'])==8
        for index,manifest in enumerate(result['workloads']):
            cell = load_bound(manifest)
            assert (cell['case'],cell['order'])==(index//2,ORDERS[index%2])
            stored = load_bound(cell['output'])
            query = queries[cell['case']]
            replay = evaluate(query,cell['order'],native)
            value = replay.pop('value')
            assert value==stored
            assert logical(replay)==logical({k:cell[k] for k in replay})
            assert cell['summary']==metrics(value)
            checked += verify_graph(stored,query)
            if stored.get('source_check',{}).get('status')=='positive_scores_agree':
                scored += 1
                max_delta = max(max_delta,stored['source_check']['delta'])
            bulk += cell['output']['bytes']
            cells.append(cell)
            guard(wall,cpu,bulk)
            print(f"Audited case{cell['case']}-{cell['order']}: {stored['status']}",flush=True)
            gc.collect()
        assert result['summary']==summarize(cells) and result['bulk_bytes']==bulk
        assert array_identity(source)==identity
        assert result['resources']['wall_seconds']<=600 and result['resources']['cpu_seconds']<=500
        assert result['resources']['peak_rss_bytes']<=2*1024**3 and bulk<=128*1024**2
        save_new(OUT/'audit.json',{'status':'PASS_full_compiled_inventory_replay',
            'result':artifact(OUT/'result.json'),'calls':len(cells),'independent_supported_keys':checked,
            'independent_first_key_source_checks':scored,'maximum_source_log_delta':max_delta,
            'alternate_iterative_graph_polynomials':True,'same_author':True,
            'independent_agent_review':False,'replay_shares_compiler_sampler_native_backend':True,
            'resources':resource_report(wall,cpu),'paid_spend_usd':0})
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/'audit-failure.json',{'error':repr(exc),'audited_calls':len(cells),
            'no_retry':True,'resources':resource_report(wall,cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__=='__main__':
    audit()
