"""Post-outcome rational proof checks for cross-record shared-key correlation.

No actual corpus/source loading, fitted search, model prediction or training.
The context rows below are the old finite mathematical source fixture.
"""
from __future__ import annotations

import argparse
import itertools
import json
from fractions import Fraction as F

from scripts.run_source_bellman001 import OUT,save_new

ROWS=((F(1,3),F(2,3)),(F(3,4),F(1,4)),(F(1,5),F(4,5)))
UNITS=6
GEOMETRY=(F(1,4)*F(3,4))**2


def formula(a,b,partial,u,v):
    free=[r for r,k in enumerate(partial) if k<0]
    mean_a=sum((a[r] for r,k in enumerate(partial) if k==u),F(0))+sum((a[r] for r in free),F(0))/UNITS
    mean_b=sum((b[r] for r,k in enumerate(partial) if k==v),F(0))+sum((b[r] for r in free),F(0))/UNITS
    overlap=sum((a[r]*b[r] for r in free),F(0))
    covariance=overlap*(F(int(u==v),UNITS)-F(1,UNITS**2))
    return GEOMETRY*(mean_a*mean_b+covariance),GEOMETRY*mean_a*mean_b,covariance


def enumerate_full_keys_and_letters(a,b,partial,u,v):
    free=[r for r,k in enumerate(partial) if k<0]
    total=F(0)
    for codes in itertools.product(range(UNITS),repeat=len(free)):
        key=list(partial)
        for r,c in zip(free,codes,strict=True):
            key[r]=c
        for x,y in itertools.product(range(len(a)),repeat=2):
            if key[x]==u and key[y]==v:
                total+=GEOMETRY*a[x]*b[y]/UNITS**len(free)
    return total


def check():
    checked=0
    for ca,cb in itertools.product(range(3),repeat=2):
        a,b=ROWS[ca],ROWS[cb]
        for partial in itertools.product(range(-1,6),repeat=2):
            for u,v in itertools.product(range(2),repeat=2):
                exact,mean_product,covariance=formula(a,b,partial,u,v)
                assert exact==enumerate_full_keys_and_letters(a,b,partial,u,v)
                assert exact>=0 and mean_product>=0
                assert covariance>=0 if u==v else covariance<=0
                checked+=1
    same,mean,_=formula(ROWS[0],ROWS[0],(-1,-1),0,0)
    different,other_mean,_=formula(ROWS[0],ROWS[0],(-1,-1),0,1)
    assert same==F(17,4608) and different==F(1,2304) and mean==other_mean==F(1,1024)
    return {"status":"PASS_exact_rational_shared_key_moments","context_partial_key_observation_cases":checked,
        "same_glyph":{"true":str(same),"mean_product":str(mean),"ratio":str(same/mean)},
        "different_glyph":{"true":str(different),"mean_product":str(mean),"ratio":str(different/mean)},
        "formula":"rho^2*(1-rho)^2*(mu_A[u]*mu_B[v] + dot(p_A_free,p_B_free)*(delta_uv/U - 1/U^2))",
        "scope":"Exactly two unfinished records with one observed glyph each; fixed supplied contexts and partial dictionary; no full-text approximation guarantee.",
        "post_outcome_analysis":True,"empirical_panel_calls":0,"paid_spend_usd":0}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--save",action="store_true")
    args=parser.parse_args()
    result=check()
    if args.save:
        save_new(OUT/"shared-key-moments-post-outcome.json",result)
    print(json.dumps(result,indent=2,allow_nan=False))
