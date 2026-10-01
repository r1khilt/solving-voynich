"""Finite exact arithmetic witnesses; no empirical cipher panel or training."""
import itertools
import json
import math

from tests.test_source_bellman_guide import full_future, tail_for
from tests.test_source_state_lattice import source
from voynich.source_bellman_guide import actions, backup
from voynich.source_state_lattice import State


def check():
    p,goto=source()
    observed=[(g,) for g in range(2)]+list(itertools.product(range(2),repeat=2))
    checked,max_delta,witnesses=0,0.,{}
    for schedule in ("balanced","sequential"):
        for cipher in itertools.product(observed,repeat=2):
            root=State((0,0),(0,0),(-1,-1))
            states=sorted({root,*[s for s,_ in actions(root,cipher,p,goto,glyphs=2,rho=.25,schedule=schedule)]})
            tail=tail_for(cipher,p)
            for s in states:
                rational=full_future(s,cipher,p,goto)
                truth=float(rational)
                full=math.exp(backup(s,sum(map(len,cipher)),cipher,p,goto,tail,glyphs=2,rho=.25,schedule=schedule))
                delta=abs(full-truth)
                assert delta<=2e-14
                max_delta=max(max_delta,delta)
                iid=math.exp(tail(s))
                if abs(iid-truth)>1e-12:
                    key="iid_underestimates" if iid<truth else "iid_overestimates"
                    witness={"cipher":cipher,"schedule":schedule,"offsets":s.offsets,
                        "contexts":s.contexts,"key":s.key,"exact_future_rational":str(rational),
                        "absolute_iid_probability_error":abs(iid-truth),
                        "iid_future":iid,"one_backup_future":math.exp(backup(s,1,cipher,p,goto,tail,glyphs=2,rho=.25,schedule=schedule))}
                    if key not in witnesses or witness["absolute_iid_probability_error"]>witnesses[key]["absolute_iid_probability_error"]:
                        witnesses[key]=witness
                checked+=1
    assert set(witnesses)=={"iid_underestimates","iid_overestimates"}
    return {"status":"PASS_finite_mathematical_checks","prefix_states":checked,
        "maximum_full_depth_vs_rational_probability_delta":max_delta,"witnesses":witnesses,
        "rho":"1/4","alphabet_rows":2,"glyphs":2,"units":6,"empirical_calls":0,
        "recovery_qualification":False,"paid_spend_usd":0}


if __name__=="__main__":
    print(json.dumps(check(),indent=2,allow_nan=False))
