"""Fixed-temperature SMC over full-record support under the uniform key prior.

Weights use the old key before resampling/mutation. Structural rejection is
exact; source/graph/work failures propagate and never become zero likelihood.
Finite particles and floating arithmetic do not certify posterior coverage.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from voynich.coordinated_dictionary import OPERATION_TICKETS,pair_involution
from voynich.inventory_bdd import InventoryProfile
from voynich.source_key_particles import categorical


def supported_proposals(keys,profile,rng,*,kernel='supported_refresh'):
    """Fixed tickets; the refresh component alone draws prior | full support.

    All components have the same MH ratio L(new)^beta/L(old)^beta for the
    uniform free-row prior. Uniform supported refresh is symmetric on support.
    Retain identity moves, duplicate codes, and the original operation tickets.
    """
    if kernel not in ('row','coordinated','supported_refresh'):
        raise ValueError('Registered inventory kernel required')
    free,units = profile.free,profile.bdd.units
    candidate = keys.copy()
    counts = dict.fromkeys(sorted(set(OPERATION_TICKETS)),0)
    conditional_draws = 0
    if not len(free):
        return candidate,counts,conditional_draws
    for i,key in enumerate(keys):
        op = 'row' if kernel=='row' else OPERATION_TICKETS[int(rng.integers(32))]
        counts[op] += 1
        if op=='refresh' and kernel=='supported_refresh':
            candidate[i] = profile.sample(rng)
            conditional_draws += 1
        elif op=='refresh':
            candidate[i,free] = rng.integers(len(units),size=len(free))
        elif op in ('row','pair'):
            selected = rng.choice(free,size=1 if op=='row' else min(2,len(free)),replace=False)
            candidate[i,selected] = rng.integers(len(units),size=len(selected))
        elif len(free)>=2:
            a,b = rng.choice(free,size=2,replace=False)
            candidate[i,a],candidate[i,b] = pair_involution(key[a],key[b],units,op)
    return candidate,counts,conditional_draws


def run_tempered_inventory_smc(bridge,profile,*,betas,particles=32,seed=0,
                               mutations=4,kernel='supported_refresh',observe=None):
    """gamma_beta = prior * 1_Bfull * L^beta; Z_0 = prior(Bfull).

    The schedule is fixed before sampling. Always resample at each increment,
    then mutate using kernels invariant for that new temperature. No repair,
    zero substitution, adaptive schedule, caching or population refill.
    """
    if not isinstance(profile,InventoryProfile):
        raise ValueError('Qualified inventory profile required')
    schedule = tuple(betas)
    if (not 2<=len(schedule)<=257 or any(isinstance(b,(bool,np.bool_)) for b in schedule)
            or type(particles) is not int or not 1<=particles<=128
            or type(seed) is not int or seed<0 or type(mutations) is not int or not 0<=mutations<=8
            or kernel not in ('row','coordinated','supported_refresh')):
        raise ValueError('Bounded fixed inventory-tempering configuration required')
    schedule = tuple(float(b) for b in schedule)
    if (any(not math.isfinite(b) for b in schedule) or schedule[0]!=0 or schedule[-1]!=1
            or any(a>=b for a,b in zip(schedule,schedule[1:]))):
        raise ValueError('Fixed strictly increasing temperature schedule from0to1 required')
    bdd = profile.bdd
    if (not bdd.closed or bdd.records!=bridge.cipher or bdd.units!=bridge.units
            or len(profile.partial)!=bridge.rows or any(bridge.offsets)
            or not profile.total or not 0<profile.prior_event_probability<=1):
        raise ValueError('Whole closed records, matching all-row profile and positive support required')
    # No source-zero assumption may be silently inferred from graph support.
    for start in range(0,len(bridge.p),4096):
        rows = bridge.p[start:start+4096]
        if np.any(~np.isfinite(rows)) or np.any(rows<=0):
            raise ValueError('Full-support equivalence requires strictly positive source entries')
    rng = np.random.default_rng(seed)
    keys = np.array([profile.sample(rng) for _ in range(particles)],dtype=np.int32)
    conditional_draws = particles
    scores = bridge.log_values(keys,bridge.lengths,closed=True)
    if not np.all(np.isfinite(scores)):
        raise ArithmeticError('Supported initial key has invalid full likelihood; no repair')
    probability = profile.prior_event_probability
    initial_log_z = math.log(probability.numerator)-math.log(probability.denominator)
    log_z,trace = initial_log_z,[]
    initial_hash = hashlib.sha256(keys.tobytes()).hexdigest()
    for stage,(old_beta,beta) in enumerate(zip(schedule,schedule[1:])):
        increment = (beta-old_beta)*scores
        log_sum = float(np.logaddexp.reduce(increment))
        if not math.isfinite(log_sum):
            raise ArithmeticError('Invalid incremental normalizer')
        weights = np.exp(increment-log_sum)
        parents = categorical(weights,rng.random(particles))
        log_mean = log_sum-math.log(particles)
        log_z += log_mean
        keys,scores = keys[parents].copy(),scores[parents].copy()
        accepted = changed = rejected = proposed = refresh = 0
        operations = dict.fromkeys(sorted(set(OPERATION_TICKETS)),0)
        for _ in range(mutations if len(profile.free) else 0):
            candidates,counts,draws = supported_proposals(keys,profile,rng,kernel=kernel)
            conditional_draws += draws
            refresh += draws
            support = np.array([bdd.accepts(list(map(int,k))) for k in candidates])
            candidate_scores = np.full(particles,-math.inf)
            if np.any(support):
                evaluated = bridge.log_values(candidates[support],bridge.lengths,closed=True)
                if not np.all(np.isfinite(evaluated)):
                    raise ArithmeticError('Supported proposed key has invalid full likelihood; no substitution')
                candidate_scores[support] = evaluated
            with np.errstate(divide='ignore'):
                log_draw = np.log(rng.random(particles))
            accept = support&(log_draw<np.minimum(0.,beta*(candidate_scores-scores)))
            accepted += int(accept.sum())
            changed += int(np.count_nonzero(accept&np.any(candidates!=keys,axis=1)))
            rejected += int(np.count_nonzero(~support))
            proposed += particles
            keys[accept],scores[accept] = candidates[accept],candidate_scores[accept]
            for op,count in counts.items():
                operations[op] += count
        row = {'stage':stage,'previous_beta':old_beta,'beta':beta,
            'incremental_ess':float(1/np.square(weights).sum()),
            'maximum_incremental_weight':float(weights.max()),'parents':parents.tolist(),
            'distinct_selected_parents':int(len(np.unique(parents))),
            'distinct_keys_after_mutation':int(len(np.unique(keys,axis=0))),
            'accepted_proposals':accepted,'changed_proposals':changed,'proposals':proposed,
            'support_rejected_without_scoring':rejected,'supported_refresh_draws':refresh,
            'operation_counts':operations,'increment_log_mean':log_mean,
            'log_evidence_estimate':log_z,'key_bank_sha256':hashlib.sha256(keys.tobytes()).hexdigest()}
        trace.append(row)
        if observe is not None:
            observe(row)
    return {'status':'complete_tempered_particles','keys':keys,'key_log_likelihoods':scores,
        'log_evidence_estimate':log_z,'trace':trace,'tables':bridge.tables,'edges':bridge.edges,
        'initialization':{'prior_full_support_probability':str(probability),
            'initial_log_normalizer':initial_log_z,'initial_key_bank_sha256':initial_hash,
            'support_conditioned_dictionary_draws':conditional_draws,
            'integer_rejection_union_bound':f'{conditional_draws*(len(profile.free)+2)}/{2**64}'},
        'posterior_is_finite_approximation':True,'uniform_free_row_prior_only':True,
        'schedule_fixed_before_sampling':True}
