"""Terminal joint reading/used-dictionary independence-MH score helpers.

Unused iid-uniform rows are integrated out. A proposal's density must be that
of a stochastic COMPLETE action path under one frozen policy, not a key
marginal, greedy output law or success-conditioned/modified proposal. Failed
draws are self-loops. These helpers neither sample nor certify a supplied law.
"""

from __future__ import annotations

import math
from numbers import Real

from voynich.source_action_proposal import ReadingEnvironment, ReadingState


def _finite_real(value):
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)


def joint_reading_log_weight(environment, state, *, source_log_probability,
                             proposal_log_probability, temperature=1, uniform_row_prior=True):
    """Log [Q(texts) U**(-visited_rows) / q(action_path | observation)].

    Q includes every complete record's EOS and reset. Neither proposal nor
    source score is length-normalized. The deterministic environment schedule
    makes a compatible reading/used-key pair identify one complete action path.
    Only the terminal posterior target with iid-uniform row prior is supported;
    likelihood tempering over whole keys would need an additional marginal.
    Caller must verify both scores, policy identity and full proposal support.
    """
    if (not isinstance(environment, ReadingEnvironment) or not isinstance(state, ReadingState)
            or not _finite_real(temperature) or temperature != 1 or uniform_row_prior is not True
            or any(not _finite_real(v) or v > 0
                   for v in (source_log_probability, proposal_log_probability))):
        raise ValueError('Finite probability logs, terminal temperature and iid-uniform row prior required')
    environment.validate(state)
    if environment.selected_record(state) is not None:
        raise ValueError('Only a literally complete reading may receive a target/proposal weight')
    visited = sum(k >= 0 for k in state.key)
    value = float(source_log_probability)-visited*math.log(len(environment.pool))-float(proposal_log_probability)
    if not math.isfinite(value):
        raise ArithmeticError('Joint reading importance weight overflowed')
    return value


def independence_log_acceptance(old_log_weight, new_log_weight):
    """Stable log min(1, w_new/w_old); FAILED proposals bypass this as self-loops."""
    if any(not _finite_real(v) for v in (old_log_weight, new_log_weight)):
        raise ValueError('Finite old and new complete-state log weights required')
    difference = float(new_log_weight)-float(old_log_weight)
    if not math.isfinite(difference):
        raise ArithmeticError('Acceptance log-ratio overflowed')
    return min(0., difference)
