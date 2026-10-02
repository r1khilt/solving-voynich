"""Independent gate arithmetic: missing results, dependence and thresholds."""

import copy

import pytest

from scripts import run_tempered_recovery001 as run
from scripts.summarize_tempered_recovery001 import independent_summary


def cells():
    out = []
    for case in range(8):
        for seed in run.SEEDS:
            for arm in run.ARMS:
                recovery = {'kind':'shuffle','accuracy_not_defined':True} if case%2 else {
                    'kind':'positive','selected_matches':5,'used_rows':10,
                    'selected_complete_used_key':False,'bank_contains_complete_used_key':False,
                    'edit_errors_with_failed_readings_counted_full_length':100,'true_letters':128,'exact_records':0}
                out.append({'case':case,'seed':seed,'arm':arm,'recovery':recovery,
                    'summary':{'status':'complete_tempered_particles','edges':100,
                        'log_evidence_estimate':-100-case+(seed-run.SEEDS[0]),'maximum_incremental_weight':.4}})
    return out


def test_independent_rollup_matches_fixed_design_and_qualification_thresholds():
    panel = cells()
    for c in panel:
        if c['arm']=='supported_refresh' and c['case']%2==0:
            c['recovery'].update(selected_matches=6,edit_errors_with_failed_readings_counted_full_length=90)
    panel[3]['recovery'].update(selected_complete_used_key=True,exact_records=1)
    result = independent_summary(panel)
    assert result==run.summarize(panel)
    assert result['combined_qualification_gate']
    assert result['arms']['supported_refresh']['true_letters']==1024
    assert result['arms']['supported_refresh']['used_row_denominator']==80
    weaker = copy.deepcopy(panel)
    weaker[3]['recovery']['selected_matches'] = 5
    assert not independent_summary(weaker)['exploratory_recovery_gate']


@pytest.mark.parametrize('case',[0,1])
def test_failed_calls_keep_denominators_and_do_not_create_evidence(case):
    panel = cells()
    failed = next(c for c in panel if c['case']==case and c['arm']=='supported_refresh')
    failed['summary'] = {'status':'call_work_cap','edges':run.CALL_EDGES+1}
    if case==0:
        failed['recovery'].update(selected_matches=None,selected_complete_used_key=False,
            bank_contains_complete_used_key=False,exact_records=0,
            edit_errors_with_failed_readings_counted_full_length=128)
    summary = independent_summary(panel)
    assert summary==run.summarize(panel)
    arm = summary['arms']['supported_refresh']
    assert arm['used_row_denominator']==80 and arm['true_letters']==1024
    assert arm['complete_calls']==15 and arm['call_work_caps']==1
    assert arm['seed_log_evidence_spreads'][case] is None
    assert arm['positive_minus_shuffle_log_evidence_margins'][0] is None
    assert not arm['calibration_gate'] and not arm['positive_shuffle_gate']
    assert arm['edit_errors']==(828 if case==0 else 800)


@pytest.mark.parametrize('change',['seed_spread','weight','null_margin'])
def test_evidence_or_weight_failures_remain_visible(change):
    panel = cells()
    if change=='seed_spread':
        panel[4]['summary']['log_evidence_estimate'] += 3
    elif change=='weight':
        panel[0]['summary']['maximum_incremental_weight'] = .50001
    else:
        panel[8]['summary']['log_evidence_estimate'] = -99
    result = independent_summary(panel)
    assert result==run.summarize(panel)
    arm = result['arms']['supported_prior']
    assert not (arm['positive_shuffle_gate'] if change=='null_margin' else arm['calibration_gate'])


def test_duplicate_or_missing_calls_and_hidden_failed_accuracy_are_refused():
    panel = cells()
    with pytest.raises(ValueError,match='64 distinct'):
        independent_summary(panel[:-1])
    with pytest.raises(ValueError,match='64 distinct'):
        independent_summary([*panel[:-1],panel[0]])
    panel[0]['summary'] = {'status':'compiler_cap','edges':0}
    with pytest.raises(AssertionError):
        independent_summary(panel)
