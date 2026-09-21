"""Real subprocess controls for the bounded parallel supervisor; no model training."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SPEC = importlib.util.spec_from_file_location('campaign', Path(__file__).resolve().parents[1]/'scripts/run_parallel_campaign.py')
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def plan(jobs):
    return {'campaign': 'test', 'max_seconds': 10, 'rss_ceiling_gib': 1, 'jobs': jobs}


def job(name, code, seconds=5):
    return {'name': name, 'command': [sys.executable, '-c', code], 'max_seconds': seconds}


def test_jobs_really_overlap_and_outputs_are_isolated(tmp_path):
    jobs = [job(name, "import time; print('ready', flush=True); time.sleep(.2); print('done')") for name in ['a', 'b']]
    report = campaign.run_campaign(plan(jobs), tmp_path/'out', enforce_clean=False, poll_seconds=.02)
    assert report['state'] == 'completed'
    assert max(r['started_elapsed_seconds'] for r in report['jobs'].values()) < min(r['finished_elapsed_seconds'] for r in report['jobs'].values())
    assert (tmp_path/'out/a.log').read_text() == 'ready\ndone\n'
    assert (tmp_path/'out/b.log').read_text() == 'ready\ndone\n'
    with pytest.raises(ValueError, match='empty'):
        campaign.run_campaign(plan(jobs), tmp_path/'out', enforce_clean=False)


def test_failed_job_does_not_cancel_independent_job(tmp_path):
    jobs = [job('failure', 'raise SystemExit(7)'), job('success', "print('complete')")]
    report = campaign.run_campaign(plan(jobs), tmp_path/'out', enforce_clean=False, poll_seconds=.02)
    assert report['state'] == 'incomplete'
    assert report['jobs']['failure']['returncode'] == 7
    assert report['jobs']['success']['state'] == 'completed'


def test_deadline_stops_own_process_and_preserves_report(tmp_path):
    jobs = [job('short', 'import time; time.sleep(30)', seconds=.1)]
    report = campaign.run_campaign(plan(jobs), tmp_path/'out', enforce_clean=False, poll_seconds=.02)
    assert report['jobs']['short']['state'] == 'job_deadline'
    assert report['jobs']['short']['returncode'] < 0
    assert json.loads((tmp_path/'out/status.json').read_text())['state'] == 'incomplete'


def test_memory_guard_is_scoped_and_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'process_group_rss', lambda groups: dict.fromkeys(groups, 2*2**30))
    report = campaign.run_campaign(plan([job('large', 'import time; time.sleep(30)')]), tmp_path/'out', enforce_clean=False, poll_seconds=.02)
    assert report['jobs']['large']['state'] == 'campaign_rss_ceiling'
    assert report['jobs']['large']['peak_rss_bytes'] == 2*2**30


@pytest.mark.parametrize('change', [
    {'max_seconds': 28801}, {'max_seconds': float('nan')}, {'rss_ceiling_gib': 49},
    {'jobs': [job('../unsafe', 'pass')]}, {'jobs': [job('same', 'pass')]*2},
    {'jobs': [{'name': 'shell', 'command': 'echo no', 'max_seconds': 1}]},
])
def test_invalid_plans_rejected(change):
    value = {**plan([job('a', 'pass')]), **change}
    with pytest.raises(ValueError):
        campaign.validate_plan(value)
