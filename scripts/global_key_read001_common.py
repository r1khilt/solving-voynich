"""Published terminal search banks admitted to a separate exposed reading phase."""
import json

from scripts.confirm002_common import NAMES, panel_inputs, observed
from scripts.run_global_key_search001 import PATHS as SEARCH_PATHS, OUT as FIT, admitted as search_admission
from scripts.run_key_bank_read002 import load_source
from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_latin_source_model001 import ROOT, artifact

EXP = 'GLOBAL-KEY-READ-001'
OUT, BULK = ROOT/'results'/EXP, ROOT/'outputs'/EXP
WORKERS = 4
PATHS = sorted(set(SEARCH_PATHS+[
    'scripts/global_key_read001_common.py', 'scripts/predict_global_key_read001.py',
    'scripts/audit_global_key_read001.py', 'scripts/evaluate_global_key_read001.py',
    'tests/test_global_key_read001.py', 'docs/experiments/GLOBAL-KEY-READ-001.md',
    'results/BLIND-CHANNEL-CONFIRM-002/evaluation.json',
    'results/BLIND-CHANNEL-CONFIRM-002/evaluation-audit.json',
]))


def terminal_inventory(campaign, started):
    """Reject incomplete, duplicated, reordered or inconsistent terminal panels."""
    if ([row['case'] for row in campaign['results']] != list(NAMES)
            or started['cases'] != list(NAMES) or started['workers'] != 8
            or started['freeze'] != campaign['freeze']):
        raise ValueError('Entire original terminal fitting queue required')
    for row in campaign['results']:
        for stage in ('fit', 'audit'):
            code = row[stage+'_returncode']
            if code is not None and (type(code) is not int):
                raise ValueError('Invalid process exit code')
            if code is None and stage+'_status' not in row and not (stage == 'audit' and row['fit_returncode'] != 0):
                raise ValueError('Nonterminal or unexplained missing process outcome')
        if row['fit_returncode'] != 0 and row['audit_returncode'] is not None:
            raise ValueError('Audit unexpectedly scheduled for failed fit')
    complete = all(row['fit_returncode'] == 0 and row['audit_returncode'] == 0 for row in campaign['results'])
    if campaign['status'] != ('complete' if complete else 'failures_retained'):
        raise ValueError('Campaign completion label contradicts process inventory')


def fit_status(freeze):
    require_frozen(freeze, PATHS)
    campaign = json.loads((FIT/'campaign.json').read_text())
    started = json.loads((FIT/'campaign-started.json').read_text())
    terminal_inventory(campaign, started)
    fit_panel, benchmark = search_admission(campaign['freeze'])
    panel, _ = panel_inputs(freeze)
    paths = [str((FIT/name).relative_to(ROOT)) for name in ('campaign.json', 'campaign-started.json')]
    admitted = {}
    for process in campaign['results']:
        name = process['case']
        process_path = FIT/f'{name}-process.json'
        if json.loads(process_path.read_text()) != process:
            raise ValueError('Terminal process record differs from campaign')
        paths.append(str(process_path.relative_to(ROOT)))
        paths += [str(p.relative_to(ROOT)) for p in FIT.glob(name+'*.json')]
        if process['fit_returncode'] == 0 and process['audit_returncode'] == 0:
            path = FIT/f'{name}.json'
            value = json.loads(path.read_text())
            audit = json.loads((FIT/f'{name}-audit.json').read_text())
            old_fit = ROOT/f'results/BLIND-CHANNEL-CONFIRM-002/{name}-fit.json'
            parent = ROOT/f'results/BLIND-CHANNEL-CONFIRM-002/{name}-parent.json'
            if (value['status'] != 'complete_fit_only' or value['case'] != name
                    or value['freeze'] != campaign['freeze']
                    or value['fit'] != fit_panel['cases'][name]['fit']
                    or value['fit'] != panel['cases'][name]['fit']
                    or value['old_fit'] != artifact(old_fit) or value['parent'] != artifact(parent)
                    or value['transfer_or_answer_files_opened'] is not False
                    or audit['status'] != 'PASS' or audit['case'] != name
                    or audit['result'] != artifact(path)
                    or audit['auditor'] != artifact(ROOT/'scripts/audit_global_key_search001.py')
                    or audit['full_fit_scores_accounted'] != value['unique_full_fit_scores_including_warm_and_neighborhood']
                    or audit['transfer_or_answer_files_opened'] is not False):
                raise ValueError('Search bank lacks matching complete fit-only audit')
            for field in ('reader_bank', 'old_bank', 'warm', 'search', 'progress'):
                if artifact(ROOT/value[field]['path']) != value[field]:
                    raise ValueError('Frozen fitting archive changed: '+field)
            admitted[name] = {**value, 'bank': value['reader_bank'], 'global_fit': artifact(path)}
        else:
            admitted[name] = None
    require_frozen(freeze, sorted(set(paths)))
    return panel, benchmark, admitted, campaign


__all__ = ['EXP', 'OUT', 'BULK', 'ROOT', 'NAMES', 'WORKERS', 'fit_status', 'observed', 'load_source']
